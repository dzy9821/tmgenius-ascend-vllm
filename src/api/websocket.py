"""
WebSocket 端点 /tuling/ast/v3 —— 核心处理逻辑。

处理流程：
  1. 连接 → 检查并发上限 → 启动握手超时
  2. status=0 → 校验 → 初始化会话 → 回复握手成功
  3. status=1 → 解码音频 → VAD → 若触发断句 → 异步后台 ASR+ITN → 推送结果
  4. status=2 → 刷空缓冲区 → 等待所有后台 ASR 任务完成 → 推送终态结果 → 断开

ASR 异步处理：VAD 触发断句后，ASR+ITN 推理通过 asyncio.create_task() 在后台
执行，不阻塞音频帧的持续接收。多个并发 ASR 任务通过 session.send_lock 保证
WebSocket 写入不会交错。结果的 segId 和时间戳是自包含的，客户端可按需重排序。

每个 segment 响应的 header.message 包含 JSON 格式的耗时信息：
  {"asr_ms": 123.4, "total_ms": 145.6}
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected

from src.api.connection_manager import connection_manager
from src.api.metrics import (
    asr_connections_current,
    asr_errors_total,
    asr_processing_latency_ms,
    asr_segments_total,
)
from src.api.session import ASRSession
from src.core.config import settings
from src.core.logging import get_logger, trace_id_var
from src.models.schemas import (
    CWItem,
    ClientMessage,
    ResponseHeader,
    ResponsePayloadWrapper,
    ResultPayload,
    ServerMessage,
    WSItem,
)
from src.services.asr_service import ASRError, ASRService, build_hotword_context
from src.services.itn_pool import ITNPool
from src.utils.audio import decode_base64_opus, decode_base64_pcm, samples_to_cs, samples_to_ms

logger = get_logger(__name__)

router = APIRouter()

# 服务实例（由 main.py 生命周期管理器初始化）
asr_service: ASRService = ASRService()
itn_pool: ITNPool = ITNPool()


@router.websocket("/tuling/ast/v3")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """ASR 实时流式转录 WebSocket 端点。"""

    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else 0
    logger.info(
        "New WebSocket connection attempt: client=%s:%s, active_slots=%d/%d",
        client_host, client_port,
        connection_manager._active_count, connection_manager._max_connections,
    )

    # ---- 并发控制 ----
    if not connection_manager.try_acquire():
        await websocket.close(code=1013, reason="Try Again Later")
        logger.warning(
            "Connection rejected: max connections reached, client=%s:%s",
            client_host, client_port,
        )
        return

    logger.info("Connection slot acquired: client=%s:%s", client_host, client_port)
    await websocket.accept()
    session = None
    connection_slot_released = False

    try:
        # ---- 握手阶段（带超时） ----
        session = await _handle_handshake(websocket)
        if session is None:
            connection_manager.release_slot()
            connection_slot_released = True
            await _close_connection(websocket)
            return

        # 注册连接
        connection_manager.register(session.sid, session.trace_id)
        asr_connections_current.inc()
        trace_id_var.set(session.trace_id)

        # 回复握手成功
        await _send_response(websocket, session, status=0, seg_id=0)
        session.set_streaming()

        # 处理握手帧中携带的首帧音频
        if session._first_audio_payload is not None:
            await _handle_audio_frame(websocket, session, session._first_audio_payload)
            session._first_audio_payload = None

        logger.info("Connection opened: sid=%s, trace=%s, biz_id=%s", session.sid, session.trace_id, session.biz_id)

        # ---- 流式处理循环 ----
        while True:
            raw = await websocket.receive_text()
            msg = ClientMessage.model_validate_json(raw)

            if msg.header.status == 1:
                await _handle_audio_frame(websocket, session, msg)

            elif msg.header.status == 2:
                logger.info(
                    "Client initiated graceful close: sid=%s, trace=%s, segs=%d",
                    session.sid,
                    session.trace_id,
                    session.seg_id,
                )
                await _handle_end_frame(websocket, session)
                await _close_connection(websocket, session)
                break

    except WebSocketDisconnect:
        logger.info(
            "Client disconnected (wire close, no status=2): sid=%s, trace=%s, biz_id=%s, segs=%d, state=%s",
            session.sid if session else "?",
            session.trace_id if session else "?",
            session.biz_id if session else "?",
            session.seg_id if session else 0,
            session.state.value if session else "no_session",
        )
    except asyncio.TimeoutError:
        logger.warning("Handshake timeout")
        asr_errors_total.labels(error_type="handshake_timeout").inc()
        await _close_connection(websocket, session)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        asr_errors_total.labels(error_type="internal").inc()
        try:
            await _send_error(websocket, session, str(exc), status=2)
        except Exception:
            pass
        await _close_connection(websocket, session)
    finally:
        if session:
            pending_tasks = len([t for t in session._pending_asr_tasks if not t.done()])
            logger.info(
                "Cleaning up session: sid=%s, state=%s, pending_asr_tasks=%d",
                session.sid, session.state.value, pending_tasks,
            )
            session.close()  # 取消后台 ASR 任务 + 从 VAD 批处理器注销
            connection_manager.unregister(session.sid)
            connection_slot_released = True
            asr_connections_current.dec()
            logger.info(
                "Session cleanup complete: sid=%s, remaining_slots=%d/%d",
                session.sid,
                connection_manager._active_count, connection_manager._max_connections,
            )
        elif not connection_slot_released:
            connection_manager.release_slot()
            connection_slot_released = True
            logger.info("Released slot for session-less connection")


# ============================================================
# 内部处理函数
# ============================================================


async def _handle_handshake(websocket: WebSocket) -> ASRSession | None:
    """等待并处理握手帧；除并发上限外不主动关闭连接。"""
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(),
            timeout=settings.HANDSHAKE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise

    msg = ClientMessage.model_validate_json(raw)
    logger.info(
        "Handshake frame received: status=%d, traceId=%s, bizId=%s",
        msg.header.status,
        msg.header.traceId,
        msg.header.bizId,
    )
    if msg.header.status != 0:
        logger.warning(
            "First message must be handshake (status=0), got status=%d, traceId=%s — connection will be closed",
            msg.header.status,
            msg.header.traceId,
        )
        return None

    session = ASRSession(
        trace_id=msg.header.traceId,
        biz_id=msg.header.bizId,
        app_id=msg.header.appId or "",
    )
    logger.info(
        "ASRSession created: sid=%s, trace=%s, biz_id=%s, vad_instance=%s",
        session.sid, session.trace_id, session.biz_id, id(session.vad),
    )

    # 追加客户端热词（与环境变量默认热词合并）
    if msg.payload and msg.payload.text and msg.payload.text.text:
        client_ctx = build_hotword_context(msg.payload.text.text)
        if client_ctx:
            base = build_hotword_context(settings.HOTWORDS)
            session.hotword_context = f"{base}\n{client_ctx}" if base else client_ctx

    # 握手帧可能同时携带首帧音频数据
    if msg.payload and msg.payload.audio:
        session._first_audio_payload = msg

    return session


async def _handle_audio_frame(
    websocket: WebSocket,
    session: ASRSession,
    msg: ClientMessage,
) -> None:
    """处理音频数据帧。

    VAD 触发断句后，ASR+ITN 以后台异步任务执行，不阻塞后续音频帧的接收。
    """
    if not msg.payload or not msg.payload.audio:
        return

    # 追加客户端热词（与环境变量默认热词合并）
    if msg.payload.text and msg.payload.text.text:
        client_ctx = build_hotword_context(msg.payload.text.text)
        if client_ctx:
            base = build_hotword_context(settings.HOTWORDS)
            session.hotword_context = f"{base}\n{client_ctx}" if base else client_ctx

    # 解码音频（PCM 或 Opus → int16）
    enc = msg.payload.audio.encoding
    if enc == "opus":
        pcm_int16 = decode_base64_opus(
            msg.payload.audio.audio, session.get_opus_decoder()
        )
    elif enc is None:
        pcm_int16 = decode_base64_pcm(msg.payload.audio.audio)
    else:
        logger.warning("Unsupported encoding %r, ignoring frame", enc)
        return

    # 喂入 VAD（通过全局批处理器异步推理）
    segments = await session.vad.feed_audio(pcm_int16)

    # 累计音频采样数，用于诊断网络延迟
    session._accumulated_audio_samples += len(pcm_int16)
    acc_audio_ms = samples_to_ms(session._accumulated_audio_samples)
    conn_ms = int((time.monotonic() - session._connection_start_time) * 1000)
    gap_ms = conn_ms - acc_audio_ms
    logger.debug(
        "Audio frame: sid=%s, frame_smps=%d, acc_audio_ms=%d, conn_ms=%d, gap_ms=%d",
        session.sid,
        len(pcm_int16),
        acc_audio_ms,
        conn_ms,
        gap_ms,
    )

    # 对每个触发的语音段，启动后台 ASR+ITN 任务（不阻塞音频接收）
    for seg in segments:
        task = asyncio.create_task(
            _process_segment(websocket, session, seg)
        )
        session.track_asr_task(task)


async def _handle_end_frame(websocket: WebSocket, session: ASRSession) -> None:
    """处理结束帧：刷空 VAD 缓冲区，等待所有 ASR 任务完成，合并 flush 文本后推送终态。"""
    session.set_closing()

    # 强制刷出残余音频（如有，标记为 final，与 status=2 捆绑发送）
    seg = session.vad.flush()
    if seg is not None:
        task = asyncio.create_task(
            _process_segment(websocket, session, seg, is_final=True)
        )
        session.track_asr_task(task)

    # 等待所有后台 ASR 任务完成，确保结果全部推送后再发终态
    await session.wait_pending_asr()

    # 按终态合并规则组装 status=2 响应
    async with session.send_lock:
        if session._mid_stream_segment_sent:
            # 有中间结果：以上一句 JSON 为基础合并 flush 文本
            data = json.loads(session._last_sent_result_json)
            data["header"]["status"] = 2
            if session._final_result_json is not None:
                flush_data = json.loads(session._final_result_json)
                # 拼接文本
                data["payload"]["result"]["ws"][0]["cw"][0]["w"] += (
                    flush_data["payload"]["result"]["ws"][0]["cw"][0]["w"]
                )
                # 延伸结束时间
                data["payload"]["result"]["ed"] = flush_data["payload"]["result"]["ed"]
                data["payload"]["result"]["ws"][0]["cw"][0]["we"] = (
                    flush_data["payload"]["result"]["ws"][0]["cw"][0]["we"]
                )
            logger.info(
                "End frame merge: seg_id=%s, text=%s",
                data["payload"]["result"]["segId"],
                data["payload"]["result"]["ws"][0]["cw"][0]["w"],
            )
            await websocket.send_text(json.dumps(data, ensure_ascii=False))
        elif session._final_result_json is not None:
            # 全程一段（R3）：flush 结果直接作为 status=2 发送
            data = json.loads(session._final_result_json)
            data["header"]["status"] = 2
            logger.info(
                "End frame R3: text=%s",
                data["payload"]["result"]["ws"][0]["cw"][0]["w"],
            )
            await websocket.send_text(json.dumps(data, ensure_ascii=False))
        else:
            # 无任何识别结果
            logger.info("End frame: no text to send")
            await _send_response(websocket, session, status=2, seg_id=0)


async def _close_connection(
    websocket: WebSocket,
    session: ASRSession | None = None,
) -> None:
    """服务端主动关闭 WebSocket 连接，避免无限等待客户端断开。"""
    try:
        await asyncio.wait_for(websocket.close(), timeout=3.0)
    except Exception:
        pass


async def _process_segment(
    websocket: WebSocket,
    session: ASRSession,
    seg: dict,
    is_final: bool = False,
) -> None:
    """对一个语音段执行 ASR → ITN → 推送结果（后台任务）。

    此函数作为 asyncio.Task 在后台运行，不阻塞主循环的音频接收。
    通过 session.send_lock 保证多个并发任务的 WebSocket 写入互斥。
    常规段以 status=1 发送并记录快照；is_final=True 的 flush 段仅保存
    识别结果到 session，由 _handle_end_frame 合并进上一句后以 status=2 发送。
    """
    t0 = time.monotonic()
    seg_id = session.next_seg_id()

    audio_int16 = seg["audio"]
    start_sample = seg["start_sample"]
    end_sample = seg["end_sample"]

    try:
        # ASR 推理（单独计时）
        t_asr_start = time.monotonic()
        raw_text = await asr_service.recognize(
            audio_int16,
            sr=16000,
            context=session.hotword_context,
        )
        t_asr_end = time.monotonic()
        asr_ms = (t_asr_end - t_asr_start) * 1000

        # ITN 后处理（通过多进程池）
        final_text = await itn_pool.normalize(raw_text)

        total_ms = (time.monotonic() - t0) * 1000

        # 构建结果并推送
        bg_ms = samples_to_ms(start_sample)
        ed_ms = samples_to_ms(end_sample)
        bg_cs = samples_to_cs(start_sample)
        ed_cs = samples_to_cs(end_sample)

        ws_item = WSItem(
            bg=bg_cs,
            cw=[
                CWItem(
                    w=final_text,
                    wp="n",
                    wb=bg_cs,
                    we=ed_cs,
                )
            ],
        )

        result = ResultPayload(
            segId=seg_id,
            bg=bg_ms,
            ed=ed_ms,
            msgtype="sentence",
            ws=[ws_item],
        )

        # 在 header.message 中附带耗时 JSON，客户端可解析
        timing_msg = json.dumps({
            "asr_ms": round(asr_ms, 1),
            "total_ms": round(total_ms, 1),
        })

        response = ServerMessage(
            header=ResponseHeader(
                code=0,
                message=timing_msg,
                sid=session.sid,
                traceId=session.trace_id,
                status=1,  # 始终以 status=1 发送，终态由 _handle_end_frame 发送
            ),
            payload=ResponsePayloadWrapper(result=result),
        )

        result_json = response.model_dump_json()

        if is_final:
            # 暂存 flush 段完整 JSON，推进序号但不发送
            session._final_result_json = result_json
            await session.push_result_in_order(websocket, seg_id, "")
        else:
            await session.push_result_in_order(websocket, seg_id, result_json)
            # 记录上一句快照，供终态合并使用（多任务并发时只保留最大 segId）
            if seg_id >= session._last_sent_seg_id:
                session._last_sent_seg_id = seg_id
                session._last_sent_result_json = result_json
                session._mid_stream_segment_sent = True

        asr_processing_latency_ms.observe(total_ms)
        asr_segments_total.inc()

        audio_ms = len(audio_int16) / 16.0
        conn_ms_at_spawn = int((t0 - session._connection_start_time) * 1000)
        logger.info(
            "Segment processed: seg_id=%d, text=%s, audio=%.0fms, pos=[%d-%d]ms, conn=%dms, asr=%.0fms, total=%.0fms",
            seg_id,
            final_text,
            audio_ms,
            bg_ms,
            ed_ms,
            conn_ms_at_spawn,
            asr_ms,
            total_ms,
        )

    except (WebSocketDisconnect, ClientDisconnected):
        logger.info(
            "Client disconnected while sending segment: sid=%s, seg_id=%d",
            session.sid,
            seg_id,
        )
        # 不 re-raise：后台任务中的异常不应传播到主循环
    except asyncio.CancelledError:
        logger.debug("ASR task cancelled: sid=%s, seg_id=%d", session.sid, seg_id)
    except ASRError as exc:
        logger.error("vLLM error on segment %d: %s", seg_id, exc)
        asr_errors_total.labels(error_type="asr_inference").inc()
        try:
            await session.push_result_in_order(websocket, seg_id, "")
            async with session.send_lock:
                await _send_error(
                    websocket,
                    session,
                    f"vLLM returned {exc.status_code}: {exc}",
                )
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Error processing segment %d: %s", seg_id, exc)
        asr_errors_total.labels(error_type="asr_inference").inc()
        try:
            # 即使处理失败，也必须推送空结果以推进序号，否则后续所有结果将永久卡在缓冲区
            await session.push_result_in_order(websocket, seg_id, "")
            async with session.send_lock:
                await _send_error(websocket, session, f"ASR error: {exc}")
        except Exception:
            pass


async def _send_response(
    websocket: WebSocket,
    session: ASRSession,
    status: int,
    seg_id: int,
    text: str = "",
) -> None:
    """发送一个简单的状态响应。"""
    result = ResultPayload(segId=seg_id, msgtype="sentence")
    if text:
        result.ws = [WSItem(cw=[CWItem(w=text)])]

    response = ServerMessage(
        header=ResponseHeader(
            code=0,
            message="success",
            sid=session.sid if session else "",
            traceId=session.trace_id if session else "",
            status=status,
        ),
        payload=ResponsePayloadWrapper(result=result),
    )
    await websocket.send_text(response.model_dump_json())


async def _send_error(
    websocket: WebSocket,
    session: ASRSession | None,
    detail: str,
    status: int = 1,
) -> None:
    """推送错误消息。status 默认为 1（segment 级错误），主循环异常时传 2（会话终结）。"""
    error_resp = {
        "header": {
            "code": -1,
            "message": detail,
            "sid": session.sid if session else "",
            "traceId": session.trace_id if session else "",
            "status": status,
        },
    }
    await websocket.send_text(json.dumps(error_resp, ensure_ascii=False))
