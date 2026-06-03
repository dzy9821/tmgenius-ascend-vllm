"""
ASR 会话状态 management —— 每个 WebSocket 连接对应一个 ASRSession。
"""

from __future__ import annotations

import asyncio
import enum
import random
import string
import time

import numpy as np

from src.core.config import settings
from src.services.asr_service import build_hotword_context
from src.services.vad_service import TenVADSession
from src.utils.audio import OpusDecoder


def _generate_sid() -> str:
    """生成会话 ID，格式 AST_XXXXXXXXXXXX。"""
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=13))
    return f"AST_{suffix}"


class SessionState(enum.Enum):
    HANDSHAKING = "handshaking"
    STREAMING = "streaming"
    CLOSING = "closing"


class ASRSession:
    """
    单个 WebSocket 连接的会话上下文。

    维护段序号、时间偏移、热词等。
    每个 Session 持有独立的 TenVADSession 实例，连接创建时初始化，
    连接关闭时通过 close() 方法显式释放。

    ASR 推理采用异步后台任务模式：VAD 触发断句后，ASR+ITN 处理通过
    asyncio.create_task() 在后台执行，不阻塞音频帧的持续接收与 VAD 处理。
    _send_lock 保证多个并发 ASR 任务推送结果时不会交错写入 WebSocket。
    """

    def __init__(self, trace_id: str, biz_id: str, app_id: str = "") -> None:
        self.sid: str = _generate_sid()
        self.trace_id: str = trace_id
        self.biz_id: str = biz_id
        self.app_id: str = app_id
        self.state: SessionState = SessionState.HANDSHAKING

        # 段序号（每次断句 +1）
        self.seg_id: int = 0

        # 热词上下文（默认从环境变量 HOTWORDS 读取，客户端可追加）
        self.hotword_context: str = build_hotword_context(settings.HOTWORDS)

        # 每连接独立的 VAD 会话
        self.vad: TenVADSession = TenVADSession(sid=self.sid)

        # Opus 解码器（延迟创建，仅 encoding=opus 的连接需要）
        self._opus_decoder: OpusDecoder | None = None

        # ---- 异步 ASR 任务管理 ----
        # WebSocket 发送锁：防止多个并发 ASR 后台任务同时写入 WebSocket
        self._send_lock: asyncio.Lock = asyncio.Lock()
        # 活跃 ASR 后台任务列表
        self._pending_asr_tasks: list[asyncio.Task] = []

        # ---- 结果顺序保证 ----
        self._next_send_seg_id: int = 0
        self._result_buffer: dict[int, str] = {}
        self._final_result_json: str | None = None  # flush 段暂存，与 status=2 捆绑发送

        # ---- 握手帧携带的首帧音频 ----
        self._first_audio_payload = None  # type: ignore

        # ---- 伪流式 Progressive 状态 ----
        self._progressive_seg_id: int = 0                      # progressive 独立 segId 计数器
        self._progressive_task: asyncio.Task | None = None      # 当前进行中的 progressive 任务
        self._progressive_audio_frames: list[np.ndarray] = []   # 自上次发送以来累积的语音帧（片段模式，非全量）
        self._progressive_total_samples: int = 0                # 缓冲区内累计采样数
        self._progressive_start_sample: int = 0                 # 缓冲区首帧对应的全局采样偏移
        self._progressive_last_send_time: float = 0.0           # 上一次发起 progressive 的时间（0 = 未开始）
        self._progressive_accumulated_text: str = ""             # 当前语音段内已识别文本的累加结果

        # ---- 音频到达延时诊断 ----
        self._connection_start_time: float = 0.0  # 流式开始时刻 (time.monotonic)
        self._accumulated_audio_samples: int = 0  # 累计收到的音频采样数

    async def push_result_in_order(self, websocket, seg_id: int, response_json: str) -> None:
        """保证按 seg_id 顺序推送结果，解决短句先于长句返回导致的乱序问题。"""
        async with self._send_lock:
            self._result_buffer[seg_id] = response_json
            # 当等待的下一个段号已经准备好时，依次全部发出
            while self._next_send_seg_id in self._result_buffer:
                msg = self._result_buffer.pop(self._next_send_seg_id)
                if msg:  # 空字符串代表该段处理失败，仅推进序号不发送内容
                    await websocket.send_text(msg)
                self._next_send_seg_id += 1

    @property
    def send_lock(self) -> asyncio.Lock:
        """WebSocket 发送锁，多个并发 ASR 任务共享。"""
        return self._send_lock

    def track_asr_task(self, task: asyncio.Task) -> None:
        """注册一个 ASR 后台任务，并清理已完成的旧任务。"""
        # 清理已完成的任务，避免列表无限增长
        self._pending_asr_tasks = [
            t for t in self._pending_asr_tasks if not t.done()
        ]
        self._pending_asr_tasks.append(task)

    async def wait_pending_asr(self) -> None:
        """等待所有活跃 ASR 后台任务完成（用于发送终态前的同步）。"""
        if self._pending_asr_tasks:
            await asyncio.gather(*self._pending_asr_tasks, return_exceptions=True)
            self._pending_asr_tasks.clear()

    def cancel_pending_asr(self) -> None:
        """取消所有活跃 ASR 后台任务（连接异常关闭时调用）。"""
        for task in self._pending_asr_tasks:
            if not task.done():
                task.cancel()
        self._pending_asr_tasks.clear()

    def close(self) -> None:
        """释放资源：取消后台任务 + 取消 progressive + 从 VAD 注销 + 销毁 Opus 解码器。"""
        self.cancel_pending_asr()
        self.cancel_progressive()
        self.vad.close()
        if self._opus_decoder is not None:
            self._opus_decoder.close()
            self._opus_decoder = None

    # ---- 伪流式 Progressive 状态管理 ----

    def append_progressive_audio(self, frame: np.ndarray, frame_start_sample: int) -> None:
        """追加一帧到 progressive 片段缓冲。

        当缓冲区为空时（新语音段开始或上次发送后重置），自动记录起始时间和采样位置，
        解决首次 progressive 因 _progressive_last_send_time=0 立即触发的问题。
        """
        if not self._progressive_audio_frames:
            self._progressive_last_send_time = time.monotonic()
            self._progressive_start_sample = frame_start_sample
        self._progressive_audio_frames.append(frame)
        self._progressive_total_samples += len(frame)

    def pop_progressive_audio(self) -> tuple[np.ndarray, int, int]:
        """取出当前累积的片段音频，返回 (audio_int16, start_sample, end_sample)，并重置缓冲区。"""
        if not self._progressive_audio_frames:
            return np.array([], dtype=np.int16), 0, 0
        audio = np.concatenate(self._progressive_audio_frames)
        start_sample = self._progressive_start_sample
        end_sample = start_sample + self._progressive_total_samples
        self._progressive_audio_frames = []
        self._progressive_total_samples = 0
        self._progressive_start_sample = 0
        self._progressive_last_send_time = time.monotonic()
        return audio, start_sample, end_sample

    def next_progressive_seg_id(self) -> int:
        """获取并递增 progressive 独立 segId。"""
        current = self._progressive_seg_id
        self._progressive_seg_id += 1
        return current

    def cancel_progressive(self) -> None:
        """取消进行中的 progressive 任务，清空片段缓冲区，重置 segId。

        VAD 断句时和连接关闭时调用，确保新语音段从头开始。
        """
        if self._progressive_task is not None and not self._progressive_task.done():
            self._progressive_task.cancel()
        self._progressive_task = None
        self._progressive_audio_frames = []
        self._progressive_total_samples = 0
        self._progressive_start_sample = 0
        self._progressive_last_send_time = 0.0
        self._progressive_seg_id = 0
        self._progressive_accumulated_text = ""

    def next_seg_id(self) -> int:
        """获取当前段号并递增。"""
        current = self.seg_id
        self.seg_id += 1
        return current

    def set_streaming(self) -> None:
        self.state = SessionState.STREAMING
        self._connection_start_time = time.monotonic()

    def get_opus_decoder(self) -> OpusDecoder:
        """获取或延迟创建当前连接的 Opus 解码器。"""
        if self._opus_decoder is None:
            self._opus_decoder = OpusDecoder()
        return self._opus_decoder

    def set_closing(self) -> None:
        self.state = SessionState.CLOSING
