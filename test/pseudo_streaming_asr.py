"""
伪流式 ASR 测试 —— 滑动窗口模拟实时流式识别（异步版）。

按真实时间节奏每隔 step 秒发出一个窗口请求（不等前一个返回），
结果按实际返回顺序展示。

用法：
    python test/pseudo_streaming_asr.py <wav_file> [options]

示例：
    python test/pseudo_streaming_asr.py 120报警电话16k.wav
    python test/pseudo_streaming_asr.py audio.wav --window 3.0 --step 0.5
    python test/pseudo_streaming_asr.py audio.wav --base-url http://10.0.0.1:28856/v1
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import time
import base64
import re

import numpy as np

# ============================================================
# 音频工具
# ============================================================


def load_wav(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载 WAV 文件，返回 float32 音频和采样率。"""
    import soundfile as sf

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 简单重采样
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sr

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, sr


def encode_audio_to_data_url(audio_f32: np.ndarray, sr: int) -> str:
    """将 float32 音频编码为 data:audio/wav;base64,... 格式。"""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio_f32, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def clean_asr_output(text: str) -> str:
    """清洗 ASR 模型输出，移除 <asr_text> 等标记。"""
    text = text.strip()
    if "<asr_text>" not in text:
        return text
    parts = re.split(r"(?:language\s+[^\s<]+)?<asr_text>", text, flags=re.IGNORECASE)
    return "".join(part.strip() for part in parts if part.strip()).strip()


# ============================================================
# ASR 调用（异步版本，使用 AsyncOpenAI）
# ============================================================


def create_async_asr_client(base_url: str, api_key: str = "EMPTY"):
    """创建异步 OpenAI 兼容客户端。"""
    import httpx
    from openai import AsyncOpenAI

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)

    http_client = httpx.AsyncClient(trust_env=False)
    return AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)


async def asr_recognize(
    client,
    audio_f32: np.ndarray,
    sr: int,
    model: str,
    context: str = "",
) -> str:
    """异步调用 vLLM ASR 接口识别音频。"""
    data_url = encode_audio_to_data_url(audio_f32, sr)

    messages: list[dict] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": data_url}},
            ],
        }
    )

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return clean_asr_output(content if isinstance(content, str) else str(content))


# ============================================================
# 伪流式测试主逻辑（异步版）
# ============================================================


async def run_pseudo_streaming(
    wav_path: str,
    base_url: str = "http://localhost:28856/v1",
    api_key: str = "EMPTY",
    model: str = "Qwen3-ASR-0.6B",
    step: float = 0.4,
    window: float = 2.0,
    context: str = "",
) -> None:
    """
    滑动窗口伪流式 ASR 测试（异步版）。

    按真实时间节奏每隔 step 秒发出请求，不等前一个返回。
    结果按实际返回先后顺序展示。
    """
    audio, sr = load_wav(wav_path, target_sr=16000)
    total_duration = len(audio) / sr
    step_samples = int(sr * step)
    window_samples = int(sr * window)
    total_steps = (len(audio) + step_samples - 1) // step_samples

    print("=" * 74)
    print("  伪流式 ASR 测试（滑动窗口 · 异步实时调度）")
    print("=" * 74)
    print(f"  音频文件  : {wav_path}")
    print(f"  音频时长  : {total_duration:.2f}s")
    print(f"  采样率    : {sr} Hz")
    print(f"  窗口大小  : {window}s")
    print(f"  步进间隔  : {step}s")
    print(f"  重叠比例  : {(window - step) / window * 100:.0f}%")
    print(f"  总推理次数: {total_steps}")
    print(f"  模型      : {model}")
    print(f"  API 地址  : {base_url}")
    if context:
        print(f"  热词/提示 : {context}")
    print("=" * 74)
    print()

    client = create_async_asr_client(base_url=base_url, api_key=api_key)

    results: list[dict] = []
    results_lock = asyncio.Lock()
    t_global_start = time.monotonic()

    async def _recognize_one(step_id: int):
        """单个窗口的识别任务，完成后立即打印。"""
        window_end_sample = min((step_id + 1) * step_samples, len(audio))
        window_start_sample = max(0, window_end_sample - window_samples)
        window_time_start = window_start_sample / sr
        window_time_end = window_end_sample / sr
        window_dur = window_time_end - window_time_start
        time_label = f"{window_time_start:.2f}s → {window_time_end:.2f}s"
        audio_window = audio[window_start_sample:window_end_sample]

        t_send = time.monotonic() - t_global_start
        t_start = time.monotonic()
        try:
            text = await asr_recognize(
                client=client,
                audio_f32=audio_window,
                sr=sr,
                model=model,
                context=context,
            )
        except Exception as exc:
            text = f"[ERROR: {exc}]"
        t_elapsed = (time.monotonic() - t_start) * 1000
        t_recv = time.monotonic() - t_global_start

        result = {
            "step_id": step_id + 1,
            "window_range": time_label,
            "window_duration_ms": int(window_dur * 1000),
            "asr_latency_ms": round(t_elapsed, 1),
            "send_time": round(t_send, 3),
            "recv_time": round(t_recv, 3),
            "raw_text": text,
        }

        async with results_lock:
            arrival_idx = len(results) + 1
            results.append(result)
            print(
                f"  [返回 {arrival_idx:3d}/{total_steps}]  "
                f"步号={step_id + 1:3d}  "
                f"窗口={time_label:>20s}  "
                f"发送={t_send:6.2f}s  "
                f"返回={t_recv:6.2f}s  "
                f"耗时={t_elapsed:7.1f}ms"
            )
            print(f"             识别文本: {text}")

    # 按真实时间调度：每隔 step 秒发射一个任务
    tasks: list[asyncio.Task] = []
    for i in range(total_steps):
        scheduled_time = i * step
        now = time.monotonic() - t_global_start
        wait = scheduled_time - now
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(_recognize_one(i)))

    await asyncio.gather(*tasks)

    t_total_elapsed = (time.monotonic() - t_global_start) * 1000

    # ---- 汇总报告 ----
    print()
    print("=" * 74)
    print("  汇总报告")
    print("=" * 74)

    latencies = [r["asr_latency_ms"] for r in results]
    valid = sum(1 for r in results if not r["raw_text"].startswith("[ERROR"))

    print(f"  总耗时      : {t_total_elapsed:.0f}ms ({t_total_elapsed / 1000:.2f}s)")
    print(f"  音频时长    : {total_duration:.2f}s")
    print(f"  RTF         : {t_total_elapsed / 1000 / total_duration:.2f}")
    print(f"  总推理次数  : {total_steps}")
    print(f"  成功识别    : {valid}/{total_steps}")
    print(f"  平均延迟    : {sum(latencies) / len(latencies):.1f}ms")
    print(f"  最小延迟    : {min(latencies):.1f}ms")
    print(f"  最大延迟    : {max(latencies):.1f}ms")

    # 最大同时在飞请求数
    events = []
    for r in results:
        events.append((r["send_time"], +1))
        events.append((r["recv_time"], -1))
    events.sort()
    max_inflight = 0
    current = 0
    for _, delta in events:
        current += delta
        max_inflight = max(max_inflight, current)
    print(f"  最大并发    : {max_inflight}")
    print()
    print("=" * 74)

    # ---- 逐步明细（按返回顺序） ----
    print()
    print("  逐步明细（按实际返回顺序）:")
    header = (
        f"  {'序号':>4s}  {'步号':>4s}  {'窗口范围':>20s}  "
        f"{'发送':>7s}  {'返回':>7s}  {'延迟ms':>7s}  识别文本"
    )
    print(header)
    print("  " + "-" * 85)
    for idx, r in enumerate(results, 1):
        print(
            f"  {idx:4d}  "
            f"{r['step_id']:4d}  "
            f"{r['window_range']:>20s}  "
            f"{r['send_time']:6.2f}s  "
            f"{r['recv_time']:6.2f}s  "
            f"{r['asr_latency_ms']:7.1f}  "
            f"{r['raw_text']}"
        )
    print()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="伪流式 ASR 测试：滑动窗口 + 异步实时调度，结果按返回顺序展示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认：2s 窗口，0.4s 步进
  python test/pseudo_streaming_asr.py 120报警电话16k.wav

  # 自定义窗口和步进
  python test/pseudo_streaming_asr.py audio.wav --window 3.0 --step 0.5

  # 自定义 API 地址
  python test/pseudo_streaming_asr.py audio.wav --base-url http://10.0.0.1:28856/v1
        """,
    )
    parser.add_argument(
        "wav", metavar="WAV_FILE",
        help="音频文件路径（WAV 格式，建议 16kHz 单声道）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_API_BASE", "http://localhost:28856/v1"),
        help="vLLM API 地址 (默认: http://localhost:28856/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API Key (默认: EMPTY)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL_NAME", "Qwen3-ASR-0.6B"),
        help="模型名称 (默认: Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--step", type=float, default=0.4,
        help="步进间隔秒数 (默认: 0.4)",
    )
    parser.add_argument(
        "--window", type=float, default=2.0,
        help="窗口大小秒数 (默认: 2.0)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="热词/系统提示词，如 '热词：张三丰、武当山'",
    )

    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print(f"ERROR: 文件不存在: {args.wav}")
        sys.exit(1)

    if args.window < args.step:
        print(f"ERROR: 窗口大小 ({args.window}s) 不能小于步进间隔 ({args.step}s)")
        sys.exit(1)

    asyncio.run(
        run_pseudo_streaming(
            wav_path=args.wav,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            step=args.step,
            window=args.window,
            context=args.context,
        )
    )


if __name__ == "__main__":
    main()
