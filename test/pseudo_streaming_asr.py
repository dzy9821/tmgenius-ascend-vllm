"""
伪流式 ASR 测试 —— 模拟实时流式识别。

将音频按 0.4 秒间隔分段，每次将 **累积到当前时刻的所有音频** 送入 ASR 推理，
观察识别结果随音频增长的变化过程。

用法：
    python test/pseudo_streaming_asr.py <wav_file> [options]

示例：
    python test/pseudo_streaming_asr.py 120报警电话16k.wav
    python test/pseudo_streaming_asr.py audio.wav --base-url http://127.0.0.1:15002/v1
    python test/pseudo_streaming_asr.py audio.wav --chunk-mode incremental --interval 0.2

模式说明：
    cumulative  (默认) 每次发送从头到当前时刻的累积音频，观察增量识别变化
    incremental         每次只发送当前 0.4s 片段，独立识别每段
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import base64
import re
from typing import Optional

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
# ASR 调用（复用正式代码的 vLLM OpenAI 兼容接口方式）
# ============================================================


def create_asr_client(base_url: str, api_key: str = "EMPTY"):
    """创建 OpenAI 兼容客户端。"""
    import httpx
    from openai import OpenAI

    # 清除代理环境变量
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)

    http_client = httpx.Client(trust_env=False)
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)


def asr_recognize(
    client,
    audio_f32: np.ndarray,
    sr: int,
    model: str,
    context: str = "",
) -> str:
    """调用 vLLM ASR 接口识别音频。"""
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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return clean_asr_output(content if isinstance(content, str) else str(content))


# ============================================================
# 伪流式测试主逻辑
# ============================================================


def run_pseudo_streaming(
    wav_path: str,
    base_url: str = "http://127.0.0.1:15002/v1",
    api_key: str = "EMPTY",
    model: str = "Qwen3-ASR-1.7B",
    interval: float = 0.4,
    context: str = "",
    chunk_mode: str = "cumulative",
) -> None:
    """
    伪流式 ASR 测试。

    Args:
        wav_path: WAV 文件路径
        base_url: vLLM API 地址
        api_key: API Key
        model: 模型名称
        interval: 分段间隔（秒），默认 0.4s
        context: 热词/系统提示词
        chunk_mode: "cumulative" 累积模式 | "incremental" 增量模式
    """
    # 加载音频
    audio, sr = load_wav(wav_path, target_sr=16000)
    total_duration = len(audio) / sr
    chunk_samples = int(sr * interval)
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    print("=" * 70)
    print(f"  伪流式 ASR 测试")
    print("=" * 70)
    print(f"  音频文件  : {wav_path}")
    print(f"  音频时长  : {total_duration:.2f}s")
    print(f"  采样率    : {sr} Hz")
    print(f"  分段间隔  : {interval}s")
    print(f"  总分段数  : {total_chunks}")
    print(f"  识别模式  : {chunk_mode}")
    print(f"  模型      : {model}")
    print(f"  API 地址  : {base_url}")
    if context:
        print(f"  热词/提示 : {context}")
    print("=" * 70)
    print()

    # 创建客户端
    client = create_asr_client(base_url=base_url, api_key=api_key)

    results: list[dict] = []
    t_total_start = time.monotonic()

    for i in range(total_chunks):
        chunk_start_sample = i * chunk_samples
        chunk_end_sample = min((i + 1) * chunk_samples, len(audio))
        chunk_time_start = chunk_start_sample / sr
        chunk_time_end = chunk_end_sample / sr

        # 根据模式选择送入 ASR 的音频
        if chunk_mode == "cumulative":
            # 累积模式：从头到当前时刻
            audio_to_recognize = audio[:chunk_end_sample]
            mode_label = f"0.00s → {chunk_time_end:.2f}s"
        else:
            # 增量模式：只送当前片段
            audio_to_recognize = audio[chunk_start_sample:chunk_end_sample]
            mode_label = f"{chunk_time_start:.2f}s → {chunk_time_end:.2f}s"

        # ASR 推理
        t_start = time.monotonic()
        try:
            text = asr_recognize(
                client=client,
                audio_f32=audio_to_recognize,
                sr=sr,
                model=model,
                context=context,
            )
        except Exception as exc:
            text = f"[ERROR: {exc}]"
        t_elapsed = (time.monotonic() - t_start) * 1000  # ms

        result = {
            "chunk_id": i + 1,
            "time_range": mode_label,
            "audio_duration_ms": int((chunk_end_sample - (0 if chunk_mode == "cumulative" else chunk_start_sample)) / sr * 1000),
            "asr_latency_ms": round(t_elapsed, 1),
            "text": text,
        }
        results.append(result)

        # 实时打印结果
        print(
            f"  [{i + 1:3d}/{total_chunks}]  "
            f"时间={mode_label:>18s}  "
            f"耗时={t_elapsed:7.1f}ms  "
            f"文本: {text}"
        )

    t_total_elapsed = (time.monotonic() - t_total_start) * 1000

    # ---- 汇总报告 ----
    print()
    print("=" * 70)
    print("  汇总报告")
    print("=" * 70)

    latencies = [r["asr_latency_ms"] for r in results]
    texts = [r["text"] for r in results if r["text"] and not r["text"].startswith("[ERROR")]

    print(f"  总耗时      : {t_total_elapsed:.0f}ms ({t_total_elapsed / 1000:.2f}s)")
    print(f"  音频时长    : {total_duration:.2f}s")
    print(f"  RTF         : {t_total_elapsed / 1000 / total_duration:.2f}")
    print(f"  总分段数    : {total_chunks}")
    print(f"  成功识别    : {len(texts)}/{total_chunks}")
    print(f"  平均延迟    : {sum(latencies) / len(latencies):.1f}ms")
    print(f"  最小延迟    : {min(latencies):.1f}ms")
    print(f"  最大延迟    : {max(latencies):.1f}ms")
    print()

    if chunk_mode == "cumulative" and texts:
        print("  最终识别结果:")
        print(f"    {texts[-1]}")
    elif chunk_mode == "incremental" and texts:
        print("  拼接识别结果:")
        full_text = "".join(texts)
        print(f"    {full_text}")

    print()
    print("=" * 70)

    # ---- 逐段明细 ----
    print()
    print("  逐段明细:")
    print(f"  {'段号':>4s}  {'时间范围':>18s}  {'音频ms':>7s}  {'延迟ms':>7s}  文本")
    print("  " + "-" * 66)
    for r in results:
        print(
            f"  {r['chunk_id']:4d}  "
            f"{r['time_range']:>18s}  "
            f"{r['audio_duration_ms']:7d}  "
            f"{r['asr_latency_ms']:7.1f}  "
            f"{r['text']}"
        )
    print()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="伪流式 ASR 测试：按固定间隔分段，逐段调用 ASR 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认累积模式，每 0.4s 推理一次
  python test/pseudo_streaming_asr.py 120报警电话16k.wav

  # 增量模式，每段独立识别
  python test/pseudo_streaming_asr.py audio.wav --chunk-mode incremental

  # 自定义间隔和 API 地址
  python test/pseudo_streaming_asr.py audio.wav --interval 0.2 --base-url http://10.0.0.1:15002/v1
        """,
    )
    parser.add_argument(
        "wav", metavar="WAV_FILE",
        help="音频文件路径（WAV 格式，建议 16kHz 单声道）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_API_BASE", "http://127.0.0.1:15002/v1"),
        help="vLLM API 地址 (默认: http://127.0.0.1:15002/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API Key (默认: EMPTY)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL_NAME", "Qwen3-ASR-1.7B"),
        help="模型名称 (默认: Qwen3-ASR-1.7B)",
    )
    parser.add_argument(
        "--interval", type=float, default=0.4,
        help="分段间隔秒数 (默认: 0.4)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="热词/系统提示词，如 '热词：张三丰、武当山'",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["cumulative", "incremental"],
        default="cumulative",
        help="识别模式: cumulative=累积(默认) | incremental=增量",
    )

    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print(f"ERROR: 文件不存在: {args.wav}")
        sys.exit(1)

    run_pseudo_streaming(
        wav_path=args.wav,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        interval=args.interval,
        context=args.context,
        chunk_mode=args.chunk_mode,
    )


if __name__ == "__main__":
    main()
