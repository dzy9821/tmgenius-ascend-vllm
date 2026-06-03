"""
伪流式 ASR 测试 —— 滑动窗口模拟实时流式识别。

每隔 0.4 秒步进，取 **2 秒窗口** 的音频送入 ASR 推理，
相邻窗口重叠 1.6 秒。对重叠部分的识别结果进行去重拼接，
输出最终的连续识别文本。

用法：
    python test/pseudo_streaming_asr.py <wav_file> [options]

示例：
    python test/pseudo_streaming_asr.py 120报警电话16k.wav
    python test/pseudo_streaming_asr.py audio.wav --window 3.0 --step 0.5
    python test/pseudo_streaming_asr.py audio.wav --base-url http://10.0.0.1:28856/v1
"""

from __future__ import annotations

import argparse
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


# 匹配非中文字符
_NON_CHINESE_RE = re.compile(r"[^\u4e00-\u9fff]+")


def strip_punctuation(text: str) -> str:
    """只保留中文字符，去掉其他所有内容。"""
    return _NON_CHINESE_RE.sub("", text)


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
    text = clean_asr_output(content if isinstance(content, str) else str(content))
    return strip_punctuation(text)


# ============================================================
# 滑动窗口文本对齐
# ============================================================


def _find_overlap(prev_text: str, new_text: str) -> int:
    """找 prev_text 的最长后缀与 new_text 前缀的精确匹配长度。"""
    max_check = min(len(prev_text), len(new_text))
    for k in range(max_check, 0, -1):
        if prev_text[-k:] == new_text[:k]:
            return k
    return 0


# ============================================================
# 伪流式测试主逻辑
# ============================================================


def run_pseudo_streaming(
    wav_path: str,
    base_url: str = "http://localhost:28856/v1",
    api_key: str = "EMPTY",
    model: str = "Qwen3-ASR-0.6B",
    step: float = 0.4,
    window: float = 2.0,
    context: str = "",
) -> None:
    """
    滑动窗口伪流式 ASR 测试。

    策略（比例提交法）：
      - 每次窗口滑动时，前一个窗口"滑出"的音频比例对应的文本被提交（定稿）
      - 当前窗口的完整识别结果作为"待定"文本，随时被下一个窗口替换
      - 最后一个窗口的剩余文本全部提交

    Args:
        wav_path: WAV 文件路径
        base_url: vLLM API 地址
        api_key: API Key
        model: 模型名称
        step: 步进间隔（秒），默认 0.4s
        window: 窗口大小（秒），默认 2.0s
        context: 热词/系统提示词
    """
    # 加载音频
    audio, sr = load_wav(wav_path, target_sr=16000)
    total_duration = len(audio) / sr
    step_samples = int(sr * step)
    window_samples = int(sr * window)
    total_steps = (len(audio) + step_samples - 1) // step_samples

    print("=" * 74)
    print("  伪流式 ASR 测试（滑动窗口 · 比例提交）")
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

    # 创建客户端
    client = create_asr_client(base_url=base_url, api_key=api_key)

    results: list[dict] = []
    t_total_start = time.monotonic()

    # ---- 对齐提交状态 ----
    committed_text = ""       # 已定稿的文本（不再变化）
    prev_text = ""            # 上一个窗口的识别结果（待定）

    for i in range(total_steps):
        # 窗口起止（向前取 window 秒，但不超过音频开头）
        window_end_sample = min((i + 1) * step_samples, len(audio))
        window_start_sample = max(0, window_end_sample - window_samples)

        window_time_start = window_start_sample / sr
        window_time_end = window_end_sample / sr
        window_dur = window_time_end - window_time_start
        time_label = f"{window_time_start:.2f}s → {window_time_end:.2f}s"

        # 取窗口音频
        audio_window = audio[window_start_sample:window_end_sample]

        # ASR 推理
        t_start = time.monotonic()
        try:
            text = asr_recognize(
                client=client,
                audio_f32=audio_window,
                sr=sr,
                model=model,
                context=context,
            )
        except Exception as exc:
            text = f"[ERROR: {exc}]"
        t_elapsed = (time.monotonic() - t_start) * 1000  # ms

        # ---- 后缀-前缀对齐提交 ----
        if text and not text.startswith("[ERROR"):
            if prev_text:
                overlap = _find_overlap(prev_text, text)
                if overlap > 0:
                    # 提交 prev_text 中重叠之前的部分（不重叠 = 已滑出窗口）
                    committed_text += prev_text[:-overlap]
                else:
                    # 完全没有重叠，prev_text 全部提交
                    committed_text += prev_text
            prev_text = text

        # 当前展示文本 = 已提交 + 当前窗口待定
        display_text = committed_text + prev_text

        result = {
            "step_id": i + 1,
            "window_range": time_label,
            "window_duration_ms": int(window_dur * 1000),
            "asr_latency_ms": round(t_elapsed, 1),
            "raw_text": text,
            "committed": committed_text,
            "display": display_text,
        }
        results.append(result)

        # 实时打印：展示已累积的拼接文本
        print(
            f"  [{i + 1:3d}/{total_steps}]  "
            f"窗口={time_label:>20s}  "
            f"耗时={t_elapsed:7.1f}ms"
        )
        print(f"             累积文本: {display_text}")

    # 最后一个窗口的剩余文本全部提交
    committed_text += prev_text
    final_text = committed_text

    t_total_elapsed = (time.monotonic() - t_total_start) * 1000

    # ---- 汇总报告 ----
    print()
    print("=" * 74)
    print("  汇总报告")
    print("=" * 74)

    latencies = [r["asr_latency_ms"] for r in results]
    valid_count = sum(1 for r in results if not r["raw_text"].startswith("[ERROR"))

    print(f"  总耗时      : {t_total_elapsed:.0f}ms ({t_total_elapsed / 1000:.2f}s)")
    print(f"  音频时长    : {total_duration:.2f}s")
    print(f"  RTF         : {t_total_elapsed / 1000 / total_duration:.2f}")
    print(f"  总推理次数  : {total_steps}")
    print(f"  成功识别    : {valid_count}/{total_steps}")
    print(f"  平均延迟    : {sum(latencies) / len(latencies):.1f}ms")
    print(f"  最小延迟    : {min(latencies):.1f}ms")
    print(f"  最大延迟    : {max(latencies):.1f}ms")
    print()
    print("  最终拼接结果:")
    print(f"    {final_text}")
    print()
    print("=" * 74)

    # ---- 逐步明细 ----
    print()
    print("  逐步明细:")
    print(f"  {'步号':>4s}  {'窗口范围':>20s}  {'延迟ms':>7s}  窗口原文")
    print("  " + "-" * 72)
    for r in results:
        print(
            f"  {r['step_id']:4d}  "
            f"{r['window_range']:>20s}  "
            f"{r['asr_latency_ms']:7.1f}  "
            f"{r['raw_text']}"
        )
    print()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="伪流式 ASR 测试：滑动窗口方式，每步发送固定窗口音频并去重拼接结果",
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

    run_pseudo_streaming(
        wav_path=args.wav,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        step=args.step,
        window=args.window,
        context=args.context,
    )


if __name__ == "__main__":
    main()
