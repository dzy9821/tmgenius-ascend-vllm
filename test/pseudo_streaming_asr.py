"""
伪流式 ASR 测试 —— 按步进切片，异步实时调度。

每隔 step 秒发出一个音频切片请求（不等前一个返回），
结果按实际返回顺序展示，并实时显示累积拼接文本。

用法:
    python test/pseudo_streaming_asr.py <wav_file> [options]
"""

from __future__ import annotations

import argparse, asyncio, base64, io, os, re, sys, time
import numpy as np


# ---- 音频工具 ----

def load_wav(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        indices = np.linspace(0, len(audio) - 1, int(len(audio) * target_sr / sr))
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sr
    return audio.astype(np.float32), sr


def audio_to_data_url(audio: np.ndarray, sr: int) -> str:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"


_ASR_TAG_RE = re.compile(r"(?:language\s+[^\s<]+)?<asr_text>", re.IGNORECASE)
_NON_CHINESE_RE = re.compile(r"[^\u4e00-\u9fff]+")


def clean_text(text: str) -> str:
    """移除 <asr_text> 标记，只保留中文字符。"""
    text = text.strip()
    if "<asr_text>" in text:
        text = "".join(p.strip() for p in _ASR_TAG_RE.split(text) if p.strip())
    return _NON_CHINESE_RE.sub("", text)


# ---- ASR 客户端 ----

def create_client(base_url: str, api_key: str = "EMPTY"):
    import httpx
    from openai import AsyncOpenAI
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    return AsyncOpenAI(
        base_url=base_url, api_key=api_key,
        http_client=httpx.AsyncClient(trust_env=False),
    )


async def asr_recognize(client, audio: np.ndarray, sr: int, model: str, context: str = "") -> str:
    messages: list[dict] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({
        "role": "user",
        "content": [{"type": "audio_url", "audio_url": {"url": audio_to_data_url(audio, sr)}}],
    })
    resp = await client.chat.completions.create(model=model, messages=messages)
    return clean_text(resp.choices[0].message.content or "")


# ---- 主逻辑 ----

async def run(
    wav_path: str,
    base_url: str = "http://localhost:28856/v1",
    api_key: str = "EMPTY",
    model: str = "Qwen3-ASR-0.6B",
    step: float = 0.4,
    context: str = "",
) -> None:
    audio, sr = load_wav(wav_path)
    total_dur = len(audio) / sr
    step_n = int(sr * step)
    total_steps = (len(audio) + step_n - 1) // step_n

    print("=" * 74)
    print(f"  伪流式 ASR | {wav_path} | {total_dur:.2f}s | step={step}s | {total_steps}步")
    print(f"  模型: {model}  API: {base_url}")
    print("=" * 74, "\n")

    client = create_client(base_url, api_key)
    step_texts: dict[int, str] = {}
    lock = asyncio.Lock()
    t0 = time.monotonic()
    latencies: list[float] = []
    done_count = 0

    async def _do(i: int):
        nonlocal done_count
        chunk = audio[i * step_n : min((i + 1) * step_n, len(audio))]
        t_s = f"{i * step:.2f}s→{min((i + 1) * step, total_dur):.2f}s"

        t_send = time.monotonic() - t0
        t1 = time.monotonic()
        try:
            text = await asr_recognize(client, chunk, sr, model, context)
        except Exception as e:
            text = f"[ERR:{e}]"
        lat = (time.monotonic() - t1) * 1000
        t_recv = time.monotonic() - t0

        async with lock:
            done_count += 1
            latencies.append(lat)
            if not text.startswith("[ERR"):
                step_texts[i] = text
            concat = "".join(step_texts.get(s, "") for s in range(total_steps))
            print(
                f"  [{done_count:3d}/{total_steps}] "
                f"步{i+1:3d} {t_s:>16s}  "
                f"发{t_send:5.2f}s 回{t_recv:5.2f}s "
                f"{lat:6.0f}ms  "
                f"本段「{text}」"
            )
            print(f"          累积: {concat}")

    # 按真实时间调度
    tasks = []
    for i in range(total_steps):
        wait = i * step - (time.monotonic() - t0)
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(_do(i)))
    await asyncio.gather(*tasks)

    total_ms = (time.monotonic() - t0) * 1000
    final = "".join(step_texts.get(s, "") for s in range(total_steps))

    print(f"\n{'=' * 74}")
    print(f"  总耗时: {total_ms:.0f}ms  RTF: {total_ms/1000/total_dur:.2f}  "
          f"延迟: avg={sum(latencies)/len(latencies):.0f} "
          f"min={min(latencies):.0f} max={max(latencies):.0f}ms")
    print(f"  最终文本: {final}")
    print("=" * 74)


# ---- CLI ----

def main():
    p = argparse.ArgumentParser(description="伪流式 ASR 测试")
    p.add_argument("wav", help="WAV 文件路径")
    p.add_argument("--base-url", default=os.getenv("VLLM_API_BASE", "http://localhost:28856/v1"))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.getenv("VLLM_MODEL_NAME", "Qwen3-ASR-0.6B"))
    p.add_argument("--step", type=float, default=0.4, help="步进秒数 (默认 0.4)")
    p.add_argument("--context", default="", help="热词/提示词")
    a = p.parse_args()

    if not os.path.exists(a.wav):
        sys.exit(f"ERROR: 文件不存在: {a.wav}")

    asyncio.run(run(a.wav, a.base_url, a.api_key, a.model, a.step, a.context))


if __name__ == "__main__":
    main()
