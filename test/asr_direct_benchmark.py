#!/usr/bin/env python3
"""
ASR 直接调用性能测试 —— 音频一次性传入，绕过 VAD/WebSocket。

零第三方依赖，仅使用 Python 标准库。

在指定并发下，通过 HTTP 直接调用 vLLM OpenAI 兼容接口，
测量 ASR 处理时间（min / max / avg / P95）。

用法：
    python test/asr_direct_benchmark.py
    python test/asr_direct_benchmark.py --audio 30s_16k.wav --levels 1,5
    python test/asr_direct_benchmark.py --url http://10.0.0.5:15002/v1 --levels 1,5,10
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request
import wave
from dataclasses import dataclass

# ============================================================
# 默认配置（可通过命令行或环境变量覆盖）
# ============================================================

DEFAULT_URL = os.getenv("VLLM_API_BASE", "http://127.0.0.1:15002/v1")
DEFAULT_MODEL = os.getenv("VLLM_MODEL_NAME", "Qwen3-ASR-1.7B")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100.0)
    return s[min(idx, len(s) - 1)]


def _clean_text(text: str) -> str:
    """清洗 ASR 输出（去除 <asr_text> 等标记）。"""
    text = text.strip()
    if "<asr_text>" in text:
        parts = re.split(r"(?:language\s+[^\s<]+)?<asr_text>", text, flags=re.IGNORECASE)
        return "".join(part.strip() for part in parts if part.strip()).strip()
    return text


# ============================================================
# 音频加载 & 编码（纯 stdlib：wave + base64）
# ============================================================

def load_wav_as_data_url(path: str) -> tuple[str, float]:
    """加载 WAV 文件，验证格式，返回 (data:audio/wav;base64,... 字符串, 时长秒数)。

    要求：16kHz, mono, 16-bit。不符合则报错。
    """
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit, got {wf.getsampwidth()*8}-bit")
        if wf.getframerate() != 16000:
            raise ValueError(f"Expected 16kHz, got {wf.getframerate()}Hz")
        n_frames = wf.getnframes()
        duration = n_frames / 16000.0

    # 直接读取文件字节（已含完整 WAV 头），base64 编码
    with open(path, "rb") as f:
        wav_bytes = f.read()

    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64}", duration


# ============================================================
# 单次 ASR 调用（纯 stdlib：urllib）
# ============================================================

@dataclass
class CallResult:
    call_id: int
    success: bool
    elapsed_ms: float
    text: str = ""
    error: str = ""


def _do_one_call(
    url: str,
    model: str,
    data_url: str,
    call_id: int,
    timeout: float = 300.0,
) -> CallResult:
    """同步 HTTP 调用 vLLM chat/completions，返回结果。"""
    t0 = time.monotonic()
    try:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ]},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            text = data["choices"][0]["message"]["content"]
            elapsed = (time.monotonic() - t0) * 1000.0
            return CallResult(call_id, True, elapsed, text=_clean_text(text))

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000.0
        return CallResult(call_id, False, elapsed, error=str(e))


# ============================================================
# 并发调度
# ============================================================

def run_level(
    url: str,
    model: str,
    data_url: str,
    concurrency: int,
    timeout: float = 300.0,
) -> dict:
    """运行一个并发级别，所有调用同时发出（ThreadPoolExecutor）。"""
    print(f"  Running {concurrency} concurrent calls...")
    t0 = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_do_one_call, url, model, data_url, i, timeout)
            for i in range(concurrency)
        ]
        results = [f.result() for f in futures]

    wall = time.monotonic() - t0
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    elapsed = [r.elapsed_ms for r in ok]
    return {
        "concurrency": concurrency,
        "wall_time_s": wall,
        "success": len(ok),
        "failed": len(fail),
        "elapsed_ms": elapsed,
        "errors": [r.error for r in fail],
    }


# ============================================================
# 报告输出
# ============================================================

def print_report(
    audio_path: str,
    audio_dur: float,
    results: list[dict],
    api_base: str,
    model: str,
) -> str:
    W = 78
    lines = []
    L = lines.append
    L("")
    L("╔" + "═" * W + "╗")
    L("║" + f"{'ASR 直接调用性能测试报告':^{W}}" + "║")
    L("╚" + "═" * W + "╝")
    L("")
    L(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"  服务地址: {api_base}")
    L(f"  模型名称: {model}")
    L(f"  音频文件: {os.path.basename(audio_path)} ({audio_dur:.2f}s)")
    L(f"  测试模式: 完整音频一次性传入（无 VAD，无 WebSocket，纯 stdlib HTTP）")
    L("")
    L("─" * W)
    L("  结果汇总")
    L("─" * W)
    L("")
    L(f"  {'并发':>4}  {'成功':>4}  {'失败':>4}  {'min(ms)':>9}  {'avg(ms)':>9}  "
      f"{'max(ms)':>9}  {'P95(ms)':>9}  {'墙钟(s)':>8}")
    L(f"  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")

    for lr in results:
        t = lr["elapsed_ms"]
        if t:
            s = sorted(t)
            L(f"  {lr['concurrency']:>4}  {lr['success']:>4}  {lr['failed']:>4}  "
              f"{s[0]:>9.1f}  {sum(s)/len(s):>9.1f}  {s[-1]:>9.1f}  "
              f"{_percentile(s, 95):>9.1f}  {lr['wall_time_s']:>8.2f}")
        else:
            L(f"  {lr['concurrency']:>4}  {lr['success']:>4}  {lr['failed']:>4}  "
              f"{'--':>9}  {'--':>9}  {'--':>9}  {'--':>9}  {lr['wall_time_s']:>8.2f}")
    L("")
    for lr in results:
        t = lr["elapsed_ms"]
        L(f"  ▸ 并发 {lr['concurrency']}")
        if t:
            s = sorted(t)
            L(f"    min={s[0]:.1f}  avg={sum(s)/len(s):.1f}  "
              f"P50={_percentile(s,50):.1f}  P90={_percentile(s,90):.1f}  "
              f"P95={_percentile(s,95):.1f}  P99={_percentile(s,99):.1f}  "
              f"max={s[-1]:.1f}")
        if lr["failed"]:
            L(f"    失败: {lr['errors'][:3]}")
        L("")
    L("─" * W)
    L("")
    return "\n".join(lines)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ASR 直接调用性能测试（纯 stdlib）")
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"ASR 服务地址，默认: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"模型名称，默认: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--audio",
        default=os.path.join(_SCRIPT_DIR, "..", "120报警电话16k.wav"),
        help="测试音频文件 (16kHz mono 16-bit WAV)",
    )
    parser.add_argument(
        "--levels", default="1,5",
        help="并发级别（逗号分隔），默认: 1,5",
    )
    parser.add_argument(
        "--cooldown", type=float, default=5.0,
        help="并发级别间冷却时间(秒，默认 5)",
    )
    parser.add_argument("--output", "-o", default="", help="报告输出文件路径")
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="单次 HTTP 请求超时(秒，默认 300)",
    )
    args = parser.parse_args()

    # 规范化 URL（去掉尾部斜杠）
    url = args.url.rstrip("/")

    # 加载音频
    audio_path = os.path.abspath(args.audio)
    print(f"Loading: {audio_path}")
    try:
        data_url, duration = load_wav_as_data_url(audio_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Audio: {duration:.2f}s, data_url: {len(data_url)} chars")

    # 先检查 vLLM 是否可达
    print(f"Checking vLLM at {url}/models ...")
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=10) as resp:
            if resp.status != 200:
                print(f"ERROR: vLLM returned status {resp.status}")
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: vLLM unreachable at {url}: {e}")
        sys.exit(1)
    print("  OK")

    # 解析并发级别
    levels = [int(x.strip()) for x in args.levels.split(",")]

    results = []
    for i, level in enumerate(levels):
        print(f"\n{'='*60}")
        print(f"Concurrency: {level} ({i+1}/{len(levels)})")
        print(f"{'='*60}")

        lr = run_level(url, args.model, data_url, level, timeout=args.timeout)
        results.append(lr)

        t = lr["elapsed_ms"]
        if t:
            print(f"  Done. success={lr['success']}, "
                  f"min={min(t):.1f}ms, avg={sum(t)/len(t):.1f}ms, "
                  f"max={max(t):.1f}ms, P95={_percentile(t,95):.1f}ms")

        if i < len(levels) - 1 and args.cooldown > 0:
            print(f"  Cooling down {args.cooldown}s...")
            time.sleep(args.cooldown)

    report = print_report(audio_path, duration, results, url, args.model)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved → {args.output}")


if __name__ == "__main__":
    main()
