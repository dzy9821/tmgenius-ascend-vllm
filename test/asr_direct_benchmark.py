#!/usr/bin/env python3
"""
ASR 直接调用性能测试 —— 音频一次性传入，绕过 VAD/WebSocket。

在指定并发下，测量 ASR 服务直接处理时间（min / max / avg / P95）。

用法：
    python test/asr_direct_benchmark.py
    python test/asr_direct_benchmark.py --audio 120报警电话16k.wav --levels 1,5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.services.asr_service import ASRService
from src.core.config import settings


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100.0)
    return s[min(idx, len(s) - 1)]


def load_wav_pcm(path: str) -> tuple[np.ndarray, float]:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getsampwidth() == 2, "Expected 16-bit"
        assert wf.getframerate() == 16000, "Expected 16kHz"
        n = wf.getnframes()
        return np.frombuffer(wf.readframes(n), dtype=np.int16).copy(), n / 16000.0


@dataclass
class CallResult:
    call_id: int
    success: bool
    elapsed_ms: float
    text: str = ""
    error: str = ""


async def _one_call(asr: ASRService, audio: np.ndarray, call_id: int) -> CallResult:
    t0 = time.monotonic()
    try:
        text = await asr.recognize(audio, sr=16000)
        return CallResult(call_id, True, (time.monotonic() - t0) * 1000.0, text=text)
    except Exception as e:
        return CallResult(call_id, False, (time.monotonic() - t0) * 1000.0, error=str(e))


async def run_level(asr: ASRService, audio: np.ndarray, concurrency: int) -> dict:
    print(f"  Running {concurrency} concurrent calls...")
    t0 = time.monotonic()
    results = await asyncio.gather(*[_one_call(asr, audio, i) for i in range(concurrency)])
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


def print_report(audio_path: str, audio_dur: float, results: list[dict], api_base: str) -> str:
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
    L(f"  模型名称: {settings.VLLM_MODEL_NAME}")
    L(f"  音频文件: {os.path.basename(audio_path)} ({audio_dur:.2f}s)")
    L(f"  测试模式: 完整音频一次性传入（无 VAD，无 WebSocket）")
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
            L(f"    min={s[0]:.1f}  avg={sum(s)/len(s):.1f}  P50={_percentile(s,50):.1f}  "
              f"P90={_percentile(s,90):.1f}  P95={_percentile(s,95):.1f}  "
              f"P99={_percentile(s,99):.1f}  max={s[-1]:.1f}")
        if lr["failed"]:
            L(f"    失败: {lr['errors']}")
        L("")
    L("─" * W)
    L("")
    return "\n".join(lines)


async def amain(args):
    audio_path = os.path.abspath(args.audio)
    print(f"Loading: {audio_path}")
    audio, dur = load_wav_pcm(audio_path)
    print(f"Audio: {len(audio)} samples, {dur:.2f}s")

    levels = [int(x.strip()) for x in args.levels.split(",")]

    api_base = args.url or settings.VLLM_API_BASE
    asr = ASRService(api_base=api_base)
    await asr.startup()

    if not await asr.is_available():
        print(f"ERROR: vLLM unreachable at {api_base}")
        await asr.shutdown()
        sys.exit(1)

    results = []
    for i, level in enumerate(levels):
        print(f"\n{'='*60}\nConcurrency: {level} ({i+1}/{len(levels)})\n{'='*60}")
        lr = await run_level(asr, audio, level)
        results.append(lr)
        t = lr["elapsed_ms"]
        if t:
            print(f"  Done. success={lr['success']}, min={min(t):.1f}ms, avg={sum(t)/len(t):.1f}ms, "
                  f"max={max(t):.1f}ms, P95={_percentile(t,95):.1f}ms")
        if i < len(levels) - 1 and args.cooldown > 0:
            await asyncio.sleep(args.cooldown)

    await asr.shutdown()

    report = print_report(audio_path, dur, results, api_base)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved → {args.output}")


def main():
    parser = argparse.ArgumentParser(description="ASR 直接调用性能测试")
    parser.add_argument("--url", default="", help="ASR 服务地址，默认使用 VLLM_API_BASE 环境变量")
    parser.add_argument("--audio", default=os.path.join(os.path.dirname(__file__), "..", "120报警电话16k.wav"))
    parser.add_argument("--levels", default="1,5")
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output", "-o", default="")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
