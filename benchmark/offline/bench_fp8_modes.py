"""FP8 Performance Benchmark Script.

Compares performance between:
1. BF16 baseline
2. FP8 weight dequantization
3. FP8 input quantization (per_tensor)
4. FP8 input quantization (per_token)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List

import torch

# Set environment variables before imports
os.environ["MINISGL_OVERLAP_EXTRA_SYNC"] = "1"
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "12.0"
os.environ["MINISGL_ATTENTION_BACKEND"] = "fi"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

from minisgl.core import SamplingParams
from minisgl.llm import LLM


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    mode: str
    total_tokens: int
    total_time: float
    throughput: float
    tokens_per_second: float


def run_benchmark(
    llm: LLM,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    warmup: int = 3,
    iterations: int = 5,
) -> BenchmarkResult:
    """Run benchmark and return results."""
    from random import randint, seed

    seed(42)

    # Generate random prompts
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(16, max_output_len))
        for _ in range(num_seqs)
    ]

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        llm.generate(["Hello"], SamplingParams(temperature=0.1, max_tokens=5))

    # Benchmark
    print(f"  Running benchmark ({iterations} iterations)...")
    times = []
    total_tokens = 0

    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = llm.generate(prompt_token_ids, sampling_params)

        torch.cuda.synchronize()
        end = time.perf_counter()

        iteration_time = end - start
        times.append(iteration_time)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        print(f"    Iteration {i+1}: {iteration_time:.2f}s, {total_tokens / iteration_time:.2f} tok/s")

    avg_time = sum(times) / len(times)
    throughput = total_tokens / avg_time

    return BenchmarkResult(
        mode="",
        total_tokens=total_tokens,
        total_time=avg_time,
        throughput=throughput,
        tokens_per_second=throughput,
    )


def main():
    parser = argparse.ArgumentParser(description="FP8 Performance Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-FP8", help="Model path")
    parser.add_argument("--num-seqs", type=int, default=1, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=256, help="Max input length")
    parser.add_argument("--max-output-len", type=int, default=32, help="Max output length")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["bf16", "fp8_dequant", "fp8_input_quant"],
        choices=["bf16", "fp8_dequant", "fp8_input_quant", "fp8_input_quant_per_token"],
        help="Modes to benchmark",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()

    print("=" * 60)
    print("FP8 Performance Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Num sequences: {args.num_seqs}")
    print(f"Max input length: {args.max_input_len}")
    print(f"Max output length: {args.max_output_len}")
    print(f"Modes: {args.modes}")
    print("=" * 60)

    results: List[BenchmarkResult] = []

    for mode in args.modes:
        print(f"\n--- Testing mode: {mode} ---")

        try:
            if mode == "bf16":
                llm = LLM(
                    args.model,
                    max_seq_len_override=2048,
                    max_extend_tokens=2048,
                    cuda_graph_max_bs=0,
                    page_size=64,
                    dtype=torch.bfloat16,
                    fp8_keep_quantized=False,
                    use_fp8_input_quant=False,
                    attention_backend="fi",
                )
            elif mode == "fp8_dequant":
                llm = LLM(
                    args.model,
                    max_seq_len_override=2048,
                    max_extend_tokens=2048,
                    cuda_graph_max_bs=0,
                    page_size=64,
                    dtype=torch.bfloat16,
                    fp8_keep_quantized=True,
                    use_fp8_input_quant=False,
                    attention_backend="fi",
                )
            elif mode == "fp8_input_quant":
                llm = LLM(
                    args.model,
                    max_seq_len_override=2048,
                    max_extend_tokens=2048,
                    cuda_graph_max_bs=0,
                    page_size=64,
                    dtype=torch.bfloat16,
                    fp8_keep_quantized=True,
                    use_fp8_input_quant=True,
                    fp8_input_scale_method="per_tensor",
                    attention_backend="fi",
                )
            elif mode == "fp8_input_quant_per_token":
                llm = LLM(
                    args.model,
                    max_seq_len_override=2048,
                    max_extend_tokens=2048,
                    cuda_graph_max_bs=0,
                    page_size=64,
                    dtype=torch.bfloat16,
                    fp8_keep_quantized=True,
                    use_fp8_input_quant=True,
                    fp8_input_scale_method="per_token",
                    attention_backend="fi",
                )
            else:
                print(f"Unknown mode: {mode}")
                continue

            result = run_benchmark(
                llm,
                args.num_seqs,
                args.max_input_len,
                args.max_output_len,
                args.warmup,
                args.iterations,
            )
            result.mode = mode
            results.append(result)

            # Clean up
            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in mode {mode}: {e}")
            import traceback

            traceback.print_exc()

    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<30} {'Time (s)':<12} {'Throughput (tok/s)':<20}")
    print("-" * 60)

    baseline_throughput = None
    for result in results:
        if result.mode == "bf16":
            baseline_throughput = result.throughput

    for result in results:
        speedup = ""
        if baseline_throughput and result.mode != "bf16":
            speedup_pct = (result.throughput / baseline_throughput - 1) * 100
            speedup = f" ({speedup_pct:+.1f}%)"

        print(f"{result.mode:<30} {result.total_time:<12.2f} {result.throughput:<20.2f}{speedup}")

    print("=" * 60)


if __name__ == "__main__":
    main()