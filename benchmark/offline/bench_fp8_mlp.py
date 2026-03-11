"""FP8 MLP Layer Micro-benchmark.

Compares MLP layer performance between:
1. BF16 GEMM
2. FP8 weight dequantization
3. FP8 input quantization
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
from minisgl.distributed import set_tp_info
from minisgl.kernel.fp8_gemm import convert_bf16_weight_to_fp8
from minisgl.kernel.input_quant import quantize_input_to_fp8


def setup_tp():
    """Setup tensor parallelism info for single GPU."""
    from minisgl.distributed.info import _TP_INFO

    if _TP_INFO is None:
        set_tp_info(0, 1)


def create_test_weights(
    hidden_size: int, intermediate_size: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test weights for MLP layers."""
    # gate_up_proj weight: [2 * intermediate_size, hidden_size]
    gate_up_weight = torch.randn(
        2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )
    gate_up_fp8, gate_up_scale = convert_bf16_weight_to_fp8(gate_up_weight)

    # down_proj weight: [hidden_size, intermediate_size]
    down_weight = torch.randn(
        hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    )
    down_fp8, down_scale = convert_bf16_weight_to_fp8(down_weight)

    return gate_up_weight, gate_up_fp8, gate_up_scale, down_weight, down_fp8, down_scale


def benchmark_bf16_gemm(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Benchmark BF16 GEMM path."""
    import torch.nn.functional as F

    # Warmup
    for _ in range(warmup):
        gate_up = F.linear(x, gate_up_weight)
        # SiLU activation
        gate, up = gate_up.chunk(2, dim=-1)
        y = F.silu(gate) * up
        output = F.linear(y, down_weight)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        gate_up = F.linear(x, gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        y = F.silu(gate) * up
        output = F.linear(y, down_weight)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations


def benchmark_fp8_input_quant(
    x: torch.Tensor,
    gate_up_fp8: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_fp8: torch.Tensor,
    down_scale: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Benchmark FP8 input quantization path."""
    from minisgl.kernel.fp8_gemm import fp8_gemm

    M, K = x.shape
    N_gate_up = gate_up_fp8.shape[0]
    N_down = down_fp8.shape[0]

    # Warmup
    for _ in range(warmup):
        # gate_up projection
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")
        w_scale = gate_up_scale.amax().float()
        gate_up = fp8_gemm(x_fp8, x_scale, gate_up_fp8.T, w_scale, out_dtype=torch.float16)

        # SiLU activation
        gate, up = gate_up.chunk(2, dim=-1)
        y = torch.nn.functional.silu(gate.to(torch.bfloat16)) * up.to(torch.bfloat16)

        # down projection
        y_fp8, y_scale = quantize_input_to_fp8(y, scale_method="per_tensor")
        w_scale = down_scale.amax().float()
        output = fp8_gemm(y_fp8, y_scale, down_fp8.T, w_scale, out_dtype=torch.float16)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")
        w_scale = gate_up_scale.amax().float()
        gate_up = fp8_gemm(x_fp8, x_scale, gate_up_fp8.T, w_scale, out_dtype=torch.float16)

        gate, up = gate_up.chunk(2, dim=-1)
        y = torch.nn.functional.silu(gate.to(torch.bfloat16)) * up.to(torch.bfloat16)

        y_fp8, y_scale = quantize_input_to_fp8(y, scale_method="per_tensor")
        w_scale = down_scale.amax().float()
        output = fp8_gemm(y_fp8, y_scale, down_fp8.T, w_scale, out_dtype=torch.float16)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations


def main():
    print("=" * 60)
    print("FP8 MLP Layer Micro-benchmark")
    print("=" * 60)

    setup_tp()

    # Test configurations
    configs = [
        {"batch": 1, "seq": 16, "hidden": 4096, "intermediate": 11008},
        {"batch": 4, "seq": 16, "hidden": 4096, "intermediate": 11008},
        {"batch": 16, "seq": 16, "hidden": 4096, "intermediate": 11008},
        {"batch": 32, "seq": 16, "hidden": 4096, "intermediate": 11008},
    ]

    print(f"\n{'Batch':<8} {'Seq':<8} {'BF16 (ms)':<15} {'FP8 Input Quant (ms)':<20} {'Speedup':<10}")
    print("-" * 70)

    for config in configs:
        batch = config["batch"]
        seq = config["seq"]
        hidden = config["hidden"]
        intermediate = config["intermediate"]

        M = batch * seq
        K = hidden

        # Create input
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # Create weights
        (
            gate_up_weight,
            gate_up_fp8,
            gate_up_scale,
            down_weight,
            down_fp8,
            down_scale,
        ) = create_test_weights(hidden, intermediate)

        # Run benchmarks
        bf16_time = benchmark_bf16_gemm(x, gate_up_weight, down_weight) * 1000
        fp8_time = benchmark_fp8_input_quant(
            x, gate_up_fp8, gate_up_scale, down_fp8, down_scale
        ) * 1000

        speedup = bf16_time / fp8_time

        print(f"{batch:<8} {seq:<8} {bf16_time:<15.2f} {fp8_time:<20.2f} {speedup:<10.2f}x")

        # Clean up
        del x, gate_up_weight, gate_up_fp8, gate_up_scale, down_weight, down_fp8, down_scale
        torch.cuda.empty_cache()

    print("=" * 60)


if __name__ == "__main__":
    main()