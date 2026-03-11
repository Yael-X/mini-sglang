"""FP8 Input Quantization vs Weight Dequantization Performance Comparison.

This benchmark compares two FP8 inference modes:
1. FP8 Input Quantization: BF16 input → FP8 quant → FP8×FP8 GEMM → FP16 output
2. FP8 Weight Dequantization: BF16 input → Dequant weights → BF16×BF16 GEMM → BF16 output
"""

import os
import time
import torch

os.environ["MINISGL_OVERLAP_EXTRA_SYNC"] = "1"
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "12.0"
os.environ["MINISGL_ATTENTION_BACKEND"] = "fi"

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def run_benchmark(mode: str, num_seqs: int = 8, input_len: int = 128, output_len: int = 64):
    """Run benchmark for specified mode.
    
    Args:
        mode: "input_quant" or "weight_dequant"
        num_seqs: Number of sequences
        input_len: Input sequence length
        output_len: Output sequence length
    
    Returns:
        dict with performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {mode.upper()}")
    print(f"{'='*60}")
    print(f"Config: {num_seqs} sequences, input={input_len}, output={output_len}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Initialize LLM
    if mode == "input_quant":
        llm = LLM(
            "Qwen/Qwen3-8B-FP8",
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
    else:  # weight_dequant
        llm = LLM(
            "Qwen/Qwen3-8B-FP8",
            max_seq_len_override=2048,
            max_extend_tokens=2048,
            cuda_graph_max_bs=0,
            page_size=64,
            dtype=torch.bfloat16,
            fp8_keep_quantized=True,
            use_fp8_input_quant=False,  # Use weight dequantization
            attention_backend="fi",
        )
    
    # Warm up
    print("Warming up...")
    llm.generate(["Hello"], SamplingParams(temperature=0.1, max_tokens=5))
    print("Warm-up complete.")
    
    # Generate random inputs
    torch.manual_seed(42)
    prompt_token_ids = [
        torch.randint(100, 10000, (input_len,)).tolist()
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]
    
    # Run benchmark
    print(f"\nRunning benchmark with {num_seqs} sequences...")
    torch.cuda.synchronize()
    t_start = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params)
    torch.cuda.synchronize()
    t_end = time.time()
    
    # Calculate metrics
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    elapsed = t_end - t_start
    throughput = total_tokens / elapsed
    
    print(f"\n{'='*60}")
    print(f"Results ({mode.upper()})")
    print(f"{'='*60}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} tok/s")
    print(f"  Per-sequence throughput: {throughput/num_seqs:.2f} tok/s/seq")
    
    # Memory stats
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    used_mem = total_mem - free_mem
    print(f"  GPU Memory: {used_mem:.2f} / {total_mem:.2f} GiB")
    
    return {
        "mode": mode,
        "num_seqs": num_seqs,
        "input_len": input_len,
        "output_len": output_len,
        "total_tokens": total_tokens,
        "elapsed_time": elapsed,
        "throughput": throughput,
        "throughput_per_seq": throughput / num_seqs,
        "gpu_memory_used": used_mem,
    }


def main():
    print("="*60)
    print("FP8 Input Quantization vs Weight Dequantization Benchmark")
    print("="*60)
    
    results = []
    
    # Test different configurations
    configs = [
        (1, 64, 32),    # Single sequence, short
        (1, 128, 64),   # Single sequence, medium
        (4, 128, 64),   # Multi-sequence, medium
        (8, 128, 64),   # Multi-sequence, medium (higher concurrency)
    ]
    
    for num_seqs, input_len, output_len in configs:
        print(f"\n\n{'#'*60}")
        print(f"# Configuration: {num_seqs} sequences, input={input_len}, output={output_len}")
        print(f"{'#'*60}")
        
        # Test weight dequantization first (baseline)
        try:
            result_dequant = run_benchmark(
                "weight_dequant", num_seqs, input_len, output_len
            )
            results.append(result_dequant)
        except Exception as e:
            print(f"Weight dequant benchmark failed: {e}")
            result_dequant = None
        
        # Clear memory between tests
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # Test input quantization
        try:
            result_input_quant = run_benchmark(
                "input_quant", num_seqs, input_len, output_len
            )
            results.append(result_input_quant)
        except Exception as e:
            print(f"Input quant benchmark failed: {e}")
            result_input_quant = None
        
        # Clear memory between tests
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # Print comparison
        if result_dequant and result_input_quant:
            speedup = result_input_quant["throughput"] / result_dequant["throughput"]
            print(f"\n{'='*60}")
            print(f"Performance Comparison")
            print(f"{'='*60}")
            print(f"  Weight Dequant: {result_dequant['throughput']:.2f} tok/s")
            print(f"  Input Quant:    {result_input_quant['throughput']:.2f} tok/s")
            print(f"  Speedup:        {speedup:.2f}x")
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Mode':<20} {'Throughput':<15} {'Mem (GiB)':<10}")
    print(f"{'-'*60}")
    for r in results:
        config = f"{r['num_seqs']}seq x {r['input_len']}+{r['output_len']}"
        mode = "Input Quant" if r['mode'] == 'input_quant' else "Weight Dequant"
        print(f"{config:<30} {mode:<20} {r['throughput']:>8.2f} tok/s {r['gpu_memory_used']:>8.2f}")
    
    # Save results
    import json
    with open("/tmp/fp8_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/fp8_benchmark_results.json")


if __name__ == "__main__":
    main()
