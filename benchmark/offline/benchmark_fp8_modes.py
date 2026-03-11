"""FP8 Input Quantization vs Weight Dequantization Performance Comparison.

Usage:
  python benchmark_fp8_modes.py input_quant   # Test FP8 input quantization
  python benchmark_fp8_modes.py weight_dequant # Test FP8 weight dequantization
"""

import os
import sys
import time
from random import randint, seed

import torch

os.environ["MINISGL_OVERLAP_EXTRA_SYNC"] = "1"
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "12.0"
os.environ["MINISGL_ATTENTION_BACKEND"] = "fi"

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "input_quant"
    
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    print("="*60)
    print(f"FP8 Benchmark: {mode.upper()}")
    print("="*60)
    print(f"Model: Qwen/Qwen3-8B-FP8")
    print(f"Sequences: {num_seqs}, Input: ~{max_input_len}, Output: ~{max_ouput_len}")
    print()

    if mode == "input_quant":
        llm = LLM(
            "Qwen/Qwen3-8B-FP8",
            max_seq_len_override=4096,
            max_extend_tokens=16384,
            cuda_graph_max_bs=256,
            page_size=256,
            dtype=torch.bfloat16,
            fp8_keep_quantized=True,
            use_fp8_input_quant=True,
            fp8_input_scale_method="per_tensor",
            attention_backend="fi",
        )
        print("[CONFIG] FP8 Input Quantization: ENABLED")
        print("[CONFIG] Scale Method: per_tensor")
    else:
        llm = LLM(
            "Qwen/Qwen3-8B-FP8",
            max_seq_len_override=4096,
            max_extend_tokens=16384,
            cuda_graph_max_bs=256,
            page_size=256,
            dtype=torch.bfloat16,
            fp8_keep_quantized=True,
            use_fp8_input_quant=False,
            attention_backend="fi",
        )
        print("[CONFIG] FP8 Weight Dequantization: ENABLED")
        print("[CONFIG] Input Quantization: DISABLED")
    
    print()
    print("Generating test data...")
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    
    print("Warming up...")
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))
    
    print("Running benchmark...")
    torch.cuda.synchronize()
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    torch.cuda.synchronize()
    t = time.time() - t
    
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    used_mem = total_mem - free_mem
    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Mode: {mode.upper()}")
    print(f"Total: {total_tokens} tokens")
    print(f"Time: {t:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"GPU Memory: {used_mem:.2f} / {total_mem:.2f} GiB")
    print("="*60)
    
    # Save result
    import json
    result = {
        "mode": mode,
        "total_tokens": total_tokens,
        "elapsed_time": round(t, 2),
        "throughput": round(throughput, 2),
        "gpu_memory_used": round(used_mem, 2),
    }
    output_file = f"/tmp/fp8_{mode}_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    main()
