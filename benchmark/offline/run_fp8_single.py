#!/usr/bin/env python3
"""Single run FP8 benchmark for specified mode."""

import os
import sys
import time
import torch

os.environ["MINISGL_OVERLAP_EXTRA_SYNC"] = "1"
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "12.0"
os.environ["MINISGL_ATTENTION_BACKEND"] = "fi"

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "input_quant"
    num_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    input_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    output_len = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    
    print(f"Mode: {mode}, Seqs: {num_seqs}, Input: {input_len}, Output: {output_len}")
    
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
        print("[INFO] Using FP8 input quantization with scale method: per_tensor")
    else:
        llm = LLM(
            "Qwen/Qwen3-8B-FP8",
            max_seq_len_override=2048,
            max_extend_tokens=2048,
            cuda_graph_max_bs=0,
            page_size=64,
            dtype=torch.bfloat16,
            fp8_keep_quantized=True,
            use_fp8_input_quant=False,
            attention_backend="fi",
        )
        print("[INFO] Using FP8 weight dequantization mode")
    
    # Warm up
    llm.generate(["Hello"], SamplingParams(temperature=0.1, max_tokens=5))
    
    # Generate
    torch.manual_seed(42)
    prompt_token_ids = [
        torch.randint(100, 10000, (input_len,)).tolist()
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]
    
    torch.cuda.synchronize()
    t_start = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params)
    torch.cuda.synchronize()
    t_end = time.time()
    
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    elapsed = t_end - t_start
    throughput = total_tokens / elapsed
    
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    used_mem = total_mem - free_mem
    
    # Output as JSON
    import json
    result = {
        "mode": mode,
        "num_seqs": num_seqs,
        "input_len": input_len,
        "output_len": output_len,
        "total_tokens": total_tokens,
        "elapsed_time": round(elapsed, 3),
        "throughput": round(throughput, 2),
        "gpu_memory_used": round(used_mem, 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
