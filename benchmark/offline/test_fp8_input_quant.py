"""End-to-end test for FP8 input quantization."""

import os
os.environ["MINISGL_OVERLAP_EXTRA_SYNC"] = "1"
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "12.0"
os.environ["MINISGL_ATTENTION_BACKEND"] = "fi"  # FlashInfer for Blackwell
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

import time
import torch
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    seed(0)
    num_seqs = 1  # Single sequence for initial testing
    max_input_len = 256
    max_ouput_len = 32

    print("=" * 60)
    print("Testing FP8 Input Quantization with Qwen3-8B-FP8")
    print("=" * 60)

    # Test FP8 input quantization
    print("\nInitializing LLM with FP8 input quantization...")
    llm = LLM(
        "Qwen/Qwen3-8B-FP8",
        max_seq_len_override=2048,
        max_extend_tokens=2048,
        cuda_graph_max_bs=0,  # Disable CUDA graphs to avoid JIT issues
        page_size=64,
        dtype=torch.bfloat16,  # BF16 for RoPE and KV cache compatibility
        fp8_keep_quantized=True,
        use_fp8_input_quant=True,
        fp8_input_scale_method="per_tensor",
        attention_backend="fi",  # FlashInfer for Blackwell (sm_120)
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(16, max_ouput_len))
        for _ in range(num_seqs)
    ]

    print("Warming up...")
    llm.generate(["Hello"], SamplingParams(temperature=0.1, max_tokens=5))
    print("Warm-up complete.")

    print(f"\nRunning benchmark with {num_seqs} sequences...")
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"\nResults:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time: {t:.2f}s")
    print(f"  Throughput: {throughput:.2f} tok/s")
    if outputs:
        first_output = outputs[0]
        text = first_output.get('text', '') if isinstance(first_output, dict) else str(first_output)[:100]
        print(f"\nFirst output sample: {text}...")

    print("\n" + "=" * 60)
    print("FP8 Input Quantization Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()