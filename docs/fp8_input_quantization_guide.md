# FP8 Input Quantization Guide

This document describes the FP8 Input Quantization feature in Mini-SGLang, which enables native FP8×FP8 Tensor Core GEMM operations for improved inference performance.

## Overview

FP8 Input Quantization is an optimization technique that quantizes input activations to FP8 format during inference, enabling native FP8 Tensor Core matrix multiplication. This approach differs from the default FP8 weight dequantization:

| Mode | Weight | Activation | GEMM Type | Memory | Performance |
|------|--------|------------|-----------|--------|-------------|
| BF16 | BF16 | BF16 | BF16×BF16 | High | Baseline |
| FP8 Weight Dequant | FP8 | BF16 | BF16×BF16 | Medium | +10-20% latency |
| **FP8 Input Quant** | FP8 | FP8 | **FP8×FP8** | Medium | **Better throughput** |

## Quick Start

### Command Line

```bash
# Basic usage with FP8 input quantization
python -m minisgl --model "Qwen/Qwen3-8B-FP8" \
    --fp8-keep-quantized \
    --fp8-input-quant

# With per-tensor scaling (default, faster)
python -m minisgl --model "Qwen/Qwen3-8B-FP8" \
    --fp8-keep-quantized \
    --fp8-input-quant \
    --fp8-scale-method per_tensor

# With per-token scaling (more precise, requires M divisible by 32)
python -m minisgl --model "Qwen/Qwen3-8B-FP8" \
    --fp8-keep-quantized \
    --fp8-input-quant \
    --fp8-scale-method per_token
```

### Python API

```python
from minisgl.llm import LLM
from minisgl.core import SamplingParams

# Initialize with FP8 input quantization
llm = LLM(
    "Qwen/Qwen3-8B-FP8",
    fp8_keep_quantized=True,
    use_fp8_input_quant=True,
    fp8_input_scale_method="per_tensor",
)

# Generate text
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
print(outputs[0]["text"])
```

## Requirements

1. **GPU Architecture**: NVIDIA GPU with FP8 Tensor Core support (SM 89+)
   - RTX 40 series (Ada Lovelace)
   - RTX 50 series (Blackwell)
   - H100/H200 (Hopper)

2. **Model Format**: FP8 quantized model (e.g., `Qwen/Qwen3-8B-FP8`)

3. **Attention Backend**: FlashInfer recommended for Blackwell GPUs (SM 12.0)

## Architecture

### Layer Precision Strategy

The FP8 input quantization is applied selectively based on layer requirements:

| Layer Type | Precision Mode | Reason |
|------------|----------------|--------|
| QKV Projection | BF16 (weight dequant) | RoPE requires BF16 precision |
| O Projection | BF16 (weight dequant) | Precision sensitive |
| **gate_up_proj** | **FP8 input quant** | Best fit for GEMM |
| **down_proj** | **FP8 input quant** | Best fit for GEMM |

### Data Flow

```
Input (BF16)
    ↓
┌─────────────────────────────────────┐
│         Attention Layer             │
│  QKV: BF16 (weight dequant)         │
│  O:   BF16 (weight dequant)         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│            MLP Layer                │
│  gate_up: Quantize → FP8×FP8 GEMM   │
│  activation (SiLU)                  │
│  down:    Quantize → FP8×FP8 GEMM   │
└─────────────────────────────────────┘
    ↓
Output (FP16 → BF16)
```

## Scale Methods

### Per-Tensor Scaling (Recommended)

- **Description**: Single scale value for entire activation tensor
- **Performance**: Faster, lower overhead
- **Precision**: Slightly less precise
- **Compatibility**: Works with all tensor sizes

### Per-Token Scaling

- **Description**: Individual scale per token
- **Performance**: Slightly slower due to scale computation
- **Precision**: More precise for varying input magnitudes
- **Limitations**: Requires M (batch × seq) to be divisible by 32 for GEMM

> **Note**: Per-token scaling with `torch._scaled_mm` has specific alignment requirements:
> - `scale_a` must be `(M/32, 1)` contiguous
> - `scale_b` must be `(1, N)` contiguous
> - M must be divisible by 32

## Performance Benchmarks

### Test Configuration

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 5060 Ti 16GB |
| Architecture | Blackwell (SM 12.0) |
| Model | Qwen3-8B-FP8 |
| Attention Backend | FlashInfer |
| CUDA Graph | Disabled |

### Memory Usage

| Mode | Weight Memory | KV Cache Space | Total |
|------|---------------|----------------|-------|
| BF16 | 16.0 GB | < 0.5 GB | ~16.5 GB |
| FP8 Weight Dequant | 9.4 GB | 4-5 GB | ~14 GB |
| FP8 Input Quant | 9.4 GB | 4-5 GB | ~14 GB |

### Throughput

| Mode | Single Sequence | Multi-Sequence |
|------|-----------------|----------------|
| BF16 | Baseline | Baseline |
| FP8 Weight Dequant | -10% | -10% |
| FP8 Input Quant | **+5-15%** | **+10-20%** |

*Note: Actual performance varies based on hardware and workload.*

## Known Limitations

1. **Model Requirements**:
   - Only works with FP8 quantized models
   - Requires `--fp8-keep-quantized` flag

2. **Attention Layers**:
   - FP8 input quantization is NOT applied to attention layers
   - QKV and O projections use BF16 weight dequantization
   - This is intentional for RoPE compatibility

3. **Per-Token Scaling**:
   - Requires M (batch × seq) divisible by 32 for GEMM
   - May not work with all batch sizes
   - Falls back to per-tensor if incompatible

4. **CUDA Graph**:
   - May have compatibility issues with FP8 GEMM
   - Recommend disabling with `--cuda-graph-max-bs 0` for initial testing

5. **Blackwell GPUs (RTX 50 series)**:
   - Must use FlashInfer attention backend (`--attention-backend fi`)
   - TRTLLM backend not supported on SM 12.0

## Troubleshooting

### RuntimeError: Invalid scaling configuration

This error occurs when per-token scaling dimensions don't meet requirements:
- Ensure M (batch × seq) is divisible by 32
- Or use `--fp8-scale-method per_tensor`

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce `--max-running-req`
2. Reduce `--max-prefill-length`
3. Disable CUDA graphs: `--cuda-graph-max-bs 0`

### Slow Initial Inference

First inference may be slow due to:
- JIT kernel compilation
- CUDA graph capture (if enabled)

Subsequent inferences will be faster.

## Related Documentation

- [FP8 Adaptation Report](./fp8_adaptation_report.md) - Implementation details
- [FP8 Inference Flow](./fp8_inference_flow_mermaid.md) - Architecture diagram
- [Qwen3-8B-FP8 Test Guide](./qwen3_8b_fp8_test_guide.md) - Step-by-step testing