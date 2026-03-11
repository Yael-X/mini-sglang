# FP8 Input Quantization Performance Report

**Date:** 2026-03-11  
**Model:** Qwen/Qwen3-8B-FP8  
**GPU:** RTX 5060 Ti (Blackwell, sm_120)  
**Backend:** FlashInfer

---

## Executive Summary

FP8 Input Quantization 相比 FP8 Weight Dequantization 实现了 **3.57x 性能提升**！

| 指标 | Weight Dequant | Input Quant | 提升 |
|------|----------------|-------------|------|
| **吞吐量** | 3.47 tok/s | 12.37 tok/s | **3.57x** |
| **显存占用** | 15.93 GiB | 15.01 GiB | -0.92 GiB |
| **延迟 (64 tokens)** | 18.44s | 5.18s | **3.56x** |

---

## Benchmark Configuration

```
Model: Qwen/Qwen3-8B-FP8
Input Length: 128 tokens
Output Length: 64 tokens
Batch Size: 1 sequence
Attention Backend: FlashInfer (fi)
CUDA Graphs: Disabled (JIT compatibility)
```

---

## Detailed Results

### FP8 Weight Dequantization (Baseline)

**Mode:** BF16 input → Dequant weights → BF16×BF16 GEMM → BF16 output

```json
{
  "mode": "weight_dequant",
  "num_seqs": 1,
  "input_len": 128,
  "output_len": 64,
  "total_tokens": 64,
  "elapsed_time": 18.443,
  "throughput": 3.47,
  "gpu_memory_used": 15.93
}
```

### FP8 Input Quantization (New)

**Mode:** BF16 input → FP8 quant → FP8×FP8 GEMM → FP16 output

```json
{
  "mode": "input_quant",
  "num_seqs": 1,
  "input_len": 128,
  "output_len": 64,
  "total_tokens": 64,
  "elapsed_time": 5.176,
  "throughput": 12.37,
  "gpu_memory_used": 15.01
}
```

---

## Performance Analysis

### Throughput Comparison

```
Weight Dequant: ████████░░░░░░░░░░░░ 3.47 tok/s
Input Quant:    ████████████████████ 12.37 tok/s
                          ▲
                    3.57x faster
```

### Latency Comparison (64 tokens)

```
Weight Dequant: ██████████████████ 18.44s
Input Quant:    █████ 5.18s
                  ▲
            3.56x faster
```

### Memory Usage

```
Weight Dequant: ████████████████ 15.93 GiB
Input Quant:    ███████████████  15.01 GiB
                 ▲
           0.92 GiB saved (5.8%)
```

---

## Architecture Comparison

### FP8 Weight Dequantization

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ BF16 Input  │ →  │ FP8 Weight   │ →  │ BF16×BF16   │ →  │ BF16 Output  │
│             │    │ Dequantize   │    │ GEMM        │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                         │
                   ┌─────▼─────┐
                   │ FP8 Weight│
                   │ + Scales  │
                   └───────────┘
```

**Characteristics:**
- 每个 token 都需要反量化 FP8 权重
- GEMM 操作在 BF16 精度下执行
- 计算开销较大

### FP8 Input Quantization (New)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ BF16 Input  │ →  │ FP8 Quantize │ →  │ FP8×FP8     │ →  │ FP16 Output  │
│             │    │ (per-tensor) │    │ Tensor Core │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                              │
                                        ┌─────▼─────┐
                                        │ FP8 Weight│
                                        │ + Scales  │
                                        └───────────┘
```

**Characteristics:**
- 输入激活量化为 FP8（一次）
- 利用 Blackwell FP8 Tensor Core
- 原生 FP8×FP8 GEMM，性能最优

---

## Key Optimizations

### 1. Native FP8 Tensor Core GEMM

Blackwell 架构的 FP8 Tensor Core 专为 FP8×FP8 矩阵乘法优化：
- 权重反量化方案：BF16×BF16 GEMM（未利用 FP8 Tensor Core）
- 输入量化方案：FP8×FP8 GEMM（充分利用 FP8 Tensor Core）

### 2. Reduced Memory Bandwidth

输入量化方案的优势：
- 权重保持 FP8 格式（无需反量化读取）
- 输入激活只需量化一次
- 减少内存带宽压力

### 3. Mixed Precision Strategy

```
MLP Layers:     FP8 Input Quant (performance-critical)
Attention QKV:  BF16 (RoPE compatibility)
Attention O:    BF16 (KV cache compatibility)
LM Head:        BF16 (vocab projection)
```

---

## Usage

### Enable FP8 Input Quantization

```bash
python -m minisgl \
    --model-path "Qwen/Qwen3-8B-FP8" \
    --fp8-keep-quantized \
    --fp8-input-quant \
    --fp8-scale-method per_tensor \
    --attention-backend fi
```

### Python API

```python
from minisgl.llm import LLM

llm = LLM(
    "Qwen/Qwen3-8B-FP8",
    fp8_keep_quantized=True,
    use_fp8_input_quant=True,
    fp8_input_scale_method="per_tensor",
    attention_backend="fi",
)
```

---

## Known Limitations

1. **Attention Layer Compatibility**
   - QKV 投影必须使用 BF16（RoPE 需要）
   - KV Cache 必须使用 BF16
   - 因此 Attention 层不使用 FP8 输入量化

2. **Blackwell-Specific**
   - 当前实现在 sm_120 (RTX 5060 Ti) 上验证
   - 其他架构可能需要调整 attention backend

3. **Per-Tensor Scaling**
   - 当前仅支持 per-tensor 缩放
   - Per-token 缩放需要额外工作

---

## Conclusion

FP8 Input Quantization 通过利用 Blackwell 架构的原生 FP8 Tensor Core，实现了 **3.57x 性能提升**，同时减少了 5.8% 的显存占用。

**推荐配置：**
- 使用 `--fp8-input-quant` 启用 FP8 输入量化
- 使用 `--attention-backend fi` 确保 Blackwell 兼容性
- 使用 `--fp8-scale-method per_tensor` 获得最佳性能

---

## Appendix: Benchmark Scripts

- `run_fp8_single.py` - 单次运行 benchmark
- `benchmark_fp8_compare.py` - 对比两种模式
- `test_fp8_input_quant.py` - 端到端功能测试

**Results Location:** `/tmp/fp8_benchmark_results.json`
