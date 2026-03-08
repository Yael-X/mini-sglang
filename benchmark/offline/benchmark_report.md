# Mini-SGLang Offline Benchmark 测试报告

## 测试环境

- **平台**: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- **GPU**: NVIDIA GPU (14.71 GiB 可用显存)
- **模型**: Qwen/Qwen3-0.6B
- **Python**: 3.12

## 测试配置

| 参数 | 值 |
|------|-----|
| 序列数 (num_seqs) | 256 |
| 输入长度范围 | 100-1024 tokens (随机) |
| 输出长度范围 | 100-1024 tokens (随机) |
| max_seq_len_override | 4096 |
| max_extend_tokens | 16384 |
| cuda_graph_max_bs | 256 |
| temperature | 0.6 |
| ignore_eos | True |

## 运行命令

```bash
cd /home/xy/sgl/mini-sglang
source .venv/bin/activate

# 运行 benchmark (必须设置此环境变量)
MINISGL_OVERLAP_EXTRA_SYNC=1 python benchmark/offline/bench.py
```

或设置永久环境变量：

```bash
export MINISGL_OVERLAP_EXTRA_SYNC=1
python benchmark/offline/bench.py
```

## 测试结果

| 指标 | 值 |
|------|-----|
| 总 Token 数 | 133,966 tok |
| 耗时 | 53.24 s |
| 吞吐量 | 2516.27 tok/s |

**显存使用情况：**
- 加载模型前: 14.71 GiB
- KV Cache 分配: 11.80 GiB (110447 pages)
- 初始化后剩余: 0.86 GiB
- CUDA Graph 捕获后剩余: 0.51 GiB

## 踩坑记录

### 问题 1: CUDA Illegal Memory Access

**错误信息：**
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
```

**错误位置：**
```python
# python/minisgl/scheduler/scheduler.py:83
copy_done.synchronize()
```

**原因分析：**

Mini-SGLang 使用 overlap scheduling 来重叠调度和执行，以提高 GPU 利用率。但在某些情况下，CUDA 流之间的同步存在竞争条件，导致在访问 GPU 到 CPU 的异步拷贝结果时发生非法内存访问。

具体来说，在 `overlap_loop` 中：
1. 当前批次在 engine stream 中执行
2. 上一批次的结果在 scheduler stream 中处理
3. 两个流之间的同步时序可能出错，导致访问未完成的内存拷贝

**解决方案：**

设置环境变量 `MINISGL_OVERLAP_EXTRA_SYNC=1`，这会在 `_forward` 方法中添加额外的流同步：

```python
# python/minisgl/scheduler/scheduler.py:176-178
if ENV.OVERLAP_EXTRA_SYNC:
    self.stream.synchronize()
```

**相关 Issue：** https://github.com/sgl-project/mini-sglang/issues/58

### 问题 2: CUDA Graph 捕获失败（潜在问题）

如果在显存不足的情况下，CUDA Graph 捕获可能失败。建议：
- 确保 GPU 有足够显存（本测试使用了约 14.2 GiB）
- 可通过降低 `cuda_graph_max_bs` 来减少显存占用

## 其他可选配置

### 禁用 Overlap Scheduling（用于对比测试）

```bash
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python benchmark/offline/bench.py
```

### 测试不同模型

修改 `benchmark/offline/bench.py` 中的模型路径：
```python
llm = LLM(
    "Qwen/Qwen3-1.7B",  # 或 "Qwen/Qwen3-4B", "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_len_override=4096,
    max_extend_tokens=16384,
    cuda_graph_max_bs=256
)
```

## 总结

1. Mini-SGLang 在单卡上实现了约 2516 tok/s 的吞吐量
2. 必须设置 `MINISGL_OVERLAP_EXTRA_SYNC=1` 环境变量才能稳定运行
3. CUDA Graph 优化带来了显著的性能提升（支持 bs=1 到 bs=256 的多种批次大小）