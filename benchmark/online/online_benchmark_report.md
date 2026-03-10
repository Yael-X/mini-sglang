# Mini-SGLang Online Benchmark 测试报告

## 测试环境

- **平台**: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16 GiB)
- **模型**: Qwen/Qwen3-0.6B
- **Python**: 3.12

## 测试配置

### 服务器配置

| 参数 | 值 |
|------|-----|
| 模型路径 | Qwen/Qwen3-0.6B |
| 端口 | 1919 |
| max_seq_len_override | 4096 |
| max_extend_tokens | 16384 |
| cuda_graph_max_bs | 256 |
| 内存使用 | ~15.8 GiB |

### Benchmark 配置

| 参数 | 值 |
|------|-----|
| 并发请求数 | 64 |
| 输入长度范围 | 1-1024 tokens (随机) |
| 输出长度范围 | 16-1024 tokens (随机) |
| 测试批次 | [64] |

## 运行命令

### 1. 启动服务器

```bash
cd /home/xy/sgl/mini-sglang
source .venv/bin/activate

# 必须设置此环境变量
MINISGL_OVERLAP_EXTRA_SYNC=1 python -m minisgl \
    --model-path Qwen/Qwen3-0.6B \
    --port 1919 \
    --max-seq-len-override 4096 \
    --max-extend-length 16384 \
    --cuda-graph-max-bs 256
```

### 2. 运行 Benchmark 客户端

```bash
# 在另一个终端
cd /home/xy/sgl/mini-sglang
source .venv/bin/activate
MINISGL_OVERLAP_EXTRA_SYNC=1 python benchmark/online/bench_simple.py
```

## 测试结果

### 配置一：短输入 (MAX_INPUT=1024)

| 参数 | 值 |
|------|-----|
| 并发请求数 | 64 |
| 输入长度范围 | 1-1024 tokens |
| 服务器 max_seq_len_override | 4096 |

#### 吞吐量指标

| 指标 | 值 |
|------|-----|
| 总请求数 | 64 |
| 总 Token 数 | 36,319 |
| 测试时长 | 14.94 s |
| Token 吞吐量 | **2430 tok/s** |
| 请求吞吐量 | **4.28 req/s** |

#### 延迟指标

| 指标 | 平均值 | P50 | P90 | P99 | 最大值 |
|------|--------|-----|-----|-----|--------|
| TTFT (首Token延迟) | 856 ms | 820 ms | 1078 ms | 1078 ms | 1078 ms |
| TPOT (每Token延迟) | 15 ms | 15 ms | 17 ms | 19 ms | 982 ms |
| E2E (端到端延迟) | 9.62 s | 10.46 s | 14.46 s | 14.94 s | 14.94 s |

---

### 配置二：长输入 (MAX_INPUT=8192)

| 参数 | 值 |
|------|-----|
| 并发请求数 | 64 |
| 输入长度范围 | 1-8192 tokens |
| 服务器 max_seq_len_override | 16384 |

#### 吞吐量指标

| 指标 | 值 |
|------|-----|
| 总请求数 | 64 |
| 总 Token 数 | 31,168 |
| 测试时长 | 52.83 s |
| Token 吞吐量 | **590 tok/s** |
| 请求吞吐量 | **1.21 req/s** |

#### 延迟指标

| 指标 | 平均值 | P50 | P90 | P99 | 最大值 |
|------|--------|-----|-----|-----|--------|
| TTFT (首Token延迟) | 14465 ms | 10460 ms | 34266 ms | 40700 ms | 40700 ms |
| TPOT (每Token延迟) | 36 ms | 32 ms | 35 ms | 254 ms | 3229 ms |
| E2E (端到端延迟) | 31.82 s | 33.97 s | 51.50 s | 52.82 s | 52.82 s |

---

### 性能对比

| 指标 | 短输入 (1024) | 长输入 (8192) | 变化 |
|------|---------------|---------------|------|
| Token 吞吐量 | 2430 tok/s | 590 tok/s | **-76%** |
| 请求吞吐量 | 4.28 req/s | 1.21 req/s | **-72%** |
| TTFT (avg) | 856 ms | 14465 ms | **+1590%** |
| TPOT (avg) | 15 ms | 36 ms | **+140%** |
| E2E (avg) | 9.62 s | 31.82 s | **+231%** |

**原因分析：**
- 长输入序列需要更多的 prefill 计算时间
- KV Cache 占用更大，内存带宽压力增加
- Prefill 阶段是计算密集型，Decode 阶段是内存密集型

### 显存使用

| 阶段 | 显存使用 |
|------|----------|
| 加载模型前 | 14.71 GiB 可用 |
| KV Cache 分配 | 11.80 GiB (110447 pages) |
| 初始化后剩余 | 0.84 GiB |
| CUDA Graph 捕获后 | 0.55 GiB |
| 运行时峰值 | ~15.8 GiB |

## 踩坑记录

### 问题 1: 服务器启动缺少环境变量

**错误信息：**
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**原因：** 服务器未设置 `MINISGL_OVERLAP_EXTRA_SYNC=1` 环境变量，导致 overlap scheduling 中的 CUDA 流同步竞争条件。

**解决方案：**
```bash
MINISGL_OVERLAP_EXTRA_SYNC=1 python -m minisgl ...
```

### 问题 2: 输入长度超出服务器限制

**错误信息：**
```
WARNING Input sequence length 5939 exceeds 4096, request 1 is dropped.
```

**原因：** Benchmark 配置的 `MAX_INPUT=8192` 超过了服务器的 `max_seq_len_override=4096` 限制。

**解决方案：** 修改 `benchmark/online/bench_simple.py` 中的 `MAX_INPUT` 值：
```python
MAX_INPUT = 1024  # 原值为 8192
```

### 问题 3: GPU 利用率观察

**现象：** Benchmark 结束后 GPU 利用率显示 34%。

**实际情况：** 在 Benchmark 运行期间，GPU 利用率是较高的，只是测试完成得很快（~15秒），用户在结束后查看时利用率已经下降。

**性能分析：**
- Qwen3-0.6B 是小模型，计算量相对有限
- Decode 阶段是内存带宽受限而非计算受限
- 在线场景下 Continuous Batching 能有效利用 GPU 资源

**优化建议：**
- 使用更大的模型以进一步提高 GPU 利用率
- 增加并发请求数进行压力测试
- 延长测试时间以观察稳态性能

## Online vs Offline Benchmark 对比

| 指标 | Offline | Online |
|------|---------|--------|
| 吞吐量 | 2516 tok/s | 2430 tok/s |
| 延迟 (TTFT) | N/A | 856 ms avg |
| 延迟 (TPOT) | N/A | 15 ms avg |
| 适用场景 | 批量推理 | 实时服务 |
| GPU 利用率 | 较高 | 较低 |

---

## Trace-based Benchmark (待复测验证)

### 测试配置

| 参数 | 值 |
|------|-----|
| 请求数 | 200 |
| Trace 来源 | Qwen Bailian 使用痕迹 |
| 服务器 max_seq_len_override | 32768 |
| Scale 参数 | [0.4, 0.6, 0.8] |

### 初步结果

| Scale | 吞吐量 | 请求吞吐量 | TTFT (avg) | TPOT (avg) | E2E (avg) |
|-------|---------------|------------|------------|------------|-----------|
| 0.4 | 1301 | 3.64 | 884 ms | 32 ms | 12.4 s |
| 0.6 | 1235 | 3.45 | 23 ms | 8 ms | 3.0 s |
| 0.8 | 991 | 2.77 | 21 ms | 8 ms | 2.7 s |

### Scale 参数说明

- **Scale 越小**：请求到达间隔越短，系统负载越高
- **Scale 越大**：请求到达间隔越长，系统空闲时间越多

### 踩坑记录

1. **输入长度超限**：Trace 中存在 `input_length=16744` 的请求，需要 `max_seq_len_override≥32768`
2. **超时配置**：需要增加 httpx 客户端超时时间
   ```python
   timeout = httpx.Timeout(600.0, connect=60.0)
   async with OpenAI(..., timeout=timeout) as client:
   ```
3. **请求卡住问题**：输入长度超过服务器限制时，请求被丢弃但客户端仍在等待响应

### 运行命令

```bash
# 启动服务器
MINISGL_OVERLAP_EXTRA_SYNC=1 python -m minisgl \
    --model-path Qwen/Qwen3-0.6B \
    --port 1919 \
    --max-seq-len-override 32768 \
    --max-extend-length 16384 \
    --cuda-graph-max-bs 256

# 运行 benchmark
MINISGL_OVERLAP_EXTRA_SYNC=1 python benchmark/online/bench_qwen.py
```

### 备注

此结果需要复测验证，可能存在以下问题：
- 请求卡住导致部分数据不准确
- 需要检查服务器日志确认所有请求正常处理
- 建议增加错误处理和重试机制