# Qwen3-8B-FP8 推理服务测试指南

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| 显存 | 16 GB |
| 架构 | Blackwell (SM 12.0) |
| CUDA | 13.0 |
| Python | 3.12 |

## 关键发现

### ⚠️ RTX 50 系列 GPU 注意事项

**问题**: RTX 5060 Ti (Blackwell 架构, SM 12.0) 目前不被 TRTLLM 后端支持。

**解决方案**: 必须使用 `--attention-backend fi` (FlashInfer) 参数。

```
错误: RuntimeError: Unsupported architecture
原因: TRTLLM 后端不支持 SM 12.0
```

### 网络问题

如果无法访问 huggingface.co，需要使用本地模型路径：

```bash
# 查找本地模型路径
find ~/.cache/huggingface/hub -name "config.json" -path "*Qwen3-8B-FP8*"
```

## 启动命令

### 基础命令

```bash
python -m minisgl \
  --model-path "Qwen/Qwen3-8B-FP8" \
  --fp8-keep-quantized \
  --attention-backend fi \
  --port 1919
```

### 使用本地路径（离线环境）

```bash
LOCAL_MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/<commit-hash>"

python -m minisgl \
  --model-path "$LOCAL_MODEL_PATH" \
  --fp8-keep-quantized \
  --attention-backend fi \
  --port 1919
```

### 完整推荐参数

```bash
python -m minisgl \
  --model-path "Qwen/Qwen3-8B-FP8" \
  --fp8-keep-quantized \
  --attention-backend fi \
  --port 1919 \
  --cuda-graph-max-bs 4 \
  --page-size 1 \
  --memory-ratio 0.85
```

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--fp8-keep-quantized` | 保持 FP8 权重不反量化 | **必须启用** |
| `--attention-backend fi` | 使用 FlashInfer 后端 | RTX 50 系列必须 |
| `--cuda-graph-max-bs` | CUDA Graph 最大批次 | 4 |
| `--memory-ratio` | KV Cache 显存比例 | 0.85 |
| `--page-size` | KV Cache 页大小 | 1 |

## 显存占用

| 项目 | 大小 |
|------|------|
| FP8 权重 | ~7.0 GB |
| Scale 张量 | ~0.1 GB |
| Embeddings (BF16) | ~2.4 GB |
| KV Cache (BF16) | ~3.6 GB |
| CUDA Graph | ~1.0 GB |
| **总计** | **~14.5 GB** |
| **剩余** | **~1.5 GB** |

## 测试方法

### 方法 1: 使用测试脚本（推荐）

测试脚本位置: `client_qwen3_8b.py`

```bash
# 启动推理服务（终端 1）
python -m minisgl \
  --model-path "Qwen/Qwen3-8B-FP8" \
  --fp8-keep-quantized \
  --attention-backend fi \
  --port 1919

# 运行测试脚本（终端 2）
python client_qwen3_8b.py "你好，请介绍一下你自己"
python client_qwen3_8b.py "写一个快速排序算法"
python client_qwen3_8b.py "解释一下什么是机器学习"
```

测试脚本输出示例：
```
=== Qwen3-8B-FP8 推理测试 ===
服务器: http://localhost:1919/v1/chat/completions
问题: 你好，请介绍一下你自己
--------------------------------------------------
嗯，用户让我简单介绍一下自己。首先，我需要明确自己的身份和功能...
==================================================
总耗时: 3.45s
首token延迟 (TTFT): 0.52s
生成tokens: 156
吞吐量: 45.22 tokens/s
```

### 方法 2: 使用 curl

```bash
curl -X POST http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-FP8",
    "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己。"}],
    "max_tokens": 100
  }'
```

### 方法 3: 使用 Python OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:1919/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-FP8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## 完整测试流程

```bash
# 1. 进入项目目录
cd /home/xy/sgl/mini-sglang

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 启动推理服务
python -m minisgl \
  --model-path "Qwen/Qwen3-8B-FP8" \
  --fp8-keep-quantized \
  --attention-backend fi \
  --port 1919 \
  --cuda-graph-max-bs 4 \
  --memory-ratio 0.85

# 4. 在另一个终端运行测试
python client_qwen3_8b.py "你好"
```

## 常见问题

### 1. 端口被占用

```bash
# 检查端口
ss -tlnp | grep 1919

# 杀死占用进程
pkill -f minisgl
```

### 2. 显存不足

如果遇到 OOM：
1. 降低 `--memory-ratio` (如 0.75)
2. 减少 `--cuda-graph-max-bs` (如 2)
3. 设置 `--num-pages` 限制 KV Cache

### 3. TRTLLM 不支持架构

```
RuntimeError: Unsupported architecture
```

**解决**: 添加 `--attention-backend fi`

### 4. 网络超时

```
MaxRetryError: Failed to establish a new connection
```

**解决**: 使用本地模型路径

### 5. 测试脚本连接失败

```
连接错误: [Errno 111] Connection refused
```

**解决**: 确认推理服务已启动，检查端口是否正确

## 性能优化建议

1. **CUDA Graph**: 启用可提升 decode 阶段性能
2. **Radix Cache**: 默认启用，提升前缀复用
3. **Chunked Prefill**: 通过 `--max-prefill-length` 控制

## 相关文档

- [FP8 适配报告](./fp8_adaptation_report.md)
- [Mini-SGLang README](../CLAUDE.md)

---

*测试日期: 2026-03-08*
*GPU: RTX 5060 Ti 16GB*