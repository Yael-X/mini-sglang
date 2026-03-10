# Mini-SGLang 本地部署指南

> 本文档记录了在 WSL2 环境下部署 Mini-SGLang 推理框架的完整流程。

## 目录

- [环境概述](#环境概述)
- [部署流程](#部署流程)
  - [1. 安装 CUDA Toolkit](#1-安装-cuda-toolkit)
  - [2. 克隆项目](#2-克隆项目)
  - [3. 创建虚拟环境并安装依赖](#3-创建虚拟环境并安装依赖)
  - [4. 下载模型权重](#4-下载模型权重)
  - [5. 启动推理服务](#5-启动推理服务)
- [使用方法](#使用方法)
- [常见问题](#常见问题)
- [源码学习指南](#源码学习指南)

---

## 环境概述

### 硬件配置

| 组件 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| CPU | 16 cores |
| RAM | 15GB |
| 系统 | WSL2 (Ubuntu 24.04) |

### 软件版本

| 软件 | 版本 |
|------|------|
| Python | 3.12.3 |
| CUDA Toolkit | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.57.3 |
| Mini-SGLang | 0.1.0 |

---

## 部署流程

### 1. 安装 CUDA Toolkit

**重要**: RTX 5060 Ti 是 Blackwell 架构 (SM 120a)，需要 CUDA 12.8+ 才能支持。

```bash
# 添加 NVIDIA CUDA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装 CUDA Toolkit 12.8
sudo apt install -y cuda-toolkit-12-8

# 配置环境变量
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

**验证输出**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
```

### 2. 克隆项目

```bash
cd /home/xy/sgl
git clone https://github.com/Yael-X/mini-sglang.git
cd mini-sglang
```

### 3. 创建虚拟环境并安装依赖

```bash
# 安装 uv 包管理器
pipx install uv

# 创建虚拟环境
~/.local/bin/uv venv --python=3.12
source .venv/bin/activate

# 安装 mini-sglang 及依赖
~/.local/bin/uv pip install -e .
```

**已安装的关键依赖**:
- `torch==2.10.0` - PyTorch 深度学习框架
- `transformers==4.57.3` - HuggingFace Transformers
- `flashinfer-python==0.5.3` - 高性能注意力机制
- `sgl-kernel==0.3.21` - SGLang CUDA kernels

### 4. 下载模型权重

由于当前 transformers 版本限制，使用 Qwen3-0.6B 模型：

```bash
# 首次运行时会自动下载，也可预下载：
source .venv/bin/activate
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B'); \
  AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')"
```

**模型缓存位置**: `~/.cache/huggingface/hub/`

### 5. 启动推理服务

```bash
source .venv/bin/activate
python -m minisgl --model "Qwen/Qwen3-0.6B" --port 8000
```

**首次启动日志**:
```
[INFO] Free memory before loading model: 14.71 GiB
[INFO] Allocating 110447 pages for KV cache, K + V = 11.80 GiB
[INFO] Auto-selected attention backend: fi
[INFO] Capturing CUDA graphs with sizes: [1, 2, 4, 8, ...]
[INFO] API server is ready to serve on 127.0.0.1:8000
```

---

## 使用方法

### 启动推理服务

```bash
# 进入项目目录
cd /home/xy/sgl/mini-sglang

# 激活虚拟环境
source .venv/bin/activate

# 启动服务器
python -m minisgl --model "Qwen/Qwen3-0.6B" --port 8000
```

### API 调用示例

**查看可用模型**:
```bash
curl http://localhost:8000/v1/models
```

**对话补全 (Chat Completion)**:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 100
  }'
```

**Python 客户端**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)
print(response.choices[0].message.content)
```

### 交互式 Shell 模式

```bash
source .venv/bin/activate
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

- 输入问题直接对话
- 使用 `/reset` 清除对话历史

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径或 HuggingFace ID | 必填 |
| `--port` | 服务端口 | 8000 |
| `--host` | 绑定地址 | 127.0.0.1 |
| `--tp` | Tensor Parallelism GPU 数量 | 1 |
| `--shell` | 启动交互式 Shell | False |

---

## 常见问题

### 1. CUDA 架构不支持错误

**错误信息**: `nvcc fatal: Unsupported gpu architecture 'compute_120a'`

**原因**: RTX 5060 Ti (Blackwell 架构) 需要 CUDA 12.8+

**解决**: 升级 CUDA Toolkit 到 12.8 或更高版本

### 2. transformers 版本冲突

**错误信息**: `'Qwen3Config' object has no attribute 'rope_theta'`

**原因**: transformers 版本过高/过低与 mini-sglang 不兼容

**解决**: 安装兼容版本
```bash
pip install "transformers>=4.56.0,<=4.57.3"
```

### 3. Qwen3.5 不支持

**原因**: Qwen3.5 需要更新的 transformers 版本，但 mini-sglang 当前限制为 4.x

**解决**: 使用 Qwen3 系列模型（如 Qwen3-0.6B, Qwen3-4B）

### 4. 首次启动慢

**原因**: 首次启动需要 JIT 编译 CUDA kernels（FlashInfer, sgl_kernel）

**说明**: 编译结果会缓存到 `~/.cache/flashinfer/` 和 `~/.cache/tvm-ffi/`，后续启动会快很多

---

## 源码学习指南

Mini-SGLang 代码量约 5000 行，非常适合学习 LLM 推理系统原理。

### 核心文件

| 文件 | 功能 | 学习重点 |
|------|------|----------|
| `python/minisgl/engine/engine.py` | 推理引擎核心 | 模型加载、KV Cache 管理、前向传播 |
| `python/minisgl/scheduler/scheduler.py` | 请求调度器 | 请求队列、批处理、优先级调度 |
| `python/minisgl/radix_cache.py` | Radix Cache | 前缀共享、内存优化 |
| `python/minisgl/model_runner.py` | 模型执行 | 模型推理、采样策略 |
| `python/minisgl/attention/fi.py` | FlashInfer 注意力 | 高性能注意力实现 |
| `python/minisgl/server/api.py` | OpenAI 兼容 API | API 服务器实现 |

### 建议学习路径

1. **入口**: `python/minisgl/__main__.py` → 了解启动流程
2. **调度**: `scheduler/scheduler.py` → 理解请求调度逻辑
3. **引擎**: `engine/engine.py` → 核心推理流程
4. **缓存**: `radix_cache.py` → KV Cache 优化技术
5. **注意力**: `attention/fi.py` → FlashInfer 实现

### 调试技巧

```bash
# 启用详细日志
MINISGL_DEBUG=1 python -m minisgl --model "Qwen/Qwen3-0.6B"

# 禁用 CUDA Graph (调试用)
MINISGL_DISABLE_CUDA_GRAPH=1 python -m minisgl --model "Qwen/Qwen3-0.6B"
```

---

## 项目结构

```
/home/xy/sgl/mini-sglang/
├── .venv/                    # Python 虚拟环境
├── python/minisgl/           # 源代码
│   ├── __main__.py          # 入口文件
│   ├── engine/              # 推理引擎
│   ├── scheduler/           # 调度器
│   ├── attention/           # 注意力实现
│   ├── models/              # 模型定义
│   └── server/              # API 服务器
├── pyproject.toml           # 项目配置
└── README.md                # 项目说明
```

---

## 参考链接

- [Mini-SGLang GitHub](https://github.com/Yael-X/mini-sglang)
- [SGLang 官方文档](https://github.com/sgl-project/sglang)
- [FlashInfer 文档](https://flashinfer.ai/)
- [Qwen 模型仓库](https://huggingface.co/Qwen)

---

*文档生成时间: 2026-03-07*