# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-SGLang is a lightweight (~5,000 lines) yet high-performance LLM inference framework. It implements key optimizations from SGLang including Radix Cache, Chunked Prefill, Overlap Scheduling, Tensor Parallelism, FlashAttention, and FlashInfer integration.

## Commands

### Installation
```bash
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

### Running the Server
```bash
# Single GPU
python -m minisgl --model-path "Qwen/Qwen3-0.6B"

# Multi-GPU with Tensor Parallelism
python -m minisgl --model-path "meta-llama/Llama-3.1-70B-Instruct" --tensor-parallel-size 4 --port 30000

# Interactive shell mode
python -m minisgl --model-path "Qwen/Qwen3-0.6B" --shell-mode
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/kernel/test_store.py

# Run with coverage
pytest --cov=minisgl tests/
```

### Linting & Formatting
```bash
ruff check python/minisgl
black python/minisgl
mypy python/minisgl
```

## Architecture

Mini-SGLang is a distributed system with these independent processes communicating via ZMQ (control) and NCCL (tensor data):

- **API Server** (`server/api_server.py`): FastAPI frontend, OpenAI-compatible `/v1/chat/completions`
- **Tokenizer Worker** (`tokenizer/`): Text-to-tokens conversion
- **Detokenizer Worker** (`tokenizer/`): Tokens-to-text conversion
- **Scheduler Workers** (`scheduler/`): One per GPU (TP rank), manages computation and KV cache

### Request Flow
1. User → API Server → Tokenizer
2. Tokenizer → Scheduler (Rank 0) → Broadcast to all Schedulers
3. All Schedulers schedule request, trigger Engine computation
4. Scheduler (Rank 0) → Detokenizer → API Server → User

### Key Modules (`python/minisgl/`)

| Module | Purpose |
|--------|---------|
| `core.py` | `Req`, `Batch`, `Context`, `SamplingParams` dataclasses |
| `engine/` | `Engine` class - manages model, KVCache, attention backend, CUDA graph |
| `scheduler/` | `Scheduler` class - manages Engine per TP worker |
| `models/` | LLM implementations (Llama, Qwen3), weight loading from HuggingFace |
| `layers/` | Building blocks with TP support (linear, layernorm, embedding, RoPE) |
| `attention/` | FlashAttention and FlashInfer backend interfaces |
| `kvcache/` | KVCache pool/manager, Radix and Naive strategies |
| `message/` | ZMQ message types between processes |
| `kernel/` | Custom CUDA kernels via tvm-ffi |

### Key CLI Arguments
- `--model-path`: HuggingFace model ID or local path
- `--tensor-parallel-size` / `--tp-size`: Number of GPUs for TP
- `--attention-backend` / `--attn`: Attention backend (fa, fi, or "fa,fi" for prefill,decode)
- `--cache-type`: KV cache strategy ("radix" or "naive")
- `--max-prefill-length`: Chunk size for Chunked Prefill
- `--cuda-graph-max-bs`: CUDA graph max batch size (0 to disable)
- `--host` / `--port`: Server binding

## Supported Models

- Llama-3 series
- Qwen-3 series