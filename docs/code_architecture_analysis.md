# mini-sglang 代码架构分析与学习指南

## 1. 项目概览 (Project Overview)
- **一句话总结**：mini-sglang 以“可读、可跑、可扩展”的方式实现了 SGLang 推理系统最核心的在线服务链路：OpenAI 兼容 API、多进程 Tokenize/Detokenize、请求调度（含 prefill/decode）、KV Cache（含 Radix 前缀复用）与模型执行（含 CUDA graph 与高性能 attention backend）。
- **相对完整 SGLang 的简化点（推断）**：
  - 代码规模更小（README 明确约 5k Python 行），更偏教学与原理展示。
  - 分布式能力以 TP 为主，缺少完整生产系统常见的复杂容灾与多层调度机制。
  - 内核优化聚焦于接入 FlashAttention/FlashInfer 与少量自定义 kernel，未覆盖完整项目中更丰富/更深的融合优化矩阵。
  - 若干组件保留了 TODO 或简化实现（例如缓存管理策略与完整性检查）。

## 2. 核心架构图解 (Architecture & Modules)

### 2.1 目录结构与模块职责
- `python/minisgl/server`：前端 API 与进程启动编排（FastAPI + uvicorn + 子进程拉起）。
- `python/minisgl/tokenizer`：Tokenizer/Detokenizer worker，承接文本与 token 的双向转换。
- `python/minisgl/message`：跨进程消息协议（TokenizeMsg/UserMsg/DetokenizeMsg/UserReply 等）。
- `python/minisgl/scheduler`：调度中枢，负责 prefill/decode 编排、缓存匹配、显存页分配、结果回传。
- `python/minisgl/engine`：单 TP rank 的执行引擎，封装模型、KV cache、attention backend、采样与图执行。
- `python/minisgl/kvcache`：KV 缓存管理策略（naive/radix），其中 radix_manager 是前缀复用关键。
- `python/minisgl/models` + `python/minisgl/layers`：模型定义与算子层（以 Llama/Qwen 为主）。
- `python/minisgl/attention`：attention backend 抽象与具体实现（fa/fi/hybrid）。

### 2.2 关键模块定位

#### A) Scheduler（调度器）
- 入口：`scheduler/scheduler.py::Scheduler.run_forever`。
- 关键循环：
  - `overlap_loop`：边处理上一批结果边调度下一批，实现 CPU/GPU overlap。
  - `_schedule_next_batch`：优先 prefill，再 decode（目前策略较直接）。
- 请求队列处理：
  - 新请求进入 `PrefillManager.pending_list`。
  - `PrefillManager.schedule_next_batch` 按预算/资源挑选可运行请求。
  - `DecodeManager.running_reqs` 保存持续解码中的请求集合。

#### B) Memory Manager（内存管理 & Radix KV Cache）
- 调度侧资源门面：`scheduler/cache.py::CacheManager`。
  - `allocate`：优先用 free slots，不够则触发 `manager.evict(...)`。
  - `free_and_cache_finished_req`：请求结束后写回前缀缓存并释放可回收页。
- Radix 前缀树核心：`kvcache/radix_manager.py`。
  - `match_prefix`：沿树匹配已缓存前缀，返回 `RadixCacheHandle` 与页索引。
  - `insert_prefix`：把新前缀片段插入树，形成可复用缓存。
  - `lock_handle`：通过 ref_count 控制节点是否可驱逐（evictable/protected）。
  - `evict`：按叶节点时间戳（近似 LRU）驱逐未保护节点。

#### C) Model Runner（模型执行器）
- 执行入口：`engine/engine.py::Engine.forward_batch`。
  - 在 `Context.forward_batch` 中设置当前 batch 上下文。
  - 可走 CUDA graph replay 或普通 `model.forward()`。
  - 采样器 `Sampler.sample` 生成 next token。
- 模型调用路径示例（Llama）：
  - `models/llama.py::LlamaForCausalLM.forward` 读取全局 context 的 `batch.input_ids`。
  - `layers/attention.py::AttentionLayer.forward` 调用 `ctx.attn_backend.forward(...)`，并配合 KV 写入与 metadata。

#### D) HTTP/API Server（请求入口）
- API 定义：`server/api_server.py`。
  - `/generate` 与 `/v1/chat/completions` 接收请求。
  - 请求封装为 `TokenizeMsg` 发往 tokenizer worker。
- 启动编排：`server/launch.py::launch_server`。
  - 拉起 scheduler 进程（每 TP rank 一个）、tokenizer 与 detokenizer 子进程。

## 3. 数据流全链路追踪 (Data Flow Trace)

以一次“生成请求（Generation Request）”为例：

1. **Request (HTTP Ingress)**
   - 客户端调用 `/v1/chat/completions`。
   - API server 创建 `uid`，封装 `TokenizeMsg(uid, text, sampling_params)` 并通过 ZMQ 发给 tokenizer。

2. **Tokenizer**
   - `tokenizer/server.py::tokenize_worker` 收到 `TokenizeMsg`。
   - `TokenizeManager.tokenize(...)` 把文本转为 `input_ids`。
   - 转换为 `UserMsg(uid, input_ids, sampling_params)` 发送给 scheduler backend。

3. **Scheduler**
   - `Scheduler._process_one_msg` 接收 `UserMsg`，做长度安全检查与 `max_tokens` 截断，再交给 `PrefillManager.add_one_req`。
   - `PrefillManager.schedule_next_batch` 结合 token budget、table slot、cache 空间构造 prefill batch。

4. **Memory Alloc (KV pages + prefix reuse)**
   - `CacheManager.match_req` 调用 radix manager `match_prefix` 找可复用前缀。
   - `Scheduler._prepare_batch` 计算 `needed_size = sum(extend_len)`，`cache_manager.allocate(...)` 分配页。
   - 将页写入 `page_table`，并调用 `attn_backend.prepare_metadata(batch)`。

5. **Model Forward**
   - `Scheduler._forward` 设置 `batch.input_ids` 后调用 `engine.forward_batch`。
   - `Engine.forward_batch` 中执行模型前向（graph replay 或普通 forward），并完成采样。
   - attention backend 在 forward 中将新 K/V 写入 KV cache，再执行 paged attention。

6. **Decode（持续迭代）**
   - 每轮 forward 后 `req.complete_one()` 推进 `cached_len/device_len`。
   - `DecodeManager.filter_reqs` 保留仍可 decode 的请求，实现 continuous batching（running 集合持续滚动）。

7. **Output**
   - scheduler 将 next token 封装 `DetokenizeMsg` 发回 detokenizer。
   - detokenizer 生成增量字符串 `UserReply(incremental_output, finished)`。
   - API server 以 SSE 方式流式返回给客户端，直到 `[DONE]`。

## 4. 核心算法代码定位 (Key Algorithms)

### 4.1 Radix Attention（前缀树缓存共享）
- **核心文件**：`python/minisgl/kvcache/radix_manager.py`。
- **关键函数**：
  - `_walk(input_ids)`：按 token 逐层下钻，匹配节点 key；若部分匹配则 `_split_at` 分裂节点。
  - `match_prefix(input_ids)`：返回“最长前缀句柄 + 对应缓存页索引串”。
  - `insert_prefix(input_ids, indices)`：把未命中后缀插入树，记录页索引。
  - `lock_handle` / `evict`：引用计数保护 + 叶子时间戳驱逐。
- **工作机制（简述）**：
  1) 新请求先匹配历史前缀并命中一段已有 KV 页；
  2) 仅对未命中后缀做 prefill 计算与页分配；
  3) 请求结束后把新前缀写回 radix tree；
  4) 显存紧张时驱逐“未被引用的最旧叶子节点”。

### 4.2 Continuous Batching（连续批处理）
- **核心文件**：
  - `scheduler/decode.py`（`running_reqs` 生命周期管理）；
  - `scheduler/prefill.py`（新请求按预算持续注入）；
  - `scheduler/scheduler.py`（主循环选择 prefill 或 decode batch）。
- **核心逻辑**：
  - 并非“一次性静态 batch”，而是每步迭代都根据当前状态重新组 batch。
  - 已进入 decode 的请求留在 `running_reqs` 中持续推进，新请求可被穿插进 prefill。
  - `overlap_loop` 将“上一批结果处理”和“下一批计算”并行化，进一步提高吞吐。

## 5. 循序渐进的学习计划 (Step-by-Step Study Plan)

### 阶段 1：先看系统边界与消息协议
- **目标**：理解进程拓扑、入口出口与消息形状。
- **建议文件**：
  - `docs/structures.md`
  - `server/launch.py`
  - `server/api_server.py`
  - `message/*.py`
- **重点函数/类**：`launch_server`、`FrontendManager`、`TokenizeMsg/UserMsg/DetokenizeMsg/UserReply`。

### 阶段 2：看请求状态与调度数据结构
- **目标**：吃透 `Req` / `Batch` 生命周期与 prefill/decode 切换。
- **建议文件**：
  - `core.py`
  - `scheduler/prefill.py`
  - `scheduler/decode.py`
  - `scheduler/scheduler.py`
- **重点函数/类**：`Req.complete_one`、`PrefillAdder.try_add_one`、`DecodeManager.filter_reqs`、`Scheduler._schedule_next_batch`。

### 阶段 3：看 KV Cache 与 Radix 前缀复用
- **目标**：理解缓存命中、保护、插入、驱逐闭环。
- **建议文件**：
  - `scheduler/cache.py`
  - `kvcache/radix_manager.py`
  - `kvcache/naive_manager.py`（用于对照）
- **重点函数/类**：`match_prefix`、`insert_prefix`、`lock_handle`、`evict`、`free_and_cache_finished_req`。

### 阶段 4：看执行引擎与模型前向
- **目标**：把“调度输出”映射到“模型执行输入”。
- **建议文件**：
  - `engine/engine.py`
  - `layers/attention.py`
  - `attention/fi.py`（或 `fa.py`）
  - `models/llama.py`（或 qwen 系列）
- **重点函数/类**：`Engine.forward_batch`、`prepare_metadata`、`AttentionLayer.forward`、`LlamaForCausalLM.forward`。

### 阶段 5：回到端到端时序与性能特性
- **目标**：理解为什么它快，以及何处还能优化。
- **建议文件**：
  - `docs/features.md`
  - `scheduler/scheduler.py`（overlap loop）
  - `engine/graph.py`（CUDA graph）
- **重点函数/类**：`overlap_loop`、`GraphRunner` 相关 capture/replay 路径。

### 3 个 Insightful Questions（阅读时深挖）
1. **Radix cache 的命中收益与碎片代价如何平衡？**
   - 当前实现中，节点分裂与页索引拼接会带来什么元数据成本？在高并发短请求分布下收益是否仍显著？
2. **Prefill 优先策略是否始终最优？**
   - `_schedule_next_batch` 目前“prefill first”会如何影响 tail latency？在哪些 workload 下应改为 decode-first 或混合优先级？
3. **Overlap scheduling 与 CUDA graph 的交互边界在哪里？**
   - 当 batch 形状频繁变化、图命中率下降时，overlap 还能否抵消 launch overhead？如何设计自适应阈值？
