# mini-sglang 代码级深度笔记（调度 / KV Cache / 引擎 / 并发通信）

> 面向：按代码逐行对照学习。以下内容严格对应当前仓库实现。

## 一、调度模块（`scheduler.py` / `prefill.py` / `decode.py`）

### 1) 请求状态机（Req 生命周期）

mini-sglang 并没有显式 `enum state`，而是通过“容器归属 + 关键字段”隐式表达状态：

- **Pending**：请求刚进 `PrefillManager.pending_list`，类型为 `PendingReq`。
- **Running/Prefill**：被 `PrefillManager.schedule_next_batch` 选中后变成 `Req` 或 `ChunkedReq`，进入 prefill batch。
- **Running/Decode**：经过一次 forward 后，`DecodeManager.filter_reqs` 把仍 `can_decode=True` 的请求放入 `running_reqs`。
- **Finished**：`_process_last_data` 判断达到 `max_tokens` 或 EOS，发送 finished，随后释放 table slot + cache handle。

`Req` 的关键字段变化：

- `cached_len`: 表示“前缀已在 KV cache 中可复用的长度”，每次 `complete_one()` 会先同步到 `device_len`。
- `device_len`: 当前 GPU 侧序列长度（含输入 + 已生成 token）。每次 decode 成功后 `+1`。
- `max_device_len = input_len + output_len`：请求硬上限。
- `input_ids`: CPU 张量，`append_host` 持续把新 token 追加进去（这里相当于你问题里的 `output_ids` 语义）。

特别注意：这个仓库没有独立 `output_ids` 字段，而是把输出 token 直接 append 到 `Req.input_ids`。

### 2) Continuous Batching：prefill 与 decode 的取舍

`Scheduler._schedule_next_batch` 当前策略是 **prefill-first**：

1. 先尝试 `prefill_manager.schedule_next_batch(prefill_budget)`；
2. 若 prefill 取不到，再走 `decode_manager.schedule_next_batch()`。

因此当前代码**不做同一个 batch 的 prefill/decode 混合**，而是通过连续循环在批次级切换 phase。实现上每个 `Batch` 有单一 `phase`（`"prefill"` 或 `"decode"`）。

但它支持“系统层面的并行推进”：

- prefill batch 在跑时，历史 decode 请求会被 `inflight_tokens` 计入预留预算，避免 cache 过度承诺；
- overlap loop 把“上一批结果处理”与“下一批 forward”流水并行。

### 3) Overlap Loop（CPU/GPU 流水并行）

`overlap_loop(last_data)` 的核心步骤：

1. 根据是否有 in-flight 工作决定 `receive_msg` 是否阻塞；
2. 吃掉新消息，更新 pending 队列；
3. 立刻调度并准备下一批（CPU 侧元数据、page table 写入等）；
4. 在 engine stream 上发起 GPU forward；
5. 同时处理上一批结果（D2H token + finished 回收）；
6. 返回当前 batch 作为下次 `last_data`。

ASCII 时序（简化）：

```
Time --->

CPU(main stream):   recv+schedule(B1)  prepare(B1)   process(B0 result)  recv+schedule(B2)  prepare(B2)  process(B1 result)
GPU(engine stream):                 forward(B1) ----------------------->                 forward(B2) ----------------------->

Data handoff:        last_data=B0 ---------------------> last_data=B1 ---------------------> last_data=B2
```

### 4) 关键代码导读（精选 26 行，逐行注释）

以下是最关键的一段（来自 `scheduler/scheduler.py`，为理解方便添加注释）：

```python
# 1) 调度策略：prefill 优先，取不到再 decode
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)  # 先吃新请求
    or self.decode_manager.schedule_next_batch()                   # 再推进旧请求
)

# 2) 计算本批需要的新页数（仅为 extend 段分配）
needed_size = sum(r.extend_len for r in batch.reqs)
out_loc = self.cache_manager.allocate(needed_size)

# 3) 为 CUDA Graph 可能需要的 batch padding
if padding_size := self.engine.graph_runner.pad_batch(batch):
    out_loc = F.pad(out_loc, (0, padding_size), value=self.engine.dummy_page)

batch.out_loc = out_loc
batch.positions = _make_positions(batch, self.device)             # [cached_len, ..., device_len-1]
input_mapping = _make_input_tuple(batch, self.device)             # (table_idx, position)
write_mapping = _make_write_tuple(batch, self.device)             # 每个 req 下一 token 写回位置

# 4) 先写 page_table，再让 attention backend 基于 page_table 组 metadata
self.page_table[input_mapping] = batch.out_loc
self.engine.attn_backend.prepare_metadata(batch)

# 5) 真正前向：从 token_pool gather 输入 token
batch.input_ids = self.token_pool[input_mapping]
forward_output = self.engine.forward_batch(batch, sample_args)

# 6) 把采样出的 next token 写回 token_pool 的“下一个位置”
self.token_pool[output_mapping] = forward_output.next_tokens_gpu

# 7) 更新 decode 集（只保留 can_decode 的请求）
self.decode_manager.filter_reqs(forward_input.batch.reqs)

# 8) 处理上一批 D2H 结果：append_host + finished 判定
req.append_host(next_token_id.unsqueeze(0))
finished = not req.can_decode
if not req.sampling_params.ignore_eos:
    finished |= next_token == self.eos_token_id

# 9) finished 请求先从 decode_manager 移除
if finished:
    self.finished_reqs.add(req)
    self.decode_manager.remove_req(req)

# 10) 非 ongoing 的 finished 请求做最终资源回收（table slot + cache）
self.table_manager.free(req.table_idx)
self.cache_manager.free_and_cache_finished_req(
    req.cache_handle,
    req.input_ids[: req.cached_len],
    self.page_table[req.table_idx, : req.cached_len],
)
```

---

## 二、Radix Attention 与缓存管理（`kvcache/radix_manager.py` / `scheduler/cache.py`）

### 1) Radix Tree 结构与存储

`RadixTreeNode` 是“压缩前缀树”节点：

- `children: Dict[int, RadixTreeNode]`：按“下一 token id”分叉；
- `_key`: 当前边压缩后的 token 子串；
- `_value`: 与 `_key` 对齐的 page index 子串（逻辑上每 token 对应 page）；
- `_length`: 该压缩段长度；
- `ref_count`: 引用计数（>0 不可驱逐）；
- `timestamp`: 最近访问时间（LRU 依据）。

**例子**（概念化）

先插入 `Hello World`（token 序列记为 `H e l l o _ W o r l d`）：

- root 下出现一条压缩边 `[H...d]`，value 为对应 page 索引串。

再插入 `Hello SGLang`：

- walk 到公共前缀 `Hello _` 后遇到分歧；
- 触发 `_split_at(pos)`：把原节点切成“公共前缀节点 + 原后缀子节点”；
- 新请求的后缀 `SGLang` 作为公共前缀节点的新 child。

这就是 radix 的“节点分裂”核心：只在首次分歧点切分，避免逐 token 建树。

### 2) 前缀匹配：`match_prefix(input_ids)` 与 handle 语义

匹配流程：

1. `_walk` 从 root 开始；
2. 用当前 token 命中 child；
3. `get_match_len` 对压缩边做快速比较；
4. 若部分匹配，立即 `_split_at` 并返回 split 后节点；
5. 完全匹配则继续深入，直到输入耗尽或 child 不存在。

返回值：

- `RadixCacheHandle(prefix_len, node)`：记录“匹配长度 + 匹配终点节点”；
- 以及拼接后的 matched page indices 张量。

为什么要 `ref_count + lock/unlock`：

- scheduler 选中请求后，会 `lock(handle)`，沿路径把节点 `ref_count += 1`；
- 这样 evict 时不会删掉正在被请求使用的前缀；
- 请求结束后 `unlock`，`ref_count` 归零的节点重新变为可驱逐。

### 3) 驱逐策略：`evict()` 如何做 LRU

流程：

1. 先 DFS 收集 `ref_count==0` 的 leaf；
2. 用 `heapq`（按 `timestamp` 最小堆）反复弹出最老 leaf；
3. 删除该 leaf 并回收其 `_value`（即 page indices）；
4. 若父节点因此变叶子且 `ref_count==0`，继续入堆（级联回收）。

LRU 时间戳更新点：`_walk` 每次成功访问完整节点时 `node.timestamp = tic`。

### 4) 逻辑页 vs 物理页映射

- 对 Scheduler 来说，`Req.table_idx + position` 映射到 `page_table[table_idx, pos]`，这是“逻辑页表视图”；
- 对 KV cache 来说，`page index` 是底层物理 cache page 槽位。

`CacheManager.allocate(needed_len)` 会从 `_free_slots`（或先 evict）拿到物理页 index，之后 scheduler 写入 `page_table[input_mapping] = out_loc`，模型 attention 再按 page_table 去读写真正 KV。

---

## 三、Forward 与硬件加速（`engine/` + `layers/attention.py`）

### 1) Scheduler 如何准备输入 Tensor

`_prepare_batch` 做了 4 件关键事：

1. 分配新页 `out_loc`；
2. 构造 `positions`；
3. 构造二维索引 `input_mapping=(table_idx, positions)`，用于从 `token_pool` gather；
4. 构造 `write_mapping=(table_idx, device_len or -1)`，用于写回 next token。

`_forward` 中 `batch.input_ids = token_pool[input_mapping]`，这一步把多请求“逻辑拼接”为单一连续输入向量。

### 2) Metadata 到 GPU 的关键信息

以 FI/FA backend 为例，metadata 典型包含：

- `cu_seqlens_q`, `cu_seqlens_k`（ragged batch 边界）；
- `indices`（拼接后的 paged block/page table 索引）；
- `last_page_len`、`seq_lens`；
- 头数、head_dim、dtype、wrapper。

这些信息是 paged attention kernel 的“索引描述符”，替代传统 dense attention 的 `[B, S, S]` mask 语义。

### 3) Backend 抽象与选择（FI / FA / Hybrid）

- `create_attention_backend("auto")` 按 GPU 架构选择：
  - Blackwell 倾向 `fi`；
  - Hopper 默认 `fa,fi`（prefill 用 FA，decode 用 FI）；
  - 更旧架构走 `fi`。
- 若传入 `"fa,fi"`，会构建 `HybridBackend`，在 `batch.is_prefill` / `is_decode` 处分流。

### 4) CUDA Graph：capture / replay / 动态 batch

`GraphRunner` 策略是 **多 batch-size 预录制** + **padding 到最近可用 bs**：

- `_determine_cuda_graph_bs` 生成要 capture 的 bs 列表（如 1,2,4,8,...）；
- `_capture_graphs` 对每个 bs 建 `CUDAGraph`；
- `pad_batch` 把 decode batch 补到下一个支持的 bs（用 dummy req）；
- `replay` 时 copy 当前 batch 数据进 capture buffer，再 `g.replay()`。

因此不是单图通吃，也不是纯 bucketing 丢弃；是“离散 bs 图池 + 运行时补齐”。

### 5) 采样逻辑（Top-K/Top-P 在哪里做）

`Sampler.sample` 在 GPU 侧完成：

- greedy：`torch.argmax(logits, dim=-1)`（GPU）；
- 非 greedy：调用 `flashinfer.sampling` 的 softmax + top-k/top-p 采样 kernel。

最后仅把选中的 `next_tokens_gpu` 异步拷回 CPU（用于 detokenizer/回包）。

---

## 四、进程模型与消息通信（`server/` + `message/`）

### 1) 启动拓扑（`launch_server`）

默认启动：

- `world_size` 个 Scheduler 进程（TP rank）；
- `num_tokenizer` 个 Tokenizer 进程；
- 1 个 Detokenizer 进程；
- 1 个 API 主进程（FastAPI/uvicorn）。

通信协议：主要是 **ZeroMQ**（Push/Pull + Pub/Sub） + TP 内部 `torch.distributed`。

### 2) HTTP -> UserMsg 路径（以 `TokenizeMsg` / `UserMsg` 为主线）

1. FastAPI `/generate` 或 `/v1/chat/completions` 收到请求；
2. 封装 `TokenizeMsg(uid, text, sampling_params)` 发到 tokenizer 队列；
3. tokenizer worker 把文本编码成 `input_ids`；
4. 生成 `UserMsg(uid, input_ids, sampling_params)` 发给 scheduler backend 队列；
5. scheduler 收到 `UserMsg`，转入 `PrefillManager.pending_list`。

### 3) 生成 token 如何回流前端（增量 detokenize）

1. Scheduler 对每步采样结果发 `DetokenizeMsg(uid, next_token, finished)`；
2. Detokenizer 维护每个 uid 的 `DecodeStatus`（含 offsets）；
3. 只对“新增 token 段”做 `batch_decode`，并通过 `sent_offset` 仅发送增量字符串；
4. 转成 `UserReply(uid, incremental_output, finished)` 回前端，FastAPI 以 SSE 流式输出。

它避免整句反复 decode 的关键是：`read_offset/surr_offset/sent_offset` 三重偏移。

### 4) 高并发瓶颈（基于当前代码结构）

可能瓶颈点：

- **单 Scheduler 线程循环**：每个 TP rank 内调度逻辑是单线程 event loop；
- **tokenize/detokenize Python 侧开销**：尤其 tokenizer 目前 TODO 提示“未做批量 tokenization 优化”；
- **ZMQ hops 增加延迟**：Frontend <-> Tokenizer <-> Scheduler 多跳；
- **CPU 元数据构建 + H2D/D2H**：虽然 overlap 已缓解，但超高 QPS 仍可能成为瓶颈；
- **TP 多卡同步与广播**：多 rank 模式下 rank0 广播消息与 barrier 也会增加尾延迟。

