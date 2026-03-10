# Mini-SGLang 推理流程图（含 FP8 适配点）

本文给出从启动参数到在线推理的整体链路，并标注 FP8 动态反量化相关的适配点。

## 1. 整体流程（Mermaid）

```mermaid
flowchart TD
    A[启动 minisgl 进程\npython -m minisgl ...] --> B[parse_args\n读取 --fp8-keep-quantized]
    B --> C[构造 ServerArgs / EngineConfig\n写入 fp8_keep_quantized]
    C --> D[Engine.__init__]

    D --> E[create_model(model_config, use_fp8=fp8_keep_quantized)]
    E --> E1{use_fp8?}
    E1 -- 否 --> E2[构建常规 Linear 层\nLinearQKVMerged / LinearRowParallel...]
    E1 -- 是 --> E3[构建 FP8 Linear 层\nFp8LinearQKVMerged / Fp8LinearRowParallel...]

    D --> F[load_weight(model_path, device, fp8_keep_quantized)]
    F --> G{检测量化类型\nconfig.json.quantization_config.quant_method}

    G -- fp8 + keep_quantized=true --> H[加载 FP8 原始权重与 scale\n_ load_fp8_weights_without_dequant]
    H --> H1[合并 qkv / gate_up 及对应 scale\n_ merge_fp8_state_dict]
    H1 --> H2[按 TP 分片 weight 与 scale\n_ shard_fp8_weight/_shard_fp8_scale]
    H2 --> I[BaseOP.load_state_dict\n映射到 weight_fp8 + weight_scale]

    G -- fp8 + keep_quantized=false --> J[加载并离线反量化为 BF16\n_ load_fp8_weights -> _dequantize_fp8_linear]
    J --> J1[再进入常规 weight 参数]

    G -- 非 fp8 --> K[常规 BF16/FP16 权重加载]

    I --> L[进入推理循环 forward]
    J1 --> L
    K --> L

    L --> M[每层 Linear.forward]
    M --> N{是否 FP8 Linear?}
    N -- 否 --> O[F.linear(x, weight, bias)]
    N -- 是 --> P[Fp8LinearMixin._forward_fp8]

    P --> Q[Fp8DequantBuffer.get_instance\n按 CUDA stream 复用 BF16 buffer]
    Q --> R[dequantize_fp8_block\nFP8 x scale -> BF16 临时权重]
    R --> S[F.linear(x, dequant_weight, bias)]

    O --> T[Attention/MLP 聚合 -> logits]
    S --> T
    T --> U[Scheduler/Server 流式输出 tokens]
```

---

## 2. FP8 适配点与“为什么需要适配”

### 适配点 A：CLI 与配置层新增 `--fp8-keep-quantized`
- **位置**：`server/args -> EngineConfig`
- **为什么需要适配**：
  - FP8 路径是“行为切换”而不是纯性能优化，必须由参数显式控制。
  - 需要在启动期就决定模型结构（FP8 Linear 还是普通 Linear）和权重加载策略（保留量化 or 离线反量化）。
  - 避免默认行为改变影响现有 BF16/FP16 用户。

### 适配点 B：模型构建支持 `use_fp8`（Linear 模块替换）
- **位置**：`create_model -> 各模型类 -> layers/utils`
- **为什么需要适配**：
  - FP8 权重不再是 `weight` 单张量，而是 `weight_fp8 + weight_scale` 二元结构。
  - 前向逻辑需要先反量化再 GEMM，普通 Linear 无法直接复用。
  - TP 下不同线性层分片规则不同（QKV/O/RowParallel），需要对应 FP8 版本保证 shape/通信一致。

### 适配点 C：权重加载器识别 FP8 模型并走分支
- **位置**：`models/weight.py`
- **为什么需要适配**：
  - FP8 safetensors 中 scale 与权重分开存储，命名规则与普通 BF16 权重不同。
  - q/k/v 与 gate/up 在框架内部是 merge 后算子，必须同时 merge 对应 scale 才能数值对齐。
  - TP 场景下不仅 weight 要分片，scale 也必须按同一维度分片，否则反量化形状/数值都会错。

### 适配点 D：`BaseOP.load_state_dict` 增加 FP8 参数映射
- **位置**：`layers/base.py`
- **为什么需要适配**：
  - 传统 `state_dict` 仅匹配 `weight`；FP8 需要把 `weight` 映射到 `weight_fp8`，并关联 `weight_scale`。
  - 若不做映射，FP8 权重会被当作“unexpected keys”或丢失 scale，导致前向不可用。

### 适配点 E：新增反量化 kernel（`dequantize_fp8_block`）
- **位置**：`kernel/fp8.py`
- **为什么需要适配**：
  - 目标是“运行时反量化 + 显存友好”，需要避免每次创建大中间张量。
  - block-wise scale 与权重做 broadcast 乘法，能保持与量化格式一致的数学语义。
  - 把反量化从加载时迁移到前向时，是释放常驻显存（为 KV cache 腾空间）的关键。

### 适配点 F：共享缓冲区管理（`Fp8DequantBuffer`）
- **位置**：`kernel/fp8.py`
- **为什么需要适配**：
  - 每个线性层每次都临时 `torch.empty` 会造成显存碎片和 allocator 抖动。
  - 复用“按需扩张”的 BF16 buffer，可降低频繁分配释放成本。
  - 按 CUDA stream 隔离可避免多流并发下的写覆盖风险。

### 适配点 G：Engine 中保留 FP8 dtype，不强转为全局 dtype
- **位置**：`engine.py::_load_weight_state_dict`
- **为什么需要适配**：
  - 如果把 FP8 权重统一 `.to(self.dtype)`，就会变回 BF16/FP16，直接失去显存节省目标。
  - 必须仅转换非 FP8 参数（如 norm/embedding 部分），FP8 权重保持 float8 原生格式。

---

## 3. 一句话总结

FP8 适配本质是把“**加载期一次性反量化**”改造成“**前向期按层动态反量化 + 缓冲复用**”，从而用更低常驻显存换取可接受的计算开销，保证中小显存卡能跑通较大参数模型。
