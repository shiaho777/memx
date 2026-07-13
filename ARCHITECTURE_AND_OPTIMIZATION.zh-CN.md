# MemX 架构与优化

说明 MemX 的分层、LLM 驻留编排方式，以及内存与速度的评估方法。英文版：[ARCHITECTURE_AND_OPTIMIZATION.md](ARCHITECTURE_AND_OPTIMIZATION.md)。

## 问题

本地 LLM 与其它大块进程内缓存需要在控制 RSS 的同时保留大量张量。量化改精度，mmap 交给 OS page cache，offload 框架把张量搬到磁盘或其它设备——这些路径都成立。MemX 切的是另一条：在**进程私有匿名内存**里保留原始精度张量，冷页压缩，计算时只流式驻留一小段 working set。

FullHost 正确性门槛：解压必须还原宿主写入的字节；Qwen3.5-0.8B FullHost 以 output sum **`-24.360558`** 作为 bitexact 门禁。

## 分层

```text
Host（C / Python / Torch）
  显式 context、配额、tensor 描述符
  epoch + WS 意图（HOT / PREFETCH / RETIRE）
  FullHost 参考：run_qwen.py
        │
        ▼  memx_runtime.h / memx_runtime.py
控制面
  context、tensor 策略、驻留编排、遥测
        │
        ▼
页状态机
  RESIDENT → COMPRESSING → COMPRESSED → fault / HOT
  write_seq、dirty 中止、CAS 提交、整页内容校验
        │
        ▼
压缩虚拟池
  Metal 辅助压缩、tensor codec、free-extent 回收
  信号驱动解压（TLS scratch / 映射页）
```

对应源码：

| 部分 | 路径 |
|------|------|
| 公开 API | `include/memx_runtime.h` |
| 实现 | `libmemx3.m` |
| Python | `python/memx_runtime.py` |
| FullHost | `run_qwen.py` |

## 分配与张量策略

Context 持有托管分配与配额。宿主通过 `malloc` / `mmap` 或 `malloc_tensor` + `memx_runtime_tensor_desc_t`（role、dtype、layout、flags、shape）分配。描述符只影响 codec 候选与驻留策略，不改写张量内容。

区间 API 覆盖 flag 更新、prefetch、mark-access、seal、force-compress、purge。统计项包括 pool pressure、碎片、codec 节省、fault、以及 per-context 用量。

## 页生命周期（正确性）

压缩路径：

1. 进入 `PAGE_COMPRESSING`，写入压缩 meta。
2. 仅在内容仍匹配时 CAS 到 `PAGE_COMPRESSED`（`page_compress_content_ok`，整页比对）。
3. 写后 dirty / `write_seq` 变化则中止提交。

解压走 TLS scratch。Darwin 上 free 后改写前使用 `MADV_FREE_REUSE`。空闲 LLM 页可谨慎 WRITE-protect；顺序 dirty hold 降低 thrash。Race 覆盖见 `test_compressing_race`。

FullHost 相关操作约束：

- matmul 内不做 `seal_flush`。
- Final 路径不用 `reseal_weights()` 做整包粗暴 reseal。
- 下一块 col-strip 只 prefetch；冷却当前 strip 时不要把整窗重新 pin 热。

## 驻留编排器

编排器把压缩池变成可流式推进的 LLM working set。

### Epoch

| 阶段 | 常量 | 作用 |
|------|------|------|
| Load | `MEMX_EPOCH_LOAD` | 托管张量，允许初始压缩 |
| Compress | `MEMX_EPOCH_COMPRESS` | 后台 seal / 压力回收 |
| Infer | `MEMX_EPOCH_INFER` | 限制 hot budget，推进窗口 |
| Final | `MEMX_EPOCH_FINAL` | 回收 tracks，终端 seal / purge |

```c
memx_runtime_context_begin_epoch(ctx, phase, hot_budget_bytes);
memx_runtime_context_apply_ws(ctx, intents, n);
memx_runtime_context_ws_advance(ctx, ptr, hot_off, hot_len, prefetch_len, flags);
memx_runtime_context_ws_close(ctx, ptr, flags);
memx_runtime_context_end_epoch(ctx, seal_tracked);
```

### 意图 flag

| Flag | 行为 |
|------|------|
| `MEMX_WS_FLAG_HOT` | 为计算保持区间驻留 |
| `MEMX_WS_FLAG_PREFETCH` | 预热后续区间，不扩大 durable hot set |
| `MEMX_WS_FLAG_RETIRE` | trail 标冷；策略允许时异步 seal |
| `MEMX_WS_FLAG_RETIRE_SYNC` | 同步 seal trail |
| `MEMX_WS_FLAG_MARK_ACCESS` | 只记账，不做完整 pin |
| `MEMX_WS_FLAG_NO_ASYNC` | 强制同步路径 |

每个 context 跟踪有限数量的窗口。已覆盖且 flag 不变的区间跳过重复工作。Prefetch 上限跟 pool pressure 走，并与 hot 增长分离。Trail 默认 cold-mark；只有 RETIRE / trail-seal 策略要求时才 seal。

### FullHost 流程（`run_qwen.py`，`MEMX_WS_ORCH=1`）

1. 权重托管进 MemX 张量（BF16，分块转换）。
2. Compress epoch，等待后台压缩。
3. Infer epoch，设定 hot budget。
4. 每个 matmul / layer：推进热窗、prefetch 下一块 / 下一算子、在热路径外 retire 上一块。
5. 结束 infer；Final epoch + purge / seal，压终端 RSS。

密集 strip 尽量批进 `apply_ws`。

## 内存下降从何而来

同一机器、同一权重：

| 模式 | 驻留内容 | 0.8B 量级 |
|------|----------|-----------|
| 朴素托管 | 整包 BF16 权重 | ~1.6–1.8 GB RSS |
| MemX FullHost 收敛后 | 压缩页 + 小热窗 | final ~110–120 MB |

机制：

1. 权重页 bitexact 压缩（tensor codec），不是量化。
2. 匿名压缩池，而不是所有页长期 dirty 驻留。
3. 仅对当前 matmul strip / layer 流式 HOT。
4. Prefetch 与 durable hot 分离。
5. Trail retire 让上一窗口回到压缩态。
6. Final seal / purge 在推理后把 RSS 压下去。

干净参考（Qwen3.5-0.8B，bitexact `-24.360558`）：

| 路径 | Infer wall | Infer RSS | Final RSS |
|------|------------|-----------|-----------|
| extreme10 | **0.386 s** | **348 MB** | **111 MB** |
| orchestrator 干净路径（orch4） | 0.450 s | 335 MB | 119 MB |

相对 hosted ~1663 MB，final ~111 MB 约 **15×**。压缩与 final seal 充分结算后，FullHost “Saved” 可到约 93%。优先看干净顺序日志；macOS dirty RSS 尖峰（常 1000 MB+）是系统噪声，不当产品指标。

## 相关系统

| 路线 | 例子 | 与 MemX 的差异 |
|------|------|----------------|
| 量化 | llama.cpp/GGUF、AWQ、GPTQ、bitsandbytes | 更少比特，精度不同 |
| mmap 加载 | GGUF / safetensors mmap | 依赖 OS cache；无宿主主导的压缩 WS |
| Offload | FlexGen、DeepSpeed ZeRO-Inference | 以 CPU/NVMe 搬运为主杠杆 |
| 系统压缩内存 | macOS memory compression | 系统级，不是 tensor-WS + bitexact 池 API |

MemX 把 bitexact 宿主字节、页压缩匿名内存、Transformer 描述符/codec、以及带池遥测的 epoch/WS 编排合在一起。

## 构建、测试、FullHost

```bash
make all
make test-explicit
make test-compressing-race
make test-python-bitexact
```

```bash
MEMX_OP_LEVEL_WS=1 MEMX_BLOCK_WS=1 MEMX_STREAM_WS=1 \
MEMX_BLOCK_PREFETCH=1 MEMX_MATMUL_CHUNK=384 MEMX_OP_FORCE_COOL=0 \
MEMX_COL_STRIP=1 MEMX_STREAM_TRAIL_SEAL=0 MEMX_STREAM_END_SEAL=0 \
MEMX_POST_HOST_FORCE=1 MEMX_FINAL_FORCE=1 MEMX_FINAL_PURGE=1 \
MEMX_FINAL_SEAL_PASSES=2 MEMX_WAIT_S=25 MEMX_ALIVE_S=0 \
MEMX_ADAPTIVE_CHUNK=1 MEMX_COLD_ASYNC_SEAL=1 MEMX_WS_ORCH=1 \
DYLD_LIBRARY_PATH=build python3 -u run_qwen.py --model-path .local/Qwen3.5-0.8B-hf
```

长任务建议 detached（`Popen(..., start_new_session=True)`）；日志放 `/tmp/memx_opt/`。

## 优化方向

优先改结构，少堆 env：

- HOT 窗口已覆盖时跳过 syscall。
- 每个 op/layer 把 matmul 意图批进一次 `apply_ws`。
- trail seal 离开 infer 热路径。
- prefetch 与 hot 驻留增长分离。
- 受 pressure 约束的 prefetch 上限。
- 在 bitexact 前提下提高真实 weight/KV 压缩率的 codec。

目标：干净追平或超过 extreme10（wall ≤ 0.386 s，infer RSS ≤ 350 MB，final ≤ 111 MB，bitexact）；在 macOS 内存污染下稳住 final seal；0.5B/0.8B 多轮均值作回归门禁；让非 demo 宿主更少手写 strip 逻辑。

## 门禁

- `make test-explicit`
- `make test-compressing-race`
- Python bitexact 套件
- FullHost 0.8B output sum 精确等于 `-24.360558`
- 优先 clean final RSS，而非污染快照

## 运行要求

Apple Silicon，macOS 13+，Xcode CLT。压缩路径依赖 Metal。面向大块托管分配；不可压页存 raw。仅显式宿主接入。

## 许可证

MIT
