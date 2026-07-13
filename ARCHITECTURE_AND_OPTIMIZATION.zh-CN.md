# MemX 架构与优化

MemX 是面向 Apple Silicon 的可嵌入压缩内存运行时。宿主把大块缓冲显式放进托管虚拟池；冷页压缩，按需 fault 回填，并通过流式工作集编排驻留窗口。LLM 托管目标是：**保持原始精度 bitexact**，同时把进程 RSS 压到远低于模型原始体积。

English version: [ARCHITECTURE_AND_OPTIMIZATION.md](ARCHITECTURE_AND_OPTIMIZATION.md)

## 设计论点

主流 LLM 内存方案通常只走三条路之一：

1. **量化**：用更少比特表示权重（llama.cpp / GGUF、AWQ、GPTQ、bitsandbytes）。
2. **磁盘映射**：依赖 OS page cache / demand paging（GGUF 或 safetensors 的 mmap）。
3. **卸载**：把张量调度到 CPU / NVMe（FlexGen、DeepSpeed ZeRO-Inference）。

MemX 走第四条路：

**在进程私有匿名内存里保留原始张量字节（bitexact），再压缩，并只流式驻留当前工作集。**

同一台机器上可以出现巨大的 RSS 差距，且数值结果不变——fault / 解压后，宿主看到的仍是精确的 BF16 / FP16 / FP32 字节。

## 系统分层

```text
┌─────────────────────────────────────────────────────────────┐
│ Host（C / Python / Torch）                                  │
│  - 显式 context / quota / tensor 描述符                     │
│  - epoch + working-set 意图（HOT / PREFETCH / RETIRE）      │
│  - 可选 FullHost demo：run_qwen.py                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ memx_runtime.h / memx_runtime.py
┌───────────────────────────▼─────────────────────────────────┐
│ Runtime 控制面                                              │
│  - 命名 context 与配额                                      │
│  - tensor role / dtype / layout / flags                     │
│  - 驻留编排器（epochs、WS tracks、pressure）                │
│  - 遥测（pool pressure、codec 节省、fault）                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ 页状态机                                                    │
│  PAGE_NONE → PAGE_RESIDENT → PAGE_COMPRESSING               │
│            → PAGE_COMPRESSED → PAGE_HOT / fault 回填        │
│  write_seq + dirty 中止、CAS 提交、内容校验                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ 压缩 + 虚拟池                                               │
│  - Metal 辅助压缩路径                                       │
│  - tensor 编解码：FP16/BF16 split、delta-split、bitplane、  │
│    sparse-byte、exp-pack、zlib 混合                         │
│  - 压缩存储 + free-extent 回收                              │
│  - 信号驱动解压到 TLS scratch / 映射页                      │
└─────────────────────────────────────────────────────────────┘
```

## 核心运行时接口

公开头文件：`include/memx_runtime.h`  
实现：`libmemx3.m`  
Python 绑定：`python/memx_runtime.py`

### 分配与策略

- Context 创建 / 销毁 / 配额
- 托管 `malloc` / `calloc` / `realloc` / `posix_memalign` / `mmap`
- 通过 `memx_runtime_tensor_desc_t` 做张量语义分配
- 区间 flag 更新、prefetch、mark-access
- Seal / force-compress / purge
- Pressure 与 per-context 统计

Tensor 描述符只是元数据。它们不会量化，也不会改写模型数据。它们按角色（weight、KV cache、activation、embedding、temporary）选择 codec 候选与驻留策略。

### 页生命周期保证

关键正确性约束：

1. 压缩提交时先在 `PAGE_COMPRESSING` 下写 meta，再 CAS 到 `PAGE_COMPRESSED`。
2. `write_seq` + 写后 dirty 中止，防止撕裂提交。
3. `page_compress_content_ok` 在接受压缩结果前做整页内容比对。
4. 对空闲 LLM 页谨慎 WRITE-protect；顺序 dirty hold 避免 thrash。
5. 解压走 TLS scratch；Darwin 上 free 后改写前用 `MADV_FREE_REUSE`。
6. Race 压力测试（`test_compressing_race`）必须保持通过。
7. FullHost 0.8B bitexact 门禁：**Output sum = -24.360558**。
8. Final 路径不要用 `reseal_weights()` 做粗暴整包重托管。
9. matmul 热路径禁止 `seal_flush`。
10. Col-strip 下一块只做 prefetch；冷却当前 strip 时不要把整窗重新 pin 热。

## 驻留编排器（Residency Orchestrator）

编排器把“压缩池”升级成“LLM 流式驻留”。

### Epoch

| 阶段 | 常量 | 意图 |
|------|------|------|
| Load | `MEMX_EPOCH_LOAD` | 托管张量，允许初始压缩 |
| Compress | `MEMX_EPOCH_COMPRESS` | 后台 seal / pressure reclaim |
| Infer | `MEMX_EPOCH_INFER` | 限制 hot budget，流式推进 WS |
| Final | `MEMX_EPOCH_FINAL` | 回收 tracks，final seal / purge |

API：

- `memx_runtime_context_begin_epoch(ctx, phase, hot_budget_bytes)`
- `memx_runtime_context_apply_ws(ctx, intents, n)`
- `memx_runtime_context_ws_advance(ctx, ptr, hot_off, hot_len, prefetch_len, flags)`
- `memx_runtime_context_ws_close(ctx, ptr, flags)`
- `memx_runtime_context_end_epoch(ctx, seal_tracked)`

### Working-set 意图

`memx_runtime_ws_intent_t` 携带：

- pointer + offset + length
- 可选 prefetch length
- priority
- flags：

| Flag | 含义 |
|------|------|
| `MEMX_WS_FLAG_HOT` | 为计算保持区间驻留 |
| `MEMX_WS_FLAG_PREFETCH` | 提前提前 fault，但不永久扩大 durable hot set |
| `MEMX_WS_FLAG_RETIRE` | 标记 trail 为冷；策略允许时异步 seal |
| `MEMX_WS_FLAG_RETIRE_SYNC` | 同步 seal trail |
| `MEMX_WS_FLAG_MARK_ACCESS` | 只做访问记账，不做完整 pin 语义 |
| `MEMX_WS_FLAG_NO_ASYNC` | 强制同步路径 |

Context 跟踪有限数量的 working-set 窗口。已覆盖且 flag 相同的区间走 skip 快路径，避免 syscall 抖动。Prefetch 预算受 pressure 约束，并与 hot 驻留增长分离。Lazy trail release 默认只 cold-mark；只有 RETIRE / trail-seal 策略要求时才 seal。

### 宿主接入模式（FullHost）

`run_qwen.py` 在 `MEMX_WS_ORCH=1`（默认）时使用编排器：

1. 把权重托管进 MemX 张量（BF16 路径，分块 half 转换）。
2. 进入 compress epoch；等待后台压缩。
3. 进入 infer epoch 并设置 hot budget。
4. 每个 matmul / layer：
   - pin 或 advance 热窗口
   - prefetch 下一块 strip / 下一算子
   - retire 上一块 strip（cold-mark；seal 离开热路径）
5. 结束 infer epoch。
6. Final epoch + purge / seal passes，把终端 RSS 压下去。

Col-strip 与 block/stream WS 在密集意图时尽量批量走 `apply_ws`。

## 为什么同一台机器上 RSS 能差这么大

同样硬件、同样权重，两种驻留模式：

| 模式 | 内存里住什么 | 0.8B 典型结果 |
|------|--------------|---------------|
| 朴素托管 | 整包 BF16 权重常驻 | ~1.6–1.8 GB RSS 量级 |
| MemX FullHost 压缩后 | 压缩页 + 极小热窗口 | final ~110–120 MB 量级 |

机制栈：

1. **权重页 bitexact 压缩**（split / tensor codecs），不是量化。
2. **匿名压缩池**，而不是让所有页一直 dirty 驻留。
3. **流式 working set**：只有当前 matmul strip / layer 窗口是 HOT。
4. **Prefetch 与 hot 分离**：未来页可以预热，但不会永久撑大 hot set。
5. **Trail retire**：上一个窗口回到 cold/compressed，不阻塞计算。
6. **Final seal / purge**：推理后对 tracked 窗口再压缩，final RSS 塌缩。

Qwen3.5-0.8B 干净 FullHost 参考（bitexact `-24.360558`）：

| 路径 | Infer wall | Infer RSS | Final RSS |
|------|------------|-----------|-----------|
| extreme10 历史最佳速度（干净内存） | **0.386s** | **348 MB** | **111 MB** |
| orchestrator 干净路径（orch4） | 0.450s | 335 MB | 119 MB |

相对 hosted 权重 footprint（~1663 MB）：

- Final RSS ~111 MB → 最佳约束路径上约 **15×** 进程驻留缩减。
- FullHost 汇总里的 “Saved” 在压缩与 final seal 充分结算后，可相对模型/托管体积报告约 93%。

macOS dirty RSS（经常 1000 MB+）不当真赢；系统内存污染主导方差。优先看 `/tmp/memx_opt/` 下干净顺序 FullHost 日志。

## 竞争格局

类似，但不是同一架构：

| 类别 | 例子 | 优化什么 | 与 MemX 的差距 |
|------|------|----------|----------------|
| 量化 | llama.cpp/GGUF、bitsandbytes、AWQ、GPTQ | 每权重比特数 | 改变精度；不是原始张量 bitexact |
| mmap 加载 | GGUF mmap、safetensors mmap | 加载路径 / OS cache | 仍付 OS 驻留成本；没有进程内压缩 + WS 编排 |
| Offload 框架 | FlexGen、DeepSpeed ZeRO-Inference | CPU/NVMe 调度 | 偏磁盘/宿主搬运；产品形态不同 |
| 系统压缩内存 | macOS compressor | 系统级机会式压缩 | 不是宿主主导的 tensor WS + bitexact 池遥测 |

MemX 差异化：

- 原始精度 **bitexact** 宿主视图
- 显式 API 下的页压缩 **匿名** 驻留
- Transformer 感知描述符 + tensor codecs
- epoch + working-set 编排器，用于流式 LLM 窗口
- pressure / codec / fault 遥测，服务宿主策略

## 仓库结构

```text
include/memx_runtime.h     公开 C API
libmemx3.m                 运行时 + 压缩器 + 编排器
python/memx_runtime.py     ctypes 绑定 + WS helpers
run_qwen.py                FullHost LLM 驻留 demo
examples/                  嵌入式宿主示例
tests/                     Explicit runtime、race、Python bitexact
benchmarks/                仅 runtime-native 基准
MemXApp/                   本地 dashboard shell
ARCHITECTURE_AND_OPTIMIZATION.md
ARCHITECTURE_AND_OPTIMIZATION.zh-CN.md
```

本地资产默认忽略：`.local/`、`build/`、`MemXApp.app/`。

## 构建与验证

```bash
make all
make examples
make test-explicit
make test-compressing-race
make test-python-bitexact
```

嵌入式 demo：

```bash
make example-embedded
```

FullHost 0.8B 基线（权重在 `.local/Qwen3.5-0.8B-hf`）：

```bash
MEMX_OP_LEVEL_WS=1 MEMX_BLOCK_WS=1 MEMX_STREAM_WS=1 \
MEMX_BLOCK_PREFETCH=1 MEMX_MATMUL_CHUNK=384 MEMX_OP_FORCE_COOL=0 \
MEMX_COL_STRIP=1 MEMX_STREAM_TRAIL_SEAL=0 MEMX_STREAM_END_SEAL=0 \
MEMX_POST_HOST_FORCE=1 MEMX_FINAL_FORCE=1 MEMX_FINAL_PURGE=1 \
MEMX_FINAL_SEAL_PASSES=2 MEMX_WAIT_S=25 MEMX_ALIVE_S=0 \
MEMX_ADAPTIVE_CHUNK=1 MEMX_COLD_ASYNC_SEAL=1 MEMX_WS_ORCH=1 \
DYLD_LIBRARY_PATH=build python3 -u run_qwen.py --model-path .local/Qwen3.5-0.8B-hf
```

长等待建议 detached FullHost：

```python
subprocess.Popen(..., start_new_session=True)
```

日志习惯放在 `/tmp/memx_opt/`。

## 优化哲学

架构优先，旋钮其次。

值得做：

- 已覆盖 HOT 窗口减少 syscall
- 每个 op/layer 把 matmul 意图批进一次 `apply_ws`
- trail seal 离开 infer 热路径
- prefetch 增长与 hot 驻留增长分离
- pressure-aware prefetch 上限
- 在 bitexact 前提下提高真实张量压缩率的 codec 选择

避免：

- 把随机 env flag 堆叠当成“架构”
- 把 dirty 1000 MB+ RSS 当速度胜利
- final 路径整包 reseal 所有权重
- 冷却当前 strip 时把整窗重新 pin
- 在代码里写注释（项目规则：代码不写注释）

## 近期架构目标

1. 干净打赢 extreme10：wall ≤ 0.386s，infer RSS ≤ 350 MB，final ≤ 111 MB，bitexact `-24.360558`。
2. 窗口已覆盖时，track-local delta-HOT 零 syscall 快路径。
3. 在 macOS 内存污染条件下稳住 final seal。
4. 干净机器上保留 0.5B / 0.8B 多轮均值作为回归门禁。
5. 扩展编排器宿主体验，让非 demo 运行时不必重写 strip 逻辑也能用 epoch/WS。

## 正确性门禁（不可回退）

- `make test-explicit`
- `make test-compressing-race`
- Python bitexact 测试
- FullHost 0.8B output sum **精确等于** `-24.360558`
- 优先 clean final RSS，而不是污染系统快照

## 要求与限制

- Apple Silicon Mac，macOS 13+，Xcode CLT
- 压缩路径依赖 Metal
- 主要面向大块托管分配
- 不可压数据回退到 raw page 存储
- 只支持显式宿主接入；不是系统级注入产品

## 许可证

MIT
