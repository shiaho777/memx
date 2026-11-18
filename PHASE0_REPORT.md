# MemX 阶段 0 体检报告（2026-11-16）

## 11-18 自主迭代：代码自审 → 6 项修复（全程 ≤50MB 验证）

用户授权自主迭代后，我自审代码找问题、修复、实测，循环 7 轮：

### 修复清单（全部实测验证）
1. **移位未定义行为**（我此前引入）：churn 退避 `32u << (streak-4)` 在 streak≥36 时 UB。改为显式饱和（shift≥4 → 255）。
2. **初始化冗余清零**：dedup 表（851KB）与物化缓存（4MB）对匿名 mmap 做了多余 memset（内核已零填充）。移除后初始化 RSS 11.4MB → **10.6MB**，物化缓存改按需提交。
3. **后台压缩器空转扫描**：res_list 为空时每周期扫 65536 页元数据。降频为每 16 空闲周期一次，空闲 3s CPU 降至 ~0.06s。
4. **绑定层悬空视图防护**：Allocation.free() 后 buffer()/torch_tensor() 原会访问已释放内存（段错误风险）。加显式 ValueError 防护 + 清空缓存视图。
5. **store 提交上下文泄漏**：do_commit 成功/失败路径都未释放托管分配，context_destroy 返回 EBUSY 被忽略 → 每次提交泄漏一个上下文。改为跟踪分配指针、销毁前统一释放。
6. **capsule 导出/挂载/段物化错误路径审查**：资源释放完整、段边界在清单加载时已校验（10471 行），无需改动。

### 验证
- 每轮修复后全量门禁 GATE_EXIT=0（test/explicit/race/python/core/sim/audit/posix/store）。
- 初始化 RSS 10.6MB，流式测试 21.5MB，通用基准 47MB——全部 ≤50MB。
- 未跑大模型、未 commit/push（遵守授权边界）。

### 自主迭代方法论
找问题（代码自审 + 基准实测）→ 最小修复 → 门禁 + 量化验证 → 下一轮。不做无收益优化（异步预取已回退），不夸大数字。

## 11-18：异步预取探索（诚实结论：无收益）+ B 轨存储基准

### 15. 异步预取流水线探索
- 假设：流式 decode 时后台异步物化下下层，与当前层计算重叠。
- 实测（纯 C 基准 `bench_stream_prefetch`，16 层 ×256KB，模拟计算窗口）：
  - 无预取：0.334 ms/层；提前 1 层预取：0.349 ms；提前 2 层：0.347 ms；1ms 计算窗口：仍无收益。
  - 异步 ON hits=32 vs OFF hits=172——异步路径反而更差。
- **根因**：物化已优化到 ~0.03ms/层，计算主导，重叠窗口太小；预取调用本身开销 > 收益。
- **结论**：回退 Python 预取调用（无效），保留 C 基准作为测量记录。**不做无收益的优化**。

### 16. B 轨存储竞争力基准（MemX vs zlib）
- `bench_generic` 缩到 4MB/类（峰值 47MB，守住 50MB）：
  - zeros：MemX 0.01x vs zlib 0.00x，fault 2978 MB/s
  - text：MemX 0.01x vs zlib 0.01x，fault 1805 MB/s
  - struct_i32：**MemX 0.39x vs zlib 0.87x**（2.2× 更优），fault 1481 MB/s
- `bench_capsule`（512 页）：export 1.0ms，attach 0.03ms，batch_v 比 single 快 5.6×。
- **诚实判断**：MemX 对结构化数据压缩比显著优于 zlib；对近随机权重压缩比低（信息论上限）。存储产品竞争力在结构化/冷数据场景。

## 11-17 深夜：多页物化并行解压流水线（大范围 +17%）

### 14. materialize_range 并行分支
- **优化**：≥32 页、页对齐、无 dtype 转换的范围物化改用 `dispatch_apply` 并发直出解压；失败原子标记 + 串行回退。`MEMX_MAT_PAR=0` 可关。
- **阈值调优（实测驱动）**：初版 ≥4 页并行，小张量（16 页）反而慢 ~15%（分发开销 > 收益）；提高到 ≥32 页后小张量回到串行快路径。
- **A/B 实测**（纯 C 基准，≤50MB）：
  - 小张量 8×256KB（串行路径）：~11.5 GB/s（不回归）
  - 大张量 4×1MB（并行路径）：串行 ~9.3 GB/s → 并行 ~10.5 GB/s（**+17%**）
- 正确性：流式测试新增 64 页 ×3 轮并行用例，字节全一致；全量门禁 GATE_EXIT=0；测试峰值 27.8MB。

## 11-17 晚：物化热路径零拷贝优化（A/B 实测 +10%）

### 12. materialize_range 零拷贝直出
- **瓶颈**：流式 decode 每层物化走"解压到栈缓冲 pagebuf → 再 memcpy 到调用方 dst"双重拷贝。
- **优化**：整页、无 dtype 转换时直接解压到调用方缓冲区，失败回退旧路径；部分页/转换路径不变。
- **A/B 实测**（`make bench-materialize`，8×256KB 张量 ×20 轮，峰值 RSS 22MB）：
  - 旧路径：9980 / 10109 / 10293 MB/s（均值 ~10.1 GB/s）
  - 新路径：11083 / 11192 / 11038 MB/s（均值 ~11.1 GB/s）→ **+10%**
  - 不可压缩数据：13.2 GB/s
- 新增 `benchmarks/bench_materialize.c` + `make bench-materialize`（纯 C，≤50MB）。
- 全量门禁 GATE_EXIT=0（test-python-transformer 并发负载下偶发时序失败，单独重跑 3/3 通过，非回归）。

### 13. 当前优化状态小结
- 初始化 RSS：212MB → 11MB（关闭扫描移除 + 元数据惰性提交）。
- 物化吞吐：+10%（零拷贝直出）。
- 流式工作集：2 槽双缓冲 + 跨轮次重置修复，小数据 3 轮字节全一致。
- 测试体系：纯 C 门禁 21.5MB，torch/大模型测试 opt-in。

## 11-17 A 轨攻坚：真实 Transformer 前向 + 挖出并修复核心数据损坏 bug

### 1. 把 Demo 变成真推理（python/memx_llama.py）
新建真实 Llama 架构前向引擎：RMSNorm、RoPE、GQA attention + 真实 KV cache、SiLU 门控 MLP、逐 token 贪心生成。**不是**原 run_qwen.py 的合成 matmul 链。
- 逐组件与 HF 参考对拍 bit-exact：embed=0、attention 残差 0.0039、MLP=0、单层 final 0.0078（均 bf16 舍入级）。
- 修正一处精度顺序 bug：RMSNorm 须先转回 bf16 再乘权重（与 HF 一致），否则偏差 0.0625 经 MLP 放大成 5.875。
- MPS 流式前向验证：42 层 prefill 1.298s，argmax 正确（'a'）。MPS bf16 矩乘 5900 GFLOP/s，比 CPU bf16（290）快 20×——A 轨性能出路。
- 发现 torch 2.8 在本机报 "NO AVX"，CPU bf16 走标量 fallback（2.49 t/s 基线慢的根因）。

### 2. 核心数据损坏 bug（真凶，已修复）
**现象**：真实 bf16 权重经 MemX 压缩-解压后全 0，managed 生成退化。但 capsule CRC 校验全过（CRC 只验压缩流完整性，不验解压正确性）。
**根因**：`encode_tensor_page_one` 的 codec 竞争共用 `codec_tmp` 栈缓冲。当 sticky 已跑过 zsplit（`zsplit_done=1`）或 exp_pack→zsplit 替换分支复用 `zsplit_done` 时，**跳过重新压缩，直接 `memcpy(tensor_dst, codec_tmp, zsplit_csz)`，但 codec_tmp 已被后续 ZLIB 竞争覆盖成 ZLIB 流**。结果：元数据 `out_codec=0x86`(zsplit)/`csz=11956`，实际流头 `0x85`(ZLIB)/需 13256 字节 → 解压校验 `8+zlen>cs` 失败 → **静默 memset 填零**。
**铁证**：编码出口诊断 `out_codec=86 dst[2]=85`；缺页解压 `codec=85 tmp 全 0`。
**修复**（libmemx3.m 两处）：zsplit 竞争分支与 exp_pack→zsplit 替换分支**总是重新压缩到 codec_tmp**，确保流内容与元数据一致。
**验证**：单张量 512 页往返 `readback equal: True maxdiff: 0.0`；全量门禁 `GATE_EXIT=0` 无回归。

### 3. 意义
这是 MemX 作为"bit-exact 压缩内存"的**致命正确性 bug**——此前所有走该路径的权重都被静默损坏，而 CRC 门禁完全没抓到。修复后 bit-exact 契约才真正成立。同时暴露测试盲区：现有 capsule 测试用随机/合成数据，未覆盖真实权重分布触发的 codec 竞争路径。新增 `tests/test_real_generation_bitexact.py` 用真实模型 + 真实生成补上这一门禁。

### 4. 端到端生成门禁通过 —— 不需要重启（纠正此前误判）
**此前"需重启"的结论是错的**，两处误判已纠正：
- 误判一：把"裸 C 探针缺页冻结"等同于"MemX 跑不了"。实测对比：裸探针（无端口重置）冻结，但 **MemX 进程（有端口接管修复）缺页读回 `readback equal: True maxdiff: 0.0` 完全正常**。MemX 绕开了坏掉的系统崩溃报告路径，自身缺页投递是好的。
- 误判二：把 MPS 门禁物化阶段的"慢"误判为"死锁"（等待不足即认定 GONE）。加阶段日志后确认进程 184% CPU 活跃计算，只是物化 381 张量耗时。

**最终验证（CPU 与 MPS 双路径均通过）**：
```
TOKEN_MATCH=True
ref tokens    = managed tokens = [996, 911, 35, 594, 1043, 2137, 35, 594]
WEIGHT_BITEXACT=True
OK real generation bitexact
```
- CPU：4 token，managed_generate 0.917s。
- MPS：8 token，managed_generate 0.726s，prefetch_hits 38614。
- 真实 Llama 42 层前向 + 逐 token 贪心生成，MemX 托管权重与纯 torch 参考**逐 token 完全一致**，权重字节位精确。
- 纯 MPS、MemX+MPS 小测试、MPS 参考生成均独立验证正常，无系统级阻塞。

**结论：A 轨核心契约（bit-exact 真实生成）已闭环，无需重启。** 系统裸探针的间歇性异常投递故障不影响 MemX（端口接管修复生效）。`decompress_compressed_page` 信号上下文 alloc_mutex 已改递归锁加固。

### 5. 本轮代码改动清单
- `libmemx3.m`：① 异常端口接管（BAD_ACCESS→自有信号处理器，MEMX_KEEP_EXC_PORT=1 可关）；② codec 竞争两处缓冲修复（zsplit 竞争 + exp_pack→zsplit 替换总是重新压缩）；③ alloc_mutex 改递归锁（缓解信号重入）。
- `platform/posix/memx_posix.c`：异常端口接管（__APPLE__ 条件编译）。
- `python/memx_llama.py`：新增真实 Llama 前向 + 生成引擎。
- `tests/test_real_generation_bitexact.py`：新增真实生成位精确门禁。
- `benchmarks/bench_baseline_torch.py`：新增诚实基线（含退化检测）。
- `run_qwen.py:754`：修 ctypes.memmove 缓冲区类型错误。

### 6. B 轨存储修复 + 真实 FullHost 演示（run_real.py）
- **res_list churn 修复**：不可压缩页（codec_fail_streak≥4）指数退避（cooldown 32<<streak，clamp 255），通用压缩失败也累积 streak。验证：64 页随机数据 force_compress 后 4s 内 `compressions delta=0`（此前每轮重试）。
- **store 服务并发提交互斥**：新增 g_commit_mu 包裹 do_commit，序列化世代号计算+导出，防并发 COMIT 撞世代。`make test-store-service` 通过（含 concurrent）。
- **run_real.py 真实 FullHost 演示**（替换合成 Demo 的独立入口）：托管 381 张量→压缩收敛→物化→真实贪心生成→诚实 SUMMARY。MPS 16 token 实测：generate 2.7s（~5.5 tok/s 端到端）、WEIGHT_BITEXACT=True。
- **暴露的真问题（下一步优化）**：① 物化 4.7GB 耗时 ~151s（每页多拷贝+解压+锁）；② 压缩未充分 settled 时物化导致峰值 phys ~10.6GB（双份权重）。已改 run_real 等压缩收敛再物化以降低峰值。

### 7. run_real 最终实测（MPS，16 token）与诚实局限
```
hosted 381 tensors 2.6s | materialize 4.4s | generate 1.86s (~8 tok/s)
logical 4.69GB | compressed saved 134MB | peak phys 10.7GB | WEIGHT_BITEXACT=True
```
- **物化优化生效**：151s → 4.4s（30×）。
- **诚实局限一（压缩比）**：真实 bf16 权重近随机，无损压缩仅省 ~2.8%。这是信息论上限，不是 bug。MemX 的内存价值不在"压缩省内存"，而在"冷页不常驻 + 流式工作集"。
- **诚实局限二（峰值内存）**：run_real 为验证 bit-exact 采用"全量物化到 MPS"，峰值=托管+MPS 双份≈10.7GB，**未体现流式优势**。真正展示 MemX 价值需"逐层流式物化+ retire"（run_qwen 已有 materialize_weight_range/warm/cool 基础设施），使 MPS 只驻留当前层工作集。这是 A 轨下一步核心。
- **结论**：bit-exact 真实生成契约已闭环且可复跑；压缩比与峰值内存的诚实数字已给出，不夸大。下一步攻流式工作集生成以真正降低峰值内存。

## 11-17 下午：50MB 测试约束下的流式优化 + 初始化内存暴涨修复

**用户约束**：测试环境内存 ≤50MB，不再主动跑大模型脚本。已存入长期记忆。

### 8. 流式工作集生成（StreamedWeights）
- `python/memx_llama.py` 新增 StreamedWeights：2 槽双缓冲，逐层从 MemX 物化 + 预取下一层，前向开始 `reset_stream()`。
- **修复跨轮次槽位污染 bug**：`_next` 指针在 decode 轮次间不回绕，导致第二轮拿到上一轮残留层数据（首 token 对、后续全错）。修复：forward 入口显式重置 + 越界抛 IndexError。
- 真实模型验证（约束前已完成）：流式峰值 phys 10.7GB→7.4GB；修复后小数据验证 3 轮 ×4 层 ×9 张量字节全一致。

### 9. 初始化内存暴涨根因修复（212MB→11MB）
- **根因**：`fini_memx` 关闭期遍历全部 npages（24GB 机器 =3.1M 页）做 mprotect 恢复，触碰整个 201MB PageMeta 数组 → 任何进程 init+shutdown 峰值 RSS 212MB。
- **修复**：移除关闭期全量扫描（vmem 随后即 munmap，无需逐页恢复）。init-only 峰值 212MB→**11MB**。
- 这是真实产品缺陷：小内存场景（嵌入式/多实例）下每个 MemX 进程白占 200MB。

### 10. 50MB 合规测试体系
- 新增 `tests/test_stream_materialize.c`（纯 C，无 torch）：36 张量 ×3 轮流式物化 + 双槽预取，108 项字节校验全过，**峰值 RSS 21.5MB**。已挂入 `make test`。
- torch 版测试（import 即 176MB）改为显式 opt-in：`MEMX_ALLOW_TORCH_TESTS=1` / `MEMX_ALLOW_LARGE_TESTS=1`，默认 SKIP。
- 诊断中确认：materialize_range 的 dst_cap 必须 ≥ length（EMSGSIZE=40），测试槽缓冲需按最大张量分配。
- 全量门禁（含 test-store-service）GATE_EXIT=0。

### 11. 遗留与下一步
- 流式生成的真实模型端到端验证（修复后）需用户 opt-in 跑 `run_real.py --stream --check`（我不主动跑）。
- B 轨剩余：store 基准对比 zlib/mmap、SDK 打磨。
- A 轨剩余：流式 + MPS 的吞吐优化（当前流式 decode 每层物化开销大于计算）。

## 21:40 重大突破：根因定位 + 修复已落地，全量门禁转绿

**用户已重启机器**（uptime 4.5h）。重启后异常投递**短暂恢复又失效**，证明不是 uptime/内存压力问题。本轮深挖定位到真正根因并修复：

### 根因（系统级，非 MemX bug）
- 进程的 `EXC_BAD_ACCESS` 异常被注册到系统崩溃报告端口（ReportCrash，behavior=`EXCEPTION_STATE_IDENTITY|MACH_EXCEPTION_CODES`）。
- 该端口在本机**不投递消息**：缺页线程永久冻结，ReportCrash 进程 0% CPU 干等（`mach_msg_server_once`），重启后无任何新崩溃报告生成。
- 软件信号（kill 发信号）始终正常；调试器托管（debugserver 自己的异常端口）也正常 → 断点精确锁定在"内核异常 → ReportCrash 端口"这一段。
- `kern.task_exc_guard_default=153`（只读，用户态改不了）；SIP 开启；无第三方内核扩展。

### 修复（已写入代码，默认生效）
在运行时初始化安装信号处理器后，主动接管 BAD_ACCESS 异常端口：
`task_set_exception_ports(mach_task_self(), EXC_MASK_BAD_ACCESS, MACH_PORT_NULL, EXCEPTION_DEFAULT, THREAD_STATE_NONE)`
让缺页直接走 BSD 信号 → MemX 自己的处理器，绕开失效的系统崩溃端口。
- 落点：`libmemx3.m`（主运行时）+ `platform/posix/memx_posix.c`（POSIX 适配层，`__APPLE__` 条件编译，保持可移植）。
- 开关：`MEMX_KEEP_EXC_PORT=1` 可关闭（保留系统崩溃报告）。
- 对 fork/exec 子进程同样生效（vessel 子进程受益）。
- **副作用须知**：接管后 MemX 能处理的缺页正常恢复；但真正的野指针崩溃会挂起而非干净退出（崩溃报告端口被清空）。这是"自包含缺页处理"与"系统崩溃报告"的取舍，故做成可选开关。

### 验证（全部不用外部包装器，直接跑）
- `make test test-explicit test-compressing-race test-python core-test core-sim core-audit posix-adapter` → **GATE_EXIT=0，全绿**。
- 唯一偶发：`test_python_torch_transformer` 在并发套件里偶发"roles did not compress"，单独重跑即过 → 测试隔离/时序问题，非修复回归。
- FullHost 演示在 **MiniCPM5-2B（4.69GB）** 上完整跑通（原 Qwen3.5-0.8B 权重缺失，仅剩 config/tokenizer）。

### 诚实基线（MiniCPM5-2B，真实贪心生成，统一 phys_footprint 口径）
| 方案 | 解码吞吐 | 峰值 phys | 备注 |
|---|---|---|---|
| torch CPU bf16（8 线程） | **2.49 t/s** | 291 MB | 可信；权重经 safetensors mmap，phys 低 |
| torch MPS bf16 | ~~15.74 t/s~~ **作废** | 6051 MB | 退化成全吐 token 0（eos=0），空文本，非有效生成 |
| llama.cpp Q4_K_M（2B） | 无法加载 | — | 本机 llama.cpp 不识 `minicpm5` pre-tokenizer |
| llama.cpp Q4_K_M（Qwen3-0.6B） | **231.7 t/s**（tg64）/ 3979 t/s（pp128） | 456 MB 模型 | 量化对手参考，小一个量级的模型 |

**关键判断**：MemX 无损压缩在 2B 模型上对 CPU bf16 基线没有吞吐优势（基线本就靠 mmap 省内存）；量化对手快两个数量级。MemX 的差异化只能在"bit-exact 不可妥协 + 内存极度受限"的交集里找，不在通用 LLM 推理速度。

### FullHost 现状（坐实"Demo"判断）
- SUMMARY：`Output sum: 0.000000`、`Hosted weights: 0 tensors`、`Infer wall 1.563s`。
- 前向是前 3 层 matmul 链 + 合成 KV，**非真实 Transformer、无逐 token 生成**；门禁值只打印无断言。
- 内存数字（phys 1053MB / vessel 5MB / x-factor）是真实的，但"推理"是空壳。
- 本轮顺手修了 `run_qwen.py:754` 的 `ctypes.memmove` 缓冲区类型错误（bytearray→c_char 数组），否则 MiniCPM 上直接崩。

## 16:23 复验更新（用户反馈"应该没事了"之后）

- 机器**未重启**（uptime 仍 24 天），负载从 ~18 降到 ~7，swap 从 20.8GB 降到 2.8GB——内存压力明显缓解。
- 复测结果：异常投递**短暂恢复过一次**（一次 PROT_NONE 读成功触发处理器），随后**再次全面失效**：PROT_NONE 读、只读页写、空指针读全部重新冻结；软件信号始终正常。
- 新证据：`kern.task_exc_guard_default = 153`；调试器托管下同一异常仍可正常捕获（EXC_BAD_ACCESS code=2），确认断点在"异常→普通用户态进程"投递环节。
- **新发现（影响 B 轨判断）**：capsule 显式路径并非完全免疫——后台压缩器会把正在写入的页降为只读/无保护，慢速写入（如 Python 逐字节填充）触发写缺页后同样冻结。已用 C 慢写探针复现（`build/slow_fill`）。快速一次性写入 + 立即 force_compress + export + attach + materialize 的纯显式流程**可以完整跑通且位精确**（`build/seg_probe2`：32 页 compress→export→attach→segment materialize bitexact=1，verify bad=0）。
- 结论不变：**需要重启机器**。压力缓解不能修复异常投递链路；重启前 A 轨无法实测，B 轨只能做"快写+显式压缩"的受限验证。

## 结论先行

1. 代码可以正常构建；不依赖缺页机制的核心测试全部通过。
2. **本机当前无法向用户态投递硬件内存异常（EXC_BAD_ACCESS）**：任何程序访问受保护内存页时线程永久冻结，信号处理器不执行。这是系统级故障，与 MemX 代码无关（已用不含 MemX 的 30 行最小 C 程序复现）。
3. MemX 的透明压缩/解压架构完全依赖缺页信号投递，因此本机当前所有缺页驱动路径（test-explicit、posix-adapter、FullHost 推理）不可用。**建议重启机器恢复**，重启后重跑全部门禁。

## 环境

| 项 | 值 |
|---|---|
| 机器 | Mac16,7 · Apple M4 Pro · 24 GB |
| 系统 | macOS 26.6.2 (25G83)，已连续运行 24 天 |
| 内存状态 | swap 已用 20.8 / 22.5 GB，压缩器占用约 11 GB，负载 ~18，多次 JetsamEvent |
| 工具链 | Xcode clang 17 可用；`~/.swiftly` 的 clang 包装器损坏（报 Swift 6.3.0 toolchain 缺失），构建需 `PATH=/usr/bin:...` 前置绕过 |
| 依赖 | torch 2.8.0 / transformers 4.57.6 / MLX / llama.cpp（homebrew）均可用 |

## 构建与测试结果

| 项 | 结果 |
|---|---|
| `make all`（PATH 修正后） | ✅ 通过 |
| `make core-test` | ✅ codec 往返 / hook vtable / freestanding 核心 |
| `make core-sim`（16K + 4K） | ✅ 完整 embedder 契约（fault+commit+install，纯内存模拟，不依赖真实缺页） |
| `make core-audit` | ✅ 未定义符号仅 memcpy 族 |
| `make test`（test-explicit 起） | ❌ 挂起（原因见下，非代码断言失败） |
| `make posix-adapter` | ❌ 挂起（同一系统原因） |

## 系统故障证据链（全部可复现）

最小复现：mmap 一页 PROT_NONE，安装 SIGSEGV/SIGBUS 处理器后读取 → 线程冻结，处理器不执行，进程 0% CPU，可被 SIGTERM/SIGKILL 正常终止。

| 实验 | 结果 |
|---|---|
| 软件信号（kill SIGSEGV / SIGUSR1，含跨线程） | ✅ 处理器正常执行 |
| 懒分配零页读、文件映射读、fork COW 写 | ✅ 正常（内核内部 VM 故障不受影响） |
| 调试器（lldb/debugserver）托管下触发同一异常 | ✅ 正常捕获 EXC_BAD_ACCESS（系统日志确认 debugserver 设置异常端口成功） |
| 无调试器时：PROT_NONE 读 / 只读页写 / 降权后访问 | ❌ 全部冻结（读、写、主线程、子线程、子进程均复现） |
| 冻结期间再发 SIGALRM/SIGUSR1 | ❌ 不投递（线程已冻死在异常等待中） |
| 访问前先 mprotect 恢复读写 | ✅ 正常（不产生异常即不冻结） |
| MemX 调试副本：处理器安装确认正确（sigaction 回读一致、SA_ONSTACK 备用栈就绪），主动触发缺页 | ❌ 冻结在首次缺页，处理器入口跟踪从未打印 |

伴随现象：ReportCrash 多次长时间停滞、`sample`/`spindump` 对系统进程超时、崩溃报告延迟数分钟才落盘。机器 24 天未重启、swap 近满、内存压力事件频发，异常投递链路（exc 消息 → 用户态）处于失效状态。

## 对改造计划的影响

- **A 轨（LLM 流式推理）**：缺页驱动的透明解压是核心路径，必须在异常投递恢复后才能实测与验收。
- **B 轨（capsule 存储）**：显式 API 路径（malloc → 写入 → seal/force_compress → capsule_export → attach → materialize/verify，走 pread 与显式解压）不依赖硬件缺页，**可以立即推进**。
- 基线对比中的 torch 原生 / mmap / llama.cpp 侧不依赖缺页信号，可先行测量。

## 可用模型（已勘察）

| 模型 | 位置 | 用途 |
|---|---|---|
| MiniCPM5-2B（llama 架构，BF16，4.7 GB，42 层） | `~/Desktop/模型/0-/MiniCPM5-2B/hf-bf16` | A 轨大模型主目标（2B 级，24GB 机器可承载） |
| MiniCPM5-2B Q4_K_M / F16 GGUF | 同上目录 | llama.cpp 量化/全精度基线 |
| Qwen3-0.6B Q4_K_M GGUF | `~/Desktop/模型/0-/Qwen3-0.6B` | 小模型基线 |
| OLMoE-1B-7B Q2_K 等 MoE GGUF | `~/Desktop/模型/moe` | 备选 |
| Qwen3.5-0.8B-hf | 备份目录仅剩配置/分词器，**权重缺失** | README 门禁模型需重新获取或改用 MiniCPM5-2B 重建门禁 |

## 其他发现（代码层，待后续阶段处理）

- `run_qwen.py` 的"推理"是前 3 层权重的 matmul 链 + 合成 KV 负载，**不是真实 Transformer 前向，也没有逐 token 生成**；`-24.360558` 只是打印值，脚本内无断言。A 轨需要补真实前向与生成循环，门禁值需在新模型上重建。
- 托管上限默认 1800 MB、quota 4096 MB，2B 模型（4.7 GB）需调整。
- 层键匹配硬编码 `range(3)`（只算前 3 层），扩展模型时要改。