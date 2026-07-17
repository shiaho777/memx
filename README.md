# MemX — ~100× Lower LLM Memory on Apple, Original Precision

**Run large language models on Apple Silicon with a fraction of the usual memory footprint — without quantizing weights, without changing numerics, and without treating the model as a permanent multi‑gigabyte RSS bill.**

MemX is an explicit compressed-memory runtime for Mac. It turns cold weight pages into a durable **capability plane** (vault + capsule), streams only the hot working set during inference, and keeps BF16/FP16/FP32 bytes **bit-exact**. On Qwen3.5‑0.8B FullHost, stable measured planes land around **~100×** as a headline figure, with a transparent breakdown:

| Plane | What it measures | Stable measured scale (0.8B, ~1.66 GB weights) |
|-------|------------------|-----------------------------------------------|
| **Phys footprint** | `task_vm_info` physical charge | **~9×** (about 180 MB class) |
| **Engine cold** | MemX structural / resident engine surface after seal | **~100×** (about 16 MB class) |
| **Vessel capability** | Separate process that attaches a capsule and materializes pages — no Torch, no full weight RSS | **~300×** (about 5 MB class) |

Headline **~100×** is the stable center of those measured planes (phys / engine / vessel), not a single opaque “RSS always drops 100×” claim. Host `ps` RSS can still look large when macOS retains external file-cache / COW mappings; that is OS accounting, not “weights must live as process RAM.”

Correctness gate for the FullHost path: output sum **`-24.360558`** (bitexact).

---

## What problem this solves

Local LLM hosts usually pay for weights three times:

1. **Load cost** — full tensors appear in process memory  
2. **Idle cost** — cold layers keep charging RAM long after they are needed  
3. **Product confusion** — quantization, mmap, and disk offload all “save memory,” but each changes a different contract (numerics, OS cache, or I/O path)

MemX takes another cut:

- Keep **original precision** in process-owned anonymous memory  
- **Compress cold pages** with tensor-aware codecs (Metal-assisted where useful)  
- Drive residency with an explicit **working-set orchestrator**  
- Export a **Sovereign Capsule Relay (SCR)** so the weight plane becomes a transferable capability (`spill` + `ledger` + compact `rank.map`), not a forever-resident heap

You are not forced into INT4/INT8 as the product story. Decompress restores the same bytes the host wrote.

---

## How it feels in practice

### FullHost LLM path (`run_qwen.py`)

1. Host weights into MemX-managed tensors  
2. Background / vault-native compression  
3. Optional **capsule export** (APFS clone-native when possible)  
4. Infer with a streaming hot window (materialize / TCA / sovereign CRW paths)  
5. Post-infer fold + phoenix seal for engine-cold surface  
6. Optional **ultralite vessel** process: attach capsule → `materialize_rank` → report capability RSS

### Capability plane (SCR)

After compression, MemX can export:

| Artifact | Role |
|----------|------|
| `spill.bin` | Compressed page fabric (clone/hardlink when the FS allows) |
| `ledger.bin` | Page catalog |
| `rank.map` | Dense O(1) rank index (`csz` + `off`) for lite attach |

A vessel process can hold the **entire weight capability** in single-digit megabytes and materialize pages on demand. That is the architectural point: **weights are a capability, not permanent process RSS.**

API surface (C / Python):

- `memx_runtime_capsule_export` / `attach` / `detach`  
- `memx_runtime_capsule_materialize` / `materialize_v` / `materialize_rank`  
- `memx_runtime_capsule_pidx_at` / `stats`  
- Vessel binary: `build/memx_capsule_vessel` (`tools/memx_capsule_vessel.c`)

---

## Primary surfaces

| Artifact | Path |
|----------|------|
| C API | [`include/memx_runtime.h`](include/memx_runtime.h) |
| Runtime | `build/libmemx_runtime.dylib` (from `libmemx3.m`) |
| Python bindings | [`python/memx_runtime.py`](python/memx_runtime.py) |
| FullHost LLM harness | [`run_qwen.py`](run_qwen.py) |
| Capsule vessel | [`tools/memx_capsule_vessel.c`](tools/memx_capsule_vessel.c) |
| Embed example | [`examples/embedded_runtime_demo.c`](examples/embedded_runtime_demo.c) |
| Architecture deep dive | [`ARCHITECTURE_AND_OPTIMIZATION.md`](ARCHITECTURE_AND_OPTIMIZATION.md) |

---

## Measured results (stable, 0.8B)

Workload: Qwen3.5‑0.8B FullHost, bitexact **`-24.360558`**, Apple Silicon.

Representative stable SCR path (v81-class):

- **Infer wall:** ~0.40 s (matmul strip path; environment-dependent)  
- **Phys footprint:** ~9× vs model size  
- **Engine cold estimate:** ~100×  
- **Vessel RSS:** ~5 MB → **~300×** capability plane vs ~1.66 GB weights  
- **Capsule export:** milliseconds when APFS clone succeeds (`clone=1`, `rank=1`)

Older constrained FullHost paths (pre-SCR headline) also showed large resident reductions with non-destructive materialize / fused ND cache (infer RSS hundreds of MB vs multi‑GB hosted weights). Prefer the **three-plane table** above when citing SCR-era numbers.

---

## Integration model

The host opts in per allocation. Create a context, set a quota, allocate through MemX, optionally describe tensors (weight, KV, activation, …). Residency can be driven by range flags or by the working-set orchestrator (`begin_epoch` / `apply_ws` / `ws_advance` / `end_epoch`).

```c
#include "memx_runtime.h"

memx_runtime_context_t *ctx = NULL;
memx_runtime_context_create("kv-cache", &ctx);
memx_runtime_context_set_quota(ctx, 512ULL << 20);

void *buf = memx_runtime_context_malloc(ctx, 128ULL << 20);

memx_runtime_pressure_t pressure;
memx_runtime_get_pressure(&pressure);

memx_runtime_context_free(ctx, buf);
memx_runtime_context_destroy(ctx);
memx_runtime_shutdown();
```

Tensor-aware allocation (metadata only; contents stay bitexact):

```c
memx_runtime_tensor_desc_t desc = {
    .struct_size = sizeof(desc),
    .role = MEMX_TENSOR_ROLE_KV_CACHE,
    .dtype = MEMX_TENSOR_DTYPE_FP16,
    .layout = MEMX_TENSOR_LAYOUT_BLOCKED,
    .flags = MEMX_TENSOR_FLAG_SEQUENTIAL | MEMX_TENSOR_FLAG_COLD,
    .rank = 4,
    .shape = {1, 32, 4096, 128}
};

void *kv = memx_runtime_context_malloc_tensor(ctx, 512ULL << 20, &desc);

memx_runtime_context_update_tensor_flags_range(
    ctx, kv, old_token_offset, old_token_bytes,
    MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY);
```

Codecs used for tensor pages (FP16/BF16 split, delta-split, bitplane16, sparse-byte, exp-pack, zlib hybrids) decompress to the original bytes. Pages that do not compress usefully fall back to raw storage.

---

## Runtime capabilities

- Managed `malloc` / `calloc` / `realloc` / aligned alloc / `mmap` with named contexts and quotas  
- Tensor descriptors and range flag updates  
- Working-set epochs and intents (HOT / PREFETCH / RETIRE)  
- Non-destructive `materialize_range` / `materialize_tile` for strip matmul  
- Weight archives (`export_archive` / `import_archive`)  
- Vault-native pool, sovereign CRW, TCA sticky paths  
- Phoenix seal for engine-cold structural reclaim  
- **Sovereign Capsule Relay:** export / attach / rank materialize / ultralite vessel  

FullHost knobs commonly used in demos (see `run_qwen.py` defaults): vault-native pool, sovereign + TCA, post-infer phoenix, capsule relay / vessel, lite attach, host bind.

```bash
make all
export DYLD_LIBRARY_PATH=$PWD/build
export MEMX_MODEL_PATH=/path/to/Qwen3.5-0.8B-hf
python3 -u run_qwen.py
```

Capsule-only vessel probe against an exported directory:

```bash
export DYLD_LIBRARY_PATH=$PWD/build
export MEMX_NO_SELFTEST=1 MEMX_CAPSULE_LITE=1 MEMX_CPU_ONLY=1
./build/memx_capsule_vessel --dir /tmp/memx_capsule_<pid> --pages 32
```

---

## What MemX is not

- Not a quantization product (INT4/INT8 is not the value proposition)  
- Not “hide memory by hoping the OS compresses dirty pages”  
- Not a guarantee that every `ps` RSS sample collapses — cite **phys / engine / vessel** honestly  

MemX is a **bit-exact residency runtime**: compress when cold, stream when hot, export weights as a capability when you want the plane to leave the process.

---

## Status

Active research/engineering runtime for Apple Silicon. FullHost numbers above are from local Qwen3.5‑0.8B runs under the harness in this repo. Reproduce with `make all` and `run_qwen.py`; treat SUMMARY lines for phys, engine cold, and vessel as the authoritative multi-plane report.

License and contribution notes follow the repository defaults.
