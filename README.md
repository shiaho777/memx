# MemX — ~100× Lower LLM Memory on Apple, Original Precision

**Run LLMs on Apple Silicon with far less memory — without quantizing weights, without changing numerics, and without treating the model as a permanent multi‑gigabyte RSS bill.**

MemX is a bit-exact compressed-memory runtime for Mac. Cold weight pages leave the hot process surface as a durable **capability plane** (vault + capsule). Inference streams only a working set. BF16 / FP16 / FP32 bytes round-trip unchanged.

On **Qwen3.5‑0.8B** FullHost (~1.66 GB weights), stable measured planes center on a **~100×** headline:

| Plane | What it measures | Stable scale |
|-------|------------------|--------------|
| **Phys footprint** | `task_vm_info` physical charge | **~9×** (~180 MB class) |
| **Engine cold** | MemX structural surface after phoenix seal | **~100×** (~16 MB class) |
| **Vessel capability** | Capsule attach + materialize process (no Torch) | **~300×** (~5 MB class) |

`~100×` is the stable center of those three planes — not a claim that every `ps` RSS sample collapses. macOS can still show large **external / COW** file-cache residency; that is OS accounting, not “weights must live as process RAM.”

**Correctness gate:** FullHost output sum **`-24.360558`** (bitexact).

---

## Why MemX

Local LLM hosts usually pay three times for weights:

1. **Load** — full tensors land in process memory  
2. **Idle** — cold layers keep charging RAM  
3. **Confusion** — quantization, mmap, and disk offload all “save memory,” but each changes a different contract (numerics, OS cache, or I/O)

MemX does something else:

- Keep **original precision** in process-owned anonymous memory  
- **Compress cold pages** with tensor-aware codecs (Metal-assisted where useful)  
- Drive residency with an explicit **working-set orchestrator**  
- Export a **Sovereign Capsule Relay (SCR)** so weights become a transferable capability (`spill.bin` + `ledger.bin` + `rank.map`), not forever-resident heap

Decompress restores the same bytes the host wrote. INT4/INT8 is not the product story.

---

## Quick start

```bash
make all
export DYLD_LIBRARY_PATH=$PWD/build
export MEMX_MODEL_PATH=/path/to/Qwen3.5-0.8B-hf
python3 -u run_qwen.py
```

Capsule vessel against an exported directory:

```bash
export DYLD_LIBRARY_PATH=$PWD/build
export MEMX_NO_SELFTEST=1 MEMX_CAPSULE_LITE=1 MEMX_CPU_ONLY=1
./build/memx_capsule_vessel --dir /tmp/memx_capsule_<pid> --pages 32
```

Core tests:

```bash
make all
make test-explicit
make test-compressing-race
make test-python-bitexact
```

---

## How it works

### FullHost path (`run_qwen.py`)

1. Host weights into MemX-managed tensors  
2. Vault-native compression  
3. Optional **capsule export** (APFS clone when possible)  
4. Infer with a streaming hot window (materialize / TCA / sovereign CRW)  
5. Post-infer fold + **phoenix seal** for engine-cold surface  
6. Optional **ultralite vessel**: attach → `materialize_rank` → capability RSS  

### Capability plane (SCR)

| File | Role |
|------|------|
| `spill.bin` | Compressed page fabric (clone / hardlink when the FS allows) |
| `ledger.bin` | Page catalog |
| `rank.map` | Dense O(1) rank index (`csz` + `off`) for lite attach |

A vessel process can hold the **full weight capability** in single-digit megabytes and materialize pages on demand:

**Weights are a capability, not permanent process RSS.**

```c
memx_runtime_capsule_export(dir, &bytes);
memx_runtime_capsule_attach(dir);
memx_runtime_capsule_materialize_rank(rank, dst, dst_cap);
memx_runtime_capsule_detach();
```

Python mirrors these APIs in [`python/memx_runtime.py`](python/memx_runtime.py). Vessel binary: [`tools/memx_capsule_vessel.c`](tools/memx_capsule_vessel.c).

---

## Repository map

| Artifact | Path |
|----------|------|
| C API | [`include/memx_runtime.h`](include/memx_runtime.h) |
| Runtime | `libmemx3.m` → `build/libmemx_runtime.dylib` |
| Python | [`python/memx_runtime.py`](python/memx_runtime.py) |
| FullHost harness | [`run_qwen.py`](run_qwen.py) |
| Capsule vessel | [`tools/memx_capsule_vessel.c`](tools/memx_capsule_vessel.c) |
| Embed example | [`examples/embedded_runtime_demo.c`](examples/embedded_runtime_demo.c) |
| Architecture | [`ARCHITECTURE_AND_OPTIMIZATION.md`](ARCHITECTURE_AND_OPTIMIZATION.md) |

---

## Results (stable, 0.8B)

Workload: Qwen3.5‑0.8B FullHost, bitexact **`-24.360558`**, Apple Silicon.

Representative SCR-era run class:

| Metric | Stable reading |
|--------|----------------|
| Infer wall | ~0.40 s (matmul-strip path; machine-dependent) |
| Phys footprint | **~9×** vs model size |
| Engine cold | **~100×** |
| Vessel RSS | **~5 MB → ~300×** capability plane |
| Capsule export | milliseconds when `clone=1` |

### How to cite numbers

| Use this | When |
|----------|------|
| **~100×** | Headline multi-plane center |
| **~9× phys** | OS physical charge |
| **~100× engine** | Runtime structural cold surface |
| **~300× vessel** | Capability-plane process without Torch |
| Avoid sole `ps` RSS | External COW / file cache can dominate |

Deep dive: [`ARCHITECTURE_AND_OPTIMIZATION.md`](ARCHITECTURE_AND_OPTIMIZATION.md).

---

## Integrate

Opt in per allocation: context → quota → malloc / mmap / `malloc_tensor`. Drive residency with range flags or the working-set orchestrator (`begin_epoch` / `apply_ws` / `ws_advance` / `end_epoch`).

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

Tensor metadata (role / dtype / layout / flags) selects codecs and residency policy. Contents stay bitexact.

Also available: non-destructive `materialize_range` / `materialize_tile`, weight archives (`.mxwa`), vault-native pool, sovereign CRW, TCA sticky paths, phoenix seal.

---

## What MemX is not

- Not a quantization product  
- Not “hope the OS compresses dirty pages for you”  
- Not a promise that every process RSS sample is tiny  

MemX is a **bit-exact residency runtime**: compress when cold, stream when hot, export weights as a capability when the plane should leave the process.

---

## Requirements

Apple Silicon, macOS 13+, Xcode CLT. Metal compressor path available (CPU-only vessel paths supported). Explicit host integration only.

## License

MIT
