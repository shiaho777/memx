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
make test
make test-explicit
make test-compressing-race
make test-python-bitexact
```

---

## Architecture

```text
Host (C / Python / Torch)
  contexts, quotas, tensor descriptors
  epoch + WS intents (HOT / PREFETCH / RETIRE)
  FullHost: run_qwen.py
        │
        ▼  memx_runtime.h / memx_runtime.py
Control plane
  residency orchestrator, telemetry, capsule API
        │
        ▼
Page state machine
  RESIDENT → COMPRESSING → COMPRESSED → fault / HOT / materialize
  write_seq, dirty abort, CAS commit
        │
        ▼
Vault-native compressed pool
  tensor codecs, Metal-assisted compress, sovereign CRW / TCA
        │
        ▼
Capability plane (SCR)
  spill.bin + ledger.bin + rank.map
  lite attach · materialize_rank · ultralite vessel
```

| Piece | Path |
|-------|------|
| C API | [`include/memx_runtime.h`](include/memx_runtime.h) |
| Runtime | `libmemx3.m` → `build/libmemx_runtime.dylib` |
| Python | [`python/memx_runtime.py`](python/memx_runtime.py) |
| FullHost | [`run_qwen.py`](run_qwen.py) |
| Vessel | [`tools/memx_capsule_vessel.c`](tools/memx_capsule_vessel.c) |
| Embed demo | [`examples/embedded_runtime_demo.c`](examples/embedded_runtime_demo.c) |

### Allocation and tensor policy

Contexts own managed allocations and quotas. Hosts allocate with `malloc` / `mmap` or `malloc_tensor` plus `memx_runtime_tensor_desc_t` (role, dtype, layout, flags, shape). Descriptors select codec candidates and residency policy; they never rewrite tensor contents.

Range APIs cover flag updates, prefetch, mark-access, seal, force-compress, purge, and non-destructive materialize. Stats expose pool pressure, fragmentation, codec savings, faults, and per-context usage.

### Page lifecycle

Compress path:

1. Enter `PAGE_COMPRESSING`, write compression meta
2. CAS to `PAGE_COMPRESSED` only if content still matches
3. Abort on dirty / `write_seq` change after mutation

Decompress uses TLS scratch. Race coverage: `test_compressing_race`.

Operational rules that matter for FullHost:

- No `seal_flush` inside matmul
- Soft pre-infer trim only (`trim(32)`); hard trim before infer risks SIGBUS
- Soft post-infer vault release is OK; hard trim against `PROT_NONE` vault after soft seal is not
- Do not hard-decommit live compressed pool blobs except after durable spill + pread path

Codecs (FP16/BF16 split, delta-split, bitplane16, sparse-byte, exp-pack, zlib hybrids) decompress to original bytes. Incompressible pages store raw.

### Residency orchestrator

| Phase | Constant | Role |
|-------|----------|------|
| Load | `MEMX_EPOCH_LOAD` | Host tensors; allow initial compress |
| Compress | `MEMX_EPOCH_COMPRESS` | Background seal / reclaim under pressure |
| Infer | `MEMX_EPOCH_INFER` | Bound hot budget; advance windows |
| Final | `MEMX_EPOCH_FINAL` | Retire tracks; terminal seal / purge |

```c
memx_runtime_context_begin_epoch(ctx, phase, hot_budget_bytes);
memx_runtime_context_apply_ws(ctx, intents, n);
memx_runtime_context_ws_advance(ctx, ptr, hot_off, hot_len, prefetch_len, flags);
memx_runtime_context_ws_close(ctx, ptr, flags);
memx_runtime_context_end_epoch(ctx, seal_tracked);
```

| Flag | Behavior |
|------|----------|
| `MEMX_WS_FLAG_HOT` | Keep range resident for compute |
| `MEMX_WS_FLAG_PREFETCH` | Stage ahead without growing durable hot set |
| `MEMX_WS_FLAG_RETIRE` | Cold-mark trail; async seal when policy allows |
| `MEMX_WS_FLAG_RETIRE_SYNC` | Seal trail synchronously |
| `MEMX_WS_FLAG_MARK_ACCESS` | Access accounting without full pin |
| `MEMX_WS_FLAG_NO_ASYNC` | Force sync path |

Prefetch caps follow pool pressure and stay separate from hot growth.

### FullHost path

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
| `ledger.bin` | Page catalog (`pidx`, `csz`, `off`, codec, …) |
| `rank.map` | Dense O(1) rank index (`csz` + `off`) for lite attach |

| Mode | Memory behavior | Use |
|------|-----------------|-----|
| Full | Ledger mapped in process | Host-side tooling |
| **Lite** | Ledger on demand via `pread` | Default vessel / host bind |
| **Rank** | `materialize_rank` / dense pidx→rank | Fast materialize without binary search |

```c
memx_runtime_capsule_export(dir, &bytes);
memx_runtime_capsule_attach(dir);
memx_runtime_capsule_materialize_rank(rank, dst, dst_cap);
memx_runtime_capsule_detach();
```

**Weights are a capability, not permanent process RSS.**

Vessel (`tools/memx_capsule_vessel.c`): no Torch, lite attach + single-page scratch (`--ultra` default), reports capability RSS / phys / × vs logical page bytes — the **~300×** plane on 0.8B.

Optional host bind (`MEMX_CAPSULE_HOST_BIND`) keeps the capability live so phoenix can tear down zone surfaces without erasing the weight plane. Named spill keep (`MEMX_POOL_SPILL_KEEP`) enables path-based clone; unlinked anonymous spill forces byte copy.

### Non-destructive materialize and archives

Fault/decompress can **consume** compressed pool data into HOT residency. For read-mostly weight strips that is often wrong.

- `materialize_range` / `materialize_tile` — bitexact copy into caller buffers
- Keep-compressed policy leaves pages `PAGE_COMPRESSED`
- ND page cache + BF16→FP16 fused feed reduces extra copies
- Prefetch warms cache for the next strip without HOT growth

Weight archive (`.mxwa`): `export_archive` / `import_archive` install precompressed pages without a full BF16 inflate-then-compress cycle. FullHost hooks: `MEMX_ARCHIVE_DIR`, `MEMX_ARCHIVE_SAVE`, `MEMX_ARCHIVE_LOAD`.

`ws_tile` maps matmul column-strip geometry to HOT / PREFETCH / RETIRE without host-side page coalescing.

### Optimization principles

1. Capability plane over permanent weight RSS (SCR)
2. Vault-native durable spill over process-backed pool mirrors
3. Non-destructive materialize for read-mostly strips
4. Prefetch ≠ hot residency
5. Skip syscalls when the HOT window is already covered
6. Batch intents; keep trail seal off the infer hot path
7. Honest multi-plane metrics in SUMMARY

### Related systems

| Approach | Trade-off vs MemX |
|----------|-------------------|
| Quantization (GGUF, AWQ, GPTQ, …) | Fewer bits; different precision |
| mmap load | OS cache residency; no host-directed compressed WS API |
| Offload (FlexGen, ZeRO-Inference, …) | Movement as the main lever |
| OS compressor | System-wide; not tensor-WS + bitexact pool API |

MemX: bitexact host bytes + page-compressed anonymous memory + tensor policy + epoch/WS orchestrator + capability export.

---

## Results (stable, 0.8B)

Workload: Qwen3.5‑0.8B FullHost, bitexact **`-24.360558`**, Apple Silicon.

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

| SUMMARY line | Meaning | Cite as |
|--------------|---------|---------|
| phys_footprint | Kernel physical charge | **Phys ×** |
| Engine cold est | Resident engine + structural floor | **Engine ×** |
| Vessel RSS | Capsule process after attach/materialize | **Vessel ×** |
| `ps` RSS / external | May include file-cache COW | Context only |

Do not claim process 30×/90× from dirty multi‑GB duals, non-bitexact runs, or phys-only samples mislabeled as process RSS.

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

## Gates

- `make all` / `make test`
- `make test-explicit`
- `make test-compressing-race`
- Python bitexact suite (`make test-python-bitexact`)
- FullHost 0.8B output sum exactly **`-24.360558`**
- Prefer dual bitexact when claiming gains
- Report phys / engine / vessel; do not oversell polluted `ps` RSS

### Direction

- Host capsule-native cold resume after phoenix
- Stronger batch rank pread for lite materialize
- Larger models under the same three-plane report
- Keep bitexact and infer wall as co-equal gates with memory

---

## Requirements

Apple Silicon, macOS 13+, Xcode CLT. Metal compressor path for the main runtime; vessel can run CPU-only. Designed for large managed allocations; incompressible pages store raw. Explicit host integration only.

---

## Contributing

Changes use **Issue → PR into `main` → CI (`gate`) → merge**. See [CONTRIBUTING.md](CONTRIBUTING.md). Agents: [AGENTS.md](AGENTS.md). Linked Issues close only on merge via `Fixes #N` / `Closes #N`.

---

## License

MIT
