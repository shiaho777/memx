# MemX Architecture and Optimization

How MemX is built, how LLM residency is orchestrated, and how memory/speed are measured on Apple Silicon.

**Framing:** ~**100×** lower LLM memory as a multi-plane headline (original precision / bitexact), with transparent stable planes on Qwen3.5‑0.8B:

| Plane | Stable scale | Notes |
|-------|--------------|-------|
| Phys footprint | **~9×** | `task_vm_info` physical charge (~180 MB class) |
| Engine cold | **~100×** | Structural surface after phoenix seal (~16 MB class) |
| Vessel capability | **~300×** | Capsule process, no Torch (~5 MB class) |

Host `ps` RSS is **not** the sole scoreboard when external file-cache COW inflates samples.

**Correctness gate:** FullHost output sum **`-24.360558`**.

---

## Problem

Local LLM hosts must retain large tensors while controlling memory. Quantization, mmap-only loads, and CPU/NVMe offload each change a different contract. MemX targets another cut:

- **Original-precision** tensors in anonymous process memory  
- **Compressed when cold**, fault / materialize when hot  
- Explicit **working-set orchestration** during compute  
- **Sovereign Capsule Relay (SCR)** so the weight plane can leave live process surfaces as a transferable capability  

---

## Layered design

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

| Piece | Location |
|-------|----------|
| Public API | `include/memx_runtime.h` |
| Implementation | `libmemx3.m` |
| Python | `python/memx_runtime.py` |
| FullHost | `run_qwen.py` |
| Vessel | `tools/memx_capsule_vessel.c` |

---

## Allocation and tensor policy

Contexts own managed allocations and quotas. Hosts allocate with `malloc` / `mmap` or `malloc_tensor` plus `memx_runtime_tensor_desc_t` (role, dtype, layout, flags, shape). Descriptors select codec candidates and residency policy; they never rewrite tensor contents.

Range APIs cover flag updates, prefetch, mark-access, seal, force-compress, purge, and non-destructive materialize. Stats expose pool pressure, fragmentation, codec savings, faults, and per-context usage.

---

## Page lifecycle (correctness)

Compress path:

1. Enter `PAGE_COMPRESSING`, write compression meta.  
2. CAS to `PAGE_COMPRESSED` only if content still matches (full-page check).  
3. Abort on dirty / `write_seq` change after mutation.  

Decompress uses TLS scratch. On Darwin, free/reuse advice precedes rewrite after free. Race coverage: `test_compressing_race`.

Operational rules that matter for FullHost:

- No `seal_flush` inside matmul  
- Final path does not blunt-reseal all weights via a full `reseal_weights()` hammer  
- Soft pre-infer trim only (`trim(32)`); hard trim before infer risks SIGBUS  
- Soft post-infer vault release is OK; hard trim against `PROT_NONE` vault after soft seal is not  
- Do not hard-decommit live compressed pool blobs except after durable spill + pread path  

Codecs (FP16/BF16 split, delta-split, bitplane16, sparse-byte, exp-pack, zlib hybrids) decompress to original bytes. Incompressible pages store raw.

---

## Residency orchestrator

Turns a compressed pool into a streamable LLM working set.

### Epochs

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

### Intent flags

| Flag | Behavior |
|------|----------|
| `MEMX_WS_FLAG_HOT` | Keep range resident for compute |
| `MEMX_WS_FLAG_PREFETCH` | Stage ahead without growing durable hot set |
| `MEMX_WS_FLAG_RETIRE` | Cold-mark trail; async seal when policy allows |
| `MEMX_WS_FLAG_RETIRE_SYNC` | Seal trail synchronously |
| `MEMX_WS_FLAG_MARK_ACCESS` | Access accounting without full pin |
| `MEMX_WS_FLAG_NO_ASYNC` | Force sync path |

Prefetch caps follow pool pressure and stay separate from hot growth.

### FullHost flow (`run_qwen.py`)

1. Host weights into MemX tensors  
2. Compress / vault-native settle  
3. Optional SCR export (+ host lite bind, vessel probe)  
4. Infer epoch: advance hot window, materialize strips, retire trail  
5. Post-infer fold (vault + structural + phoenix)  
6. Final samples: report phys / engine / vessel honestly  

---

## Sovereign Capsule Relay (SCR)

SCR exports the compressed weight fabric as a capability directory.

### Artifacts

| File | Role |
|------|------|
| `spill.bin` | Compressed pages; APFS `clonefile` / hardlink when possible |
| `ledger.bin` | Page catalog (`pidx`, `csz`, `off`, codec, …) |
| `rank.map` | Dense O(1) rank records (`csz`, `off_lo/hi`) — half the entry tax for hot paths |

### Attach modes

| Mode | Memory behavior | Use |
|------|-----------------|-----|
| Full | Ledger mapped in process | Host-side tooling |
| **Lite** | Ledger on demand via `pread` | Default vessel / host bind |
| **Rank** | `materialize_rank` / dense pidx→rank | Fast materialize without binary search |

### APIs

```text
export / attach / detach
materialize(pidx) · materialize_v(pidxs) · materialize_rank(rank)
pidx_at(rank) · stats
```

### Vessel

`tools/memx_capsule_vessel.c` → `build/memx_capsule_vessel`

- No Torch  
- Lite attach + single-page scratch (`--ultra` default)  
- Reports capability RSS / phys / × vs logical page bytes  

This is the **capability plane** measurement (~300× class on 0.8B).

### Host bind

After export, optional auto lite-attach (`MEMX_CAPSULE_HOST_BIND`) keeps the capability live so phoenix can tear down zone surfaces without erasing the weight plane.

### Export performance

Named spill keep (`MEMX_POOL_SPILL_KEEP`) enables path-based clone; unlinked anonymous spill forces byte copy. Clone path is millisecond-class for multi‑GB logical spills.

---

## Non-destructive materialize and archives

### Materialize (infer fusion)

Fault/decompress can **consume** compressed pool data into HOT residency. For read-mostly weight strips that is often wrong.

- `materialize_range` / `materialize_tile` — bitexact copy into caller buffers  
- Keep-compressed policy leaves pages `PAGE_COMPRESSED`  
- ND page cache + BF16→FP16 fused feed reduces extra copies  
- Prefetch warms cache for the next strip without HOT growth  

### Weight archive (`.mxwa`)

`export_archive` / `import_archive` install precompressed pages without a full BF16 inflate-then-compress cycle. FullHost hooks: `MEMX_ARCHIVE_DIR`, `MEMX_ARCHIVE_SAVE`, `MEMX_ARCHIVE_LOAD`.

### Tile WS

`ws_tile` maps matmul column-strip geometry to HOT / PREFETCH / RETIRE without host-side page coalescing.

---

## Memory planes (how to read SUMMARY)

| Line | Meaning | Cite as |
|------|---------|---------|
| phys_footprint | Kernel physical charge | **Phys ×** |
| Engine cold est | Resident engine + structural floor | **Engine ×** |
| Vessel RSS | Capsule process after attach/materialize | **Vessel ×** |
| `ps` RSS / external | May include file-cache COW | Context only |

**Do not** claim process 30×/90× from dirty multi‑GB duals, non-bitexact runs, or phys-only samples mislabeled as process RSS.

Headline **~100×** = stable center of phys / engine / vessel planes on 0.8B FullHost bitexact runs.

---

## Related systems

| Approach | Trade-off vs MemX |
|----------|-------------------|
| Quantization (GGUF, AWQ, GPTQ, …) | Fewer bits; different precision |
| mmap load | OS cache residency; no host-directed compressed WS API |
| Offload (FlexGen, ZeRO-Inference, …) | Movement as the main lever |
| OS compressor | System-wide; not tensor-WS + bitexact pool API |

MemX: bitexact host bytes + page-compressed anonymous memory + tensor policy + epoch/WS orchestrator + capability export.

---

## Build, test, FullHost

```bash
make all
make test-explicit
make test-compressing-race
make test-python-bitexact
```

```bash
export DYLD_LIBRARY_PATH=$PWD/build
export MEMX_MODEL_PATH=.local/Qwen3.5-0.8B-hf
python3 -u run_qwen.py
```

SCR-oriented defaults live in `run_qwen.py` (`MEMX_POOL_VAULT_NATIVE`, sovereign/TCA, phoenix, capsule relay/vessel/lite/host-bind). Long runs work better detached; logs often under `/tmp/memx_opt/`.

Vessel-only:

```bash
export DYLD_LIBRARY_PATH=$PWD/build
export MEMX_NO_SELFTEST=1 MEMX_CAPSULE_LITE=1 MEMX_CPU_ONLY=1
./build/memx_capsule_vessel --dir /tmp/memx_capsule_<pid> --pages 32
```

---

## Optimization principles

Prefer structural changes over flag stacks:

1. Capability plane over permanent weight RSS (SCR)  
2. Vault-native durable spill over process-backed pool mirrors  
3. Non-destructive materialize for read-mostly strips  
4. Prefetch ≠ hot residency  
5. Skip syscalls when the HOT window is already covered  
6. Batch intents; keep trail seal off the infer hot path  
7. Honest multi-plane metrics in SUMMARY  

### Direction

- Host capsule-native cold resume after phoenix  
- Stronger batch rank pread for lite materialize  
- Larger models under the same three-plane report  
- Keep bitexact and infer wall as co-equal gates with memory  

---

## Gates

- `make test-explicit`  
- `make test-compressing-race`  
- Python bitexact suite  
- FullHost 0.8B output sum exactly **`-24.360558`**  
- Prefer dual bitexact when claiming gains  
- Report phys / engine / vessel; do not oversell polluted `ps` RSS  

---

## Requirements

Apple Silicon, macOS 13+, Xcode CLT. Metal compressor path for the main runtime; vessel can run CPU-only. Designed for large managed allocations; incompressible pages store raw. Explicit host integration only.

## License

MIT
