# MemX Architecture and Optimization

How MemX is structured, how LLM residency is orchestrated, and how we evaluate memory and speed.

## Problem

Local LLM hosts and other large in-process caches must retain large tensors while keeping process RSS under control. Changing numerical precision (quantization), relying on OS mmap alone, or scheduling disk/CPU offload are established options. MemX addresses a different cut of the problem: **original-precision tensors in anonymous process memory**, compressed when cold, with an explicit streaming working set during compute.

Correctness bar for the FullHost path: decompress always restores the same bytes the host wrote; Qwen3.5-0.8B FullHost is gated on output sum **`-24.360558`**.

## Layers

```text
Host (C / Python / Torch)
  explicit contexts, quotas, tensor descriptors
  epoch + WS intents (HOT / PREFETCH / RETIRE)
  FullHost reference: run_qwen.py
        │
        ▼  memx_runtime.h / memx_runtime.py
Control plane
  contexts, tensor policy, residency orchestrator, telemetry
        │
        ▼
Page state machine
  RESIDENT → COMPRESSING → COMPRESSED → fault / HOT
  write_seq, dirty abort, CAS commit, full-page content check
        │
        ▼
Compressed virtual pool
  Metal-assisted compression, tensor codecs, free-extent reclaim
  signal-driven decompress (TLS scratch / mapped pages)
```

Sources:

| Piece | Location |
|-------|----------|
| Public API | `include/memx_runtime.h` |
| Implementation | `libmemx3.m` |
| Python | `python/memx_runtime.py` |
| FullHost | `run_qwen.py` |

## Allocation and tensor policy

Contexts own managed allocations and quotas. Hosts allocate with `malloc` / `mmap` or `malloc_tensor` plus `memx_runtime_tensor_desc_t` (role, dtype, layout, flags, shape). Descriptors select codec candidates and residency policy; they never rewrite tensor contents.

Range APIs cover flag updates, prefetch, mark-access, seal, force-compress, and purge. Stats expose pool pressure, fragmentation, codec savings, faults, and per-context usage.

## Page lifecycle (correctness)

Compress path:

1. Enter `PAGE_COMPRESSING`, write compression meta.
2. CAS to `PAGE_COMPRESSED` only if content still matches (`page_compress_content_ok`, full page).
3. Abort on dirty / `write_seq` change after mutation.

Decompress uses TLS scratch. On Darwin, `MADV_FREE_REUSE` precedes rewrite after free. Idle LLM pages may be write-protected carefully; sequential dirty hold reduces thrash. Race coverage lives in `test_compressing_race`.

Operational rules that matter for FullHost:

- No `seal_flush` inside matmul.
- Final path does not blunt-reseal all weights via `reseal_weights()`.
- Next col-strip is prefetch-only; cooling the current strip must not re-pin the entire window.

## Residency orchestrator

The orchestrator turns a compressed pool into a streamable LLM working set.

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
| `MEMX_WS_FLAG_PREFETCH` | Stage ahead of use without growing durable hot set |
| `MEMX_WS_FLAG_RETIRE` | Cold-mark trail; async seal when policy allows |
| `MEMX_WS_FLAG_RETIRE_SYNC` | Seal trail synchronously |
| `MEMX_WS_FLAG_MARK_ACCESS` | Access accounting without full pin |
| `MEMX_WS_FLAG_NO_ASYNC` | Force sync path |

Each context tracks a bounded set of windows. Covered ranges with unchanged flags skip redundant work. Prefetch caps follow pool pressure and stay separate from hot growth. Trails cold-mark by default; seal only under RETIRE / trail-seal policy.

### FullHost flow (`run_qwen.py`, `MEMX_WS_ORCH=1`)

1. Host weights into MemX tensors (BF16, chunked conversion).
2. Compress epoch; wait for background compression.
3. Infer epoch with hot budget.
4. Per matmul / layer: advance hot window, prefetch next strip/op, retire previous strip off the hot path.
5. End infer; final epoch + purge / seal passes for terminal RSS.

Dense strip work batches through `apply_ws` when the host can.

## Memory reduction mechanism

Same machine, same weights:

| Mode | Resident content | 0.8B class |
|------|------------------|------------|
| Naive host | Full BF16 weights | ~1.6–1.8 GB RSS |
| MemX FullHost settled | Compressed pages + small hot window | ~110–120 MB final |

Stack:

1. Bitexact page compression (tensor codecs), not quantization.
2. Anonymous compressed pool instead of keeping every page dirty-resident.
3. Streaming HOT window over active matmul strips / layers.
4. Prefetch decoupled from durable hot residency.
5. Trail retire returns prior windows to compressed form.
6. Final seal / purge collapses RSS after infer.

Clean reference (Qwen3.5-0.8B, bitexact `-24.360558`):

| Path | Infer wall | Infer RSS | Final RSS |
|------|------------|-----------|-----------|
| extreme10 | **0.386 s** | **348 MB** | **111 MB** |
| orchestrator clean (orch4) | 0.450 s | 335 MB | 119 MB |
| materialize strip (`MEMX_MATERIALIZE=1`) | 0.434 s | **251 MB** | 116 MB |
| materialize + fused FP16 + ND prefetch | **0.400–0.412 s** | clean-machine ~250 class; dirty runs vary | ~116–232 |
| materialize vNext (TLS/async/parallel) | **0.397 s** | **250 MB** | **126 MB** |

Relative to ~1663 MB hosted weights, final ~111 MB is on the order of **15×**. FullHost “Saved” telemetry can report ~93% once compression and final seal settle. Prefer clean sequential logs; macOS dirty RSS spikes (often 1000 MB+) are system noise, not product wins.

## Related systems

| Approach | Examples | Trade-off vs MemX |
|----------|----------|-------------------|
| Quantization | llama.cpp/GGUF, AWQ, GPTQ, bitsandbytes | Fewer bits; different precision |
| mmap load | GGUF / safetensors mmap | OS cache residency; no host-directed compressed WS |
| Offload | FlexGen, DeepSpeed ZeRO-Inference | CPU/NVMe movement as the main lever |
| OS compressor | macOS memory compression | System-wide, not tensor-WS + bitexact pool API |

MemX combines bitexact host bytes, page-compressed anonymous memory, Transformer descriptors/codecs, and an epoch/WS orchestrator with pool telemetry.

## Build, test, FullHost

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

Long runs work better detached (`Popen(..., start_new_session=True)`); logs under `/tmp/memx_opt/`.



## Weight archive and tile residency

First architecture cut beyond pure online compress:

### Offline weight archive (`.mxwa`)

Bitexact page blobs for a managed tensor:

1. Host writes weights once (or reuses an existing managed allocation).
2. `memx_runtime_context_export_archive` force-compresses, then writes header + per-page codec/blob directory.
3. Later runs call `memx_runtime_context_import_archive` to allocate the tensor and **install precompressed pool pages** (`PAGE_COMPRESSED`) without a full BF16 inflate-then-compress cycle.
4. Torch views still point at the managed pointer; first touch faults/decompresses as usual.

FullHost hooks: `MEMX_ARCHIVE_DIR`, `MEMX_ARCHIVE_SAVE`, `MEMX_ARCHIVE_LOAD`.



### Non-destructive materialize (compute fusion step)

Fault/decompress historically **consumes** compressed pool data and leaves pages HOT/resident. For read-mostly weights that is the wrong default during strip matmul.

New APIs:

- `memx_runtime_context_materialize_range` — copy a byte range into a caller buffer
- `memx_runtime_context_materialize_tile` — gather a column strip into a dense tile buffer (page cache, no large span malloc)

With `MEMX_MATERIALIZE_KEEP_COMPRESSED`, compressed pages are peeked (pool payload → TLS decompress → `dst`) and **remain `PAGE_COMPRESSED`**. FullHost row/col matmul uses this path and can skip pin growth (`MEMX_MATERIALIZE_SKIP_PIN`).

Speed path (fused GEMM feed):

- Zone-level ND page cache (128 pages) avoids re-decompress across strips
- `MEMX_MATERIALIZE_BF16_TO_FP16` converts into the FP16 matmul buffer in one pass (no extra `tensor.half()` copy)
- `materialize_prefetch_range` warms the cache for the next strip without HOT residency
- Observed: wall **0.400–0.412 s** with bitexact `-24.360558` (vs ~0.434 s pre-fusion materialize)
- Next cut: TLS page cache (24) + global cache (256), async ND prefetch workers (4), parallel unique-page warm via `dispatch_apply`, GEMM-overlapped ND prefetch, mat chunk 512
- Clean FullHost: wall **0.397 s**, infer RSS **250 MB**, final **126 MB**, bitexact `-24.360558`

### Tile working-set API

`memx_runtime_context_ws_tile` takes row/col/elem geometry and maps column strips to page ranges, then applies HOT / PREFETCH / optional RETIRE. This moves strip coalescing from Python host loops into the runtime (step toward decompress–compute fusion).

## Optimization direction

Prefer structural changes over env-flag stacking:

- Skip syscalls when the HOT window is already covered.
- Batch matmul intents into one `apply_ws` per op/layer.
- Keep trail seal off the infer hot path.
- Separate prefetch growth from hot residency growth.
- Pressure-aware prefetch caps.
- Bitexact codecs that improve ratio on real weight/KV traffic.

Targets: cleanly match or beat extreme10 (≤0.386 s wall, ≤350 MB infer RSS, ≤111 MB final, bitexact); stabilize final seal under polluted macOS conditions; multi-run means for 0.5B/0.8B as regression gates; host ergonomics so non-demo runtimes need less strip logic.

## Gates

- `make test-explicit`
- `make test-compressing-race`
- Python bitexact suite
- FullHost 0.8B output sum exactly `-24.360558`
- Prefer clean final RSS over polluted snapshots

## Requirements

Apple Silicon, macOS 13+, Xcode CLT. Metal compressor path. Designed for large managed allocations; incompressible pages store raw. Explicit host integration only.

## License

MIT
