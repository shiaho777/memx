# MemX Architecture and Optimization

MemX is an embeddable compressed-memory runtime for Apple Silicon. Host applications allocate large buffers into a managed virtual pool; cold pages are compressed, faulted back on demand, and orchestrated as a streaming working set. The design target for LLM hosting is original-precision bitexact residency with process RSS far below raw model size.

Chinese version: [ARCHITECTURE_AND_OPTIMIZATION.zh-CN.md](ARCHITECTURE_AND_OPTIMIZATION.zh-CN.md)

## Design Thesis

Most LLM memory systems choose one of three axes:

1. **Quantize** weights so fewer bits live in RAM (llama.cpp / GGUF, AWQ, GPTQ, bitsandbytes).
2. **Map from disk** and rely on OS page cache / demand paging (mmap of GGUF or safetensors).
3. **Offload** tensors to CPU/NVMe under a host scheduler (FlexGen, DeepSpeed ZeRO-Inference).

MemX takes a fourth axis:

**Keep original tensor bytes bitexact inside process-owned anonymous memory, then compress and stream the resident working set.**

That yields large RSS gaps on the same machine without changing numerical results, because the host still sees exact BF16/FP16/FP32 bytes after fault/decompress.

## System Layers

```text
┌─────────────────────────────────────────────────────────────┐
│ Host (C / Python / Torch)                                   │
│  - explicit context create / quota / tensor descriptors     │
│  - epoch + working-set intents (HOT / PREFETCH / RETIRE)    │
│  - optional FullHost demo: run_qwen.py                      │
└───────────────────────────┬─────────────────────────────────┘
                            │ memx_runtime.h / memx_runtime.py
┌───────────────────────────▼─────────────────────────────────┐
│ Runtime control plane                                       │
│  - named contexts + quotas                                  │
│  - tensor role / dtype / layout / flags                     │
│  - residency orchestrator (epochs, WS tracks, pressure)     │
│  - telemetry (pool pressure, codec savings, faults)         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ Page state machine                                          │
│  PAGE_NONE → PAGE_RESIDENT → PAGE_COMPRESSING               │
│            → PAGE_COMPRESSED → PAGE_HOT / fault back        │
│  write_seq + dirty abort, CAS commit, content verify        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ Compression + virtual pool                                  │
│  - Metal-assisted compressor path                           │
│  - tensor codecs: FP16/BF16 split, delta-split, bitplane,   │
│    sparse-byte, exp-pack, zlib hybrids                      │
│  - compressed store + free-extent reclaim                   │
│  - signal-driven decompress into TLS scratch / mapped pages │
└─────────────────────────────────────────────────────────────┘
```

## Core Runtime Surface

Public header: `include/memx_runtime.h`  
Implementation: `libmemx3.m`  
Python bindings: `python/memx_runtime.py`

### Allocation and policy

- Context create / destroy / quota
- Managed `malloc` / `calloc` / `realloc` / `posix_memalign` / `mmap`
- Tensor-aware allocation via `memx_runtime_tensor_desc_t`
- Range flag updates, prefetch, mark-access
- Seal / force-compress / purge
- Pressure and per-context stats

Tensor descriptors are metadata only. They never quantize or rewrite model data. They select codec candidates and residency policy per role (weight, KV cache, activation, embedding, temporary).

### Page lifecycle guarantees

Critical correctness constraints:

1. Compress commit stores meta while `PAGE_COMPRESSING`, then CAS to `PAGE_COMPRESSED`.
2. `write_seq` plus dirty abort after mutation prevents torn commits.
3. `page_compress_content_ok` compares full page content before accepting compressed form.
4. WRITE-protect idle LLM pages carefully; sequential dirty hold avoids thrash.
5. Decompress uses TLS scratch; on Darwin, `MADV_FREE_REUSE` before rewrite after free.
6. Race stress (`test_compressing_race`) must stay green.
7. FullHost 0.8B bitexact gate: **Output sum = -24.360558**.
8. Final path must not call weight `reseal_weights()` as a blunt re-host.
9. No hot-path `seal_flush` inside matmul.
10. Col-strip next chunk is prefetch-only; do not re-pin the whole window while cooling the current strip.

## Residency Orchestrator

The orchestrator is the architecture that turns “compressed pool” into “LLM streaming residency.”

### Epochs

| Phase | Constant | Intent |
|-------|----------|--------|
| Load | `MEMX_EPOCH_LOAD` | Host tensors, allow initial compress |
| Compress | `MEMX_EPOCH_COMPRESS` | Background seal / pressure reclaim |
| Infer | `MEMX_EPOCH_INFER` | Bound hot budget, stream WS |
| Final | `MEMX_EPOCH_FINAL` | Retire tracks, final seal / purge |

APIs:

- `memx_runtime_context_begin_epoch(ctx, phase, hot_budget_bytes)`
- `memx_runtime_context_apply_ws(ctx, intents, n)`
- `memx_runtime_context_ws_advance(ctx, ptr, hot_off, hot_len, prefetch_len, flags)`
- `memx_runtime_context_ws_close(ctx, ptr, flags)`
- `memx_runtime_context_end_epoch(ctx, seal_tracked)`

### Working-set intents

`memx_runtime_ws_intent_t` carries:

- pointer + offset + length
- optional prefetch length
- priority
- flags:

| Flag | Meaning |
|------|---------|
| `MEMX_WS_FLAG_HOT` | Keep range resident for compute |
| `MEMX_WS_FLAG_PREFETCH` | Ahead-of-use fault without growing durable hot set |
| `MEMX_WS_FLAG_RETIRE` | Mark trail cold; async seal when policy allows |
| `MEMX_WS_FLAG_RETIRE_SYNC` | Seal trail synchronously |
| `MEMX_WS_FLAG_MARK_ACCESS` | Touch accounting without full pin semantics |
| `MEMX_WS_FLAG_NO_ASYNC` | Force sync path for that intent |

Context tracks a bounded set of working-set windows. Same-flag covered ranges take a skip fast path to avoid syscall churn. Prefetch budget is pressure-aware and separated from hot residency growth. Lazy trail release cold-marks by default; seal only when RETIRE / trail-seal policy requests it.

### Host integration pattern (FullHost)

`run_qwen.py` uses orchestrator when `MEMX_WS_ORCH=1` (default):

1. Host weights into MemX tensors (BF16 path, chunked half conversion).
2. Begin compress epoch; wait for background compression.
3. Begin infer epoch with hot budget.
4. For each matmul / layer:
   - pin or advance hot window
   - prefetch next strip / next op
   - retire previous strip (cold mark; seal off hot path)
5. End infer epoch.
6. Final epoch + purge / seal passes for terminal RSS.

Col-strip and block/stream WS batch dense intents through `apply_ws` when possible.

## Why RSS Can Differ Dramatically on One Machine

Same hardware, same weights, two residency modes:

| Mode | What lives in RAM | Typical 0.8B outcome |
|------|-------------------|----------------------|
| Naive host | Full BF16 weights resident | ~1.6–1.8 GB RSS class |
| MemX FullHost after compress | Compressed pages + tiny hot window | ~110–120 MB final RSS class |

Mechanism stack:

1. **Bitexact page compression** of weight tensors (split / tensor codecs), not quantization.
2. **Anonymous compressed pool** instead of leaving all pages dirty-resident.
3. **Streaming working set**: only the active matmul strip / layer window is HOT.
4. **Prefetch vs hot separation**: future pages can be staged without permanently expanding the hot set.
5. **Trail retire**: previous windows return to cold/compressed without blocking compute.
6. **Final seal / purge**: after infer, tracked windows re-compress so final RSS collapses.

Observed clean FullHost reference on Qwen3.5-0.8B (bitexact `-24.360558`):

| Path | Infer wall | Infer RSS | Final RSS |
|------|------------|-----------|-----------|
| extreme10 historical best speed under clean memory | **0.386s** | **348 MB** | **111 MB** |
| orchestrator clean (orch4) | 0.450s | 335 MB | 119 MB |

Memory reduction vs hosted weight footprint (~1663 MB):

- Final RSS ~111 MB → order of **~15×** process-resident reduction on the best constrained path.
- “Saved” telemetry in FullHost summary can report ~93% vs model/hosted size when compressor + final seal fully settle.

macOS dirty RSS (often 1000 MB+) is not treated as a real win; system memory pollution dominates variance. Prefer clean sequential FullHost logs under `/tmp/memx_opt/`.

## Competitive Landscape

Similar but not the same architecture:

| Class | Examples | What they optimize | Gap vs MemX |
|-------|----------|--------------------|-------------|
| Quantization | llama.cpp/GGUF, bitsandbytes, AWQ, GPTQ | Bits per weight | Changes precision; not bitexact original tensors |
| mmap load | GGUF mmap, safetensors mmap | Load path / OS cache | Still pays OS residency; no in-process compress+WS orchestrator |
| Offload frameworks | FlexGen, DeepSpeed ZeRO-Inference | CPU/NVMe staging | Disk/host transfer oriented; different product shape |
| OS compressed memory | macOS compressor | System-wide opportunistic | Not host-directed tensor WS with bitexact pool telemetry |

MemX differentiators:

- original-precision **bitexact** host view
- page-compressed **anonymous** residency under explicit API
- Transformer-aware descriptors + tensor codecs
- epoch + working-set orchestrator for streaming LLM windows
- pressure / codec / fault telemetry for host policy

## Repository Layout

```text
include/memx_runtime.h     Public C API
libmemx3.m                 Runtime + compressor + orchestrator
python/memx_runtime.py     ctypes bindings + WS helpers
run_qwen.py                FullHost LLM residency demo
examples/                  Embedded host example
tests/                     Explicit runtime, race, Python bitexact
benchmarks/                Runtime-native benches only
MemXApp/                   Local dashboard shell
ARCHITECTURE_AND_OPTIMIZATION.md
ARCHITECTURE_AND_OPTIMIZATION.zh-CN.md
```

Ignored local assets: `.local/`, `build/`, `MemXApp.app/`.

## Build and Validate

```bash
make all
make examples
make test-explicit
make test-compressing-race
make test-python-bitexact
```

Embedded demo:

```bash
make example-embedded
```

FullHost 0.8B baseline (weights under `.local/Qwen3.5-0.8B-hf`):

```bash
MEMX_OP_LEVEL_WS=1 MEMX_BLOCK_WS=1 MEMX_STREAM_WS=1 \
MEMX_BLOCK_PREFETCH=1 MEMX_MATMUL_CHUNK=384 MEMX_OP_FORCE_COOL=0 \
MEMX_COL_STRIP=1 MEMX_STREAM_TRAIL_SEAL=0 MEMX_STREAM_END_SEAL=0 \
MEMX_POST_HOST_FORCE=1 MEMX_FINAL_FORCE=1 MEMX_FINAL_PURGE=1 \
MEMX_FINAL_SEAL_PASSES=2 MEMX_WAIT_S=25 MEMX_ALIVE_S=0 \
MEMX_ADAPTIVE_CHUNK=1 MEMX_COLD_ASYNC_SEAL=1 MEMX_WS_ORCH=1 \
DYLD_LIBRARY_PATH=build python3 -u run_qwen.py --model-path .local/Qwen3.5-0.8B-hf
```

Prefer detached FullHost for long waits:

```python
subprocess.Popen(..., start_new_session=True)
```

Logs conventionally under `/tmp/memx_opt/`.

## Optimization Philosophy

Architecture first, knobs second.

Good work:

- reduce syscalls on already-covered HOT windows
- batch matmul intents into one `apply_ws` per op/layer
- keep trail seal off the infer hot path
- separate prefetch growth from hot residency growth
- pressure-aware prefetch caps
- codec selection that stays bitexact and improves compress ratio on real tensors

Avoid:

- stacking random env flags as “architecture”
- treating dirty 1000 MB+ RSS as speed wins
- resealing all weights on final path
- re-pinning whole windows while cooling the active strip
- comments in code (project rule: no code comments)

## Near-term Architecture Targets

1. Beat extreme10 cleanly: wall ≤ 0.386s, infer RSS ≤ 350 MB, final ≤ 111 MB, bitexact `-24.360558`.
2. Track-local delta-HOT zero-syscall fast path when window already covered.
3. Stabilize final seal under polluted macOS memory conditions.
4. Keep 0.5B / 0.8B multi-run means on clean machine as regression gates.
5. Expand orchestrator host ergonomics so non-demo runtimes get epoch/WS without reimplementing strip logic.

## Correctness Gates (Do Not Regress)

- `make test-explicit`
- `make test-compressing-race`
- Python bitexact tests
- FullHost 0.8B output sum **exactly** `-24.360558`
- Prefer clean final RSS over polluted system snapshots

## Requirements and Limits

- Apple Silicon Mac, macOS 13+, Xcode CLT
- Metal-backed compressor path
- Large managed allocations are the primary target
- Incompressible pages fall back to raw storage
- Explicit host integration only; not a system-wide injector product

## License

MIT
