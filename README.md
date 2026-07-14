# MemX

Compressed-memory runtime for Apple Silicon.

MemX gives a host process an explicit pool of managed virtual memory. Cold pages are compressed in the background (Metal-assisted), restored on fault, and accounted under named contexts with quotas and pressure telemetry. Tensor hosts can attach role, dtype, and access hints so residency and codecs follow real LLM access patterns without changing tensor bytes.

Primary surface:

| Artifact | Path |
|----------|------|
| C API | [`include/memx_runtime.h`](include/memx_runtime.h) |
| Runtime | `build/libmemx_runtime.dylib` (from `libmemx3.m`) |
| Python bindings | [`python/memx_runtime.py`](python/memx_runtime.py) |
| Embed example | [`examples/embedded_runtime_demo.c`](examples/embedded_runtime_demo.c) |
| FullHost LLM demo | [`run_qwen.py`](run_qwen.py) |

Architecture: [`ARCHITECTURE_AND_OPTIMIZATION.md`](ARCHITECTURE_AND_OPTIMIZATION.md)

## Why this exists

Large local models and in-process caches often need more virtual capacity than physical RAM allows, while still wanting original-precision compute. Quantization reduces bits; mmap defers to the OS page cache; offload frameworks move tensors to disk or another device. MemX keeps original BF16/FP16/FP32 bytes in process-owned anonymous memory, compresses idle pages, and streams a small hot working set during inference.

On Qwen3.5-0.8B FullHost (bitexact output sum `-24.360558`), a clean constrained path has held about **348 MB** RSS during infer and **111 MB** final after recompress; with non-destructive materialize strips, infer RSS has been measured near **251 MB** (final ~116 MB) at similar bitexact output, against ~1.6 GB hosted weights (~**15×** resident reduction on that path).

## Integration model

The host opts in per allocation. You create a context, set a quota, allocate through MemX, and optionally describe tensors (weight, KV, activation, …). Residency can be driven by range flags or by the working-set orchestrator (`begin_epoch` / `apply_ws` / `ws_advance` / `end_epoch`).

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

## Runtime capabilities

Weight archives and tile residency (architecture path):

- `memx_runtime_context_export_archive` / `import_archive` — offline bitexact page blobs (`.mxwa`); load installs compressed pages without inflating full BF16 first
- `memx_runtime_context_ws_tile` — matmul column-strip geometry → HOT/PREFETCH/RETIRE without host-side page coalescing
- `materialize_range` / `materialize_tile` — non-destructive bitexact read into a host buffer while pages stay `PAGE_COMPRESSED` (matmul strip path; `MEMX_MATERIALIZE=1`, `MEMX_MATERIALIZE_SKIP_PIN=1`)
- FullHost: set `MEMX_ARCHIVE_DIR=/path` to save/load per-tensor archives (`MEMX_ARCHIVE_SAVE=1`, `MEMX_ARCHIVE_LOAD=1`)


- Init / shutdown, managed `malloc` / `calloc` / `realloc` / aligned alloc / `mmap`
- Named contexts and per-context quotas
- Tensor descriptors and range flag updates
- Prefetch, mark-access, seal / force-compress / purge
- Epoch + working-set orchestrator for streaming LLM windows
- Pool pressure, fragmentation, codec savings, and fault telemetry

## Build

Requires Apple Silicon, macOS 13+, Xcode Command Line Tools.

```bash
make all
make examples
./build/embedded_runtime_demo
```

Link a host:

```bash
clang -O2 -Iinclude your_host.c -Lbuild -Wl,-rpath,@executable_path -lmemx_runtime -o your_host
```

Checks:

```bash
make test-explicit
make test-compressing-race
make test-python-bitexact
make benchmark-tensor-codecs
```

## FullHost demo

Place weights under `.local/` (gitignored), e.g. `.local/Qwen3.5-0.8B-hf`, or pass `--model-path`.

```bash
make all
MEMX_OP_LEVEL_WS=1 MEMX_BLOCK_WS=1 MEMX_STREAM_WS=1 \
MEMX_BLOCK_PREFETCH=1 MEMX_MATMUL_CHUNK=384 MEMX_COL_STRIP=1 \
MEMX_POST_HOST_FORCE=1 MEMX_FINAL_FORCE=1 MEMX_FINAL_PURGE=1 \
MEMX_FINAL_SEAL_PASSES=2 MEMX_WAIT_S=25 MEMX_WS_ORCH=1 \
DYLD_LIBRARY_PATH=build python3 -u run_qwen.py --model-path .local/Qwen3.5-0.8B-hf
```

## Layout

```text
include/memx_runtime.h
libmemx3.m
python/memx_runtime.py
run_qwen.py
examples/
tests/
benchmarks/
MemXApp/
ARCHITECTURE_AND_OPTIMIZATION.md
```

`build/` and `.local/` are ignored.

## License

MIT
