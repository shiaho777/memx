# MemX

> Embeddable compressed-memory runtime for Apple Silicon apps.

MemX lets a host application overcommit RAM inside its own process by routing managed allocations into a compressed virtual-memory pool. Cold pages are compressed with Metal, faulted back on demand, and tracked with per-context quotas and runtime telemetry.

This repository is now centered on the explicit runtime API:

- `include/memx_runtime.h`
- `build/libmemx_runtime.dylib`
- `examples/embedded_runtime_demo.c`

## What MemX Is For

MemX is useful when your app already owns a large memory tier and needs a better option than dropping straight to SSD swap:

- local AI and LLM runtimes
- vector databases and search engines
- in-process caches
- analytical desktop apps
- memory-heavy developer tools

The right integration model is: the host opts in, decides which buffers are MemX-managed, sets quotas per workload, and reacts to pressure telemetry.

## What MemX Is Not

- not system-wide macOS memory compression
- not a code-signing workaround
- not a finished platform product yet

## Core Runtime Surface

The explicit API already supports:

- runtime init and shutdown
- managed `malloc` / `calloc` / `realloc`
- managed `mmap` / `munmap`
- named runtime contexts
- per-context quotas
- Transformer-aware tensor descriptors for weights, KV cache, activations, and hot/no-compress regions
- ownership-aware accounting
- pressure and fragmentation telemetry
- explicit reclaim of stale compressed extents

Minimal integration shape:

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

Transformer-aware allocations let a host runtime expose model semantics without changing tensor contents:

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

memx_runtime_context_update_tensor_flags(
    ctx,
    kv,
    MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY
);

memx_runtime_context_update_tensor_flags_range(
    ctx,
    kv,
    old_token_offset,
    old_token_bytes,
    MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY
);
```

Those descriptors are bit-exact metadata. They do not quantize or rewrite model data; they give MemX enough context to choose compression, deduplication, hot-path, and prefetch policies per tensor class.

The tensor-specialized codecs are conservative and bit-exact. Warm tensor pages can use a fast FP16/BF16 split codec that separates low and high bytes and run-length encodes the high-byte stream. Cold or read-mostly tensor pages can use a higher-ratio bitplane16 codec. Pages that do not benefit fall back to the default compressor, and decompression restores the original bytes exactly.

## Quick Start

Build the runtime and the embedded host example:

```bash
make explicit-runtime examples
./build/embedded_runtime_demo
```

Compare tensor-specific bit-exact codec candidates:

```bash
make benchmark-tensor-codecs
```

Build your own host process against the explicit runtime:

```bash
clang -O2 -Iinclude your_host.c -Lbuild -Wl,-rpath,@executable_path -lmemx_runtime -o your_host
```

## Embedded Host Example

`examples/embedded_runtime_demo.c` models the intended product shape:

- create a dedicated managed context
- set a 192 MB quota
- allocate cache segments explicitly through MemX
- tag cache segments as tensor-shaped KV-cache data
- handle quota pressure by evicting host-owned objects
- inspect pressure, reclaim stats, and context accounting

This is the direction that scales into real integrations such as KV-cache tiers, tensor arenas, and compressed object stores.

## Architecture Deep Dive

See [`docs/ARCHITECTURE_AND_OPTIMIZATION.md`](docs/ARCHITECTURE_AND_OPTIMIZATION.md) for:

- residency orchestrator (epochs + working-set intents)
- page state machine and bitexact compress guarantees
- why process RSS can drop by ~15× on the same machine without quantization
- FullHost 0.8B reference numbers and correctness gates
- competitive landscape vs quant / mmap / offload systems

## Architecture

```text
Host app
  -> explicit MemX-managed allocations
  -> tensor role, dtype, layout, and hot/cold policy hints
  -> context quota and ownership tracking
  -> compressed virtual page pool
  -> Metal-assisted background compression
  -> signal-driven fault/decompress path
  -> pressure telemetry and reclaim
```

Page lifecycle:

```text
PAGE_NONE -> PAGE_RESIDENT -> PAGE_COMPRESSED -> PAGE_HOT -> PAGE_RESIDENT
```

## Current Validation

- `make test-explicit`: explicit runtime smoke test with quota, telemetry, reclaim, and integrity checks
- `make benchmark-runtime`: runtime-native benchmark suite for sparse, deduplicated, and mixed managed buffers
- `make benchmark-stress`: multi-threaded context stress benchmark
- `make benchmark-tensor-codecs`: offline FP16/BF16/KV codec comparison with bit-exact roundtrip checks
- `make test`: runtime smoke and embedded host example coverage

## Feature Highlights

- adaptive page compression with zero-heavy fast paths
- content-aware deduplication for identical compressed pages
- tensor-aware allocation metadata for Transformer weights, KV cache, activations, and embeddings
- bit-exact FP16/BF16 split and bitplane16 tensor page codecs with default-codec fallback
- codec and tensor-role telemetry for split, bitplane, weight, and KV-cache savings
- dynamic tensor flags and range-level demotion so host runtimes can keep new KV hot and later demote old token ranges
- predictive prefetch on sequential fault patterns
- per-context quotas and allocation-failure accounting
- pool pressure, fragmentation, and reclaim telemetry
- ownership tracking for explicit managed allocations

## Repository Layout

- `libmemx3.m`: core runtime implementation
- `include/memx_runtime.h`: public explicit runtime API
- `examples/`: embeddable host examples
- `tests/`: smoke and integrity tests
- `benchmarks/`: runtime-native benchmark sources only
- `MemXApp/`: local runtime dashboard work

`build/` and `.local/` are intentionally ignored so local experiments do not pollute the repo.

## Requirements

- Apple Silicon Mac
- macOS 13+
- Xcode Command Line Tools

## Limitations

- Apple Silicon only; the compressor depends on Metal
- large managed allocations are the primary target
- incompressible data falls back to raw page storage
- the current runtime is still prototype-quality and needs more host-facing ergonomics
- only explicit host integration is supported; MemX is not a generic drop-in for arbitrary processes

## Local Model Demo

`run_qwen.py` resolves the model from:

1. `--model-path`
2. `MEMX_MODEL_PATH`
3. `.local/Qwen3.5-0.8B-hf`

That keeps large checkpoints out of the repo root while preserving local experiments.

## License

MIT
