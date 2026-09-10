# memx_core embedding contract

`core/memx_core.{h,c}` is the kernel-grade engine core: the codec layer and the page
state machine. It compiles freestanding (`-ffreestanding -fno-builtin -nostdinc`,
with `core/shims/` supplying `stdint.h`/`stddef.h`) and links against exactly the
`memcpy`/`memset`/`memcmp`/`memmove` family — the only primitives an embedding
environment must provide (`make core-audit` enforces this).

## What the core provides

- Codecs (encode, zlib-free): `tensor_fp16_split_compress`,
  `tensor_fp16_delta_split_compress`, `tensor_bitplane16_compress`,
  `tensor_sparse_byte_compress`, `rle8_encode`/`rle8_decode`.
- Universal decoder `cpu_decompress` — decodes every MemX page format,
  including the zlib-family ones **through the hook vtable** (see below).
- Page state machine: `PageMeta`, state constants, CAS/write_seq/dirty protocol
  helpers (`page_compress_meta_stable`, `page_compress_content_ok`,
  `wait_decompress_complete`) and policy predicates (`page_wants_write_protect`,
  `page_stable_need`, `fault_stream_for_role`, codec eligibility).
- Page helpers: `page_is_all_zero`, `page_bytes_equal`, `fnv1a_word`,
  `memx_interleave_lo_hi`.

## What the embedder provides

1. **Fault delivery.** The runtime uses a SIGSEGV/SIGBUS handler with
   `si_addr` → page index and `si_code == SEGV_ACCERR` as the write-fault
   signal. An embedding kernel delivers faults through its own mechanism and
   calls the same protocol: on write fault, set `m->dirty = 1`,
   `m->stable_ticks = 0`, `m->write_seq++` (in that order, with a barrier),
   grant the page RW, and CAS `COMPRESSED|COMPRESSING → HOT`.
2. **Page protection.** The commit protocol performs a
   `PROT_NONE/PROT_READ/PROT_READ|PROT_WRITE` handshake around compression
   (revoke → snapshot → re-verify → commit as PROT_NONE). The embedder maps
   these to its protection primitives; the ordering is load-bearing —
   the scratch-decode-then-late-mprotect sequence is what keeps concurrent
   writers from observing partially installed pages.
3. **Serialization.** The core takes no locks and allocates nothing. The
   embedder serializes the decompress/install critical section (the runtime
   uses a single zone mutex for this) and decides threading.
4. **Memory.** All state is caller-provided: the `PageMeta` array and page
   buffers live in embedder memory. The core performs O(1) pidx-relative
   arithmetic only; it never owns address space.
5. **zlib hooks.** `memx_core_set_hooks()` installs `inflate`/`uncompress`
   implementations before first use (the runtime registers them from a library
   constructor). With no hooks, zlib-family payloads decode to an error
   (zero-fill page) instead of crashing — return a "codec unsupported" from
   your integration layer if you see it.

## SIMD

NEON paths are guarded by `MEMX_HAS_NEON` and every one has a scalar
fallback. Define `MEMX_FORCE_SCALAR` to compile scalar-only — required where
kernel SIMD context (`kernel_fp_simd` on arm64) is not guaranteed. The hosted
runtime always uses NEON where available.

## Verification

- `make core-test` — hosted harness: codec roundtrips on five page classes,
  hook vtable semantics (NULL hooks → unsupported; registered → roundtrip),
  zero-page constant, state-machine helper behavior, RLE edges.
- `make core-audit` — freestanding compile + `nm -u` whitelist:
  undefined symbols must be a subset of {memcpy, memset, memcmp, memmove}.
  With stack protector enabled, `__stack_chk_fail`/`__stack_chk_guard`
  additionally appear; the audit compiles with `-fno-stack-protector`.
- `make test` — the full hosted runtime (which links this same archive)
  stays bitexact across every suite; the capsule roundtrip gates decode.

## Non-goals

The core deliberately excludes: zlib-bound encoders (they stay runtime-side
behind TLS streams), Metal, threads, files, fault installation, and the zone
allocator. Embedding those is a porting decision, not a core property.
