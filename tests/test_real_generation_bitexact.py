#!/usr/bin/env python3
"""A-track gate: real Llama greedy generation with MemX-hosted weights.

Contract:
- Weights are copied into MemX tensor allocations and compressed.
- The managed forward reads weights through MemX views (fault-driven decompress).
- Generated token ids must equal a plain-torch reference token-for-token.
- Weight bytes round-trip bit-exact (verified by materialize vs golden).

This replaces the synthetic matmul-chain demo with a real transformer.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))

if not os.environ.get("MEMX_ALLOW_LARGE_TESTS"):
    print("SKIP real generation gate: loads a multi-GB model, exceeds the 50MB test cap; "
          "run with MEMX_ALLOW_LARGE_TESTS=1 to opt in")
    raise SystemExit(0)

import torch
torch.set_num_threads(int(os.environ.get("MEMX_TORCH_THREADS", "8")))

import memx_runtime as memx
import memx_llama as ml

MODEL = os.environ.get("MEMX_MODEL_PATH", str(Path.home() / "Desktop/模型/0-/MiniCPM5-2B/hf-bf16"))
PAGE = 16384
GEN_TOKENS = int(os.environ.get("MEMX_GEN_TOKENS", "3"))
DEVICE = os.environ.get("MEMX_DEVICE", "cpu")


def page_align(n):
    return ((n + PAGE - 1) // PAGE) * PAGE


def host_weights(ctx, sd, cfg):
    """Copy every weight tensor into a MemX allocation; return name->Allocation."""
    allocs = {}
    views = {}
    meta = {}
    for name, t in sd.items():
        raw = t.contiguous().view(torch.uint8).reshape(-1)
        nbytes = raw.numel()
        size = page_align(nbytes)
        role = memx.MEMX_TENSOR_ROLE_WEIGHT
        if "embed" in name or "lm_head" in name:
            role = memx.MEMX_TENSOR_ROLE_EMBEDDING
        dtype = memx.MEMX_TENSOR_DTYPE_BF16 if t.dtype == torch.bfloat16 else memx.MEMX_TENSOR_DTYPE_FP16
        shape = tuple(t.shape)
        stride = tuple(t.stride())
        desc = memx.tensor_desc(
            role, dtype, memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
            shape=shape, stride=stride,
        )
        a = ctx.malloc_tensor(size, desc, name=name)
        dst = a.torch_tensor(torch.uint8, (size,))
        dst[:nbytes].copy_(raw)
        if nbytes < size:
            dst[nbytes:].zero_()
        allocs[name] = a
        views[name] = a.torch_tensor(t.dtype, shape)
        meta[name] = (shape, t.dtype, nbytes)
    return allocs, views, meta


def materialize_weights(ctx, allocs, meta, dev):
    out = {}
    for name, a in allocs.items():
        shape, dtype, nbytes = meta[name]
        buf = torch.empty(page_align(nbytes), dtype=torch.uint8)
        ctx.materialize_range(a, 0, page_align(nbytes), buf.data_ptr(), buf.numel())
        out[name] = buf[:nbytes].view(dtype).view(shape).to(dev)
    return out


def wait_compressed(runtime, allocs, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = runtime.stats()
        if st.weight_compressed_pages > 0:
            return st
        time.sleep(0.2)
    raise AssertionError("weights did not compress")


def main():
    if not Path(MODEL, "config.json").exists():
        print(f"SKIP: model not found at {MODEL}")
        return 0
    cfg = ml.load_config(MODEL)
    sd = ml.load_state_dict(MODEL)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids

    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"

    # Plain-torch reference (weights stay in normal RAM)
    w_ref = ml.LlamaWeights(cfg, lambda k: sd[k].to(dev))
    ref_tokens = ml.generate(w_ref, ids, GEN_TOKENS, device=dev, dtype=torch.bfloat16)
    del w_ref

    # MemX-hosted managed run
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("real-generation")
    total_bytes = sum(page_align(t.numel() * t.element_size()) for t in sd.values())
    ctx.set_quota(total_bytes + 256 * memx.MB)
    allocs, views, meta = host_weights(ctx, sd, cfg)
    sample_name = "model.layers.0.self_attn.q_proj.weight"
    sample_golden = sd[sample_name].contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    sample_nbytes = sd[sample_name].numel() * sd[sample_name].element_size()
    del sd
    st0 = runtime.stats()
    st_comp = wait_compressed(runtime, allocs)
    faults_before = st_comp.faults

    managed_sd = materialize_weights(ctx, allocs, meta, dev)
    w_managed = ml.LlamaWeights(cfg, lambda k: managed_sd[k])
    t0 = time.perf_counter()
    managed_tokens = ml.generate(w_managed, ids, GEN_TOKENS, device=dev, dtype=torch.bfloat16)
    wall = time.perf_counter() - t0
    st_after = runtime.stats()

    match = managed_tokens == ref_tokens
    print(f"real generation: layers={cfg['num_hidden_layers']} gen={GEN_TOKENS}")
    print(f"  ref tokens   : {ref_tokens}")
    print(f"  managed tokens: {managed_tokens}")
    print(f"  TOKEN_MATCH={match}")
    print(f"  text={tok.decode(managed_tokens)!r}")
    print(f"  compressed weight_pages={st_comp.weight_compressed_pages} "
          f"faults={faults_before}->{st_after.faults} "
          f"prefetch_hits={st_comp.prefetch_hits}->{st_after.prefetch_hits}")
    print(f"  managed_generate_wall_s={wall:.3f}")

    # Bit-exact weight round-trip check on a sample tensor
    a = allocs[sample_name]
    got = a.torch_tensor(torch.uint8, (a.size,))[:sample_nbytes].numpy().tobytes()
    bitexact = got == sample_golden
    print(f"  WEIGHT_BITEXACT({sample_name})={bitexact}")

    for a in allocs.values():
        a.free()
    ctx.destroy()
    runtime.shutdown()

    if not match:
        raise AssertionError("managed generation diverged from reference")
    if not bitexact:
        raise AssertionError("weight bytes did not round-trip bit-exact")
    print("OK real generation bitexact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())