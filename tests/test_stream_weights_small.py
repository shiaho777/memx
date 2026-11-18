#!/usr/bin/env python3
"""Small-memory gate for StreamedWeights (must stay under 50MB RSS).

Synthetic tiny model: 4 layers, hidden=64. Verifies that the layer-streaming
provider (2-slot double buffer, wrap-around across passes) returns exactly the
same bytes as full materialization, in access order, over 3 passes.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))
os.environ.setdefault("MEMX_CPU_ONLY", "1")

import torch
import memx_runtime as memx
import memx_llama as ml

PAGE = 16384
LAYERS = 4
HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
INTER = 128
VOCAB = 256


def pa(n):
    return ((n + PAGE - 1) // PAGE) * PAGE


def rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)


def main():
    if not os.environ.get("MEMX_ALLOW_TORCH_TESTS"):
        print("SKIP stream weights small: torch import alone exceeds the 50MB test cap; "
              "run with MEMX_ALLOW_TORCH_TESTS=1 to opt in (C gate: make test-stream-materialize)")
        return 0
    cfg = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "intermediate_size": INTER,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "vocab_size": VOCAB,
    }
    torch.manual_seed(7)
    sd = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),
          "model.norm.weight": torch.randn(HIDDEN, dtype=torch.bfloat16),
          "lm_head.weight": torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = torch.randn(HIDDEN, dtype=torch.bfloat16)
        sd[p + "self_attn.q_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, HEADS * HEAD_DIM, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(HIDDEN, dtype=torch.bfloat16)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16)
        sd[p + "mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16)
        sd[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16)

    rt = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = rt.create_context("stream-small")
    total = sum(pa(t.numel() * t.element_size()) for t in sd.values())
    ctx.set_quota(total + 8 * memx.MB)

    allocs, meta = {}, {}
    for name, t in sd.items():
        raw = t.contiguous().view(torch.uint8).reshape(-1)
        nb = raw.numel()
        sz = pa(nb)
        role = memx.MEMX_TENSOR_ROLE_EMBEDDING if ("embed" in name or "lm_head" in name) else memx.MEMX_TENSOR_ROLE_WEIGHT
        desc = memx.tensor_desc(role, memx.MEMX_TENSOR_DTYPE_BF16, memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
                                memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
                                shape=tuple(t.shape), stride=tuple(t.stride()))
        a = ctx.malloc_tensor(sz, desc, name=name)
        a.torch_tensor(torch.uint8, (sz,))[:nb].copy_(raw)
        allocs[name] = a
        meta[name] = (tuple(t.shape), t.dtype, nb)

    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        if rt.stats().bytes_saved > 0:
            break
        time.sleep(0.2)

    full = {}
    for name, a in allocs.items():
        shape, dtype, nb = meta[name]
        buf = torch.empty(pa(nb), dtype=torch.uint8)
        ctx.materialize_range(a, 0, pa(nb), buf.data_ptr(), buf.numel())
        full[name] = buf[:nb].view(dtype).view(shape)

    w = ml.StreamedWeights(cfg, ctx, allocs, meta, device="cpu", page=PAGE)
    assert torch.equal(w.embed, full["model.embed_tokens.weight"])
    assert torch.equal(w.norm, full["model.norm.weight"])
    assert torch.equal(w.lm_head, full["lm_head.weight"])

    checks = 0
    for pas in range(3):
        w.reset_stream()
        for i in range(LAYERS):
            d = w.L[i]
            p = f"model.layers.{i}."
            for key in ml.LAYER_KEYS:
                name = p + ml.LAYER_NAME[key]
                if not torch.equal(d[key], full[name]):
                    raise AssertionError(f"pass{pas} layer{i} key{key} mismatch")
                checks += 1
    print(f"stream small: {checks} layer-tensor checks over 3 passes OK; "
          f"mat_calls={w.mat_calls} rss={rss_mb()}MB")
    if rss_mb() > 50:
        raise AssertionError(f"test exceeded 50MB cap: {rss_mb()}MB")
    for a in allocs.values():
        a.free()
    ctx.destroy()
    rt.shutdown()
    print("OK stream weights small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())