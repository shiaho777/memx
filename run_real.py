#!/usr/bin/env python3
"""Real FullHost demo: genuine Llama generation over MemX-hosted weights.

Replaces the synthetic matmul-chain demo with a real transformer forward and
greedy token loop. Reports honest metrics: decode tokens/s, peak phys_footprint,
compression ratio, and a bit-exact weight round-trip gate.

Usage:
  MEMX_MODEL_PATH=/path/to/hf-dir python3 run_real.py [--tokens N] [--device cpu|mps]
"""
import argparse
import ctypes
import ctypes.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))

import torch
torch.set_num_threads(int(os.environ.get("MEMX_TORCH_THREADS", "8")))

import memx_runtime as memx
import memx_llama as ml

PAGE = 16384


def phys_mb():
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib")
    TASK_VM_INFO = 22

    class VMInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64), ("region_count", ctypes.c_int),
            ("page_size", ctypes.c_int), ("resident_size", ctypes.c_uint64),
            ("resident_size_peak", ctypes.c_uint64), ("device", ctypes.c_uint64),
            ("device_peak", ctypes.c_uint64), ("internal", ctypes.c_uint64),
            ("internal_peak", ctypes.c_uint64), ("external", ctypes.c_uint64),
            ("external_peak", ctypes.c_uint64), ("reusable", ctypes.c_uint64),
            ("reusable_peak", ctypes.c_uint64), ("purgeable_volatile_pmap", ctypes.c_uint64),
            ("purgeable_volatile_resident", ctypes.c_uint64), ("purgeable_volatile_virtual", ctypes.c_uint64),
            ("compressed", ctypes.c_uint64), ("compressed_peak", ctypes.c_uint64),
            ("compressed_lifetime", ctypes.c_uint64), ("phys_footprint", ctypes.c_uint64),
        ]

    info = VMInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
    libc.mach_task_self.restype = ctypes.c_uint
    libc.task_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    libc.task_info.restype = ctypes.c_int
    if libc.task_info(libc.mach_task_self(), TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)) != 0:
        return None
    return int(info.phys_footprint // (1024 * 1024))


def page_align(n):
    return ((n + PAGE - 1) // PAGE) * PAGE


def resolve_model(cli):
    cands = []
    if cli:
        cands.append(Path(cli).expanduser())
    env = os.environ.get("MEMX_MODEL_PATH")
    if env:
        cands.append(Path(env).expanduser())
    cands.append(Path.home() / "Desktop/模型/0-/MiniCPM5-2B/hf-bf16")
    for c in cands:
        if (c / "config.json").exists():
            return c
    raise SystemExit("model not found; set MEMX_MODEL_PATH or --model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--stream", action="store_true",
                    help="layer-streaming working set (2 layers on device) instead of full materialize")
    ap.add_argument("--check", action="store_true",
                    help="also run plain-torch reference generation and require token match")
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    cfg = ml.load_config(model_path)
    sd = ml.load_state_dict(model_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok(args.prompt, return_tensors="pt").input_ids

    logical_bytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"[real] model={model_path.name} layers={cfg['num_hidden_layers']} "
          f"logical_weights={logical_bytes/1024**3:.2f}GB device={dev}", flush=True)
    print(f"[real] phys_before={phys_mb()}MB", flush=True)

    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("real-fullhost")
    total = sum(page_align(t.numel() * t.element_size()) for t in sd.values())
    ctx.set_quota(total + 256 * memx.MB)

    allocs, meta = {}, {}
    t_host = time.perf_counter()
    for name, t in sd.items():
        raw = t.contiguous().view(torch.uint8).reshape(-1)
        nb = raw.numel()
        sz = page_align(nb)
        role = memx.MEMX_TENSOR_ROLE_EMBEDDING if ("embed" in name or "lm_head" in name) else memx.MEMX_TENSOR_ROLE_WEIGHT
        dt = memx.MEMX_TENSOR_DTYPE_BF16 if t.dtype == torch.bfloat16 else memx.MEMX_TENSOR_DTYPE_FP16
        desc = memx.tensor_desc(role, dt, memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
                                memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
                                shape=tuple(t.shape), stride=tuple(t.stride()))
        a = ctx.malloc_tensor(sz, desc, name=name)
        a.torch_tensor(torch.uint8, (sz,))[:nb].copy_(raw)
        allocs[name] = a
        meta[name] = (tuple(t.shape), t.dtype, nb)
    host_wall = time.perf_counter() - t_host

    sample = "model.layers.0.self_attn.q_proj.weight"
    golden = sd[sample].contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    ref_tokens = None
    if args.check:
        w_ref = ml.LlamaWeights(cfg, lambda k: sd[k].to(dev))
        ref_tokens = ml.generate(w_ref, ids, args.tokens, device=dev, dtype=torch.bfloat16)
        del w_ref
        print(f"[real] reference tokens: {ref_tokens}", flush=True)
    del sd

    settle_s = float(os.environ.get("MEMX_SETTLE_S", "30"))
    deadline = time.time() + settle_s
    last = -1
    stable = 0
    while time.time() < deadline:
        st = runtime.stats()
        cur = st.bytes_saved
        if cur > 0 and cur == last:
            stable += 1
            if stable >= 6:
                break
        else:
            stable = 0
        last = cur
        time.sleep(0.5)
    st_comp = runtime.stats()
    print(f"[real] hosted {len(allocs)} tensors in {host_wall:.2f}s; "
          f"compressed_pages={st_comp.weight_compressed_pages} saved={st_comp.bytes_saved/1024**2:.0f}MB "
          f"phys_after_compress={phys_mb()}MB", flush=True)

    managed = {}
    t_mat = time.perf_counter()
    mat_wall = 0.0
    stream_stats = None
    if args.stream:
        print(f"[real] STREAM mode: 2-layer working set on {dev}", flush=True)
        w = ml.StreamedWeights(cfg, ctx, allocs, meta, device=dev)
        mat_wall = time.perf_counter() - t_mat
        print(f"[real] stream setup {mat_wall:.2f}s; generating {args.tokens} tokens...", flush=True)
        peak = phys_mb() or 0
        stop = [False]

        def sampler():
            while not stop[0]:
                p = phys_mb()
                if p and p > peak[0]:
                    peak[0] = p
                time.sleep(0.05)

        import threading
        peak = [peak]
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t0 = time.perf_counter()
        toks = ml.generate(w, ids, args.tokens, device=dev, dtype=torch.bfloat16)
        gen_wall = time.perf_counter() - t0
        stop[0] = True
        th.join(timeout=1)
        peak = peak[0]
        stream_stats = (w.mat_calls, w.mat_bytes)
    else:
        print(f"[real] materializing {len(allocs)} tensors to {dev}...", flush=True)
        for name, a in allocs.items():
            shape, dtype, nb = meta[name]
            buf = torch.empty(page_align(nb), dtype=torch.uint8)
            ctx.materialize_range(a, 0, page_align(nb), buf.data_ptr(), buf.numel())
            managed[name] = buf[:nb].view(dtype).view(shape).to(dev)
        mat_wall = time.perf_counter() - t_mat
        print(f"[real] materialized in {mat_wall:.2f}s; generating {args.tokens} tokens...", flush=True)
        w = ml.LlamaWeights(cfg, lambda k: managed[k])
        t0 = time.perf_counter()
        phys_pre_gen = phys_mb() or 0
        toks = ml.generate(w, ids, args.tokens, device=dev, dtype=torch.bfloat16)
        gen_wall = time.perf_counter() - t0
        peak = max(phys_pre_gen, phys_mb() or 0)
    text = tok.decode(toks)

    a = allocs[sample]
    got = a.torch_tensor(torch.uint8, (a.size,))[:meta[sample][2]].numpy().tobytes()
    bitexact = got == golden

    st_final = runtime.stats()
    decode_tps = (args.tokens - 1) / gen_wall if gen_wall > 0 else 0
    print("\n" + "=" * 60, flush=True)
    print("  REAL FULLHOST SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  prompt          : {args.prompt!r}", flush=True)
    print(f"  generated       : {text!r}", flush=True)
    print(f"  tokens          : {args.tokens}", flush=True)
    print(f"  generate wall   : {gen_wall:.3f}s  (~{decode_tps:.2f} tok/s end-to-end)", flush=True)
    print(f"  materialize wall: {mat_wall:.2f}s", flush=True)
    print(f"  logical weights : {logical_bytes/1024**3:.2f} GB", flush=True)
    print(f"  compressed saved: {st_comp.bytes_saved/1024**2:.0f} MB", flush=True)
    print(f"  peak phys       : {peak} MB", flush=True)
    print(f"  final phys      : {phys_mb()} MB", flush=True)
    print(f"  faults          : {st_comp.faults}->{st_final.faults}", flush=True)
    print(f"  prefetch hits   : {st_comp.prefetch_hits}->{st_final.prefetch_hits}", flush=True)
    print(f"  WEIGHT_BITEXACT : {bitexact}", flush=True)
    if stream_stats:
        calls, mbytes = stream_stats
        print(f"  STREAM mat calls: {calls}  bytes moved: {mbytes/1024**3:.2f} GB "
              f"(vs full {logical_bytes/1024**3:.2f} GB resident)", flush=True)
    token_match = None
    if ref_tokens is not None:
        token_match = toks == ref_tokens
        print(f"  TOKEN_MATCH     : {token_match}  managed={toks}", flush=True)
    print("=" * 60, flush=True)

    for a in allocs.values():
        a.free()
    ctx.destroy()
    runtime.shutdown()
    if not bitexact:
        raise SystemExit("weight round-trip not bit-exact")
    if token_match is False:
        raise SystemExit("streamed generation diverged from reference")
    print("[real] OK", flush=True)


if __name__ == "__main__":
    main()