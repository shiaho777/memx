#!/usr/bin/env python3
"""Honest baseline: native transformers / mmap load, real greedy generation.

Metrics: load wall, peak phys_footprint (task_vm_info), TTFT, decode tokens/s,
generated text hash (for cross-run consistency). No MemX involved.

Usage:
  python3 benchmarks/bench_baseline_torch.py --model <hf-dir> [--device cpu|mps]
      [--mmap] [--tokens 64] [--prompt "..."] [--threads N]
"""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import sys
import time


def phys_mb():
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib")
    TASK_VM_INFO = 22

    class VMInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("region_count", ctypes.c_int),
            ("page_size", ctypes.c_int),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_peak", ctypes.c_uint64),
            ("device", ctypes.c_uint64),
            ("device_peak", ctypes.c_uint64),
            ("internal", ctypes.c_uint64),
            ("internal_peak", ctypes.c_uint64),
            ("external", ctypes.c_uint64),
            ("external_peak", ctypes.c_uint64),
            ("reusable", ctypes.c_uint64),
            ("reusable_peak", ctypes.c_uint64),
            ("purgeable_volatile_pmap", ctypes.c_uint64),
            ("purgeable_volatile_resident", ctypes.c_uint64),
            ("purgeable_volatile_virtual", ctypes.c_uint64),
            ("compressed", ctypes.c_uint64),
            ("compressed_peak", ctypes.c_uint64),
            ("compressed_lifetime", ctypes.c_uint64),
            ("phys_footprint", ctypes.c_uint64),
        ]

    info = VMInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
    libc.mach_task_self.restype = ctypes.c_uint
    libc.task_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    libc.task_info.restype = ctypes.c_int
    kr = libc.task_info(libc.mach_task_self(), TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count))
    if kr != 0:
        return None
    return int(info.phys_footprint // (1024 * 1024))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    result = {
        "model": args.model,
        "device": args.device,
        "mmap": args.mmap,
        "dtype": args.dtype,
        "threads": args.threads,
        "tokens_requested": args.tokens,
        "phys_before_mb": phys_mb(),
    }

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.device == "mps":
        model = model.to("mps")
    model.eval()
    t_load = time.perf_counter() - t0
    result["load_wall_s"] = round(t_load, 3)
    result["phys_after_load_mb"] = phys_mb()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    if args.device == "mps":
        ids = ids.to("mps")
    prompt_len = int(ids.shape[1])

    peak = [phys_mb() or 0]
    n_gen = [0]
    t_first = [None]
    t_start = time.perf_counter()

    from transformers import LogitsProcessor

    class Meter(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if t_first[0] is None:
                t_first[0] = time.perf_counter()
            n_gen[0] += 1
            p = phys_mb()
            if p and p > peak[0]:
                peak[0] = p
            return scores

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.tokens,
            min_new_tokens=args.tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id or tok.pad_token_id or 0,
            logits_processor=[Meter()],
        )
    t_total = time.perf_counter() - t_start
    gen_ids = out[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    uniq = len(set(gen_ids.tolist()))
    degenerate = (uniq <= 1) or (len(text.strip()) == 0)
    if degenerate:
        result["WARNING_degenerate_generation"] = (
            f"only {uniq} unique token id(s), empty/constant text; "
            "throughput number is NOT a valid generation measurement"
        )

    ttft = (t_first[0] - t_start) if t_first[0] else None
    decode_wall = t_total - ttft if ttft is not None else None
    result.update({
        "prompt_tokens": prompt_len,
        "generated_tokens": int(gen_ids.shape[0]),
        "ttft_s": round(ttft, 4) if ttft else None,
        "total_wall_s": round(t_total, 3),
        "decode_tokens_per_s": round((n_gen[0] - 1) / decode_wall, 2) if decode_wall and decode_wall > 0 and n_gen[0] > 1 else None,
        "peak_phys_mb": peak[0],
        "phys_after_generate_mb": phys_mb(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text_preview": text[:120],
        "torch_version": torch.__version__,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()