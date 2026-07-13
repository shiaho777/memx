#!/usr/bin/env python3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

import memx_runtime as memx


PAGE = 16384
LAYERS = 4
HIDDEN = 128
HEADS = 4
HEAD_DIM = 32
TOKENS = 512
ACTIVE_TOKENS = 16
WEIGHT_SHAPE = (HIDDEN, HIDDEN)
KV_SHAPE = (HEADS, TOKENS, HEAD_DIM)


def wait_until(label, predicate, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(label)


def storage_size(shape, dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    elements = 1
    for dim in shape:
        elements *= dim
    raw = elements * element_size
    return ((raw + PAGE - 1) // PAGE) * PAGE


def make_weight(layer):
    rows = torch.arange(WEIGHT_SHAPE[0], dtype=torch.int32).reshape(-1, 1)
    cols = torch.arange(WEIGHT_SHAPE[1], dtype=torch.int32).reshape(1, -1)
    lo = ((rows * 5 + cols * 3 + layer * 17) & 255).to(torch.int32)
    hi = torch.full_like(lo, 0x3C)
    hi = torch.where(((rows + cols + layer) % 31) == 0, torch.full_like(hi, 0x3D), hi)
    words = ((hi << 8) | lo).to(torch.uint16)
    return words.view(torch.float16).contiguous()


def make_kv(layer):
    heads = torch.arange(HEADS, dtype=torch.int32).reshape(HEADS, 1, 1)
    tokens = torch.arange(TOKENS, dtype=torch.int32).reshape(1, TOKENS, 1)
    dims = torch.arange(HEAD_DIM, dtype=torch.int32).reshape(1, 1, HEAD_DIM)
    lo = ((tokens + dims * 7 + heads * 19) & 255).to(torch.int32)
    lo = torch.where(tokens < 256, lo, ((lo + layer * 13) & 255).to(torch.int32))
    hi = torch.where(tokens < 256, torch.full_like(lo, 0x38), torch.full_like(lo, 0x35 + layer))
    words = ((hi << 8) | lo).to(torch.uint16)
    return words.view(torch.float16).contiguous()


def make_input():
    tokens = torch.arange(ACTIVE_TOKENS, dtype=torch.int32).reshape(-1, 1)
    dims = torch.arange(HIDDEN, dtype=torch.int32).reshape(1, -1)
    lo = ((tokens * 11 + dims * 5) & 255).to(torch.int32)
    hi = torch.full_like(lo, 0x3B)
    return ((hi << 8) | lo).to(torch.uint16).view(torch.float16).contiguous()


def copy_tensor_to_allocation(tensor, allocation):
    src = tensor.view(torch.uint8).reshape(-1)
    dst = allocation.torch_tensor(torch.uint8, (allocation.size,))
    dst[: src.numel()].copy_(src)
    if src.numel() < allocation.size:
        dst[src.numel():].zero_()


def run_transformer(weights, kvs, x):
    for layer, (weight, kv) in enumerate(zip(weights, kvs)):
        x = torch.mm(x, weight.t())
        query = x[:, :HEADS * HEAD_DIM].reshape(ACTIVE_TOKENS, HEADS, HEAD_DIM).transpose(0, 1)
        keys = kv[:, :ACTIVE_TOKENS, :]
        scores = (query * keys).sum(dim=2)
        score_words = scores.contiguous().view(torch.uint16)
        kv_words = kv[:, layer:layer + ACTIVE_TOKENS, :ACTIVE_TOKENS].contiguous().view(torch.uint16)
        score_mix = score_words.unsqueeze(2).expand(-1, -1, ACTIVE_TOKENS)
        combined = torch.bitwise_xor(score_mix, kv_words)
        mixed = combined.reshape(-1).to(torch.int64)
        seed = int(mixed.sum().item()) & 0xFFFF
        update = torch.full((ACTIVE_TOKENS, HIDDEN), seed, dtype=torch.int64)
        update = torch.bitwise_xor(update, x.contiguous().view(torch.uint16).to(torch.int64))
        x = (update & 0xFFFF).to(torch.uint16).view(torch.float16)
    return x.contiguous()


def main():
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("python-torch-transformer")
    weight_allocs = []
    kv_allocs = []
    managed_weights = []
    managed_kvs = []
    baseline_weights = []
    baseline_kvs = []
    try:
        ctx.set_quota(64 * memx.MB)
        weight_size = storage_size(WEIGHT_SHAPE, torch.float16)
        kv_size = storage_size(KV_SHAPE, torch.float16)
        baseline_stats = runtime.stats()
        for layer in range(LAYERS):
            weight_desc = memx.tensor_desc(
                memx.MEMX_TENSOR_ROLE_WEIGHT,
                memx.MEMX_TENSOR_DTYPE_FP16,
                memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
                memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
                shape=WEIGHT_SHAPE,
                stride=(WEIGHT_SHAPE[1], 1),
                layer_index=layer,
            )
            kv_desc = memx.tensor_desc(
                memx.MEMX_TENSOR_ROLE_KV_CACHE,
                memx.MEMX_TENSOR_DTYPE_FP16,
                memx.MEMX_TENSOR_LAYOUT_BLOCKED,
                memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
                shape=KV_SHAPE,
                stride=(TOKENS * HEAD_DIM, HEAD_DIM, 1),
                layer_index=layer,
            )
            weight_alloc = ctx.malloc_tensor(weight_size, weight_desc, name=f"torch-layer{layer}-weight")
            kv_alloc = ctx.malloc_tensor(kv_size, kv_desc, name=f"torch-layer{layer}-kv")
            baseline_weight = make_weight(layer)
            baseline_kv = make_kv(layer)
            copy_tensor_to_allocation(baseline_weight, weight_alloc)
            copy_tensor_to_allocation(baseline_kv, kv_alloc)
            weight_allocs.append(weight_alloc)
            kv_allocs.append(kv_alloc)
            managed_weights.append(weight_alloc.torch_tensor(torch.float16, WEIGHT_SHAPE))
            managed_kvs.append(kv_alloc.torch_tensor(torch.float16, KV_SHAPE))
            baseline_weights.append(baseline_weight.clone())
            baseline_kvs.append(baseline_kv.clone())
        wait_until(
            "PyTorch transformer tensors did not compress",
            lambda: sum(a.info().compressed_pages for a in weight_allocs + kv_allocs) >= LAYERS * 2,
        )
        compressed_stats = runtime.stats()
        if compressed_stats.weight_compressed_pages == 0 or compressed_stats.kv_cache_compressed_pages == 0:
            raise AssertionError("PyTorch transformer roles did not compress")
        if compressed_stats.tensor_delta_split_pages == 0:
            raise AssertionError("PyTorch transformer FP16 tensors did not use tensor codec")
        if compressed_stats.dedup_hits <= baseline_stats.dedup_hits:
            raise AssertionError("PyTorch transformer repeated KV pages did not deduplicate")
        x = make_input()
        baseline_out = run_transformer(baseline_weights, baseline_kvs, x.clone())
        faults_before = runtime.stats().faults
        for layer, allocation in enumerate(weight_allocs):
            ctx.update_weight_window(
                allocation,
                memx.weight_window(
                    managed=(0, allocation.size),
                    hot=(0, PAGE),
                    prefetch=(PAGE, PAGE),
                ),
            )
            ctx.mark_access_range(allocation, PAGE, PAGE)
        for layer, allocation in enumerate(kv_allocs):
            ctx.update_kv_cache_window(
                allocation,
                memx.kv_cache_window(
                    managed=(0, allocation.size),
                    hot=(0, PAGE),
                    prefetch=(PAGE, PAGE),
                ),
            )
            ctx.mark_access_range(allocation, PAGE, PAGE)
        managed_out = run_transformer(managed_weights, managed_kvs, x.clone())
        stats_after = runtime.stats()
        if not torch.equal(managed_out.view(torch.uint16), baseline_out.view(torch.uint16)):
            raise AssertionError("PyTorch Transformer output changed under MemX")
        if stats_after.faults <= faults_before:
            raise AssertionError("PyTorch Transformer did not fault compressed MemX pages")
        if stats_after.prefetch_count <= compressed_stats.prefetch_count:
            raise AssertionError("PyTorch Transformer did not prefetch MemX pages")
        if stats_after.prefetch_hits <= compressed_stats.prefetch_hits:
            raise AssertionError("PyTorch Transformer did not record prefetch hits")
        cold_flags = memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY
        for allocation in weight_allocs + kv_allocs:
            ctx.update_tensor_flags(allocation, cold_flags)
        wait_until(
            "PyTorch transformer tensors did not return cold",
            lambda: sum(a.info().compressed_pages for a in weight_allocs + kv_allocs) >= LAYERS * 3,
        )
        final_stats = runtime.stats()
        ledger = memx.capacity_ledger(weight_allocs + kv_allocs, final_stats)
        if ledger["by_role"].get("weight", {}).get("logical_bytes", 0) != LAYERS * weight_size:
            raise AssertionError("PyTorch ledger weight bytes mismatch")
        if ledger["by_role"].get("kv_cache", {}).get("logical_bytes", 0) != LAYERS * kv_size:
            raise AssertionError("PyTorch ledger KV bytes mismatch")
        digest = managed_out.contiguous().view(torch.uint16).to(torch.int64).sum().item() & 0xFFFFFFFFFFFFFFFF
        print(
            "python torch transformer: "
            f"digest=0x{digest:016x} "
            f"faults={faults_before}->{stats_after.faults} "
            f"prefetch={compressed_stats.prefetch_count}->{stats_after.prefetch_count} "
            f"hits={compressed_stats.prefetch_hits}->{stats_after.prefetch_hits} "
            f"weight_pages={compressed_stats.weight_compressed_pages}->{final_stats.weight_compressed_pages} "
            f"kv_pages={compressed_stats.kv_cache_compressed_pages}->{final_stats.kv_cache_compressed_pages} "
            f"dedup={baseline_stats.dedup_hits}->{compressed_stats.dedup_hits} "
            f"ratio={ledger['effective_ratio_physical_est']:.2f}x"
        )
    finally:
        for allocation in reversed(kv_allocs):
            allocation.free()
        for allocation in reversed(weight_allocs):
            allocation.free()
        ctx.destroy()
        runtime.shutdown()


if __name__ == "__main__":
    main()
