#!/usr/bin/env python3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import memx_runtime as memx


PAGE = 16384
HALFS_PER_PAGE = PAGE // 2
MASK = (1 << 64) - 1
LAYERS = 6
WEIGHT_PAGES = 8
KV_PAGES = 12
HIDDEN = 128
HEADS = 4
HEAD_DIM = 64
STEPS = 10


def write_u16(buf, index, value):
    offset = index * 2
    buf[offset] = value & 0xFF
    buf[offset + 1] = (value >> 8) & 0xFF


def read_u16(buf, index):
    offset = index * 2
    return buf[offset] | (buf[offset + 1] << 8)


def mix(acc, value):
    acc ^= value & MASK
    acc = (acc * 0x9E3779B185EBCA87) & MASK
    acc ^= acc >> 33
    return acc & MASK


def wait_until(label, predicate, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(label)


def weight_value(layer, index):
    lo = (index + layer * 19 + index // 1024) & 0xFF
    hi = 0x3C if (index & 1023) < 1000 else 0x3D
    if ((index + layer) & 4095) == 0:
        hi ^= 0x80
    return (hi << 8) | lo


def kv_value(layer, index):
    page = index // HALFS_PER_PAGE
    in_page = index & (HALFS_PER_PAGE - 1)
    if page < 3:
        lo = in_page & 0xFF
        hi = 0x38
    else:
        lo = (in_page + layer * 7 + page * 13) & 0xFF
        hi = 0x30 + ((page + layer) & 7)
    return (hi << 8) | lo


def kv_update_value(layer, step, lane):
    lo = (step * 17 + layer * 11 + lane * 3) & 0xFF
    hi = 0x34 + ((step + layer + lane // 32) & 3)
    return (hi << 8) | lo


def fill_weight(buf, layer):
    for i in range(len(buf) // 2):
        write_u16(buf, i, weight_value(layer, i))


def fill_kv(buf, layer):
    for i in range(len(buf) // 2):
        write_u16(buf, i, kv_value(layer, i))


def allocation_pages(allocations):
    return sum(allocation.info().compressed_pages for allocation in allocations)


def run_lifecycle(ctx, runtime, weights, kvs, base_weights, base_kvs):
    base_acc = 0x123456789ABCDEF0
    managed_acc = base_acc
    stats_before = runtime.stats()
    first_window_checked = False
    for step in range(STEPS):
        for layer in range(LAYERS):
            weight = weights[layer]
            kv = kvs[layer]
            weight_buf = weight.buffer()
            kv_buf = kv.buffer()
            weight_page = (step + layer * 2) % WEIGHT_PAGES
            weight_prefetch = (weight_page + 1) % WEIGHT_PAGES
            weight_cold = (weight_page + 4) % WEIGHT_PAGES
            kv_hot = (step + layer) % KV_PAGES
            kv_prefetch = (kv_hot + 1) % KV_PAGES
            kv_cold = (kv_hot + 6) % KV_PAGES
            ctx.update_weight_window(
                weight,
                memx.weight_window(
                    managed=(0, weight.size),
                    hot=(weight_page * PAGE, PAGE),
                    prefetch=(weight_prefetch * PAGE, PAGE),
                ),
            )
            ctx.update_kv_cache_window(
                kv,
                memx.kv_cache_window(
                    managed=(0, kv.size),
                    hot=(kv_hot * PAGE, PAGE),
                    prefetch=(kv_prefetch * PAGE, PAGE),
                ),
            )
            if not first_window_checked:
                hot_info = weight.info(weight_page * PAGE, PAGE)
                prefetch_info = weight.info(weight_prefetch * PAGE, PAGE)
                cold_info = weight.info(weight_cold * PAGE, PAGE)
                if hot_info.compressed_pages != 0 or (hot_info.tensor_flags & memx.MEMX_TENSOR_FLAG_HOT) == 0:
                    raise AssertionError("weight hot window did not become resident and hot")
                if prefetch_info.compressed_pages != 0:
                    raise AssertionError("weight prefetch window stayed compressed")
                if cold_info.compressed_pages == 0:
                    raise AssertionError("weight cold window did not stay compressed")
                first_window_checked = True
            ctx.mark_access_range(weight, weight_prefetch * PAGE, PAGE)
            ctx.mark_access_range(kv, kv_prefetch * PAGE, PAGE)
            write_start = kv_hot * HALFS_PER_PAGE + ((step * 257 + layer * 53) % (HALFS_PER_PAGE - HIDDEN))
            for lane in range(HIDDEN):
                value = kv_update_value(layer, step, lane)
                write_u16(kv_buf, write_start + lane, value)
                write_u16(base_kvs[layer], write_start + lane, value)
            for lane in range(32):
                wi = weight_page * HALFS_PER_PAGE + ((lane * 193 + step * 17 + layer * 29) & (HALFS_PER_PAGE - 1))
                wp = weight_prefetch * HALFS_PER_PAGE + ((lane * 67 + step * 23 + layer * 31) & (HALFS_PER_PAGE - 1))
                wc = weight_cold * HALFS_PER_PAGE + ((lane * 89 + step * 41 + layer * 43) & (HALFS_PER_PAGE - 1))
                kh = kv_hot * HALFS_PER_PAGE + ((lane * 71 + step * 37 + layer * 13) & (HALFS_PER_PAGE - 1))
                kp = kv_prefetch * HALFS_PER_PAGE + ((lane * 97 + step * 19 + layer * 11) & (HALFS_PER_PAGE - 1))
                kc = kv_cold * HALFS_PER_PAGE + ((lane * 101 + step * 29 + layer * 7) & (HALFS_PER_PAGE - 1))
                base_term = read_u16(base_weights[layer], wi)
                base_term ^= (read_u16(base_weights[layer], wp) << 1) & 0xFFFF
                base_term ^= (read_u16(base_weights[layer], wc) << 2) & 0xFFFF
                base_term = (base_term + read_u16(base_kvs[layer], kh) * 3) & 0xFFFFFFFF
                base_term ^= (read_u16(base_kvs[layer], kp) * 5) & 0xFFFFFFFF
                base_term ^= (read_u16(base_kvs[layer], kc) * 7) & 0xFFFFFFFF
                managed_term = read_u16(weight_buf, wi)
                managed_term ^= (read_u16(weight_buf, wp) << 1) & 0xFFFF
                managed_term ^= (read_u16(weight_buf, wc) << 2) & 0xFFFF
                managed_term = (managed_term + read_u16(kv_buf, kh) * 3) & 0xFFFFFFFF
                managed_term ^= (read_u16(kv_buf, kp) * 5) & 0xFFFFFFFF
                managed_term ^= (read_u16(kv_buf, kc) * 7) & 0xFFFFFFFF
                base_acc = mix(base_acc, base_term + step + layer + lane)
                managed_acc = mix(managed_acc, managed_term + step + layer + lane)
    mid_stats = runtime.stats()
    if managed_acc != base_acc:
        raise AssertionError(f"Transformer lifecycle digest changed managed=0x{managed_acc:016x} baseline=0x{base_acc:016x}")
    if mid_stats.faults <= stats_before.faults:
        raise AssertionError("Transformer lifecycle did not exercise on-demand decompression")
    if mid_stats.prefetch_count <= stats_before.prefetch_count:
        raise AssertionError("Transformer lifecycle did not issue prefetches")
    if mid_stats.prefetch_hits <= stats_before.prefetch_hits:
        raise AssertionError("Transformer lifecycle did not record prefetch hits")
    if mid_stats.hot_resident_pages == 0 or mid_stats.no_compress_resident_pages == 0:
        raise AssertionError("Transformer lifecycle did not keep hot paths resident")
    return managed_acc, stats_before, mid_stats


def main():
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("python-transformer-lifecycle")
    weights = []
    kvs = []
    base_weights = []
    base_kvs = []
    try:
        ctx.set_quota(128 * memx.MB)
        weight_size = WEIGHT_PAGES * PAGE
        kv_size = KV_PAGES * PAGE
        kv_tokens = (kv_size // 2) // (HEADS * HEAD_DIM)
        baseline_stats = runtime.stats()
        for layer in range(LAYERS):
            weight_desc = memx.tensor_desc(
                memx.MEMX_TENSOR_ROLE_WEIGHT,
                memx.MEMX_TENSOR_DTYPE_FP16,
                memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
                memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
                shape=(weight_size // 2 // HIDDEN, HIDDEN),
                stride=(HIDDEN, 1),
                layer_index=layer,
            )
            kv_desc = memx.tensor_desc(
                memx.MEMX_TENSOR_ROLE_KV_CACHE,
                memx.MEMX_TENSOR_DTYPE_FP16,
                memx.MEMX_TENSOR_LAYOUT_BLOCKED,
                memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
                shape=(1, HEADS, kv_tokens, HEAD_DIM),
                stride=(HEADS * kv_tokens * HEAD_DIM, kv_tokens * HEAD_DIM, HEAD_DIM, 1),
                layer_index=layer,
            )
            weight = ctx.malloc_tensor(weight_size, weight_desc, name=f"layer{layer}-weight")
            kv = ctx.malloc_tensor(kv_size, kv_desc, name=f"layer{layer}-kv")
            base_weight = bytearray(weight_size)
            base_kv = bytearray(kv_size)
            fill_weight(base_weight, layer)
            fill_kv(base_kv, layer)
            weight.buffer()[:] = base_weight
            kv.buffer()[:] = base_kv
            weights.append(weight)
            kvs.append(kv)
            base_weights.append(base_weight)
            base_kvs.append(base_kv)
        wait_until(
            "Transformer tensors did not compress before lifecycle run",
            lambda: allocation_pages(weights) > LAYERS * 2 and allocation_pages(kvs) > LAYERS * 2,
        )
        compressed_stats = runtime.stats()
        if compressed_stats.weight_compressed_pages == 0 or compressed_stats.kv_cache_compressed_pages == 0:
            raise AssertionError("Transformer roles did not report compressed weight and KV pages")
        if compressed_stats.tensor_delta_split_pages == 0:
            raise AssertionError("Transformer FP16 tensors did not use delta-split telemetry")
        if compressed_stats.dedup_hits <= baseline_stats.dedup_hits:
            raise AssertionError("Transformer repeated KV prefix did not deduplicate")
        digest, stats_before, mid_stats = run_lifecycle(ctx, runtime, weights, kvs, base_weights, base_kvs)
        cold_flags = memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY
        for allocation in weights + kvs:
            ctx.update_tensor_flags(allocation, cold_flags)
        wait_until(
            "Transformer tensors did not return to compressed cold storage",
            lambda: allocation_pages(weights) + allocation_pages(kvs) >= LAYERS * (WEIGHT_PAGES + KV_PAGES) * 3 // 4,
        )
        final_stats = runtime.stats()
        ledger = memx.capacity_ledger(weights + kvs, final_stats)
        if ledger["by_role"].get("weight", {}).get("logical_bytes", 0) != LAYERS * weight_size:
            raise AssertionError("Transformer ledger did not account weight bytes")
        if ledger["by_role"].get("kv_cache", {}).get("logical_bytes", 0) != LAYERS * kv_size:
            raise AssertionError("Transformer ledger did not account KV bytes")
        if ledger["effective_ratio_physical_est"] < 2.0:
            raise AssertionError(f"Transformer lifecycle effective ratio too low: {ledger['effective_ratio_physical_est']:.2f}x")
        projection_16 = memx.capacity_projection(ledger, 16 * 1024 * memx.MB)
        projection_32 = memx.capacity_projection(ledger, 32 * 1024 * memx.MB)
        if not projection_16["meets_2x"] or not projection_32["meets_2x"]:
            raise AssertionError("Transformer lifecycle projection did not meet 2x memory expansion")
        print(
            "python transformer lifecycle: "
            f"digest=0x{digest:016x} "
            f"faults={stats_before.faults}->{mid_stats.faults} "
            f"prefetch={stats_before.prefetch_count}->{mid_stats.prefetch_count} "
            f"hits={stats_before.prefetch_hits}->{mid_stats.prefetch_hits} "
            f"weight_pages={compressed_stats.weight_compressed_pages}->{final_stats.weight_compressed_pages} "
            f"kv_pages={compressed_stats.kv_cache_compressed_pages}->{final_stats.kv_cache_compressed_pages} "
            f"dedup={baseline_stats.dedup_hits}->{compressed_stats.dedup_hits} "
            f"ratio={ledger['effective_ratio_physical_est']:.2f}x "
            f"proj16={projection_16['projected_logical_gb']:.1f}GB "
            f"proj32={projection_32['projected_logical_gb']:.1f}GB"
        )
    finally:
        for allocation in reversed(kvs):
            allocation.free()
        for allocation in reversed(weights):
            allocation.free()
        ctx.destroy()
        runtime.shutdown()


if __name__ == "__main__":
    main()
