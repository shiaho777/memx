#!/usr/bin/env python3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import torch

import memx_runtime as memx
import test_python_torch_transformer as torch_case


PAGE = 16384
PRESSURE_SEEDS = 10


def wait_until(label, predicate, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(label)


def fill_pressure_seed(allocation, salt):
    tensor = allocation.torch_tensor(torch.uint8, (allocation.size,))
    page_pattern = torch.empty(PAGE, dtype=torch.uint8)
    offsets = torch.arange(PAGE, dtype=torch.int32)
    page_pattern[:] = ((offsets * 3 + salt * 17) & 255).to(torch.uint8)
    page_pattern[1::2] = 0x3C + (salt & 3)
    for page in range(allocation.size // PAGE):
        start = page * PAGE
        tensor[start:start + PAGE].copy_(page_pattern)


def build_model(ctx):
    weight_allocs = []
    kv_allocs = []
    managed_weights = []
    managed_kvs = []
    baseline_weights = []
    baseline_kvs = []
    weight_size = torch_case.storage_size(torch_case.WEIGHT_SHAPE, torch.float16)
    kv_size = torch_case.storage_size(torch_case.KV_SHAPE, torch.float16)
    for layer in range(torch_case.LAYERS):
        weight_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_WEIGHT,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
            shape=torch_case.WEIGHT_SHAPE,
            stride=(torch_case.WEIGHT_SHAPE[1], 1),
            layer_index=layer,
        )
        kv_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=torch_case.KV_SHAPE,
            stride=(torch_case.TOKENS * torch_case.HEAD_DIM, torch_case.HEAD_DIM, 1),
            layer_index=layer,
        )
        weight_alloc = ctx.malloc_tensor(weight_size, weight_desc, name=f"pressure-layer{layer}-weight")
        kv_alloc = ctx.malloc_tensor(kv_size, kv_desc, name=f"pressure-layer{layer}-kv")
        baseline_weight = torch_case.make_weight(layer)
        baseline_kv = torch_case.make_kv(layer)
        torch_case.copy_tensor_to_allocation(baseline_weight, weight_alloc)
        torch_case.copy_tensor_to_allocation(baseline_kv, kv_alloc)
        weight_allocs.append(weight_alloc)
        kv_allocs.append(kv_alloc)
        managed_weights.append(weight_alloc.torch_tensor(torch.float16, torch_case.WEIGHT_SHAPE))
        managed_kvs.append(kv_alloc.torch_tensor(torch.float16, torch_case.KV_SHAPE))
        baseline_weights.append(baseline_weight.clone())
        baseline_kvs.append(baseline_kv.clone())
    return weight_allocs, kv_allocs, managed_weights, managed_kvs, baseline_weights, baseline_kvs


def main():
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("python-torch-pressure")
    weight_allocs = []
    kv_allocs = []
    pressure_allocs = []
    rescue = None
    try:
        ctx.set_quota(128 * memx.MB)
        (
            weight_allocs,
            kv_allocs,
            managed_weights,
            managed_kvs,
            baseline_weights,
            baseline_kvs,
        ) = build_model(ctx)
        model_allocs = weight_allocs + kv_allocs
        wait_until(
            "pressure model tensors did not compress",
            lambda: sum(a.info().compressed_pages for a in model_allocs) >= torch_case.LAYERS * 2,
        )
        x = torch_case.make_input()
        baseline_out = torch_case.run_transformer(baseline_weights, baseline_kvs, x.clone())
        pressure_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(1, 4, 256, 32),
            stride=(32768, 8192, 32, 1),
            layer_index=99,
        )
        for i in range(PRESSURE_SEEDS):
            allocation = ctx.malloc_tensor(4 * memx.MB, pressure_desc, name=f"pressure-seed-{i}")
            fill_pressure_seed(allocation, i)
            pressure_allocs.append(allocation)
        wait_until(
            "pressure seed tensors did not compress",
            lambda: sum(a.info().compressed_pages for a in pressure_allocs) >= PRESSURE_SEEDS * 32,
        )
        stats_before_free = runtime.stats()
        for index, allocation in enumerate(pressure_allocs):
            if index % 2 == 0:
                allocation.free()
                pressure_allocs[index] = None
        reclaimed = runtime.reclaim()
        stats_after_reclaim = runtime.stats()
        pressure = runtime.pressure()
        if stats_after_reclaim.pool_reclaim_events <= stats_before_free.pool_reclaim_events:
            raise AssertionError("pressure test did not record reclaim events")
        if stats_after_reclaim.pool_reclaim_bytes <= stats_before_free.pool_reclaim_bytes:
            raise AssertionError("pressure test did not reclaim pool bytes")
        cursor = (pressure.pool_capacity_bytes * 95 + 99) // 100
        runtime.test_set_pool_cursor(cursor)
        pressure_high = runtime.pressure()
        if pressure_high.pool_pressure_percent < 95 or pressure_high.pool_near_full == 0:
            raise AssertionError("pressure test did not reach near-full pool telemetry")
        pressure_events_before = ctx.stats().pressure_events
        rescue = ctx.malloc(64 * 1024, name="pressure-rescue")
        rescue_buf = rescue.buffer()
        for i in range(0, rescue.size, 4096):
            rescue_buf[i] = 0xA5
        for i in range(0, rescue.size, 4096):
            if rescue_buf[i] != 0xA5:
                raise AssertionError("pressure rescue allocation changed data")
        if ctx.stats().pressure_events != pressure_events_before:
            raise AssertionError("pressure rescue allocation reported a pressure failure")
        faults_before = runtime.stats().faults
        for allocation in weight_allocs:
            ctx.update_weight_window(
                allocation,
                memx.weight_window(managed=(0, allocation.size), hot=(0, PAGE), prefetch=(PAGE, PAGE)),
            )
            ctx.mark_access_range(allocation, PAGE, PAGE)
        for allocation in kv_allocs:
            ctx.update_kv_cache_window(
                allocation,
                memx.kv_cache_window(managed=(0, allocation.size), hot=(0, PAGE), prefetch=(PAGE, PAGE)),
            )
            ctx.mark_access_range(allocation, PAGE, PAGE)
        managed_out = torch_case.run_transformer(managed_weights, managed_kvs, x.clone())
        stats_after_compute = runtime.stats()
        if not torch.equal(managed_out.view(torch.uint16), baseline_out.view(torch.uint16)):
            raise AssertionError("pressure PyTorch Transformer output changed under MemX")
        if stats_after_compute.faults <= faults_before:
            raise AssertionError("pressure PyTorch Transformer did not fault compressed pages")
        if stats_after_compute.prefetch_hits == 0:
            raise AssertionError("pressure PyTorch Transformer did not observe prefetch hits")
        cold_flags = memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY
        for allocation in model_allocs:
            ctx.update_tensor_flags(allocation, cold_flags)
        wait_until(
            "pressure model tensors did not return cold",
            lambda: sum(a.info().compressed_pages for a in model_allocs) >= torch_case.LAYERS * 3,
        )
        final_stats = runtime.stats()
        ledger = memx.capacity_ledger(model_allocs, final_stats)
        digest = managed_out.contiguous().view(torch.uint16).to(torch.int64).sum().item() & 0xFFFFFFFFFFFFFFFF
        print(
            "python torch pressure: "
            f"digest=0x{digest:016x} "
            f"faults={faults_before}->{stats_after_compute.faults} "
            f"prefetch={stats_after_reclaim.prefetch_count}->{stats_after_compute.prefetch_count} "
            f"hits={stats_after_reclaim.prefetch_hits}->{stats_after_compute.prefetch_hits} "
            f"reclaim_events={stats_before_free.pool_reclaim_events}->{stats_after_reclaim.pool_reclaim_events} "
            f"reclaimed={reclaimed} "
            f"pressure={pressure_high.pool_pressure_percent}% "
            f"frag={pressure_high.pool_fragmentation_percent}% "
            f"ratio={ledger['effective_ratio_physical_est']:.2f}x"
        )
    finally:
        if rescue is not None:
            rescue.free()
        for allocation in reversed(pressure_allocs):
            if allocation is not None:
                allocation.free()
        for allocation in reversed(kv_allocs):
            allocation.free()
        for allocation in reversed(weight_allocs):
            allocation.free()
        ctx.destroy()
        runtime.shutdown()


if __name__ == "__main__":
    main()
