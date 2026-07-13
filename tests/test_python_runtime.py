#!/usr/bin/env python3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import memx_runtime as memx


def main():
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("python-smoke")
    allocation = None
    prefix = None
    sparse = None
    pressure_seed = None
    pressure_rescue = None
    try:
        ctx.set_quota(64 * memx.MB)
        size = 8 * memx.MB
        desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(1, 4, 64, 64),
            stride=(65536, 16384, 64, 1),
            layer_index=2,
        )
        allocation = ctx.malloc_tensor(size, desc, name="smoke-kv")
        buf = allocation.buffer()
        for page in range(size // 16384):
            base = page * 16384
            for half in range(8192):
                buf[base + half * 2] = (half * 11 + page * 19) & 0xFF
                buf[base + half * 2 + 1] = 0x38 if (half & 127) < 120 else 0x39
        time.sleep(3)
        before = allocation.info(0, 12 * 16384)
        if before.compressed_pages == 0 or before.tensor_codec_pages == 0:
            raise AssertionError("cold KV range did not compress through Python binding")
        window = memx.kv_cache_window(
            managed=(0, size),
            hot=(4 * 16384, 4 * 16384),
            prefetch=(0, 4 * 16384),
        )
        prefetch_before = runtime.stats().prefetch_count
        ctx.update_kv_cache_window(allocation, window)
        prefetch_after = runtime.stats().prefetch_count
        if prefetch_after <= prefetch_before:
            raise AssertionError("KV window did not prefetch any pages")
        prefetched = allocation.info(0, 4 * 16384)
        hot = allocation.info(4 * 16384, 4 * 16384)
        cold = allocation.info(8 * 16384, 8 * 16384)
        if prefetched.compressed_pages != 0:
            raise AssertionError("prefetched Python KV range stayed compressed")
        if hot.compressed_pages != 0 or (hot.tensor_flags & memx.MEMX_TENSOR_FLAG_HOT) == 0:
            raise AssertionError("hot Python KV range did not stay hot")
        if cold.compressed_pages == 0 or cold.tensor_codec_pages == 0:
            raise AssertionError("cold Python KV tail did not stay compressed")
        stats_before_access = runtime.stats()
        faults_before = stats_before_access.faults
        hits_before = stats_before_access.prefetch_hits
        ctx.mark_access_range(allocation, 0, 4 * 16384)
        hits_after = runtime.stats().prefetch_hits
        if hits_after <= hits_before:
            raise AssertionError("prefetched Python KV access did not count as a hit")
        checksum = 0
        for half in range((4 * 16384) // 2):
            checksum += buf[half * 2]
            checksum += buf[half * 2 + 1]
        faults_after = runtime.stats().faults
        if faults_after != faults_before:
            raise AssertionError("prefetched Python KV read triggered faults")
        print(
            "python runtime smoke: "
            f"compressed={before.compressed_pages} "
            f"tensor_codec={before.tensor_codec_pages} "
            f"prefetch={prefetch_before}->{prefetch_after} "
            f"checksum={checksum}"
        )
        ctx.update_tensor_flags(allocation, memx.MEMX_TENSOR_FLAG_HOT | memx.MEMX_TENSOR_FLAG_NO_COMPRESS)
        ctx.prefetch_range(allocation, size // 2, size // 2)
        ctx.update_tensor_flags_range(
            allocation,
            0,
            size // 2,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
        )
        time.sleep(3)
        cold_half = allocation.info(0, size // 2)
        hot_half = allocation.info(size // 2, size // 2)
        if cold_half.compressed_pages == 0 or cold_half.tensor_codec_pages == 0:
            raise AssertionError("Python range flags did not make cold KV half compress")
        if hot_half.compressed_pages != 0:
            raise AssertionError("Python range flags let hot KV half compress")
        if (hot_half.tensor_flags & memx.MEMX_TENSOR_FLAG_HOT) == 0:
            raise AssertionError("Python range flags did not keep hot half marked hot")
        policy_stats = runtime.stats()
        if policy_stats.hot_resident_pages == 0 or policy_stats.no_compress_resident_pages == 0:
            raise AssertionError("Python dynamic policy did not report hot resident telemetry")
        generic_offset = None
        generic_length = 2 * 16384
        for candidate in range(4 * 16384, size // 2, generic_length):
            candidate_info = allocation.info(candidate, generic_length)
            if candidate_info.compressed_pages > 0:
                generic_offset = candidate
                break
        if generic_offset is None:
            raise AssertionError("Python dynamic policy did not leave a compressed prefetch candidate")
        generic_prefetch_before = runtime.stats()
        ctx.prefetch_range(allocation, generic_offset, generic_length)
        generic_prefetch_after = runtime.stats()
        if generic_prefetch_after.prefetch_count <= generic_prefetch_before.prefetch_count:
            raise AssertionError("Python generic prefetch did not increment prefetch count")
        if generic_prefetch_after.faults != generic_prefetch_before.faults:
            raise AssertionError("Python generic prefetch triggered a fault")
        prefetched_generic = allocation.info(generic_offset, generic_length)
        if prefetched_generic.compressed_pages != 0:
            raise AssertionError("Python generic prefetch range stayed compressed")
        hits_before_generic = runtime.stats().prefetch_hits
        ctx.mark_access_range(allocation, generic_offset, generic_length)
        hits_after_generic = runtime.stats().prefetch_hits
        if hits_after_generic <= hits_before_generic:
            raise AssertionError("Python generic prefetch access did not count as a hit")
        faults_before_generic_read = runtime.stats().faults
        generic_checksum = 0
        first_half = generic_offset // 2
        for half in range(first_half, first_half + generic_length // 2):
            generic_checksum += buf[half * 2]
            generic_checksum += buf[half * 2 + 1]
        faults_after_generic_read = runtime.stats().faults
        if faults_after_generic_read != faults_before_generic_read:
            raise AssertionError("Python generic prefetched read faulted")
        print(
            "python dynamic policy: "
            f"cold_pages={cold_half.compressed_pages} "
            f"hot_pages={hot_half.compressed_pages} "
            f"hot_resident={policy_stats.hot_resident_pages} "
            f"offset={generic_offset} "
            f"prefetch={generic_prefetch_before.prefetch_count}->{generic_prefetch_after.prefetch_count} "
            f"hits={hits_before_generic}->{hits_after_generic} "
            f"checksum={generic_checksum}"
        )
        prefix_size = 4 * memx.MB
        prefix_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(1, 4, 64, 64),
            stride=(65536, 16384, 64, 1),
            layer_index=5,
        )
        prefix = ctx.malloc_tensor(prefix_size, prefix_desc, name="prefix-kv")
        prefix_buf = prefix.buffer()
        page_pattern = bytearray(16384)
        for half in range(8192):
            page_pattern[half * 2] = (half * 3) & 0xFF
            page_pattern[half * 2 + 1] = 0x3C
        for page in range(prefix_size // 16384):
            base = page * 16384
            prefix_buf[base:base + 16384] = page_pattern
        dedup_before = runtime.stats()
        time.sleep(3)
        dedup_after = runtime.stats()
        prefix_info = prefix.info()
        if prefix_info.compressed_pages == 0:
            raise AssertionError("repeated prefix pages did not compress")
        if dedup_after.dedup_hits <= dedup_before.dedup_hits:
            raise AssertionError("repeated prefix pages did not hit dedup")
        if dedup_after.dedup_bytes_saved <= dedup_before.dedup_bytes_saved:
            raise AssertionError("repeated prefix pages did not report dedup bytes")
        faults_before_prefix = runtime.stats().faults
        for i in range(0, 16384, 257):
            if prefix_buf[i] != page_pattern[i]:
                raise AssertionError("deduped prefix data changed after decompression")
        faults_after_prefix = runtime.stats().faults
        if faults_after_prefix <= faults_before_prefix:
            raise AssertionError("deduped prefix read did not exercise decompression")
        print(
            "python prefix dedup: "
            f"compressed={prefix_info.compressed_pages} "
            f"dedup_hits={dedup_before.dedup_hits}->{dedup_after.dedup_hits} "
            f"dedup_bytes={dedup_before.dedup_bytes_saved}->{dedup_after.dedup_bytes_saved}"
        )
        sparse_size = 4 * memx.MB
        sparse_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_ACTIVATION,
            memx.MEMX_TENSOR_DTYPE_UINT8,
            memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(sparse_size,),
            stride=(1,),
            layer_index=6,
        )
        sparse = ctx.malloc_tensor(sparse_size, sparse_desc, name="sparse-activation")
        sparse_buf = sparse.buffer()
        sparse_expected = {}
        for page in range(sparse_size // 16384):
            base = page * 16384
            for j in range(32):
                off = base + 32 + j * 257
                value = (j * 7 + page * 11 + 1) & 0xFF
                sparse_buf[off] = value
                sparse_expected[off] = value
        sparse_before = runtime.stats()
        time.sleep(3)
        sparse_after = runtime.stats()
        sparse_info = sparse.info()
        if sparse_info.compressed_pages == 0:
            raise AssertionError("sparse tensor pages did not compress")
        if sparse_info.primary_codec != memx.MEMX_RUNTIME_CODEC_TENSOR_SPARSE_BYTE:
            raise AssertionError(f"sparse tensor used codec 0x{sparse_info.primary_codec:x}")
        if sparse_after.tensor_sparse_pages <= sparse_before.tensor_sparse_pages:
            raise AssertionError("sparse tensor telemetry did not increase")
        faults_before_sparse = runtime.stats().faults
        for off, expected in list(sparse_expected.items())[:128]:
            if sparse_buf[off] != expected:
                raise AssertionError(
                    f"sparse tensor changed after decompression off={off} expected={expected} got={sparse_buf[off]} codec=0x{sparse_info.primary_codec:x}"
                )
        faults_after_sparse = runtime.stats().faults
        if faults_after_sparse <= faults_before_sparse:
            raise AssertionError("sparse tensor read did not exercise decompression")
        print(
            "python sparse codec: "
            f"compressed={sparse_info.compressed_pages} "
            f"sparse_pages={sparse_before.tensor_sparse_pages}->{sparse_after.tensor_sparse_pages} "
            f"codec=0x{sparse_info.primary_codec:x}"
        )
        ledger = memx.capacity_ledger([allocation, prefix, sparse], runtime.stats())
        if ledger["logical_bytes"] != size + prefix_size + sparse_size:
            raise AssertionError("Python capacity ledger logical bytes mismatch")
        if ledger["effective_ratio_physical_est"] <= 1.0:
            raise AssertionError("Python capacity ledger did not show effective expansion")
        if ledger["by_role"].get("kv_cache", {}).get("logical_bytes", 0) != size + prefix_size:
            raise AssertionError("Python capacity ledger KV role bytes mismatch")
        if ledger["by_role"].get("activation", {}).get("logical_bytes", 0) != sparse_size:
            raise AssertionError("Python capacity ledger activation role bytes mismatch")
        if ledger["by_codec"].get("tensor_sparse_byte", {}).get("compressed_pages", 0) == 0:
            raise AssertionError("Python capacity ledger did not report sparse codec pages")
        if ledger["dedup_hits"] == 0 or ledger["dedup_bytes_saved"] == 0:
            raise AssertionError("Python capacity ledger did not carry dedup telemetry")
        if ledger["hot_resident_pages"] == 0 or ledger["no_compress_resident_pages"] == 0:
            raise AssertionError("Python capacity ledger did not carry hot resident telemetry")
        if runtime.stats().tensor_delta_split_pages == 0:
            raise AssertionError("Python runtime did not expose delta split telemetry")
        projection_16 = memx.capacity_projection(ledger, 16 * 1024 * memx.MB, usable_fraction=1.0)
        projection_32 = memx.capacity_projection(ledger, 32 * 1024 * memx.MB, usable_fraction=1.0)
        safe_projection_16 = memx.capacity_projection(ledger, 16 * 1024 * memx.MB)
        safe_projection_32 = memx.capacity_projection(ledger, 32 * 1024 * memx.MB)
        if not projection_16["meets_2x"]:
            raise AssertionError("Python capacity projection did not cross 16GB->32GB target")
        if not projection_32["meets_2x"]:
            raise AssertionError("Python capacity projection did not cross 32GB->64GB target")
        print(
            "python capacity ledger: "
            f"logical={ledger['logical_bytes']} "
            f"physical_est={ledger['physical_estimate_bytes']} "
            f"ratio={ledger['effective_ratio_physical_est']:.2f} "
            f"kv={ledger['by_role']['kv_cache']['logical_bytes']} "
            f"activation={ledger['by_role']['activation']['logical_bytes']} "
            f"sparse_pages={ledger['by_codec']['tensor_sparse_byte']['compressed_pages']} "
            f"hot={ledger['hot_resident_pages']} "
            f"proj16={projection_16['projected_logical_gb']:.1f}GB "
            f"proj32={projection_32['projected_logical_gb']:.1f}GB "
            f"safe16={safe_projection_16['projected_logical_gb']:.1f}GB "
            f"safe32={safe_projection_32['projected_logical_gb']:.1f}GB"
        )
        pressure_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(1, 2, 64, 64),
            stride=(32768, 16384, 64, 1),
            layer_index=7,
        )
        pressure_seed_size = 4 * memx.MB
        pressure_seed = ctx.malloc_tensor(pressure_seed_size, pressure_desc)
        pressure_buf = pressure_seed.buffer()
        for page in range(pressure_seed_size // 16384):
            base = page * 16384
            for half in range(8192):
                pressure_buf[base + half * 2] = (half * 5 + page * 17) & 0xFF
                pressure_buf[base + half * 2 + 1] = 0x36 + (page & 1) if (half & 63) < 60 else 0x37
        time.sleep(3)
        pressure_info = pressure_seed.info()
        if pressure_info.compressed_pages == 0 or pressure_info.compressed_bytes == 0:
            raise AssertionError("Python pressure seed did not compress")
        pressure_before_free = runtime.pressure()
        stats_before_free = runtime.stats()
        reclaim_events_before = stats_before_free.pool_reclaim_events
        reclaim_bytes_before = stats_before_free.pool_reclaim_bytes
        pool_used_before = stats_before_free.pool_used_bytes
        pressure_seed.free()
        pressure_seed = None
        reclaimed = runtime.reclaim()
        pressure_after_reclaim = runtime.pressure()
        stats_after_reclaim = runtime.stats()
        if stats_after_reclaim.pool_reclaim_events <= reclaim_events_before:
            raise AssertionError("Python reclaim did not record a reclaim event")
        if stats_after_reclaim.pool_reclaim_bytes <= reclaim_bytes_before:
            raise AssertionError("Python reclaim did not record reclaimed bytes")
        if stats_after_reclaim.pool_used_bytes >= pool_used_before:
            raise AssertionError("Python reclaim did not reduce pool usage")
        pressure_events_before = ctx.stats().pressure_events
        pressure_cursor = (pressure_after_reclaim.pool_capacity_bytes * 95 + 99) // 100
        runtime.test_set_pool_cursor(pressure_cursor)
        pressure_rescue = ctx.malloc(64 * 1024)
        rescue_buf = pressure_rescue.buffer()
        for i in range(0, pressure_rescue.size, 4096):
            rescue_buf[i] = 0xA5
        for i in range(0, pressure_rescue.size, 4096):
            if rescue_buf[i] != 0xA5:
                raise AssertionError("Python pressure rescue allocation changed data")
        pressure_events_after = ctx.stats().pressure_events
        if pressure_events_after != pressure_events_before:
            raise AssertionError("Python pressure rescue allocation reported a pressure event")
        print(
            "python pressure recovery: "
            f"compressed={pressure_info.compressed_pages} "
            f"reclaim_events={reclaim_events_before}->{stats_after_reclaim.pool_reclaim_events} "
            f"reclaimed={reclaimed} cumulative_bytes={reclaim_bytes_before}->{stats_after_reclaim.pool_reclaim_bytes} "
            f"free_extents={pressure_before_free.pool_free_extent_bytes}->{pressure_after_reclaim.pool_free_extent_bytes} "
            f"cursor95={pressure_cursor}"
        )
    finally:
        if pressure_rescue is not None:
            pressure_rescue.free()
        if pressure_seed is not None:
            pressure_seed.free()
        if sparse is not None:
            sparse.free()
        if prefix is not None:
            prefix.free()
        if allocation is not None:
            allocation.free()
        ctx.destroy()
        runtime.shutdown()


if __name__ == "__main__":
    main()
