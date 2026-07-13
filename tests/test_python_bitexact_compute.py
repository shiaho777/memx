#!/usr/bin/env python3
import ctypes
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import memx_runtime as memx


PAGE = 16384


def pattern16(i, salt):
    mantissa = (i * 17 + salt * 29) & 0x03FF
    exponent = 0x3C00 if ((i + salt) & 7) != 0 else 0x3800
    sign = 0x8000 if ((i + salt) & 31) == 0 else 0
    return sign | exponent | mantissa


def write_u16(buf, index, value):
    offset = index * 2
    buf[offset] = value & 0xFF
    buf[offset + 1] = (value >> 8) & 0xFF


def read_u16(buf, index):
    offset = index * 2
    return buf[offset] | (buf[offset + 1] << 8)


def populate(buf, half_count, salt):
    for i in range(half_count):
        write_u16(buf, i, pattern16(i, salt))


def compute_digest(weight_buf, kv_buf, hidden, tokens):
    out = bytearray(hidden * 8)
    for col in range(hidden):
        acc = 0
        for row in range(hidden):
            w = read_u16(weight_buf, row * hidden + col)
            k = read_u16(kv_buf, ((row + col) % tokens) * hidden + row)
            acc = (acc + ((w ^ ((k << 1) & 0xFFFF)) * (row + 1))) & 0xFFFFFFFFFFFFFFFF
        for token in range(tokens):
            v = read_u16(kv_buf, token * hidden + col)
            acc = (acc ^ ((v + token * 131 + col * 17) * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF
        out[col * 8:(col + 1) * 8] = acc.to_bytes(8, "little")
    return bytes(out)


def main():
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("python-bitexact")
    weight = None
    kv = None
    try:
        ctx.set_quota(96 * memx.MB)
        hidden = 128
        tokens = 128
        weight_size = PAGE * 4
        kv_size = PAGE * 4
        weight_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_WEIGHT,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(hidden, hidden),
            stride=(hidden, 1),
            layer_index=0,
        )
        kv_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_COLD | memx.MEMX_TENSOR_FLAG_READ_MOSTLY,
            shape=(1, 1, tokens, hidden),
            stride=(tokens * hidden, tokens * hidden, hidden, 1),
            layer_index=0,
        )
        weight = ctx.malloc_tensor(weight_size, weight_desc)
        kv = ctx.malloc_tensor(kv_size, kv_desc)
        weight_buf = weight.buffer()
        kv_buf = kv.buffer()
        base_weight = bytearray(weight_size)
        base_kv = bytearray(kv_size)
        populate(base_weight, weight_size // 2, 3)
        populate(base_kv, kv_size // 2, 9)
        weight_buf[:] = base_weight
        kv_buf[:] = base_kv
        baseline = compute_digest(base_weight, base_kv, hidden, tokens)
        time.sleep(3)
        weight_info = weight.info()
        kv_info = kv.info()
        if weight_info.compressed_pages == 0 or kv_info.compressed_pages == 0:
            raise AssertionError("managed tensors did not compress before bit-exact compute")
        prefetch_before = runtime.stats().prefetch_count
        w_window = memx.weight_window(
            managed=(0, weight_size),
            hot=(0, PAGE),
            prefetch=(PAGE, PAGE),
        )
        ctx.update_weight_window(weight, w_window)
        prefetch_after_weight = runtime.stats().prefetch_count
        if prefetch_after_weight <= prefetch_before:
            raise AssertionError("weight window did not prefetch any pages")
        w_hot = weight.info(0, PAGE)
        w_prefetched = weight.info(PAGE, PAGE)
        w_cold = weight.info(PAGE * 2, PAGE * 2)
        if w_hot.compressed_pages != 0 or (w_hot.tensor_flags & memx.MEMX_TENSOR_FLAG_HOT) == 0:
            raise AssertionError("weight hot layer did not stay resident")
        if w_prefetched.compressed_pages != 0:
            raise AssertionError("weight prefetched layer stayed compressed")
        if w_cold.compressed_pages == 0 or w_cold.tensor_codec_pages == 0:
            raise AssertionError("weight cold tail did not stay compressed")
        window = memx.kv_cache_window(
            managed=(0, kv_size),
            hot=(PAGE, PAGE),
            prefetch=(0, PAGE),
        )
        ctx.update_kv_cache_window(kv, window)
        faults_before = runtime.stats().faults
        managed = compute_digest(weight_buf, kv_buf, hidden, tokens)
        if managed != baseline:
            raise AssertionError("MemX managed tensor compute changed output bytes")
        cold_index = (PAGE * 2) // 2
        expected_cold = read_u16(base_weight, cold_index)
        actual_cold = read_u16(weight_buf, cold_index)
        faults_after = runtime.stats().faults
        if actual_cold != expected_cold:
            raise AssertionError("cold weight tail changed after on-demand decompression")
        if faults_after <= faults_before:
            raise AssertionError("bit-exact compute did not exercise on-demand decompression")
        print(
            "python bitexact compute: "
            f"weight_pages={weight_info.compressed_pages} "
            f"kv_pages={kv_info.compressed_pages} "
            f"faults={faults_before}->{faults_after} "
            f"digest={baseline[:8].hex()}"
        )
    finally:
        if kv is not None:
            kv.free()
        if weight is not None:
            weight.free()
        ctx.destroy()
        runtime.shutdown()


if __name__ == "__main__":
    main()
