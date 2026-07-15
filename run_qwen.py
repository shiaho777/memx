#!/usr/bin/env python3
"""Qwen3.5-0.8B workload harness for local memory profiling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

try:
    import memx_runtime as memx
except Exception:
    memx = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to a local Qwen3.5-0.8B-hf checkpoint directory")
    parser.add_argument("--memx-smoke", action="store_true", help="Run a MemX KV-cache host integration smoke before loading weights")
    return parser.parse_args()


def resolve_model_path(cli_value):
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())
    env_value = os.environ.get("MEMX_MODEL_PATH")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(ROOT / ".local" / "Qwen3.5-0.8B-hf")
    candidates.append(ROOT / "Qwen3.5-0.8B-hf")
    candidates.append(Path.home() / "Desktop" / "Qwen__Qwen2.5-1.5B-Instruct")
    candidates.append(Path.home() / "Desktop" / "Qwen3.5-0.8B-hf" / "model")

    for candidate in candidates:
        if not (candidate / "config.json").exists():
            continue
        if any((candidate / n).exists() for n in (
            "model.safetensors",
            "model.safetensors.index.json",
            "model.safetensors-00001-of-00001.safetensors",
            "pytorch_model.bin",
        )):
            return candidate
        # config-only dir is acceptable only when explicitly passed
        if cli_value and Path(cli_value).expanduser().resolve() == candidate.resolve():
            return candidate

    raise SystemExit(
        "Model not found. Pass --model-path, set MEMX_MODEL_PATH, "
        "or place the checkpoint under .local/Qwen3.5-0.8B-hf"
    )


ARGS = parse_args()
model_path = resolve_model_path(ARGS.model_path)
tag = "[" + os.environ.get("MEMX_WORKLOAD_LABEL", "Baseline workload") + "]"
print(f"{tag} Process PID: {os.getpid()}")


def run_memx_smoke():
    if memx is None:
        print(f"{tag} MemX Python binding unavailable; skipping host integration smoke")
        return
    runtime = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
    ctx = runtime.create_context("qwen-python-host")
    allocation = None
    weight = None
    try:
        ctx.set_quota(256 * memx.MB)
        size = 16 * memx.MB
        desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
            shape=(1, 8, 128, 64),
            stride=(65536, 8192, 64, 1),
        )
        allocation = ctx.malloc_tensor(size, desc, name="kv-host")
        buf = allocation.buffer()
        token_bytes = 16384
        total_tokens = size // token_bytes
        written = 0
        import ctypes
        for page in range(total_tokens):
            base = page * token_bytes
            chunk = (ctypes.c_uint8 * token_bytes)()
            for half in range(token_bytes // 2):
                chunk[half * 2] = (half * 7 + page * 13) & 0xFF
                chunk[half * 2 + 1] = 0x3A if (half & 255) < 248 else 0x3B
            ctypes.memmove(ctypes.addressof(buf) + base, chunk, token_bytes)
            written = page + 1
            if (page & 7) == 7:
                ctx.advance_kv_window(
                    allocation,
                    token_bytes=token_bytes,
                    written_tokens=written,
                    hot_tokens=8,
                    prefetch_tokens=4,
                )
        time.sleep(2)
        ctx.advance_kv_window(
            allocation,
            token_bytes=token_bytes,
            written_tokens=written,
            hot_tokens=8,
            prefetch_tokens=4,
        )
        time.sleep(2)
        cold = allocation.info(0, min(size, 16 * 16384))
        weight_size = 8 * memx.MB
        weight_desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_WEIGHT,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
            shape=(weight_size // 2,),
            stride=(1,),
        )
        weight = ctx.malloc_tensor(weight_size, weight_desc, name="weight-host")
        wbuf = weight.buffer()
        for i in range(0, weight_size, 4096):
            wbuf[i] = (i // 4096) & 0xFF
        time.sleep(2)
        ww = memx.weight_window(
            managed=(0, weight_size // 2),
            hot=(weight_size // 2, weight_size // 4),
            prefetch=(weight_size // 2 + weight_size // 4, weight_size // 4),
        )
        ctx.update_weight_window(weight, ww)
        stats = runtime.stats()
        winfo = weight.info()
        print(
            f"{tag} MemX host smoke: kv_compressed={cold.compressed_pages} "
            f"kv_codec={cold.tensor_codec_pages} weight_compressed={winfo.compressed_pages} "
            f"bitplane={stats.tensor_bitplane_pages} delta={stats.tensor_delta_split_pages} "
            f"prefetch={stats.prefetch_count} hits={stats.prefetch_hits}"
        )
    finally:
        if weight is not None:
            weight.free()
        if allocation is not None:
            allocation.free()
        ctx.destroy()
        runtime.shutdown()


if ARGS.memx_smoke:
    run_memx_smoke()
    raise SystemExit(0)


class MemXHost:
    def __init__(self, runtime, ctx):
        self.runtime = runtime
        self.ctx = ctx
        self.kv = None
        self.weights = []
        self.weight_map = {}
        self.token_bytes = 16384
        self.written_tokens = 0
        self.hot_tokens = int(os.environ.get("MEMX_KV_HOT_TOKENS", "16"))
        self.prefetch_tokens = int(os.environ.get("MEMX_KV_PREFETCH_TOKENS", "8"))
        self.weight_hot_frac = float(os.environ.get("MEMX_WEIGHT_HOT_FRAC", "0.01"))
        self.hosted_bytes = 0
        self.released_bytes = 0
        self._pin_state = {}
        self._pin_page = 16384
        self._stream_active = None
        self._stream_cursor = 0
        self._ops_since_reclaim = 0
        self._ws_orch = hasattr(ctx, "ws_advance") and os.environ.get("MEMX_WS_ORCH", "1") != "0"
        self._epoch_phase = 0
        self._hot_budget = int(os.environ.get("MEMX_WS_HOT_BUDGET_MB", "384")) * 1024 * 1024

    @classmethod
    def maybe_create(cls, quota_mb=None):
        if memx is None:
            return None
        if os.environ.get("MEMX_HOST_ENABLE", "1") not in ("1", "true", "TRUE", "yes", "YES"):
            return None
        dylib = ROOT / "build" / "libmemx_runtime.dylib"
        if not dylib.exists():
            return None
        if quota_mb is None:
            quota_mb = int(os.environ.get("MEMX_HOST_QUOTA_MB", "4096"))
        runtime = memx.Runtime(dylib)
        ctx = runtime.create_context("qwen-host")
        ctx.set_quota(quota_mb * memx.MB)
        return cls(runtime, ctx)


    def archive_dir(self):
        d = os.environ.get("MEMX_ARCHIVE_DIR", "").strip()
        if not d:
            return None
        path = Path(d)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        return path

    def archive_path_for(self, name):
        root = self.archive_dir()
        if root is None or not name:
            return None
        safe = name.replace(os.sep, "__").replace("/", "__").replace(" ", "_")
        if len(safe) > 180:
            import hashlib
            safe = hashlib.sha1(name.encode("utf-8")).hexdigest() + ".mxwa"
            return root / safe
        return root / (safe + ".mxwa")

    def alloc_kv(self, tokens=256):
        size = tokens * self.token_bytes
        desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_KV_CACHE,
            memx.MEMX_TENSOR_DTYPE_FP16,
            memx.MEMX_TENSOR_LAYOUT_BLOCKED,
            memx.MEMX_TENSOR_FLAG_SEQUENTIAL,
            shape=(tokens, self.token_bytes // 2),
            stride=(self.token_bytes // 2, 1),
        )
        self.kv = self.ctx.malloc_tensor(size, desc, name="qwen-kv")
        return self.kv

    def _seal_weight(self, alloc, nbytes):
        page = 16384
        hot = int(nbytes * self.weight_hot_frac)
        hot = (hot // page) * page
        if hot < 0:
            hot = 0
        if hot > nbytes:
            hot = nbytes
        managed = nbytes - hot
        pref = min(page * 8, hot)
        pref = (pref // page) * page
        hot_body = max(0, hot - pref)
        window = memx.weight_window(
            managed=(0, managed),
            hot=(managed, hot_body),
            prefetch=(managed + hot_body, pref),
        )
        self.ctx.update_weight_window(alloc, window)
        if managed > 0:
            self.ctx.update_tensor_flags_range(
                alloc, 0, managed,
                memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
            )

    def host_weight_tensor(self, tensor, name="weight", replace=True):
        if tensor is None or not hasattr(tensor, "numel"):
            return None
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            return None
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        shape = tuple(tensor.shape)
        src_tensor = tensor.detach()
        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()
        nbytes = src_tensor.numel() * src_tensor.element_size()
        if nbytes < 16384:
            return None
        dtype_code = memx.MEMX_TENSOR_DTYPE_FP16 if src_tensor.dtype == torch.float16 else memx.MEMX_TENSOR_DTYPE_BF16
        desc = memx.tensor_desc(
            memx.MEMX_TENSOR_ROLE_WEIGHT,
            dtype_code,
            memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
            shape=shape,
            stride=tuple(src_tensor.stride()),
        )
        arch = self.archive_path_for(name)
        use_arch = (
            arch is not None
            and os.environ.get("MEMX_ARCHIVE_LOAD", "1") != "0"
            and arch.is_file()
            and hasattr(self.ctx, "import_archive")
        )
        alloc = None
        if use_arch:
            try:
                if arch.stat().st_size > 64:
                    alloc = self.ctx.import_archive(str(arch), desc=desc, name=name)
                    if alloc is None or alloc.size != nbytes:
                        if alloc is not None:
                            try:
                                alloc.free()
                            except Exception:
                                pass
                        alloc = None
            except Exception:
                alloc = None
        if alloc is None:
            try:
                alloc = self.ctx.malloc_tensor(nbytes, desc, name=name)
            except MemoryError:
                return None
            import ctypes
            src_u8 = src_tensor.view(torch.uint8).contiguous()
            buf = alloc.buffer()
            ctypes.memmove(ctypes.addressof(buf), src_u8.data_ptr(), nbytes)
            del src_u8
            del src_tensor
            self._seal_weight(alloc, nbytes)
            if (
                arch is not None
                and os.environ.get("MEMX_ARCHIVE_SAVE", "1") != "0"
                and hasattr(self.ctx, "export_archive")
            ):
                try:
                    self.ctx.force_compress_range(alloc, 0, nbytes)
                except Exception:
                    pass
                try:
                    self.ctx.export_archive(alloc, str(arch))
                except Exception:
                    pass
        else:
            try:
                self._seal_weight(alloc, nbytes)
            except Exception:
                pass
        hosted = None
        if replace:
            try:
                hosted = alloc.torch_tensor(tensor.dtype, shape)
            except Exception:
                hosted = None
        self.weights.append(alloc)
        self.weight_map[name] = alloc
        self.hosted_bytes += nbytes
        return hosted if hosted is not None else alloc

    def host_state_dict(self, state_dict, max_bytes=None, min_bytes=16384, prefer_prefixes=None):
        if not state_dict:
            return 0, 0
        if max_bytes is None:
            max_bytes = int(os.environ.get("MEMX_HOST_MAX_BYTES", str(1800 * 1024 * 1024)))
        if prefer_prefixes is None:
            prefer_prefixes = (
                "model.language_model.embed_tokens",
                "model.language_model.layers.",
                "model.language_model.norm",
                "model.layers.",
                "model.embed_tokens",
            )
        items = []
        for k, v in state_dict.items():
            if not hasattr(v, "numel") or not hasattr(v, "dtype"):
                continue
            if v.dtype not in (torch.float16, torch.bfloat16):
                continue
            nbytes = v.numel() * v.element_size()
            if nbytes < min_bytes:
                continue
            score = nbytes
            if any(k.startswith(p) or p in k for p in prefer_prefixes):
                score += 10**12
            if "visual" in k or "mtp." in k:
                score -= 10**11
            items.append((score, nbytes, k, v))
        items.sort(reverse=True)
        n = 0
        used = 0
        for idx, (_, nbytes, k, v) in enumerate(items):
            if used + nbytes > max_bytes:
                continue
            hosted = self.host_weight_tensor(v, name=k, replace=True)
            if hosted is None:
                continue
            if torch.is_tensor(hosted):
                state_dict[k] = hosted
                self.released_bytes += nbytes
            # drop local ref to source tensor ASAP
            items[idx] = (0, 0, k, None)
            del v
            n += 1
            used += nbytes
            if (n & 15) == 0:
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
        del items
        try:
            self.runtime.reclaim()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            gc.collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        return n, used

    def host_selected_weights(self, state_dict, max_tensors=8, min_mb=1):
        n, _ = self.host_state_dict(
            state_dict,
            max_bytes=max_tensors * max(min_mb, 1) * 8 * 1024 * 1024,
            min_bytes=min_mb * 1024 * 1024,
        )
        return n

    def write_token(self, token_idx, pattern=0x3A):
        if self.kv is None:
            return
        import ctypes
        base = token_idx * self.token_bytes
        buf = self.kv.buffer()
        end = min(base + self.token_bytes, self.kv.size)
        span = end - base
        if span > 0:
            chunk = (ctypes.c_uint8 * span)()
            for j in range(0, span, 2):
                chunk[j] = ((base + j) // 2 + token_idx) & 0xFF
                if j + 1 < span:
                    chunk[j + 1] = pattern
            ctypes.memmove(ctypes.addressof(buf) + base, chunk, span)
        self.written_tokens = max(self.written_tokens, token_idx + 1)
        if (token_idx & 3) == 3:
            self.advance()

    def advance(self):
        if self.kv is None:
            return None
        return self.ctx.advance_kv_window(
            self.kv,
            token_bytes=self.token_bytes,
            written_tokens=self.written_tokens,
            hot_tokens=self.hot_tokens,
            prefetch_tokens=self.prefetch_tokens,
        )


    def reseal_weights(self):
        for name, alloc in list(self.weight_map.items()):
            try:
                self._seal_weight(alloc, alloc.size)
            except Exception:
                pass
        try:
            self.runtime.reclaim()
        except Exception:
            pass

    def cool_all_weights(self, force=False, purge=False):
        do_force = force or os.environ.get("MEMX_FORCE_COMPRESS", "0") == "1"
        do_purge = purge or os.environ.get("MEMX_PURGE", "0") == "1"
        self._pin_state.clear()
        self._stream_active = None
        self._stream_cursor = 0
        n = 0
        batch = int(os.environ.get("MEMX_COOL_RECLAIM_BATCH", "32"))
        if batch < 1:
            batch = 1
        seal = hasattr(self.ctx, "seal_range")
        for name, alloc in list(self.weight_map.items()):
            try:
                nbytes = alloc.size
                if do_purge:
                    try:
                        self.ctx.purge(alloc)
                    except Exception:
                        if seal:
                            try:
                                self.ctx.seal_range(alloc, 0, nbytes)
                            except Exception:
                                self.ctx.force_compress_range(alloc, 0, nbytes)
                        else:
                            self.ctx.update_tensor_flags_range(
                                alloc, 0, nbytes,
                                memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                            )
                            self.ctx.force_compress_range(alloc, 0, nbytes)
                elif do_force:
                    if hasattr(self.ctx, "seal_range_async"):
                        try:
                            self.ctx.seal_range_async(alloc, 0, nbytes)
                        except Exception:
                            if seal:
                                self.ctx.seal_range(alloc, 0, nbytes)
                            else:
                                self.ctx.update_tensor_flags_range(
                                    alloc, 0, nbytes,
                                    memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                                )
                                self.ctx.force_compress_range(alloc, 0, nbytes)
                    elif seal:
                        try:
                            self.ctx.seal_range(alloc, 0, nbytes)
                        except Exception:
                            self.ctx.update_tensor_flags_range(
                                alloc, 0, nbytes,
                                memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                            )
                            self.ctx.force_compress_range(alloc, 0, nbytes)
                    else:
                        self.ctx.update_tensor_flags_range(
                            alloc, 0, nbytes,
                            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                        )
                        self.ctx.force_compress_range(alloc, 0, nbytes)
                else:
                    self.ctx.update_tensor_flags_range(
                        alloc, 0, nbytes,
                        memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                    )
                n += 1
                if (do_force or do_purge) and (n % batch) == 0:
                    try:
                        self.runtime.reclaim()
                    except Exception:
                        pass
            except Exception:
                pass
        if self.kv is not None:
            try:
                nbytes = self.kv.size
                if do_purge:
                    try:
                        self.ctx.purge(self.kv)
                    except Exception:
                        if seal:
                            self.ctx.seal_range(self.kv, 0, nbytes)
                        else:
                            self.ctx.force_compress_range(self.kv, 0, nbytes)
                elif do_force:
                    if seal:
                        self.ctx.seal_range(self.kv, 0, nbytes)
                    else:
                        self.ctx.update_tensor_flags_range(
                            self.kv, 0, nbytes,
                            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                        )
                        self.ctx.force_compress_range(self.kv, 0, nbytes)
                else:
                    self.ctx.update_tensor_flags_range(
                        self.kv, 0, nbytes,
                        memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                    )
            except Exception:
                pass
        if hasattr(self.ctx, "seal_flush"):
            try:
                self.ctx.seal_flush()
            except Exception:
                pass
        try:
            self.runtime.reclaim()
        except Exception:
            pass
        if do_force or do_purge:
            try:
                self.runtime.compact()
            except Exception:
                try:
                    self.runtime.reclaim()
                except Exception:
                    pass

    def warm_weight(self, name, hot_frac=None, full=False, prefetch_only=False):
        alloc = self.weight_map.get(name)
        if alloc is None:
            return
        page = 16384
        nbytes = alloc.size
        if full:
            hot_frac = 1.0
        elif hot_frac is None:
            hot_frac = float(os.environ.get("MEMX_INFER_HOT_FRAC", "1.0"))
        hot = int(nbytes * hot_frac)
        hot = (hot // page) * page
        if hot < page and nbytes >= page:
            hot = page
        if hot > nbytes:
            hot = nbytes
        if prefetch_only:
            max_pages = int(os.environ.get("MEMX_PREFETCH_PAGES", "12"))
            if nbytes >= 8 * 1024 * 1024 and max_pages < 24:
                max_pages = 24
            elif nbytes >= 2 * 1024 * 1024 and max_pages < 16:
                max_pages = 16
            pref = min(hot if hot > 0 else min(page * max_pages, nbytes), nbytes)
            pref = (pref // page) * page
            if pref <= 0:
                return
            try:
                self.ctx.prefetch_range(alloc, 0, pref)
            except Exception:
                pass
            return
        managed = max(0, nbytes - hot)
        pref = min(page * 8, hot if hot > 0 else page * 8)
        pref = (pref // page) * page
        if pref > hot:
            pref = hot
        hot_body = max(0, hot - pref)
        try:
            window = memx.weight_window(
                managed=(0, managed),
                hot=(managed, hot_body),
                prefetch=(managed + hot_body, pref),
            )
            self.ctx.update_weight_window(alloc, window)
            if hot > 0 and os.environ.get("MEMX_WARM_PREFETCH", "1") != "0":
                try:
                    # cap eager materialization to avoid RSS spikes on large weights
                    cap_pages = int(os.environ.get("MEMX_WARM_PREFETCH_PAGES", "32"))
                    cap = min(hot, page * max(cap_pages, 1))
                    self.ctx.prefetch_range(alloc, managed, cap)
                except Exception:
                    pass
        except Exception:
            pass

    def cool_weight(self, name, force=False):
        alloc = self.weight_map.get(name)
        if alloc is None:
            return
        self._pin_state.pop(name, None)
        if self._ws_orch and force:
            if self._ws_close(alloc, retire=True, sync=False):
                return
        try:
            nbytes = alloc.size
            self.ctx.update_tensor_flags_range(
                alloc, 0, nbytes,
                memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
            )
            if force:
                try:
                    if os.environ.get("MEMX_OP_PURGE", "0") == "1":
                        self.ctx.purge(alloc)
                    else:
                        self.ctx.force_compress_range(alloc, 0, nbytes)
                except Exception:
                    pass
            elif os.environ.get("MEMX_COOL_WINDOW", "0") == "1":
                window = memx.weight_window(
                    managed=(0, nbytes),
                    hot=(0, 0),
                    prefetch=(0, 0),
                )
                self.ctx.update_weight_window(alloc, window)
        except Exception:
            pass

    def _align_range(self, offset, length, nbytes, page=None):
        if page is None:
            page = self._pin_page
        if length <= 0 or offset >= nbytes:
            return 0, 0
        if offset + length > nbytes:
            length = nbytes - offset
        off = (offset // page) * page
        end = ((offset + length + page - 1) // page) * page
        if end > nbytes:
            end = nbytes
        if end <= off:
            return 0, 0
        return off, end - off

    def _hot_range(self, alloc, off, ln, prefetch=True):
        if ln <= 0:
            return
        self.ctx.update_tensor_flags_range(
            alloc, off, ln,
            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_HOT,
        )
        if prefetch:
            try:
                page = self._pin_page
                cap_pages = int(os.environ.get("MEMX_PIN_PREFETCH_PAGES", "48"))
                if cap_pages < 1:
                    cap_pages = 1
                if ln >= page * 8 and cap_pages < 24:
                    cap_pages = 24
                cap = min(ln, page * cap_pages)
                if cap > 0:
                    self.ctx.prefetch_range(alloc, off, cap)
                if hasattr(self.ctx, "mark_access_range") and ln > 0:
                    try:
                        self.ctx.mark_access_range(alloc, off, min(ln, page * 4))
                    except Exception:
                        pass
            except Exception:
                pass

    def _cold_range(self, alloc, off, ln, force=False):
        if ln <= 0:
            return
        self.ctx.update_tensor_flags_range(
            alloc, off, ln,
            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
        )
        if force:
            try:
                if os.environ.get("MEMX_COLD_ASYNC_SEAL", "1") == "1" and hasattr(self.ctx, "seal_range_async"):
                    self.ctx.seal_range_async(alloc, off, ln)
                elif hasattr(self.ctx, "seal_range"):
                    self.ctx.seal_range(alloc, off, ln)
                else:
                    self.ctx.force_compress_range(alloc, off, ln)
            except Exception:
                try:
                    self.ctx.force_compress_range(alloc, off, ln)
                except Exception:
                    pass

    def prefetch_weight_range(self, name, offset, length):
        alloc = self.weight_map.get(name)
        if alloc is None or length <= 0:
            return
        nbytes = alloc.size
        off, ln = self._align_range(offset, length, nbytes)
        if ln <= 0:
            return
        try:
            page = self._pin_page
            cap_pages = int(os.environ.get("MEMX_BLOCK_PREFETCH_PAGES", "8"))
            if cap_pages < 1:
                cap_pages = 1
            ln = min(ln, page * cap_pages)
            if ln > 0:
                if self._ws_orch and self._ws_prefetch(alloc, off, ln):
                    return
                self.ctx.prefetch_range(alloc, off, ln)
        except Exception:
            pass

    def pin_weight_range(self, name, offset, length, prefetch=True):
        alloc = self.weight_map.get(name)
        if alloc is None or length <= 0:
            return
        nbytes = alloc.size
        off, ln = self._align_range(offset, length, nbytes)
        if ln <= 0:
            return
        end = off + ln
        if self._ws_orch:
            prev = self._pin_state.get(name)
            if prev is not None and prev[0] == off and prev[1] == end:
                return
            if prev is not None and off >= prev[0] and end <= prev[1]:
                return
            if self._ws_advance(alloc, off, ln, prefetch_next=0, mark=False):
                self._pin_state[name] = (off, end)
                return
        prev = self._pin_state.get(name)
        try:
            if prev is None:
                self._hot_range(alloc, off, ln, prefetch=prefetch)
                self._pin_state[name] = (off, end)
                return
            p0, p1 = prev
            if off >= p0 and end <= p1:
                return
            if end > p0 and off < p1:
                trail_force = os.environ.get("MEMX_PIN_TRAIL_FORCE", "0") != "0"
                if off < p0:
                    self._hot_range(alloc, off, p0 - off, prefetch=prefetch)
                if end > p1:
                    self._hot_range(alloc, p1, end - p1, prefetch=prefetch)
                if off > p0:
                    self._cold_range(alloc, p0, off - p0, force=trail_force)
                if end < p1:
                    self._cold_range(alloc, end, p1 - end, force=False)
                self._pin_state[name] = (off, end)
                return
            if end <= p0 or off >= p1:
                trail_force = os.environ.get("MEMX_PIN_TRAIL_FORCE", "0") != "0"
                self._cold_range(alloc, p0, p1 - p0, force=trail_force)
                self._hot_range(alloc, off, ln, prefetch=prefetch)
                self._pin_state[name] = (off, end)
                return
            self._hot_range(alloc, off, ln, prefetch=prefetch)
            self._pin_state[name] = (off, end)
        except Exception:
            pass

    def release_weight_range(self, name, offset, length, force=True):
        alloc = self.weight_map.get(name)
        if alloc is None or length <= 0:
            return
        nbytes = alloc.size
        off, ln = self._align_range(offset, length, nbytes)
        if ln <= 0:
            return
        end = off + ln
        prev = self._pin_state.get(name)
        try:
            if prev is not None:
                p0, p1 = prev
                if off >= p0 and end <= p1:
                    trail_force = force or (os.environ.get("MEMX_PIN_TRAIL_FORCE", "0") != "0")
                    if off > p0:
                        self._cold_range(alloc, p0, off - p0, force=trail_force)
                    keep0 = end
                    keep1 = p1
                    if keep0 < keep1:
                        self._pin_state[name] = (keep0, keep1)
                    else:
                        self._pin_state.pop(name, None)
                    self._cold_range(alloc, off, ln, force=trail_force)
                    return
                if end <= p0 or off >= p1:
                    self._cold_range(alloc, off, ln, force=force)
                    return
                self._cold_range(alloc, off, ln, force=force)
                n0 = max(p0, end)
                n1 = p1
                if n0 < n1:
                    self._pin_state[name] = (n0, n1)
                else:
                    self._pin_state.pop(name, None)
                return
            self._cold_range(alloc, off, ln, force=force)
        except Exception:
            pass

    def clear_pin_state(self, name=None):
        if name is None:
            self._pin_state.clear()
            return
        self._pin_state.pop(name, None)

    def begin_infer_epoch(self):
        if not self._ws_orch:
            return
        try:
            self.ctx.begin_epoch(memx.MEMX_EPOCH_INFER, self._hot_budget)
            self._epoch_phase = memx.MEMX_EPOCH_INFER
        except Exception:
            self._ws_orch = False

    def begin_final_epoch(self):
        if not self._ws_orch:
            return
        try:
            self.ctx.begin_epoch(memx.MEMX_EPOCH_FINAL, 0)
            self._epoch_phase = memx.MEMX_EPOCH_FINAL
        except Exception:
            pass

    def end_infer_epoch(self, seal=False):
        if not self._ws_orch:
            return
        try:
            self.ctx.end_epoch(1 if seal else 0)
            self._epoch_phase = 0
        except Exception:
            pass

    def _ws_advance(self, alloc, off, ln, prefetch_next=0, mark=False):
        if not self._ws_orch or alloc is None or ln <= 0:
            return False
        flags = memx.MEMX_WS_FLAG_HOT | memx.MEMX_WS_FLAG_PREFETCH
        if mark:
            flags |= memx.MEMX_WS_FLAG_MARK_ACCESS
        if os.environ.get("MEMX_STREAM_TRAIL_SEAL", "0") == "1" or os.environ.get("MEMX_PIN_TRAIL_FORCE", "0") != "0":
            flags |= memx.MEMX_WS_FLAG_RETIRE
        try:
            self.ctx.ws_advance(alloc, off, ln, prefetch_next, flags)
            return True
        except Exception:
            return False

    def _ws_close(self, alloc, retire=True, sync=False):
        if not self._ws_orch or alloc is None:
            return False
        flags = 0
        if retire:
            flags |= memx.MEMX_WS_FLAG_RETIRE
        if sync:
            flags |= memx.MEMX_WS_FLAG_RETIRE_SYNC
        try:
            self.ctx.ws_close(alloc, flags)
            return True
        except Exception:
            return False

    def _ws_prefetch(self, alloc, off, ln):
        if not self._ws_orch or alloc is None or ln <= 0:
            return False
        try:
            it = memx.ws_intent(
                alloc.ptr, off, ln, prefetch_length=ln,
                flags=memx.MEMX_WS_FLAG_PREFETCH,
            )
            self.ctx.apply_ws([it])
            return True
        except Exception:
            return False


    def stream_begin(self, name):
        if name is None:
            return
        if self._stream_active and self._stream_active != name:
            self.stream_end(self._stream_active, force=False)
        self._stream_active = name
        self._stream_cursor = 0

    def stream_advance(self, name, offset, length, prefetch_next=0):
        alloc = self.weight_map.get(name)
        if alloc is None or length <= 0:
            return
        if self._stream_active != name:
            self.stream_begin(name)
        nbytes = alloc.size
        off, ln = self._align_range(offset, length, nbytes)
        if ln <= 0:
            return
        page = self._pin_page
        look = max(0, int(prefetch_next))
        if look > 0:
            end_req = min(nbytes, off + ln + look)
            end_req = ((end_req + page - 1) // page) * page
            if end_req > nbytes:
                end_req = nbytes
            win_ln = max(ln, end_req - off)
        else:
            win_ln = ln
        new_end = off + win_ln
        if self._ws_orch:
            prev = self._pin_state.get(name)
            if prev is not None and prev[0] == off and prev[1] == new_end:
                self._stream_cursor = off + ln
                return
            if prev is not None and off >= prev[0] and new_end <= prev[1] and look <= 0:
                self._stream_cursor = off + ln
                return
            pref = look if look > 0 else 0
            if self._ws_advance(alloc, off, ln, prefetch_next=pref, mark=False):
                self._pin_state[name] = (off, new_end)
                self._stream_cursor = off + ln
                return
        prev = self._pin_state.get(name)
        trail_seal = os.environ.get("MEMX_STREAM_TRAIL_SEAL", "0") == "1"
        try:
            if prev is None:
                self._hot_range(alloc, off, win_ln, prefetch=True)
                self._pin_state[name] = (off, new_end)
                self._stream_cursor = off + ln
                return
            p0, p1 = prev
            if off == p0 and new_end == p1:
                self._stream_cursor = off + ln
                return
            if off >= p0 and off < p1:
                if off > p0:
                    trail_async = os.environ.get("MEMX_STREAM_TRAIL_SEAL_ASYNC", "1") == "1"
                    if trail_seal and trail_async and hasattr(self.ctx, "seal_range_async"):
                        try:
                            self.ctx.seal_range_async(alloc, p0, off - p0)
                        except Exception:
                            self._cold_range(alloc, p0, off - p0, force=False)
                    elif trail_seal and hasattr(self.ctx, "seal_range"):
                        try:
                            self.ctx.seal_range(alloc, p0, off - p0)
                        except Exception:
                            self._cold_range(alloc, p0, off - p0, force=True)
                    else:
                        self._cold_range(alloc, p0, off - p0, force=False)
                if new_end > p1:
                    self._hot_range(alloc, p1, new_end - p1, prefetch=True)
                elif new_end < p1:
                    self._cold_range(alloc, new_end, p1 - new_end, force=False)
                self._pin_state[name] = (off, new_end)
                self._stream_cursor = off + ln
                return
            if new_end <= p0 or off >= p1:
                trail_async = os.environ.get("MEMX_STREAM_TRAIL_SEAL_ASYNC", "1") == "1"
                if trail_seal and trail_async and hasattr(self.ctx, "seal_range_async"):
                    try:
                        self.ctx.seal_range_async(alloc, p0, p1 - p0)
                    except Exception:
                        self._cold_range(alloc, p0, p1 - p0, force=False)
                elif trail_seal and hasattr(self.ctx, "seal_range"):
                    try:
                        self.ctx.seal_range(alloc, p0, p1 - p0)
                    except Exception:
                        self._cold_range(alloc, p0, p1 - p0, force=True)
                else:
                    self._cold_range(alloc, p0, p1 - p0, force=False)
                self._hot_range(alloc, off, win_ln, prefetch=True)
                self._pin_state[name] = (off, new_end)
                self._stream_cursor = off + ln
                return
            self._hot_range(alloc, off, win_ln, prefetch=True)
            self._pin_state[name] = (off, new_end)
            self._stream_cursor = off + ln
        except Exception:
            pass


    def stream_end(self, name=None, force=False):
        if name is None:
            name = self._stream_active
        if name is None:
            return
        alloc = self.weight_map.get(name)
        prev = self._pin_state.pop(name, None)
        self._stream_active = None
        self._stream_cursor = 0
        if alloc is None:
            return
        if self._ws_orch:
            end_seal = force or (os.environ.get("MEMX_STREAM_END_SEAL", "0") == "1")
            if self._ws_close(alloc, retire=end_seal or force, sync=False):
                return
        try:
            if prev is not None:
                p0, p1 = prev
                end_seal = force or (os.environ.get("MEMX_STREAM_END_SEAL", "0") == "1")
                trail_async = os.environ.get("MEMX_STREAM_TRAIL_SEAL_ASYNC", "1") == "1"
                if end_seal and trail_async and hasattr(self.ctx, "seal_range_async"):
                    try:
                        self.ctx.seal_range_async(alloc, p0, p1 - p0)
                    except Exception:
                        self._cold_range(alloc, p0, p1 - p0, force=True)
                elif end_seal and hasattr(self.ctx, "seal_range"):
                    try:
                        self.ctx.seal_range(alloc, p0, p1 - p0)
                    except Exception:
                        self._cold_range(alloc, p0, p1 - p0, force=True)
                else:
                    self._cold_range(alloc, p0, p1 - p0, force=False)
            elif force:
                trail_async = os.environ.get("MEMX_STREAM_TRAIL_SEAL_ASYNC", "1") == "1"
                if trail_async and hasattr(self.ctx, "seal_range_async"):
                    try:
                        self.ctx.seal_range_async(alloc, 0, alloc.size)
                    except Exception:
                        self.cool_weight(name, force=True)
                elif hasattr(self.ctx, "seal_range"):
                    try:
                        self.ctx.seal_range(alloc, 0, alloc.size)
                    except Exception:
                        self.cool_weight(name, force=True)
                else:
                    self.cool_weight(name, force=True)
            else:
                self.cool_weight(name, force=False)
        except Exception:
            pass

    def final_seal(self, passes=2):
        self._pin_state.clear()
        self._stream_active = None
        self._stream_cursor = 0
        self.begin_final_epoch()
        if hasattr(self.ctx, "seal_flush"):
            try:
                self.ctx.seal_flush()
            except Exception:
                pass
        seal_async = hasattr(self.ctx, "seal_range_async")
        seal = hasattr(self.ctx, "seal_range")
        names = list(self.weight_map.items())
        names.sort(key=lambda kv: getattr(kv[1], "size", 0), reverse=True)
        for _pass in range(max(1, passes)):
            for i, (name, alloc) in enumerate(names):
                try:
                    if seal_async:
                        self.ctx.seal_range_async(alloc, 0, alloc.size)
                    elif seal:
                        self.ctx.seal_range(alloc, 0, alloc.size)
                    else:
                        self.ctx.update_tensor_flags_range(
                            alloc, 0, alloc.size,
                            memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                        )
                        self.ctx.force_compress_range(alloc, 0, alloc.size)
                except Exception:
                    pass
                if ((i + 1) & 15) == 0:
                    if seal_async and hasattr(self.ctx, "seal_flush"):
                        try:
                            self.ctx.seal_flush()
                        except Exception:
                            pass
                    try:
                        self.runtime.reclaim()
                    except Exception:
                        pass
            if self.kv is not None:
                try:
                    if seal_async:
                        self.ctx.seal_range_async(self.kv, 0, self.kv.size)
                    elif seal:
                        self.ctx.seal_range(self.kv, 0, self.kv.size)
                    else:
                        self.ctx.force_compress_range(self.kv, 0, self.kv.size)
                except Exception:
                    pass
            if seal_async and hasattr(self.ctx, "seal_flush"):
                try:
                    self.ctx.seal_flush()
                except Exception:
                    pass
            try:
                self.runtime.reclaim()
            except Exception:
                pass
            try:
                self.runtime.compact()
            except Exception:
                pass
        if hasattr(self.ctx, "seal_flush"):
            try:
                self.ctx.seal_flush()
            except Exception:
                pass


    def materialize_enabled(self):
        return (
            os.environ.get("MEMX_MATERIALIZE", "1") != "0"
            and hasattr(self.ctx, "materialize_range")
            and hasattr(self.ctx, "materialize_tile")
        )

    def materialize_flags_for(self, weight_dtype, out_dtype):
        flags = memx.MEMX_MATERIALIZE_KEEP_COMPRESSED | memx.MEMX_MATERIALIZE_ALLOW_RESIDENT
        if (
            weight_dtype == torch.bfloat16
            and out_dtype == torch.float16
            and os.environ.get("MEMX_MATERIALIZE_FP16", "1") != "0"
        ):
            flags |= memx.MEMX_MATERIALIZE_BF16_TO_FP16
        return flags

    def materialize_weight_range(self, name, offset, length, tensor, weight_dtype=None):
        alloc = self.weight_map.get(name)
        if alloc is None or tensor is None:
            return False
        if not self.materialize_enabled():
            return False
        try:
            if not tensor.is_contiguous():
                return False
            nbytes = int(tensor.numel()) * int(tensor.element_size())
            if nbytes < int(length):
                return False
            if int(tensor.element_size()) <= 0:
                return False
            flags = self.materialize_flags_for(weight_dtype, tensor.dtype)
            self.ctx.materialize_range(
                alloc, int(offset), int(length), int(tensor.data_ptr()), nbytes, flags=flags
            )
            return True
        except Exception:
            return False

    def materialize_weight_col_tile(self, name, rows, cols, col_start, col_n, elem, tensor, weight_dtype=None):
        alloc = self.weight_map.get(name)
        if alloc is None or tensor is None:
            return False
        if not self.materialize_enabled():
            return False
        try:
            if not tensor.is_contiguous():
                return False
            need = int(rows) * int(col_n) * int(elem)
            nbytes = int(tensor.numel()) * int(tensor.element_size())
            if nbytes < need or int(elem) != int(tensor.element_size()):
                return False
            flags = self.materialize_flags_for(weight_dtype, tensor.dtype)
            # source elem is always 2 for bf16/fp16 weights; dst stride uses tensor elem
            src_elem = 2 if weight_dtype in (torch.float16, torch.bfloat16) else int(elem)
            self.ctx.materialize_tile(
                alloc, int(rows), int(cols), int(src_elem), int(col_start), int(col_n),
                int(tensor.data_ptr()), nbytes, int(col_n) * int(elem),
                flags=flags,
            )
            return True
        except Exception:
            return False

    def materialize_prefetch_weight_range(self, name, offset, length):
        alloc = self.weight_map.get(name)
        if alloc is None or not self.materialize_enabled():
            return
        if not hasattr(self.ctx, "materialize_prefetch_range"):
            return
        try:
            self.ctx.materialize_prefetch_range(alloc, int(offset), int(length))
        except Exception:
            pass

    def materialize_prefetch_weight_col(self, name, rows, cols, col_start, col_n, elem):
        alloc = self.weight_map.get(name)
        if alloc is None or not self.materialize_enabled():
            return
        if not hasattr(self.ctx, "materialize_prefetch_range"):
            return
        try:
            # approximate page cover of the strip via first/last row endpoints
            if rows <= 0 or cols <= 0 or col_n <= 0 or elem <= 0:
                return
            row_bytes = cols * elem
            b0 = col_start * elem
            b1 = (rows - 1) * row_bytes + (col_start + col_n) * elem
            if b1 <= b0:
                return
            self.ctx.materialize_prefetch_range(alloc, b0, b1 - b0)
        except Exception:
            pass

    def pin_weight_col_block(self, name, rows, cols, col_start, col_n, elem, prefetch=True):
        alloc = self.weight_map.get(name)
        if alloc is None or rows <= 0 or cols <= 0 or col_n <= 0 or elem <= 0:
            return
        if col_start >= cols:
            return
        if col_start + col_n > cols:
            col_n = cols - col_start
        page = self._pin_page
        nbytes = alloc.size
        row_bytes = cols * elem
        strip = col_n * elem
        col_off = col_start * elem
        if strip <= 0 or row_bytes <= 0:
            return
        ranges = []
        cur0 = -1
        cur1 = -1
        for r in range(rows):
            b = r * row_bytes + col_off
            e = b + strip
            if b >= nbytes:
                break
            if e > nbytes:
                e = nbytes
            p0 = (b // page) * page
            p1 = ((e + page - 1) // page) * page
            if p1 > nbytes:
                p1 = nbytes
            if p1 <= p0:
                continue
            if cur0 < 0:
                cur0, cur1 = p0, p1
            elif p0 <= cur1:
                if p1 > cur1:
                    cur1 = p1
            else:
                ranges.append((cur0, cur1 - cur0))
                cur0, cur1 = p0, p1
        if cur0 >= 0:
            ranges.append((cur0, cur1 - cur0))
        if not ranges:
            return
        if len(ranges) == 1 and ranges[0][1] >= nbytes - (nbytes % page or page):
            self.pin_weight_range(name, 0, nbytes, prefetch=prefetch)
            return
        cover0 = ranges[0][0]
        cover1 = ranges[-1][0] + ranges[-1][1]
        density = sum(ln for _, ln in ranges) / max(cover1 - cover0, 1)
        if density >= 0.75 or len(ranges) > 96:
            self.pin_weight_range(name, cover0, cover1 - cover0, prefetch=prefetch)
            return
        prev = self._pin_state.get(name)
        if self._ws_orch and hasattr(self.ctx, "ws_tile"):
            try:
                retire0 = 0
                retire_n = 0
                if prev is not None and (prev[0] != cover0 or prev[1] != cover1):
                    # best-effort: retire previous cover window via col geometry when possible
                    pass
                self.ctx.ws_tile(
                    alloc,
                    rows=rows,
                    cols=cols,
                    elem_size=elem,
                    col_start=col_start,
                    col_count=col_n,
                    prefetch_cols=0,
                    flags=memx.MEMX_WS_FLAG_HOT | (memx.MEMX_WS_FLAG_PREFETCH if prefetch else 0) | memx.MEMX_WS_FLAG_MARK_ACCESS,
                )
                self._pin_state[name] = (cover0, cover1)
                return
            except Exception:
                pass
        if self._ws_orch and hasattr(self.ctx, 'apply_ws'):
            try:
                intents = []
                if prev is not None and (prev[0] != cover0 or prev[1] != cover1):
                    it = memx.ws_intent(alloc.ptr, prev[0], prev[1]-prev[0], flags=memx.MEMX_WS_FLAG_RETIRE if os.environ.get('MEMX_PIN_TRAIL_FORCE','0')!='0' else 0)
                    # cold only: no HOT, no PREFETCH => no-op unless RETIRE. Force cold via retire=false path:
                    pass
                # promote cover window as one advance for better track coalescing when dense enough
                if density >= 0.55 or len(ranges) <= 4:
                    if self._ws_advance(alloc, cover0, cover1-cover0, prefetch_next=0, mark=False):
                        self._pin_state[name] = (cover0, cover1)
                        return
                intents = []
                for off, ln in ranges:
                    intents.append(memx.ws_intent(
                        alloc.ptr, off, ln, prefetch_length=0,
                        flags=memx.MEMX_WS_FLAG_HOT | (memx.MEMX_WS_FLAG_PREFETCH if prefetch else 0),
                    ))
                if intents:
                    self.ctx.apply_ws(intents)
                    self._pin_state[name] = (cover0, cover1)
                    return
            except Exception:
                pass
        if prev is not None and (prev[0] != cover0 or prev[1] != cover1):
            try:
                self._cold_range(alloc, prev[0], prev[1] - prev[0], force=False)
            except Exception:
                pass
        try:
            for off, ln in ranges:
                self._hot_range(alloc, off, ln, prefetch=prefetch)
            self._pin_state[name] = (cover0, cover1)
        except Exception:
            pass

    def prefetch_weight_col_block(self, name, rows, cols, col_start, col_n, elem):
        alloc = self.weight_map.get(name)
        if alloc is None or rows <= 0 or cols <= 0 or col_n <= 0 or elem <= 0:
            return
        if col_start >= cols:
            return
        if col_start + col_n > cols:
            col_n = cols - col_start
        page = self._pin_page
        nbytes = alloc.size
        row_bytes = cols * elem
        strip = col_n * elem
        col_off = col_start * elem
        if strip <= 0 or row_bytes <= 0:
            return
        max_pages = int(os.environ.get("MEMX_BLOCK_PREFETCH_PAGES", "12"))
        if max_pages < 1:
            max_pages = 1
        budget = page * max_pages
        used = 0
        try:
            for r in range(rows):
                if used >= budget:
                    break
                b = r * row_bytes + col_off
                e = b + strip
                if b >= nbytes:
                    break
                if e > nbytes:
                    e = nbytes
                p0 = (b // page) * page
                p1 = ((e + page - 1) // page) * page
                if p1 > nbytes:
                    p1 = nbytes
                ln = p1 - p0
                if ln <= 0:
                    continue
                if used + ln > budget:
                    ln = budget - used
                if ln <= 0:
                    break
                self.ctx.prefetch_range(alloc, p0, ln)
                used += ln
        except Exception:
            pass

    def release_weight_col_block(self, name, force=False):
        prev = self._pin_state.pop(name, None)
        alloc = self.weight_map.get(name)
        if alloc is None or prev is None:
            return
        try:
            self._cold_range(alloc, prev[0], prev[1] - prev[0], force=force)
        except Exception:
            pass

    def warm_weights(self, names, full=True, prefetch_only=False):
        ordered = list(names)
        try:
            ordered.sort(key=lambda n: self.weight_map[n].size if n in self.weight_map else 0, reverse=True)
        except Exception:
            pass
        for name in ordered:
            self.warm_weight(name, full=full, prefetch_only=prefetch_only)

    def cool_weights(self, names, reclaim=False, force=False):
        ordered = list(names)
        try:
            ordered.sort(key=lambda n: self.weight_map[n].size if n in self.weight_map else 0, reverse=True)
        except Exception:
            pass
        for name in ordered:
            self.cool_weight(name, force=force)
        if reclaim:
            try:
                self.runtime.reclaim()
            except Exception:
                pass

    @staticmethod
    def layer_id_of(name):
        import re
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            return int(m.group(1))
        m = re.search(r"\.h\.(\d+)\.", name)
        if m:
            return int(m.group(1))
        return -1

    def stats_line(self, tag_name):
        st = self.runtime.stats()
        try:
            pressure = self.runtime.pressure().pool_pressure_percent
        except Exception:
            pressure = 0
        saved_mb = int(st.bytes_saved // (1024 * 1024))
        print(
            f"{tag_name} MemX: compressed={st.compressed_pages} faults={st.faults} "
            f"prefetch={st.prefetch_count}/{st.prefetch_hits} "
            f"kv_saved_pages={st.kv_cache_compressed_pages} weight_pages={st.weight_compressed_pages} "
            f"codec={st.tensor_codec_pages} split={st.tensor_split_pages} "
            f"delta={st.tensor_delta_split_pages} bitplane={st.tensor_bitplane_pages} "
            f"sparse={st.tensor_sparse_pages} exp={getattr(st, 'tensor_exp_pack_pages', 0)} saved_mb={saved_mb} "
            f"weights_hosted={len(self.weights)} hosted_mb={self.hosted_bytes//(1024*1024)} "
            f"pressure={pressure}%"
        )

    def close(self):
        for w in self.weights:
            try:
                w.free()
            except Exception:
                pass
        self.weights = []
        self.weight_map = {}
        if self.kv is not None:
            try:
                self.kv.free()
            except Exception:
                pass
            self.kv = None
        if self.ctx is not None:
            try:
                self.ctx.destroy()
            except Exception:
                pass
            self.ctx = None
        if self.runtime is not None:
            try:
                self.runtime.shutdown()
            except Exception:
                pass
            self.runtime = None


def show_rss():
    rss = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    mb = int(rss) // 1024
    print(f"{tag} RSS: {mb} MB")
    return mb

# Load config
with open(model_path / "config.json") as f:
    config = json.load(f)
print(f"{tag} Model: {config.get('model_type','?')} hidden={config.get('hidden_size','?')} layers={config.get('num_hidden_layers','?')}")

# Load weights into malloc memory
t0 = time.time()
index_file = model_path / "model.safetensors.index.json"
if index_file.exists():
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_files = set(weight_map.values())
    state_dict = {}
    for shard in sorted(shard_files):
        shard_path = model_path / shard
        print(f"{tag} Loading {shard} ...")
        partial = load_file(shard_path, device="cpu")
        state_dict.update({k: v.clone() for k, v in partial.items()})
        del partial
else:
    single = None
    for name in (
        "model.safetensors",
        "model.safetensors-00001-of-00001.safetensors",
        "pytorch_model.bin",
    ):
        cand = model_path / name
        if cand.exists():
            single = cand
            break
    if single is None:
        raise SystemExit(f"No weight file found under {model_path}")
    print(f"{tag} Loading {single.name} ...")
    if single.suffix == ".bin":
        raw = torch.load(single, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        state_dict = {k: v.clone() if hasattr(v, "clone") else v for k, v in raw.items()}
    else:
        raw = load_file(single, device="cpu")
        state_dict = {k: v.clone() for k, v in raw.items()}
    del raw

t1 = time.time()
n_params = sum(v.numel() for v in state_dict.values())
size_gb = sum(v.nelement() * v.element_size() for v in state_dict.values()) / 1024**3
print(f"{tag} Loaded {n_params/1e6:.0f}M params ({size_gb:.2f} GB) in {t1-t0:.1f}s")
rss_after_load = show_rss()

host = MemXHost.maybe_create() if 'MemXHost' in globals() else None
if host is not None:
    host.alloc_kv(tokens=int(os.environ.get("MEMX_KV_TOKENS", "256")))
    max_bytes = int(os.environ.get("MEMX_HOST_MAX_BYTES", str(1800 * 1024 * 1024)))
    n_w, hosted = host.host_state_dict(state_dict, max_bytes=max_bytes)
    print(
        f"{tag} MemX host: kv={host.kv.size // (1024*1024)}MB weights={n_w} "
        f"hosted_mb={hosted//(1024*1024)} replaced_mb={host.released_bytes//(1024*1024)}"
    )
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    host.cool_all_weights(force=os.environ.get("MEMX_POST_HOST_FORCE", "1") != "0", purge=False)
    try:
        host.runtime.reclaim()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    host.stats_line(tag)

wait_s=float(os.environ.get("MEMX_WAIT_S","15"))
print(f"{tag} Waiting {wait_s:.0f}s for compressor...")
time.sleep(wait_s)
if host is not None:
    host.cool_all_weights(force=os.environ.get("MEMX_POST_HOST_FORCE", "1") != "0", purge=os.environ.get("MEMX_POST_HOST_PURGE", "0") == "1")
    try:
        host.runtime.reclaim()
    except Exception:
        pass
    try:
        host.runtime.compact()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    host.stats_line(tag)
rss_after_compress = show_rss()

print(f"\n{tag} Running forward pass on all layers...")
t_infer0 = time.time()
torch.manual_seed(42)
embed_keys = [k for k in state_dict.keys() if "embed_tokens" in k.lower() or k.endswith("embed_tokens.weight")]
if not embed_keys:
    embed_keys = [k for k in state_dict.keys() if "embed" in k.lower() and k.endswith(".weight")]
embed_key = embed_keys[0]
if host is not None:
    host.warm_weight(embed_key, hot_frac=float(os.environ.get("MEMX_EMBED_HOT_FRAC", "0.02")), full=False)
embed_w = state_dict[embed_key]
print(f"{tag} embed={embed_key} shape={tuple(embed_w.shape)} dtype={embed_w.dtype}")

input_ids = torch.tensor([[1, 15043, 29892, 590, 1024, 338]])
x = torch.nn.functional.embedding(input_ids, embed_w)
if x.dtype != torch.float16:
    x = x.half()
if host is not None:
    host.cool_weight(embed_key, force=os.environ.get("MEMX_OP_FORCE", "1") != "0")

layer_prefixes = (
    "model.language_model.layers.",
    "model.layers.",
    "model.model.layers.",
    "transformer.h.",
)
layer_keys = []
for pfx in layer_prefixes:
    cand = [k for k in state_dict.keys() if any(k.startswith(f"{pfx}{i}.") for i in range(3))]
    if cand:
        layer_keys = sorted(cand)
        print(f"{tag} using layer prefix {pfx} n={len(layer_keys)}")
        break
if not layer_keys:
    layer_keys = sorted([k for k in state_dict.keys() if ".layers.0." in k or ".layers.1." in k or ".layers.2." in k])
    print(f"{tag} fallback layer keys n={len(layer_keys)}")

outputs = []
running_sum = 0.0
token_step = 0
matmul_n = 0
if host is not None:
    host.cool_all_weights()

from collections import OrderedDict
layer_groups = OrderedDict()
orphan_keys = []
for key in layer_keys:
    lid = host.layer_id_of(key) if host is not None else MemXHost.layer_id_of(key)
    if lid < 0:
        orphan_keys.append(key)
    else:
        layer_groups.setdefault(lid, []).append(key)


_matmul_row_buf = {}
_matmul_col_buf = {}

def matmul_weight(x, w, host=None, name=None):
    if w is None:
        return None
    x_in = x if x.dtype == torch.float16 else x.half()
    block = os.environ.get("MEMX_BLOCK_WS", "1") != "0"
    stream = os.environ.get("MEMX_STREAM_WS", "1") != "0"
    force_cool = os.environ.get("MEMX_OP_FORCE_COOL", "0") == "1"
    look = int(os.environ.get("MEMX_BLOCK_PREFETCH", "1"))
    if w.dtype == torch.float16:
        if host is not None and name is not None and block:
            try:
                host.warm_weight(name, full=False, hot_frac=float(os.environ.get("MEMX_FP16_HOT_FRAC", "0.25")))
            except Exception:
                pass
        if w.shape[1] == x_in.shape[-1]:
            out = torch.nn.functional.linear(x_in, w)
        else:
            out = torch.nn.functional.linear(x_in, w.t())
        if host is not None and name is not None:
            try:
                host.cool_weight(name, force=force_cool)
            except Exception:
                pass
        return out
    chunk = int(os.environ.get("MEMX_MATMUL_CHUNK", "0"))
    if chunk <= 0:
        if host is not None and hasattr(host, "materialize_enabled") and host.materialize_enabled():
            chunk = int(os.environ.get("MEMX_MATMUL_CHUNK_MAT", "512"))
        else:
            chunk = 384
    if chunk < 32:
        chunk = 32
    elem = w.element_size()
    if w.shape[1] == x_in.shape[-1]:
        rows = w.shape[0]
        row_bytes = w.shape[1] * elem
        if os.environ.get("MEMX_ADAPTIVE_CHUNK", "1") != "0":
            if rows >= 4096 and chunk < 512:
                chunk = 512
            elif rows <= 1024 and chunk > 256:
                chunk = 256
        outs = []
        key = ("row_fp16", int(w.shape[1]))
        buf = _matmul_row_buf.get(key)
        if host is not None and name is not None and block and stream:
            host.stream_begin(name)
        for i in range(0, rows, chunk):
            n = min(chunk, rows - i)
            off = i * row_bytes
            ln = n * row_bytes
            if host is not None and name is not None and block:
                use_mat = host.materialize_enabled() and os.environ.get("MEMX_MATERIALIZE_SKIP_PIN", "1") != "0"
                if stream and not use_mat:
                    pref = 0
                    if look > 0 and i + chunk < rows:
                        pref = min(chunk * max(look, 2), rows - (i + chunk)) * row_bytes
                    host.stream_advance(name, off, ln, prefetch_next=pref)
                elif not use_mat:
                    host.pin_weight_range(name, off, ln, prefetch=True)
            if buf is None or buf.shape[0] < n or buf.shape[1] != w.shape[1] or buf.dtype != torch.float16:
                buf = torch.empty((chunk, w.shape[1]), dtype=torch.float16)
                _matmul_row_buf[key] = buf
            dst = buf[:n]
            filled = False
            if host is not None and name is not None and block:
                filled = host.materialize_weight_range(name, off, ln, dst, weight_dtype=w.dtype)
            if not filled:
                if w.dtype == torch.bfloat16:
                    dst.copy_(w[i:i + n].half())
                else:
                    dst.copy_(w[i:i + n])
            pref_th = None
            if host is not None and name is not None and block and look > 0:
                use_mat = host.materialize_enabled() and os.environ.get("MEMX_MATERIALIZE_SKIP_PIN", "1") != "0"
                ni = i + chunk
                if ni < rows:
                    nn = min(chunk, rows - ni)
                    if use_mat:
                        import threading
                        pref_th = threading.Thread(
                            target=host.materialize_prefetch_weight_range,
                            args=(name, ni * row_bytes, nn * row_bytes),
                            daemon=True,
                        )
                        pref_th.start()
                    elif stream:
                        try:
                            host.prefetch_weight_range(name, ni * row_bytes, nn * row_bytes)
                        except Exception:
                            pass
            outs.append(torch.nn.functional.linear(x_in, dst))
            if pref_th is not None:
                try:
                    pref_th.join(timeout=0.05)
                except Exception:
                    pass
            if host is not None and name is not None and block and not stream:
                host.release_weight_range(name, off, ln, force=False)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        if host is not None and name is not None:
            try:
                if block and stream:
                    host.stream_end(name, force=force_cool)
                else:
                    host.cool_weight(name, force=force_cool)
            except Exception:
                pass
        return out
    cols = w.shape[1]
    rows = w.shape[0]
    if os.environ.get("MEMX_ADAPTIVE_CHUNK", "1") != "0":
        if cols >= 4096 and chunk < 512:
            chunk = 512
        elif cols <= 1024 and chunk > 256:
            chunk = 256
    outs = []
    key = "col"
    buf = _matmul_col_buf.get(key)
    strip = os.environ.get("MEMX_COL_STRIP", "1") != "0"
    if host is not None and name is not None and block and stream and not strip:
        host.stream_begin(name)
    for i in range(0, cols, chunk):
        n = min(chunk, cols - i)
        if host is not None and name is not None and block:
            if strip:
                use_mat = host.materialize_enabled() and os.environ.get("MEMX_MATERIALIZE_SKIP_PIN", "1") != "0"
                if not use_mat:
                    host.pin_weight_col_block(name, rows, cols, i, n, elem, prefetch=True)
                    ni = i + chunk
                    if look > 0 and ni < cols:
                        nn = min(chunk, cols - ni)
                        try:
                            host.prefetch_weight_col_block(name, rows, cols, ni, nn, elem)
                        except Exception:
                            pass
            elif stream:
                if i == 0:
                    hf = min(0.08, max(0.02, float(n) / max(cols, 1)))
                    try:
                        host.warm_weight(name, hot_frac=hf, full=False)
                    except Exception:
                        pass
            elif i == 0:
                hf = min(0.08, max(0.02, float(chunk) / max(cols, 1)))
                try:
                    host.warm_weight(name, hot_frac=hf, full=False)
                except Exception:
                    pass
        ckey = ("col_fp16", int(rows))
        buf = _matmul_col_buf.get(ckey)
        if buf is None or buf.shape[1] < n or buf.shape[0] != rows or buf.dtype != torch.float16:
            buf = torch.empty((rows, chunk), dtype=torch.float16)
            _matmul_col_buf[ckey] = buf
        tile = buf[:, :n]
        filled = False
        if host is not None and name is not None and block and strip:
            filled = host.materialize_weight_col_tile(
                name, rows, cols, i, n, tile.element_size(), tile, weight_dtype=w.dtype
            )
            if filled:
                try:
                    host.release_weight_col_block(name, force=False)
                except Exception:
                    pass
        if not filled:
            if w.dtype == torch.bfloat16:
                tile.copy_(w[:, i:i + n].half())
            else:
                tile.copy_(w[:, i:i + n])
            if host is not None and name is not None and block and strip:
                host.release_weight_col_block(name, force=False)
        # async ND warm next strip while this GEMM runs
        pref_th = None
        if host is not None and name is not None and block and strip and look > 0 and host.materialize_enabled():
            ni = i + chunk
            if ni < cols:
                nn = min(chunk, cols - ni)
                import threading
                pref_th = threading.Thread(
                    target=host.materialize_prefetch_weight_col,
                    args=(name, rows, cols, ni, nn, elem),
                    daemon=True,
                )
                pref_th.start()
        outs.append(torch.nn.functional.linear(x_in, tile.t()))
        if pref_th is not None:
            try:
                pref_th.join(timeout=0.05)
            except Exception:
                pass
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
    if host is not None and name is not None:
        try:
            if block and stream and not strip:
                host.stream_end(name, force=force_cool)
            else:
                host.cool_weight(name, force=force_cool)
        except Exception:
            pass
    return out

def matmul_eligible(w, x_last):
    if w is None or not hasattr(w, "dtype"):
        return False
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if w.dim() != 2:
        return False
    return w.shape[1] == x_last or w.shape[0] == x_last

group_items = list(layer_groups.items())
op_mode = os.environ.get("MEMX_OP_LEVEL_WS", "1") != "0"
prefetch_ops = int(os.environ.get("MEMX_OP_PREFETCH", "1"))
if host is not None:
    try:
        host.begin_infer_epoch()
    except Exception:
        pass
with torch.no_grad():
    if op_mode:
        ordered_keys = []
        for gi, (lid, keys) in enumerate(group_items):
            ordered_keys.extend(keys)
        ordered_keys.extend(orphan_keys)

        def next_eligible(start_idx, x_last):
            for j in range(start_idx, len(ordered_keys)):
                cand = ordered_keys[j]
                ww = state_dict.get(cand)
                if matmul_eligible(ww, x_last):
                    return j, cand
            return None, None

        oi = 0
        scan = 0
        while True:
            idx, key = next_eligible(scan, x.shape[-1])
            if key is None:
                break
            scan = idx + 1
            w = state_dict[key]
            block_ws = os.environ.get("MEMX_BLOCK_WS", "1") != "0"
            if host is not None and not block_ws:
                host.warm_weight(key, full=True)
                if prefetch_ops > 0:
                    pref_scan = scan
                    for _pj in range(prefetch_ops):
                        pidx, pkey = next_eligible(pref_scan, w.shape[0] if w.shape[1] == x.shape[-1] else w.shape[1])
                        if pkey is None:
                            pidx, pkey = next_eligible(pref_scan, x.shape[-1])
                        if pkey is None:
                            break
                        host.warm_weight(pkey, full=False, prefetch_only=True)
                        pref_scan = pidx + 1
            elif host is not None and block_ws and prefetch_ops > 0 and os.environ.get("MEMX_BLOCK_OP_PREFETCH", "1") != "0":
                pref_scan = scan
                npref = min(max(prefetch_ops, 1), int(os.environ.get("MEMX_OP_PREFETCH_DEPTH", "1")))
                for _pj in range(npref):
                    pidx, pkey = next_eligible(pref_scan, w.shape[0] if w.shape[1] == x.shape[-1] else w.shape[1])
                    if pkey is None:
                        pidx, pkey = next_eligible(pref_scan, x.shape[-1])
                    if pkey is None:
                        break
                    try:
                        host.warm_weight(pkey, full=False, prefetch_only=True)
                    except Exception:
                        pass
                    pref_scan = pidx + 1
            out = matmul_weight(x, w, host=host, name=key)
            outputs.append(out.shape)
            running_sum = running_sum + float(out.sum().item())
            x = out
            matmul_n += 1
            if host is not None:
                host.write_token(token_step % 128)
                token_step += 1
                if not block_ws:
                    host.cool_weight(key, force=os.environ.get("MEMX_OP_FORCE", "1") != "0")
                if (oi & 15) == 15:
                    try:
                        host.runtime.reclaim()
                    except Exception:
                        pass
            del out
            oi += 1
    else:
        for gi, (lid, keys) in enumerate(group_items):
            candidates = []
            for key in keys:
                w = state_dict.get(key)
                if w is not None and hasattr(w, "dtype") and w.dtype in (torch.float16, torch.bfloat16, torch.float32) and getattr(w, "dim", lambda: 0)() == 2:
                    candidates.append(key)
            if host is not None and candidates:
                host.warm_weights(candidates, full=True)
                if gi + 1 < len(group_items):
                    nxt = []
                    for nk in group_items[gi + 1][1]:
                        nw = state_dict.get(nk)
                        if nw is not None and hasattr(nw, "dtype") and nw.dtype in (torch.float16, torch.bfloat16, torch.float32) and getattr(nw, "dim", lambda: 0)() == 2:
                            nxt.append(nk)
                    if nxt:
                        host.warm_weights(nxt, full=False, prefetch_only=True)
            for key in keys:
                w = state_dict.get(key)
                if not matmul_eligible(w, x.shape[-1]):
                    continue
                out = matmul_weight(x, w, host=host, name=key)
                outputs.append(out.shape)
                running_sum = running_sum + float(out.sum().item())
                x = out
                matmul_n += 1
                if host is not None:
                    host.write_token(token_step % 128)
                    token_step += 1
                del out
            if host is not None and candidates:
                host.cool_weights(candidates, reclaim=True, force=os.environ.get("MEMX_OP_FORCE", "1") != "0")
        for key in orphan_keys:
            if host is not None:
                host.warm_weight(key, full=True)
            w = state_dict[key]
            if not matmul_eligible(w, x.shape[-1]):
                if host is not None:
                    host.cool_weight(key)
                continue
            out = matmul_weight(x, w, host=host, name=key)
            outputs.append(out.shape)
            running_sum = running_sum + float(out.sum().item())
            x = out
            matmul_n += 1
            if host is not None:
                host.write_token(token_step % 128)
                token_step += 1
                host.cool_weight(key)
            del out
print(f"{tag} matmul_ops={matmul_n} op_level_ws={int(op_mode)}")

if host is not None:
    host.advance()
    host.stats_line(tag)

output_sum = running_sum
output_shape = list(outputs)
print(f"{tag} Output sum: {output_sum:.6f}")
print(f"{tag} Output shapes: {output_shape}")
infer_wall = time.time() - t_infer0
print(f"{tag} Infer wall: {infer_wall:.3f}s matmul_ops={matmul_n}")
try:
    _matmul_row_buf.clear()
    _matmul_col_buf.clear()
except Exception:
    pass
try:
    import gc
    gc.collect()
except Exception:
    pass
rss_after_infer = show_rss()

try:
    del x
except Exception:
    pass
try:
    del embed_w
except Exception:
    pass
try:
    import gc
    gc.collect()
    gc.collect()
except Exception:
    pass

wait_s=float(os.environ.get("MEMX_WAIT_S","15"))
if host is not None:
    final_force = os.environ.get("MEMX_FINAL_FORCE", "1") != "0"
    final_purge = os.environ.get("MEMX_FINAL_PURGE", "1") != "0"
    if final_force or final_purge:
        try:
            try:
                host.end_infer_epoch(seal=False)
            except Exception:
                pass
            host.final_seal(passes=1 if not final_purge else 2)
        except Exception:
            host.cool_all_weights(force=final_force, purge=final_purge)
    else:
        host.cool_all_weights(force=False, purge=False)
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
print(f"\n{tag} Waiting {wait_s:.0f}s for recompression...")
time.sleep(wait_s)
if host is not None:
    final_force = os.environ.get("MEMX_FINAL_FORCE", "1") != "0"
    final_purge = os.environ.get("MEMX_FINAL_PURGE", "1") != "0"
    passes = int(os.environ.get("MEMX_FINAL_SEAL_PASSES", "2"))
    if passes < 1:
        passes = 1
    if final_force or final_purge:
        try:
            try:
                host.end_infer_epoch(seal=False)
            except Exception:
                pass
            host.final_seal(passes=passes + (1 if final_purge else 0))
        except Exception:
            for _pass in range(passes):
                host.cool_all_weights(force=final_force, purge=final_purge)
                try:
                    host.runtime.reclaim()
                except Exception:
                    pass
                try:
                    host.runtime.compact()
                except Exception:
                    pass
    else:
        host.cool_all_weights(force=False, purge=False)
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    time.sleep(float(os.environ.get("MEMX_FINAL_SETTLE_S", "1.5")))
    try:
        host.runtime.compact()
    except Exception:
        try:
            host.runtime.reclaim()
        except Exception:
            pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    host.stats_line(tag)
rss_final = show_rss()

# Summary
print(f"\n{'='*60}")
print(f"  SUMMARY ({tag})")
print(f"{'='*60}")
print(f"  Model size:       {size_gb:.2f} GB")
print(f"  RSS after load:   {rss_after_load} MB")
print(f"  RSS after 15s:    {rss_after_compress} MB  (compressor)")
print(f"  RSS after infer:  {rss_after_infer} MB")
if "infer_wall" in globals():
    print(f"  Infer wall:       {infer_wall:.3f}s")
print(f"  RSS final:        {rss_final} MB  (recompressed)")
print(f"  Output sum:       {output_sum:.6f}")
baseline_rss = max(rss_after_load, (host.hosted_bytes if host is not None else 0) // (1024 * 1024) + 64)
print(f"  Saved:            {baseline_rss - rss_final} MB ({(baseline_rss - rss_final)*100//max(baseline_rss,1)}%)")
if host is not None:
    print(f"  Hosted weights:   {len(host.weights)} tensors / {host.hosted_bytes//(1024*1024)} MB")
    print(f"  Replaced tensors: {host.released_bytes//(1024*1024)} MB")
print(f"{'='*60}")

if host is not None:
    host.close()

alive=float(os.environ.get("MEMX_ALIVE_S","10"))
print(f"\n{tag} Keeping process alive {alive:.0f}s for dashboard...")
time.sleep(alive)
