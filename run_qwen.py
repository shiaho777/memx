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
torch.set_num_threads(int(os.environ.get("MEMX_TORCH_THREADS", "1")))
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

def _early_memx_env():
    defaults = {
        "MEMX_POOL_SPILL": "1",
        "MEMX_POOL_DETACH": "1",
        "MEMX_POOL_GHOST": "0",
        "MEMX_POOL_GHOST_FINAL": "1",
        "MEMX_POOL_MIRROR": "0",
        "MEMX_POOL_VAULT_NATIVE": "1",
        "MEMX_VAULT_PROBE": "1",
        "MEMX_FINAL_DISSOLVE": "1",
        "MEMX_VAULT_CACHE": "1",
        "MEMX_VAULT_CACHE_MB": "192",
        "MEMX_VAULT_CACHE_INFER_MB": "32",
        "MEMX_VAULT_AVCS": "1",
        "MEMX_SOV_TCA": "1",
        "MEMX_TCA_WARP": "1",
        "MEMX_TCA_NO_VAULT": "1",
        "MEMX_TCA_VAULT_DRIFT": "1",
        "MEMX_TCA_STICKY": "1",
        "MEMX_TCA_SLOTS": "1",
        "MEMX_TCA_BUDGET_MB": "64",
        "MEMX_TCA_DIFF": "1",
        "MEMX_TCA_CHRONOS": "0",
        "MEMX_STICKY_NO_CHRONOS": "1",
        "MEMX_POST_INFER_FOLD": "1",
        "MEMX_SOV_CHRONOS_PAGES": "96",
        "MEMX_VAULT_WINDOW": "0",
        "MEMX_VAULT_WINDOW_MB": "8",
        "MEMX_VAULT_WBUF": "0",
        "MEMX_VAULT_WBUF_KB": "1024",
        "MEMX_SOVEREIGN": "1",
        "MEMX_SOVEREIGN_HARD": "0",
        "MEMX_SOV_WARM": "1",
        "MEMX_SOV_STREAM": "1",
        "MEMX_SOV_CRW": "1",
        "MEMX_SOV_CRW_DIRECT": "1",
        "MEMX_SOV_CHRONOS": "1",
        "MEMX_SOV_CHRONOS_ASYNC": "1",
        "MEMX_KILL_SRC_STORAGE": "1",
        "MEMX_FINAL_DISSOLVE_WEIGHTS": "1",
        "MEMX_FINAL_SOFT_PAGEOUT": "1",
        "MEMX_FINAL_SOFT_PAGEOUT_MB": "",
        "MEMX_FINAL_SOFT_PULSE": "1",
        "MEMX_WAIT_S": "5",
        "MEMX_FINAL_TARGET_MB": "18",
        "MEMX_FINAL_OS_PRESSURE": "0",
        "MEMX_FINAL_OS_PRESSURE_LEVEL": "warn",

        "MEMX_FINAL_FOREIGN_PASSES": "5",

        "MEMX_FINAL_SOFT_PULSES": "6",
        "MEMX_FINAL_RSS_SAMPLES": "6",
        "MEMX_POST_INFER_STRUCT": "1",
        "MEMX_POST_INFER_PHOENIX": "1",
        "MEMX_FINAL_PHOENIX": "1",
        "MEMX_FINAL_CHILD_RECLAIM": "auto",
        "MEMX_HARD_COMPACT": "0",
        "MEMX_SOFT_COMPACT": "0",
        "MEMX_CAPSULE_RELAY": "1",
        "MEMX_CAPSULE_VESSEL": "1",
        "MEMX_CAPSULE_VESSEL_PAGES": "16",
        "MEMX_CAPSULE_VESSEL_BATCH": "1",
        "MEMX_POOL_SPILL_KEEP": "1",
        "MEMX_CAPSULE_LITE": "1",
        "MEMX_CAPSULE_HOST_BIND": "1",
        "MEMX_NO_SELFTEST": "0",
    }
    for k, v in defaults.items():
        if os.environ.get(k, "") == "":
            os.environ[k] = v

_early_memx_env()

try:
    import memx_runtime as memx
except Exception:
    memx = None


def _kill_tensor_storage(tensor, hard=False):
    if tensor is None or not hasattr(tensor, "numel"):
        return False
    try:
        if getattr(getattr(tensor, "device", None), "type", None) == "meta":
            return False
    except Exception:
        pass
    try:
        st = tensor.untyped_storage()
        if st is None:
            return False
        try:
            nbytes = int(st.nbytes())
        except Exception:
            try:
                nbytes = int(st.size()) * int(st.element_size())
            except Exception:
                nbytes = 0
        if nbytes <= 0:
            return False
        try:
            addr = int(st.data_ptr())
        except Exception:
            addr = 0
        if addr:
            try:
                import ctypes
                libc = ctypes.CDLL("libSystem.B.dylib")
                libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                libc.madvise.restype = ctypes.c_int
                page = 16384
                a0 = addr & ~(page - 1)
                a1 = (addr + nbytes + page - 1) & ~(page - 1)
                if a1 > a0:
                    n = a1 - a0
                    try:
                        libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(n), 7)
                    except Exception:
                        pass
                    try:
                        libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(n), 10)
                    except Exception:
                        pass
            except Exception:
                pass
        if hard or os.environ.get("MEMX_KILL_SRC_HARD", "1") not in ("0", "false", "False"):
            try:
                st.resize_(0)
                return True
            except Exception:
                pass
            try:
                tensor.resize_(0)
                return True
            except Exception:
                pass
        return True
    except Exception:
        return False

def _defer_kill_storage(tensor, bucket):
    if bucket is None or tensor is None:
        return
    try:
        if getattr(getattr(tensor, "device", None), "type", None) == "meta":
            return
        st = tensor.untyped_storage()
        if st is None:
            return
        try:
            key = int(st.data_ptr())
        except Exception:
            key = id(st)
        if key and key not in bucket:
            bucket[key] = st
    except Exception:
        pass

def _flush_kill_bucket(bucket):
    if not bucket:
        return 0
    n = 0
    import ctypes
    try:
        libc = ctypes.CDLL("libSystem.B.dylib")
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
    except Exception:
        libc = None
    for st in list(bucket.values()):
        try:
            try:
                nbytes = int(st.nbytes())
            except Exception:
                try:
                    nbytes = int(st.size()) * int(st.element_size())
                except Exception:
                    nbytes = 0
            if nbytes <= 0:
                continue
            try:
                addr = int(st.data_ptr())
            except Exception:
                addr = 0
            did = False
            if addr and libc is not None:
                page = 16384
                a0 = addr & ~(page - 1)
                a1 = (addr + nbytes + page - 1) & ~(page - 1)
                if a1 > a0:
                    span = a1 - a0
                    try:
                        libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(span), 7)
                    except Exception:
                        pass
                    try:
                        libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(span), 10)
                    except Exception:
                        pass
                    try:
                        libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(span), 4)
                        did = True
                    except Exception:
                        pass
            if os.environ.get("MEMX_KILL_SRC_HARD", "1") not in ("0", "false", "False"):
                try:
                    st.resize_(0)
                    did = True
                except Exception:
                    pass
            if did:
                n += 1
        except Exception:
            pass
    bucket.clear()
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    return n


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
tag = "[" + os.environ.get("MEMX_WORKLOAD_LABEL", os.environ.get("MEMX_TAG", "Baseline workload")) + "]"
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
        self._mat_err = None
        self.token_bytes = 16384
        self.written_tokens = 0
        self.hot_tokens = int(os.environ.get("MEMX_KV_HOT_TOKENS", "16"))
        self.prefetch_tokens = int(os.environ.get("MEMX_KV_PREFETCH_TOKENS", "8"))
        self.weight_hot_frac = float(os.environ.get("MEMX_WEIGHT_HOT_FRAC", "0.01"))
        self.hosted_bytes = 0
        self.released_bytes = 0
        self._kill_bucket = {}
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
            dst_addr = ctypes.addressof(buf)
            src_addr = int(src_u8.data_ptr())
            seal_chunk = int(os.environ.get("MEMX_HOST_SEAL_CHUNK", "0"))
            immediate = os.environ.get("MEMX_HOST_IMMEDIATE_SEAL", "1") != "0"
            if seal_chunk > 0:
                if seal_chunk < 16384:
                    seal_chunk = 16384
                seal_chunk = (seal_chunk + 16383) & ~16383
                off = 0
                while off < nbytes:
                    ln = nbytes - off
                    if ln > seal_chunk:
                        ln = seal_chunk
                    ctypes.memmove(dst_addr + off, src_addr + off, ln)
                    if immediate:
                        try:
                            if hasattr(self.ctx, "seal_range"):
                                self.ctx.seal_range(alloc, off, ln)
                            else:
                                self.ctx.force_compress_range(alloc, off, ln)
                        except Exception:
                            try:
                                self.ctx.force_compress_range(alloc, off, ln)
                            except Exception:
                                pass
                    off += ln
            else:
                ctypes.memmove(dst_addr, src_addr, nbytes)
            del src_u8
            self._seal_weight(alloc, nbytes)
            if immediate and seal_chunk <= 0:
                try:
                    if hasattr(self.ctx, "seal_range"):
                        self.ctx.seal_range(alloc, 0, nbytes)
                    else:
                        self.ctx.force_compress_range(alloc, 0, nbytes)
                except Exception:
                    try:
                        self.ctx.force_compress_range(alloc, 0, nbytes)
                    except Exception:
                        pass
            elif not immediate:
                try:
                    if hasattr(self.ctx, "seal_range"):
                        self.ctx.seal_range(alloc, 0, nbytes)
                    else:
                        self.ctx.force_compress_range(alloc, 0, nbytes)
                except Exception:
                    try:
                        self.ctx.force_compress_range(alloc, 0, nbytes)
                    except Exception:
                        pass
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
            use_meta = (
                self.materialize_enabled()
                and os.environ.get("MEMX_META_PLACEHOLDER", "1") != "0"
            )
            if use_meta:
                try:
                    hosted = torch.empty(shape, dtype=tensor.dtype, device="meta")
                except Exception:
                    hosted = None
            if hosted is None:
                try:
                    hosted = alloc.torch_tensor(tensor.dtype, shape)
                except Exception:
                    hosted = None
        self.weights.append(alloc)
        self.weight_map[name] = alloc
        self.hosted_bytes += nbytes
        if replace and os.environ.get("MEMX_KILL_SRC_STORAGE", "1") not in ("0", "false", "False"):
            try:
                bucket = getattr(self, "_kill_bucket", None)
                if bucket is None:
                    self._kill_bucket = {}
                    bucket = self._kill_bucket
                _defer_kill_storage(tensor, bucket)
                _defer_kill_storage(src_tensor, bucket)
            except Exception:
                pass
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
            if (n & 7) == 0:
                try:
                    if hasattr(self.ctx, "seal_flush"):
                        self.ctx.seal_flush()
                except Exception:
                    pass
                try:
                    self.runtime.reclaim()
                except Exception:
                    pass
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
        del items
        try:
            if hasattr(self.ctx, "seal_flush"):
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
        try:
            import gc
            gc.collect()
            gc.collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            _flush_kill_bucket(getattr(self, "_kill_bucket", None))
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
        mat = self.materialize_enabled() and os.environ.get("MEMX_MATERIALIZE_SKIP_PIN", "1") != "0"
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
                if mat and hasattr(self.ctx, "materialize_prefetch_range"):
                    self.ctx.materialize_prefetch_range(alloc, 0, pref)
                elif not mat:
                    self.ctx.prefetch_range(alloc, 0, pref)
            except Exception:
                pass
            return
        if mat:
            try:
                self.ctx.update_tensor_flags_range(
                    alloc, 0, nbytes,
                    memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                )
            except Exception:
                pass
            if hot > 0 and os.environ.get("MEMX_WARM_ND", "1") != "0":
                try:
                    cap_pages = int(os.environ.get("MEMX_WARM_PREFETCH_PAGES", "16"))
                    cap = min(hot, page * max(cap_pages, 1))
                    if hasattr(self.ctx, "materialize_prefetch_range"):
                        self.ctx.materialize_prefetch_range(alloc, 0, cap)
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
        if self._ws_orch and force and os.environ.get("MEMX_TIER_SEAL", "0") != "1":
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
                    elif hasattr(self.ctx, "seal_range_async") and os.environ.get("MEMX_TIER_SEAL", "0") == "1":
                        self.ctx.seal_range_async(alloc, 0, nbytes)
                    elif hasattr(self.ctx, "seal_range"):
                        self.ctx.seal_range(alloc, 0, nbytes)
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
        try:
            st = self.runtime.stats()
            res = int(getattr(st, "resident_pages", 0))
            comp = int(getattr(st, "compressed_pages", 0))
            if res == 0 and comp > 0 and os.environ.get("MEMX_FINAL_SKIP_IF_COLD", "1") != "0":
                if hasattr(self.ctx, "seal_flush"):
                    try:
                        self.ctx.seal_flush()
                    except Exception:
                        pass
                try:
                    if hasattr(self.runtime, "trim"):
                        self.runtime.trim(7)
                    else:
                        self.runtime.compact()
                        self.runtime.reclaim()
                except Exception:
                    pass
                try:
                    import gc
                    gc.collect()
                    gc.collect()
                except Exception:
                    pass
                return
        except Exception:
            pass
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
            if os.environ.get("MEMX_FINAL_SOFT_COMPACT", "0") != "0":
                try:
                    self.runtime.compact()
                except Exception:
                    pass
        if hasattr(self.ctx, "seal_flush"):
            try:
                self.ctx.seal_flush()
            except Exception:
                pass
        if os.environ.get("MEMX_FINAL_RECOMPRESS", "0") != "0":
            try:
                lv = int(os.environ.get("MEMX_FINAL_ZLIB_LEVEL", "9"))
            except Exception:
                lv = 6
            try:
                if hasattr(self.runtime, "recompress_begin"):
                    self.runtime.recompress_begin(lv)
                for name, alloc in names:
                    try:
                        if hasattr(self.ctx, "recompress_range"):
                            self.ctx.recompress_range(alloc, 0, alloc.size)
                        elif hasattr(self.ctx, "force_compress_range"):
                            self.ctx.force_compress_range(alloc, 0, alloc.size)
                    except Exception:
                        pass
                if hasattr(self.runtime, "recompress_end"):
                    self.runtime.recompress_end()
            except Exception:
                pass
        try:
            if hasattr(self.runtime, "trim"):
                self.runtime.trim(7)
            else:
                self.runtime.compact()
                self.runtime.reclaim()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            gc.collect()
        except Exception:
            pass
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
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
            self._mat_err = f"missing alloc/tensor for {name}"
            return False
        if not self.materialize_enabled():
            self._mat_err = "materialize disabled"
            return False
        try:
            if not tensor.is_contiguous():
                self._mat_err = f"non-contiguous {name}"
                return False
            nbytes = int(tensor.numel()) * int(tensor.element_size())
            if nbytes < int(length):
                self._mat_err = f"dst cap {nbytes}<{length} for {name}"
                return False
            if int(tensor.element_size()) <= 0:
                return False
            flags = self.materialize_flags_for(weight_dtype, tensor.dtype)
            self.ctx.materialize_range(
                alloc, int(offset), int(length), int(tensor.data_ptr()), nbytes, flags=flags
            )
            self._mat_err = None
            return True
        except OSError as e:
            self._mat_err = f"range {name} errno={getattr(e, 'errno', None)} {e}"
            if os.environ.get("MEMX_MAT_DEBUG", "0") != "0":
                print(f"{tag} materialize fail: {self._mat_err}")
            try:
                flags = self.materialize_flags_for(weight_dtype, tensor.dtype)
                flags = flags & ~memx.MEMX_MATERIALIZE_KEEP_COMPRESSED
                self.ctx.materialize_range(
                    alloc, int(offset), int(length), int(tensor.data_ptr()), nbytes, flags=flags
                )
                self._mat_err = None
                return True
            except Exception as e2:
                self._mat_err = f"range-retry {name} {e2}"
                return False
        except Exception as e:
            self._mat_err = f"range {name} {e}"
            return False

    def materialize_weight_col_tile(self, name, rows, cols, col_start, col_n, elem, tensor, weight_dtype=None):
        alloc = self.weight_map.get(name)
        if alloc is None or tensor is None:
            self._mat_err = f"missing alloc/tensor for {name}"
            return False
        if not self.materialize_enabled():
            self._mat_err = "materialize disabled"
            return False
        try:
            if int(elem) != int(tensor.element_size()):
                self._mat_err = f"elem mismatch {name} {elem} vs {tensor.element_size()}"
                return False
            if tensor.dim() != 2 or int(tensor.stride(-1)) != 1:
                if not tensor.is_contiguous():
                    self._mat_err = f"non-contiguous {name}"
                    return False
            row_stride = int(tensor.stride(0)) * int(tensor.element_size())
            dense_row = int(col_n) * int(elem)
            if row_stride < dense_row:
                self._mat_err = f"row_stride {row_stride}<{dense_row} for {name}"
                return False
            need = (int(rows) - 1) * row_stride + dense_row if int(rows) > 0 else 0
            try:
                st = tensor.untyped_storage()
                avail = int(st.size()) * int(st.element_size()) - int(tensor.storage_offset()) * int(tensor.element_size())
            except Exception:
                avail = int(tensor.numel()) * int(tensor.element_size())
            if avail < need:
                self._mat_err = f"tile cap {name} need={need} avail={avail}"
                return False
            flags = self.materialize_flags_for(weight_dtype, tensor.dtype)
            src_elem = 2 if weight_dtype in (torch.float16, torch.bfloat16) else int(elem)
            self.ctx.materialize_tile(
                alloc, int(rows), int(cols), int(src_elem), int(col_start), int(col_n),
                int(tensor.data_ptr()), avail, row_stride,
                flags=flags,
            )
            self._mat_err = None
            return True
        except OSError as e:
            self._mat_err = f"tile {name} c0={col_start} n={col_n} errno={getattr(e, 'errno', None)} {e}"
            if os.environ.get("MEMX_MAT_DEBUG", "0") != "0":
                print(f"{tag} materialize fail: {self._mat_err}")
            try:
                flags = self.materialize_flags_for(weight_dtype, tensor.dtype)
                flags = flags & ~memx.MEMX_MATERIALIZE_KEEP_COMPRESSED
                src_elem = 2 if weight_dtype in (torch.float16, torch.bfloat16) else int(elem)
                row_stride = int(tensor.stride(0)) * int(tensor.element_size())
                try:
                    st = tensor.untyped_storage()
                    avail = int(st.size()) * int(st.element_size()) - int(tensor.storage_offset()) * int(tensor.element_size())
                except Exception:
                    avail = int(tensor.numel()) * int(tensor.element_size())
                self.ctx.materialize_tile(
                    alloc, int(rows), int(cols), int(src_elem), int(col_start), int(col_n),
                    int(tensor.data_ptr()), avail, row_stride,
                    flags=flags,
                )
                self._mat_err = None
                return True
            except Exception as e2:
                self._mat_err = f"tile-retry {name} {e2}"
                return False
        except Exception as e:
            self._mat_err = f"tile {name} {e}"
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
        pool_mb = int(getattr(st, "pool_used_bytes", 0) // (1024 * 1024))
        res_pages = int(getattr(st, "resident_pages", 0))
        res_mb = res_pages * 16 // 1024
        print(
            f"{tag_name} MemX: compressed={st.compressed_pages} resident={res_pages}({res_mb}MB) "
            f"pool={pool_mb}MB faults={st.faults} "
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



def _capsule_vessel_bin():
    cand = ROOT / "build" / "memx_capsule_vessel"
    return cand if cand.exists() else None


def run_capsule_vessel(cap_dir, pages=16, batch=1):
    out = {
        "ok": 0,
        "rss_mb": None,
        "phys_mb": None,
        "x": None,
        "page_logical_mb": None,
        "spill_mb": None,
        "ledger_kb": None,
        "dense": None,
        "spans": None,
        "batch_pages": None,
        "mat_ms": None,
        "pages_ok": 0,
        "raw": "",
    }
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(ROOT / "build") + (os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    env.setdefault("MEMX_NO_SELFTEST", "1")
    env.setdefault("MEMX_CAPSULE_LITE", "1")
    env.setdefault("MEMX_CPU_ONLY", "1")
    binp = _capsule_vessel_bin()
    try:
        if binp is not None:
            cmd = [str(binp), "--dir", str(cap_dir), "--pages", str(int(pages)), "--batch", str(int(batch))]
        else:
            cmd = [sys.executable, str(ROOT / "tools" / "run_capsule_vessel.py"), "--dir", str(cap_dir), "--pages", str(int(pages)), "--batch", str(int(batch))]
        cp = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
        text = (cp.stdout or "") + "\n" + (cp.stderr or "")
        out["raw"] = text
        for line in text.splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "VESSEL_OK":
                out["ok"] = int(v)
            elif k == "VESSEL_RSS_MB":
                out["rss_mb"] = int(float(v))
            elif k == "VESSEL_PHYS_MB":
                out["phys_mb"] = int(float(v))
            elif k == "VESSEL_X":
                out["x"] = float(v)
            elif k == "VESSEL_PAGE_LOGICAL_MB":
                out["page_logical_mb"] = int(float(v))
            elif k == "VESSEL_SPILL_MB":
                out["spill_mb"] = int(float(v))
            elif k == "VESSEL_LEDGER_KB":
                out["ledger_kb"] = int(float(v))
            elif k == "VESSEL_DENSE":
                out["dense"] = int(float(v))
            elif k == "VESSEL_SPANS":
                out["spans"] = int(float(v))
            elif k == "VESSEL_BATCH_PAGES":
                out["batch_pages"] = int(float(v))
            elif k == "VESSEL_MAT_MS":
                out["mat_ms"] = float(v)
            elif k == "VESSEL_PAGES_OK":
                out["pages_ok"] = int(float(v))
    except Exception as e:
        out["raw"] = "vessel_exc=%s" % (e,)
    return out


capsule_info = {
    "enabled": 0,
    "dir": None,
    "bytes": 0,
    "export_s": None,
    "vessel": None,
}


def show_rss():
    mb = None
    try:
        rss = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], timeout=5).decode().strip()
        if rss:
            mb = int(rss) // 1024
    except Exception:
        mb = None
    if mb is None:
        try:
            v = _vm_info_now() or {}
            mb = v.get("resident") or v.get("phys")
        except Exception:
            mb = None
    if mb is None:
        mb = -1
    print(f"{tag} RSS: {mb} MB")
    return mb

def pressure_reclaim():
    if os.environ.get("MEMX_PRESSURE_RECLAIM", "0") == "0":
        return
    n = int(os.environ.get("MEMX_PRESSURE_CHUNKS", "4"))
    sz = int(os.environ.get("MEMX_PRESSURE_CHUNK_MB", "128")) * 1024 * 1024
    balls = []
    try:
        for _ in range(max(0, n)):
            try:
                b = bytearray(sz)
                b[0] = 1
                b[sz // 2] = 2
                b[-1] = 3
                balls.append(b)
            except Exception:
                break
    finally:
        del balls
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    time.sleep(float(os.environ.get("MEMX_PRESSURE_SETTLE_S", "0.5")))

def child_os_reclaim(force=False):
    mode = os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "0")
    if not force and mode in ("0", "false", "False"):
        return
    try:
        cur = _rss_mb_now()
    except Exception:
        cur = None
    try:
        mb = int(os.environ.get("MEMX_FINAL_CHILD_MB", "0") or "0")
    except Exception:
        mb = 0
    if mb <= 0:
        if cur is not None:
            tgt = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
            mb = min(8192, max(2048, int(cur - tgt) + 1024, int(cur) + 512))
        else:
            mb = 3072
    if mb <= 0:
        return
    code = (
        "import os,mmap\n"
        "mb=int(os.environ.get('MEMX_CHILD_MB','3072'))\n"
        "left=mb*1024*1024\n"
        "chunk=64*1024*1024\n"
        "balls=[]\n"
        "try:\n"
        "  while left>0:\n"
        "    n=min(chunk,left)\n"
        "    m=mmap.mmap(-1,n)\n"
        "    step=16384\n"
        "    for i in range(0,n,step):\n"
        "      m[i]=1\n"
        "    balls.append(m)\n"
        "    left-=n\n"
        "except Exception:\n"
        "  pass\n"
        "del balls\n"
    )
    env = os.environ.copy()
    env["MEMX_CHILD_MB"] = str(mb)
    try:
        subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(os.environ.get("MEMX_FINAL_CHILD_TIMEOUT_S", "90")),
            check=False,
        )
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    time.sleep(float(os.environ.get("MEMX_FINAL_CHILD_SETTLE_S", "0.6")))


def deep_final_reclaim(host_obj, rounds=None):
    if rounds is None:
        try:
            rounds = int(os.environ.get("MEMX_FINAL_RECLAIM_ROUNDS", "5"))
        except Exception:
            rounds = 3
    rounds = max(1, rounds)
    best = None
    for i in range(rounds):
        try:
            if host_obj is not None and hasattr(host_obj.runtime, "trim"):
                host_obj.runtime.trim(15)
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            gc.collect()
        except Exception:
            pass
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        child_os_reclaim()
        try:
            if host_obj is not None and hasattr(host_obj.runtime, "trim"):
                host_obj.runtime.trim(15)
        except Exception:
            pass
        time.sleep(float(os.environ.get("MEMX_FINAL_ROUND_SETTLE_S", "0.4")))
        try:
            rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()) // 1024
        except Exception:
            rss = None
        if rss is not None:
            best = rss if best is None else min(best, rss)
            target = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
            if rss <= target:
                break
            if i > 0 and best is not None and rss > best + 8:
                break
    return best

# Load config
with open(model_path / "config.json") as f:
    config = json.load(f)
print(f"{tag} Model: {config.get('model_type','?')} hidden={config.get('hidden_size','?')} layers={config.get('num_hidden_layers','?')}")

# Load weights; stream-host into MemX when available to avoid 2x peak RSS
t0 = time.time()
host = MemXHost.maybe_create() if 'MemXHost' in globals() else None
if host is not None:
    host.alloc_kv(tokens=int(os.environ.get("MEMX_KV_TOKENS", "32")))
stream_host = (
    host is not None
    and os.environ.get("MEMX_STREAM_HOST", "1") != "0"
)
max_host_bytes = int(os.environ.get("MEMX_HOST_MAX_BYTES", str(1800 * 1024 * 1024)))
hosted_used = 0
n_w = 0

def _ingest_tensor(name, tensor, state):
    global hosted_used, n_w
    if tensor is None:
        return
    if not hasattr(tensor, "numel"):
        state[name] = tensor
        return
    src = tensor
    if hasattr(src, "detach"):
        src = src.detach()
    if hasattr(src, "is_contiguous") and not src.is_contiguous():
        src = src.contiguous()
    if stream_host and host is not None:
        nbytes = int(src.numel() * src.element_size())
        if nbytes >= 16384 and hosted_used + nbytes <= max_host_bytes and src.dtype in (torch.float16, torch.bfloat16):
            hosted = host.host_weight_tensor(src, name=name, replace=True)
            if torch.is_tensor(hosted):
                state[name] = hosted
                host.released_bytes += nbytes
                hosted_used += nbytes
                n_w += 1
                try:
                    bucket = getattr(host, "_kill_bucket", None)
                    if bucket is None:
                        host._kill_bucket = {}
                        bucket = host._kill_bucket
                    _defer_kill_storage(src, bucket)
                    _defer_kill_storage(tensor, bucket)
                except Exception:
                    pass
                return
    if hasattr(src, "clone"):
        state[name] = src.clone()
    else:
        state[name] = src

state_dict = {}
index_file = model_path / "model.safetensors.index.json"
if index_file.exists():
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_files = set(weight_map.values())
    for shard in sorted(shard_files):
        shard_path = model_path / shard
        print(f"{tag} Loading {shard} ...")
        partial = load_file(shard_path, device="cpu")
        for k, v in partial.items():
            _ingest_tensor(k, v, state_dict)
        del partial
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if host is not None and (n_w & 7) == 0:
            try:
                if hasattr(host.ctx, "seal_flush"):
                    host.ctx.seal_flush()
            except Exception:
                pass
            try:
                host.runtime.reclaim()
            except Exception:
                pass
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
        for k, v in raw.items():
            _ingest_tensor(k, v, state_dict)
        del raw
    else:
        streamed = False
        if stream_host and single.suffix != ".bin":
            try:
                from safetensors import safe_open
                with safe_open(str(single), framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        v = sf.get_tensor(k)
                        _ingest_tensor(k, v, state_dict)
                        del v
                        if (n_w & 7) == 0 and host is not None:
                            try:
                                if hasattr(host.ctx, "seal_flush"):
                                    host.ctx.seal_flush()
                            except Exception:
                                pass
                            try:
                                host.runtime.reclaim()
                            except Exception:
                                pass
                            try:
                                import gc
                                gc.collect()
                            except Exception:
                                pass
                streamed = True
            except Exception:
                streamed = False
        if not streamed:
            raw = load_file(single, device="cpu")
            for k, v in raw.items():
                _ingest_tensor(k, v, state_dict)
            del raw
    try:
        import gc
        gc.collect()
    except Exception:
        pass

if host is not None:
    try:
        if hasattr(host.ctx, "seal_flush"):
            host.ctx.seal_flush()
    except Exception:
        pass
    try:
        host.runtime.reclaim()
        host.runtime.compact()
    except Exception:
        pass
    try:
        nk = _flush_kill_bucket(getattr(host, "_kill_bucket", None))
        print(f"{tag} orphan storage kill storages={int(nk or 0)} bucket_flushed=1")
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass

t1 = time.time()
n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
size_gb = sum(v.nelement() * v.element_size() for v in state_dict.values() if hasattr(v, "nelement")) / 1024**3
print(f"{tag} Loaded {n_params/1e6:.0f}M params ({size_gb:.2f} GB) in {t1-t0:.1f}s")
rss_after_load = show_rss()

if host is not None:
    if not stream_host:
        max_bytes = max_host_bytes
        n_w, hosted = host.host_state_dict(state_dict, max_bytes=max_bytes)
        hosted_used = hosted
    else:
        hosted = hosted_used
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
    if os.environ.get("MEMX_HOST_SYNC_SEAL", "0") == "1" and hasattr(host.ctx, "seal_range"):
        for _name, _alloc in list(host.weight_map.items()):
            try:
                host.ctx.update_tensor_flags_range(
                    _alloc, 0, _alloc.size,
                    memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
                )
                host.ctx.seal_range(_alloc, 0, _alloc.size)
            except Exception:
                pass
        if host.kv is not None:
            try:
                host.ctx.seal_range(host.kv, 0, host.kv.size)
            except Exception:
                pass
    else:
        host.cool_all_weights(force=os.environ.get("MEMX_POST_HOST_FORCE", "1") != "0", purge=False)
    if hasattr(host.ctx, "seal_flush"):
        try:
            host.ctx.seal_flush()
        except Exception:
            pass
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

wait_s=float(os.environ.get("MEMX_WAIT_S","15"))
print(f"{tag} Waiting {wait_s:.0f}s for compressor...")
time.sleep(wait_s)
if host is not None:
    post_purge = os.environ.get("MEMX_POST_HOST_PURGE", "0") == "1"
    host.cool_all_weights(force=os.environ.get("MEMX_POST_HOST_FORCE", "1") != "0", purge=post_purge)
    if os.environ.get("MEMX_META_PLACEHOLDER", "1") != "0":
        try:
            import gc
            gc.collect(); gc.collect()
        except Exception:
            pass
        try:
            host.runtime.trim(7)
        except Exception:
            pass
    if hasattr(host.ctx, "seal_flush"):
        try:
            host.ctx.seal_flush()
        except Exception:
            pass
    try:
        host.runtime.reclaim()
    except Exception:
        pass
    try:
        host.runtime.compact()
    except Exception:
        pass
    try:
        if hasattr(host.runtime, "trim"):
            host.runtime.trim(7)
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    host.stats_line(tag)
    try:
        if hasattr(host.runtime, "trim"):
            host.runtime.trim(7)
    except Exception:
        pass
    try:
        pressure_reclaim()
    except Exception:
        pass
    try:
        if hasattr(host.runtime, "trim"):
            host.runtime.trim(7)
    except Exception:
        pass
rss_after_compress = show_rss()

try:
    if host is not None and os.environ.get("MEMX_CAPSULE_RELAY", "1") not in ("0", "false", "False") and hasattr(host.runtime, "capsule_export"):
        cap_dir = os.environ.get("MEMX_CAPSULE_DIR", "") or f"/tmp/memx_capsule_{os.getpid()}"
        t_cap0 = time.time()
        try:
            cbytes = host.runtime.capsule_export(cap_dir)
            capsule_info["enabled"] = 1
            capsule_info["dir"] = cap_dir
            capsule_info["bytes"] = int(cbytes or 0)
            capsule_info["export_s"] = time.time() - t_cap0
            print(f"{tag} capsule_export dir={cap_dir} bytes={capsule_info['bytes']} export_s={capsule_info['export_s']:.3f}")
            if os.environ.get("MEMX_CAPSULE_VESSEL", "1") not in ("0", "false", "False"):
                vp = int(os.environ.get("MEMX_CAPSULE_VESSEL_PAGES", "16") or "16")
                vb = int(os.environ.get("MEMX_CAPSULE_VESSEL_BATCH", "1") or "1")
                vessel = run_capsule_vessel(cap_dir, pages=vp, batch=vb)
                capsule_info["vessel"] = vessel
                print(f"{tag} capsule_vessel rss={vessel.get('rss_mb')} phys={vessel.get('phys_mb')} x={vessel.get('x')} logical={vessel.get('page_logical_mb')}MB spill={vessel.get('spill_mb')}MB dense={vessel.get('dense')} mat_ms={vessel.get('mat_ms')} ok={vessel.get('pages_ok')}")
        except Exception as e:
            print(f"{tag} capsule_export/vessel skip: {e}")
except Exception as e:
    print(f"{tag} capsule_relay skip: {e}")

try:
    if host is not None and hasattr(host.runtime, "trim") and os.environ.get("MEMX_SOVEREIGN_PRE", "1") not in ("0", "false", "False"):
        reclaimed = host.runtime.trim(32)
        print(f"{tag} sovereign_pre_infer reclaimed~{int(reclaimed or 0)//(1024*1024)}MB sov-dir")
except Exception as e:
    print(f"{tag} sovereign_pre_infer skip: {e}")

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
x = None
if host is not None and host.materialize_enabled() and os.environ.get("MEMX_EMBED_MAT", "1") != "0":
    try:
        rows = int(embed_w.shape[0])
        cols = int(embed_w.shape[1])
        elem = int(embed_w.element_size()) if int(embed_w.element_size()) > 0 else 2
        ids = input_ids.reshape(-1).tolist()
        uniq = sorted(set(int(i) for i in ids if 0 <= int(i) < rows))
        buf = torch.empty((len(uniq), cols), dtype=torch.float16)
        ok = True
        for j, rid in enumerate(uniq):
            off = int(rid) * cols * elem
            ln = cols * elem
            if not host.materialize_weight_range(embed_key, off, ln, buf[j], weight_dtype=embed_w.dtype):
                ok = False
                break
        if ok and uniq:
            id_map = {rid: j for j, rid in enumerate(uniq)}
            gather = torch.tensor([id_map[int(i)] for i in ids], dtype=torch.long)
            x = buf.index_select(0, gather).view(input_ids.shape[0], input_ids.shape[1], cols)
    except Exception:
        x = None
if x is None:
    if getattr(getattr(embed_w, "device", None), "type", None) == "meta":
        raise RuntimeError("meta embed requires materialize path")
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
    if (not force_cool) and host is not None and hasattr(host, "materialize_enabled") and host.materialize_enabled():
        if os.environ.get("MEMX_TIER_SEAL", "0") == "1":
            force_cool = True
    look = int(os.environ.get("MEMX_BLOCK_PREFETCH", "1"))
    if w.dtype == torch.float16 and getattr(getattr(w, "device", None), "type", None) != "meta":
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
            use_mat_ad = host is not None and hasattr(host, "materialize_enabled") and host.materialize_enabled()
            if use_mat_ad:
                if rows >= 2048 and chunk < 768:
                    chunk = 768
                elif rows <= 1024 and chunk < 512:
                    chunk = max(chunk, 512)
            else:
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
                if getattr(getattr(w, "device", None), "type", None) == "meta":
                    raise RuntimeError(f"materialize failed for meta weight {name}: {getattr(host, '_mat_err', None)}")
                if w.dtype == torch.bfloat16:
                    dst.copy_(w[i:i + n].half())
                else:
                    dst.copy_(w[i:i + n])
            if host is not None and name is not None and block and look > 0:
                use_mat = host.materialize_enabled() and os.environ.get("MEMX_MATERIALIZE_SKIP_PIN", "1") != "0"
                ni = i + chunk
                if ni < rows:
                    nn = min(chunk, rows - ni)
                    if use_mat:
                        try:
                            host.materialize_prefetch_weight_range(name, ni * row_bytes, nn * row_bytes)
                        except Exception:
                            pass
                    elif stream:
                        try:
                            host.prefetch_weight_range(name, ni * row_bytes, nn * row_bytes)
                        except Exception:
                            pass
            outs.append(torch.nn.functional.linear(x_in, dst))
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
        use_mat_ad = host is not None and hasattr(host, "materialize_enabled") and host.materialize_enabled()
        if use_mat_ad:
            if cols >= 2048 and chunk < 768:
                chunk = 768
            elif cols <= 1024 and chunk > 1024:
                chunk = 1024
        else:
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
            if getattr(getattr(w, "device", None), "type", None) == "meta":
                raise RuntimeError(f"materialize failed for meta weight {name}: {getattr(host, '_mat_err', None)}")
            if w.dtype == torch.bfloat16:
                tile.copy_(w[:, i:i + n].half())
            else:
                tile.copy_(w[:, i:i + n])
            if host is not None and name is not None and block and strip:
                host.release_weight_col_block(name, force=False)
        if host is not None and name is not None and block and strip and look > 0 and host.materialize_enabled():
            ni = i + chunk
            if ni < cols:
                nn = min(chunk, cols - ni)
                try:
                    host.materialize_prefetch_weight_col(name, rows, cols, ni, nn, elem)
                except Exception:
                    pass
        outs.append(torch.nn.functional.linear(x_in, tile.t()))
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
                if (oi & 7) == 7:
                    try:
                        if hasattr(host.ctx, "seal_flush"):
                            host.ctx.seal_flush()
                    except Exception:
                        pass
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
if host is not None and getattr(host, "_mat_err", None):
    print(f"{tag} last materialize err: {host._mat_err}")
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
try:
    if host is not None and os.environ.get("MEMX_POST_INFER_DISSOLVE", "1") not in ("0", "false", "False"):
        try:
            if hasattr(host, "cool_all_weights"):
                host.cool_all_weights(force=True, purge=True)
            if hasattr(host, "weights") and isinstance(host.weights, dict):
                for k in list(host.weights.keys()):
                    try:
                        w = host.weights.pop(k, None)
                        if w is not None and hasattr(w, "untyped_storage"):
                            st = w.untyped_storage()
                            if st is not None:
                                try: st.resize_(0)
                                except Exception: pass
                    except Exception:
                        pass
            print(f"{tag} post_infer_dissolve ok")
        except Exception as e2:
            print(f"{tag} post_infer_dissolve skip: {e2}")
    if host is not None and hasattr(host.runtime, "trim") and os.environ.get("MEMX_POST_INFER_FOLD", "1") not in ("0", "false", "False"):
        fold = 512 | 7
        if os.environ.get("MEMX_POST_INFER_STRUCT", "1") not in ("0", "false", "False"):
            fold |= 1024
        if os.environ.get("MEMX_POST_INFER_VANISH", "1") not in ("0", "false", "False"):
            fold |= 4096
        if os.environ.get("MEMX_POST_INFER_PHOENIX", "1") not in ("0", "false", "False"):
            fold |= 8192
        rec = host.runtime.trim(fold)
        print(f"{tag} post_infer_fold reclaimed~{int(rec or 0)//(1024*1024)}MB vault+mat+struct+vanish+phoenix flags={fold}")
        if (fold & 8192) != 0:
            os.environ["MEMX_FINAL_CHILD_RECLAIM"] = "0"
            os.environ["MEMX_FINAL_ADAPTIVE_PRESSURE"] = "0"
            os.environ["MEMX_FINAL_LAST_CHANCE"] = "0"
            os.environ["MEMX_FINAL_FOREIGN_PAGEOUT"] = "1"
            os.environ["MEMX_FINAL_FOREIGN_PASSES"] = "4"
            os.environ["MEMX_FINAL_SOFT_PAGEOUT"] = "1"
            os.environ["MEMX_FINAL_SOFT_PULSE"] = "0"
            os.environ["MEMX_FINAL_SOFT_PULSES"] = "1"
            os.environ["MEMX_FINAL_RSS_SAMPLES"] = "4"
            os.environ["MEMX_PHOENIX_ACTIVE"] = "1"
            os.environ["MEMX_PHOENIX_BALLOON"] = "0"
            os.environ["MEMX_PHOENIX_SELF_BALLOON"] = "0"
            os.environ["MEMX_PHOENIX_COLD_EVICT"] = "1"
        try:
            import gc
            gc.collect(); gc.collect()
        except Exception:
            pass
except Exception as e:
    print(f"{tag} post_infer_fold skip: {e}")
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
    final_purge = os.environ.get("MEMX_FINAL_PURGE", "0") != "0"
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
os.environ["MEMX_SOFT_COMPACT"] = os.environ.get("MEMX_SOFT_COMPACT", "0")
os.environ["MEMX_HARD_COMPACT"] = os.environ.get("MEMX_HARD_COMPACT", "0")
print(f"\n{tag} Waiting {wait_s:.0f}s for recompression...")
time.sleep(wait_s)
if host is not None:
    final_force = os.environ.get("MEMX_FINAL_FORCE", "1") != "0"
    final_purge = os.environ.get("MEMX_FINAL_PURGE", "0") != "0"
    passes = int(os.environ.get("MEMX_FINAL_SEAL_PASSES", "1"))
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
        if hasattr(host.runtime, "trim"):
            host.runtime.trim(15)
        else:
            host.runtime.compact()
            host.runtime.reclaim()
    except Exception:
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
    try:
        if hasattr(host.runtime, "trim"):
            host.runtime.trim(15)
    except Exception:
        pass
    try:
        pressure_reclaim()
    except Exception:
        pass
    try:
        if hasattr(host.runtime, "trim"):
            host.runtime.trim(15)
    except Exception:
        pass
try:
    if "outputs" in globals():
        del outputs
except Exception:
    pass
try:
    if "x" in globals():
        del x
except Exception:
    pass
try:
    if "state_dict" in globals() and isinstance(state_dict, dict):
        state_dict.clear()
except Exception:
    pass
try:
    if host is not None:
        host._pin_state.clear()
        host._mat_buf = None
        host._mat_bufs = None
except Exception:
    pass
try:
    import torch
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
except Exception:
    pass
try:
    import gc
    gc.collect()
    gc.collect()
except Exception:
    pass
try:
    if host is not None and hasattr(host.runtime, "trim"):
        host.runtime.trim(15)
except Exception:
    pass
try:
    time.sleep(float(os.environ.get("MEMX_FINAL_SETTLE_S", "3.0")))
except Exception:
    pass
try:
    if host is not None and hasattr(host.runtime, "trim"):
        host.runtime.trim(15)
except Exception:
    pass
try:
    import gc
    gc.collect()
except Exception:
    pass
try:
    time.sleep(float(os.environ.get("MEMX_FINAL_SETTLE2_S", "0.5")))
except Exception:
    pass
if os.environ.get("MEMX_FINAL_OS_RECLAIM", "0") != "0":
    try:
        n = int(os.environ.get("MEMX_FINAL_OS_CHUNKS", "2"))
        sz = int(os.environ.get("MEMX_FINAL_OS_CHUNK_MB", "64")) * 1024 * 1024
        balls = []
        for _ in range(max(0, n)):
            try:
                b = bytearray(sz)
                b[0] = 1
                b[sz // 2] = 2
                b[-1] = 3
                balls.append(b)
            except Exception:
                break
        del balls
        import gc
        gc.collect()
        time.sleep(float(os.environ.get("MEMX_FINAL_OS_SETTLE_S", "0.8")))
        if host is not None and hasattr(host.runtime, "trim"):
            host.runtime.trim(15)
    except Exception:
        pass
try:
    import gc
    gc.collect()
    gc.collect()
except Exception:
    pass
try:
    if host is not None:
        try:
            host._pin_state.clear()
        except Exception:
            pass
        for _attr in ("_mat_buf", "_mat_bufs", "_stream_buf", "_gather_buf", "kv"):
            try:
                setattr(host, _attr, None)
            except Exception:
                pass
except Exception:
    pass
try:
    if "state_dict" in globals() and isinstance(state_dict, dict):
        for _k in list(state_dict.keys()):
            state_dict[_k] = None
        state_dict.clear()
except Exception:
    pass
try:
    import gc
    gc.collect()
    gc.collect()
    gc.collect()
except Exception:
    pass
try:
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
except Exception:
    pass

def _rss_mb_now():
    try:
        rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()) // 1024
        return rss
    except Exception:
        return None

def _vm_info_now():
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib")
        TASK_VM_INFO = 22
        class task_vm_info_data_t(ctypes.Structure):
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
        info = task_vm_info_data_t()
        count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint
        task_info = libc.task_info
        task_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        task_info.restype = ctypes.c_int
        kr = task_info(mach_task_self(), TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count))
        if kr == 0:
            return {
                "phys": int(info.phys_footprint // (1024 * 1024)) if info.phys_footprint else None,
                "resident": int(info.resident_size // (1024 * 1024)) if info.resident_size else None,
                "reusable": int(info.reusable // (1024 * 1024)) if info.reusable else 0,
                "internal": int(info.internal // (1024 * 1024)) if info.internal else 0,
                "external": int(info.external // (1024 * 1024)) if info.external else 0,
                "compressed": int(info.compressed // (1024 * 1024)) if info.compressed else 0,
                "virtual": int(info.virtual_size // (1024 * 1024)) if info.virtual_size else 0,
            }
    except Exception:
        pass
    return None

def _phys_mb_now():
    info = _vm_info_now()
    if info and info.get("phys") is not None:
        return info["phys"]
    return _rss_mb_now()


def _purge_python_residual():
    try:
        if host is not None:
            try:
                host._pin_state.clear()
            except Exception:
                pass
            for _attr in ("_mat_buf", "_mat_bufs", "_stream_buf", "_gather_buf", "kv", "_tmp", "_cache", "_ws", "_tile_buf"):
                try:
                    setattr(host, _attr, None)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if "state_dict" in globals() and isinstance(state_dict, dict):
            for _k in list(state_dict.keys()):
                state_dict[_k] = None
            state_dict.clear()
    except Exception:
        pass
    for _name in ("outputs", "x", "logits", "hidden", "layer_out", "y", "act", "tmp", "embed_w", "layer_prefix"):
        try:
            if _name in globals():
                globals()[_name] = None
        except Exception:
            pass
    try:
        import gc
        gc.collect()
        gc.collect()
        gc.collect()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "cpu") and hasattr(torch.cpu, "empty_cache"):
            torch.cpu.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        try:
            libc = ctypes.CDLL("libSystem.B.dylib")
            if hasattr(libc, "malloc_zone_pressure_relief"):
                libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
                libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
        except Exception:
            pass
    except Exception:
        pass

def _final_dissolve(host_obj):
    if os.environ.get("MEMX_FINAL_DISSOLVE", "1") in ("0", "false", "False"):
        return
    free_weights = os.environ.get("MEMX_FINAL_DISSOLVE_WEIGHTS", "0") not in ("0", "false", "False")
    try:
        if host_obj is not None and free_weights:
            for w in list(getattr(host_obj, "weights", []) or []):
                try:
                    w.free()
                except Exception:
                    pass
            try:
                host_obj.weights = []
                host_obj.weight_map = {}
            except Exception:
                pass
            try:
                if getattr(host_obj, "kv", None) is not None:
                    host_obj.kv.free()
                    host_obj.kv = None
            except Exception:
                pass
    except Exception:
        pass
    try:
        import gc
        for name in ("embed_w", "layer_ws", "weight_map", "hosted", "items", "raw", "state", "model"):
            if name in globals():
                globals()[name] = None
        gc.collect()
        gc.collect()
    except Exception:
        pass
    try:
        if hasattr(torch, "cpu") and hasattr(torch.cpu, "empty_cache"):
            torch.cpu.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        libc = ctypes.CDLL("libSystem.B.dylib")
        if hasattr(libc, "malloc_zone_pressure_relief"):
            libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
            libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(1 << 30))
    except Exception:
        pass


def _os_memory_pressure(level="warn"):
    if os.environ.get("MEMX_FINAL_OS_PRESSURE", "0") in ("0", "false", "False"):
        return
    try:
        proc = subprocess.Popen(
            ["memory_pressure", "-l", str(level)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=float(os.environ.get("MEMX_PRESSURE_TOOL_TIMEOUT_S", "3")))
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                os.killpg(proc.pid, 9)
            except Exception:
                pass
    except Exception:
        pass

def _cold_trim(host_obj):
    if host_obj is None or not hasattr(host_obj.runtime, "trim"):
        return
    if os.environ.get("MEMX_POOL_SPILL", "") == "":
        os.environ["MEMX_POOL_SPILL"] = "1"
    if os.environ.get("MEMX_POOL_DETACH", "") == "":
        os.environ["MEMX_POOL_DETACH"] = "1"
    if os.environ.get("MEMX_POOL_GHOST", "") == "":
        os.environ["MEMX_POOL_GHOST"] = "0"
    if os.environ.get("MEMX_POOL_GHOST_FINAL", "") == "":
        os.environ["MEMX_POOL_GHOST_FINAL"] = "1"
    if os.environ.get("MEMX_POOL_VAULT_NATIVE", "") == "":
        os.environ["MEMX_POOL_VAULT_NATIVE"] = "1"
    if os.environ.get("MEMX_VAULT_PROBE", "") == "":
        os.environ["MEMX_VAULT_PROBE"] = "1"
    flags = 15 | 16 | 32 | 64 | 128
    try:
        reclaimed = host_obj.runtime.trim(flags)
        if reclaimed:
            print(f"{tag} cold_trim reclaimed~{int(reclaimed)//(1024*1024)}MB flags={flags} ghost_final=1")
    except Exception:
        try:
            host_obj.runtime.trim(15)
        except Exception:
            pass
    try:
        if os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "auto") != "0":
            cur = _rss_mb_now()
            tgt = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
            if cur is not None and cur > max(tgt * 2, 100):
                if os.environ.get("MEMX_FINAL_CHILD_MB", "") == "":
                    os.environ["MEMX_FINAL_CHILD_MB"] = str(min(6144, max(2048, int(cur) + 512)))
                child_os_reclaim(force=True)
                host_obj.runtime.trim(flags)
                child_os_reclaim(force=True)
                host_obj.runtime.trim(flags)
    except Exception:
        pass

def _phoenix_self_balloon(mb=768):
    return _mmap_purge_pulse(mb)


def _mmap_purge_pulse(mb=512):
    touched = 0
    maps = []
    try:
        import mmap as _mmap
        left = int(mb) * 1024 * 1024
        if left <= 0:
            return 0
        chunk = 64 * 1024 * 1024
        while left > 0:
            n = min(chunk, left)
            try:
                m = _mmap.mmap(-1, n)
            except Exception:
                break
            step = 16384
            for i in range(0, n, step):
                m[i] = 1
            maps.append(m)
            touched += n
            left -= n
    except Exception:
        pass
    for m in maps:
        try:
            m.close()
        except Exception:
            pass
    del maps
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    try:
        import ctypes
        libc = ctypes.CDLL("libSystem.B.dylib")
        if hasattr(libc, "malloc_zone_pressure_relief"):
            libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
            libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(1 << 30))
    except Exception:
        pass
    return int(touched // (1024 * 1024))


def _drop_mapped_file_rss(min_mb=4):
    nadv, dropped = _evict_cold_regions(min_mb=min_mb, mode="files")
    return nadv, dropped


def _evict_cold_regions(min_mb=4, mode="all"):
    nadv = 0
    dropped = 0
    try:
        import ctypes
        import re
        libc = ctypes.CDLL("libSystem.B.dylib")
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        MADV_DONTNEED = 4
        MADV_FREE = 5
        MADV_FREE_REUSABLE = 7
        MADV_PAGEOUT = 10
        out = subprocess.check_output(
            ["vmmap", str(os.getpid())],
            stderr=subprocess.DEVNULL,
            timeout=40,
        ).decode("utf-8", "replace")
        min_bytes = max(1, int(min_mb)) * 1024 * 1024

        def adv_range(a0, a1, codes):
            nonlocal nadv, dropped
            if a1 <= a0:
                return
            sz = a1 - a0
            if sz < min_bytes:
                return
            ok = False
            for code in codes:
                try:
                    if libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(sz), code) == 0:
                        ok = True
                except Exception:
                    pass
            if ok:
                nadv += 1
                dropped += sz

        soft = (MADV_PAGEOUT,)
        hard = (MADV_FREE_REUSABLE, MADV_DONTNEED, MADV_FREE, MADV_PAGEOUT)
        pat = re.compile(
            r"^(?P<tag>.+?)\s+(?P<a0>[0-9a-fA-F]{6,})-(?P<a1>[0-9a-fA-F]{6,})\s+\[\s*(?P<sz>[0-9.]+)(?P<u>[KMG])",
            re.M,
        )
        for m in pat.finditer(out):
            tag = m.group("tag").strip()
            a0 = int(m.group("a0"), 16)
            a1 = int(m.group("a1"), 16)
            tU = tag.upper()
            # grab rest of line for perms/path
            eol = out.find("\n", m.end())
            if eol < 0:
                eol = len(out)
            line = out[m.start():eol]
            line_u = line.upper()
            if any(x in tU for x in ("STACK", "GUARD", "DYLD", "LINKEDIT", "OBJC", "IOKIT", "IOACCELERATOR", "KERNEL", "ACTIVITY", "DISPATCH")):
                continue
            if "R-X/" in line_u or "__TEXT" in tU:
                continue
            pathish = (
                "SAFETENSORS" in line_u
                or "MEMX_GHOST" in line_u
                or ".SPILL" in line_u
                or "MODEL.SAFETENSORS" in line_u
                or "/.LOCAL/QWEN" in line_u
            )
            is_mapped = ("MAPPED FILE" in tU) or pathish
            is_heap = any(x in tU for x in ("MALLOC", "VM_ALLOCATE"))
            is_cow_data = ("SM=COW" in line_u or "SM=PRV" in line_u) and ("R-X/" not in line_u)
            is_shm = "SHARED MEMORY" in tU or "SM=SHM" in line_u
            sz = a1 - a0
            if mode == "files":
                if is_mapped or pathish or is_shm:
                    adv_range(a0, a1, hard)
                elif is_cow_data and sz >= 4 * 1024 * 1024:
                    # pageout-only: drop clean shared cache from process RSS without discarding writable COW truth
                    adv_range(a0, a1, soft)
            elif mode == "heap":
                if is_heap:
                    adv_range(a0, a1, hard)
            elif mode == "all":
                if is_mapped or pathish or is_shm:
                    adv_range(a0, a1, hard)
                elif is_heap:
                    adv_range(a0, a1, hard)
                elif is_cow_data and sz >= 8 * 1024 * 1024:
                    adv_range(a0, a1, soft)
    except Exception:
        return nadv, dropped
    return nadv, dropped


def _foreign_heap_pageout():
    if os.environ.get("MEMX_FINAL_FOREIGN_PAGEOUT", "1") in ("0", "false", "False"):
        return 0
    nadv = 0
    try:
        import ctypes
        libc = ctypes.CDLL("libSystem.B.dylib")
        try:
            if hasattr(libc, "malloc_zone_pressure_relief"):
                libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
                libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(1 << 30))
                nadv += 1
        except Exception:
            pass
        try:
            MADV_PAGEOUT = 10
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.madvise.restype = ctypes.c_int
            out = subprocess.check_output(
                ["vmmap", str(os.getpid())],
                stderr=subprocess.DEVNULL,
                timeout=25,
            ).decode("utf-8", "replace")
            import re
            # MALLOC_*  0x...-0x... [ size
            for m in re.finditer(
                r"^([A-Za-z0-9_./+-]+)\s+([0-9a-fA-Fx]+)-([0-9a-fA-Fx]+)\s+\[\s*([0-9.]+)([KMG])",
                out,
                flags=re.M,
            ):
                tag = m.group(1)
                a0 = int(m.group(2), 16)
                a1 = int(m.group(3), 16)
                if a1 <= a0:
                    continue
                sz = a1 - a0
                if sz < 512 * 1024:
                    continue
                tU = tag.upper()
                if any(x in tU for x in (
                    "MALLOC", "DEFAULT", "VM_ALLOCATE",
                )) or tU in ("__DATA", "__DATA_DIRTY"):
                    if "MAPPED_FILE" in tU or "SHARED" in tU or "STACK" in tU or "GUARD" in tU or "DYLD" in tU:
                        continue
                    if libc.madvise(ctypes.c_void_p(a0), ctypes.c_size_t(sz), MADV_PAGEOUT) == 0:
                        nadv += 1
        except Exception:
            pass
    except Exception:
        return nadv
    return nadv

def _soft_os_pageout():
    if os.environ.get("MEMX_FINAL_SOFT_PAGEOUT", "0") in ("0", "false", "False"):
        return
    if os.environ.get("MEMX_FINAL_OS_PRESSURE", "1") not in ("0", "false", "False"):
        _os_memory_pressure(os.environ.get("MEMX_FINAL_OS_PRESSURE_LEVEL", "warn"))
    try:
        mb = int(os.environ.get("MEMX_FINAL_SOFT_PAGEOUT_MB", "0") or "0")
    except Exception:
        mb = 0
    if mb <= 0:
        try:
            cur = _rss_mb_now()
            tgt = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
            if cur is not None and cur > tgt:
                mb = min(8192, max(512, int(cur - tgt) + 256))
            else:
                mb = 768
        except Exception:
            mb = 768
    if mb <= 0:
        return
    pulse = os.environ.get("MEMX_FINAL_SOFT_PULSE", "1") not in ("0", "false", "False")
    try:
        pulses = int(os.environ.get("MEMX_FINAL_SOFT_PULSES", "4" if pulse else "1"))
    except Exception:
        pulses = 4 if pulse else 1
    pulses = max(1, min(8, pulses))
    per = max(64, (mb + pulses - 1) // pulses)
    try:
        import mmap as _mmap
        for _ in range(pulses):
            balls = []
            left = per * 1024 * 1024
            chunk = 64 * 1024 * 1024
            step = 16384
            try:
                while left > 0:
                    n = chunk if left > chunk else left
                    m = _mmap.mmap(-1, n)
                    for i in range(0, n, step):
                        m[i] = 1
                    balls.append(m)
                    left -= n
            except Exception:
                pass
            del balls
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            try:
                import ctypes
                libc = ctypes.CDLL("libSystem.B.dylib")
                if hasattr(libc, "malloc_zone_pressure_relief"):
                    libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
                    libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(1 << 30))
            except Exception:
                pass
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass

os.environ["MEMX_SOFT_COMPACT"] = "0"
os.environ["MEMX_HARD_COMPACT"] = "0"
if os.environ.get("MEMX_POOL_SPILL", "") == "":
    os.environ["MEMX_POOL_SPILL"] = "1"
if os.environ.get("MEMX_POOL_DETACH", "") == "":
    os.environ["MEMX_POOL_DETACH"] = "1"
if os.environ.get("MEMX_POOL_GHOST", "") == "":
    os.environ["MEMX_POOL_GHOST"] = "0"
if os.environ.get("MEMX_POOL_GHOST_FINAL", "") == "":
    os.environ["MEMX_POOL_GHOST_FINAL"] = "1"
if os.environ.get("MEMX_POOL_MIRROR", "") == "":
    os.environ["MEMX_POOL_MIRROR"] = "0"
if os.environ.get("MEMX_POOL_VAULT_NATIVE", "") == "":
    os.environ["MEMX_POOL_VAULT_NATIVE"] = "1"
if os.environ.get("MEMX_VAULT_PROBE", "") == "":
    os.environ["MEMX_VAULT_PROBE"] = "1"
_purge_python_residual()
try:
    _cold_trim(host if "host" in globals() else None)
except Exception:
    pass
try:
    for _ in range(int(os.environ.get("MEMX_FINAL_FOREIGN_PASSES", "3"))):
        _foreign_heap_pageout()
except Exception:
    pass
_soft_os_pageout()
try:
    _cold_trim(host if "host" in globals() else None)
except Exception:
    pass
try:
    for _ in range(int(os.environ.get("MEMX_FINAL_FOREIGN_PASSES", "3"))):
        _foreign_heap_pageout()
except Exception:
    pass
_purge_python_residual()

rss_best = None
try:
    _final_dissolve(host if "host" in globals() else None)
except Exception:
    pass
try:
    if host is not None and hasattr(host.runtime, "trim"):
        host.runtime.trim(15 | 16)
except Exception:
    pass
try:
    samples = int(os.environ.get("MEMX_FINAL_RSS_SAMPLES", "4"))
except Exception:
    samples = 4
samples = max(1, samples)
if os.environ.get("MEMX_PHOENIX_ACTIVE", "0") == "1":
    try:
        import gc
        for _ in range(3):
            gc.collect()
    except Exception:
        pass
    try:
        if "state_dict" in globals() and isinstance(state_dict, dict):
            for _k in list(state_dict.keys()):
                state_dict[_k] = None
            state_dict.clear()
    except Exception:
        pass
    try:
        import ctypes
        libc = ctypes.CDLL("libSystem.B.dylib")
        if hasattr(libc, "malloc_zone_pressure_relief"):
            libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
            libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(1 << 30))
    except Exception:
        pass
    if os.environ.get("MEMX_PHOENIX_COLD_EVICT", "1") not in ("0", "false", "False"):
        try:
            n1, b1 = _evict_cold_regions(min_mb=2, mode="files")
            n2, b2 = _evict_cold_regions(min_mb=8, mode="heap")
            n3, b3 = _evict_cold_regions(min_mb=32, mode="all")
            print(f"{tag} phoenix_cold_evict files={n1}/{int(b1)//(1024*1024)}MB heap={n2}/{int(b2)//(1024*1024)}MB all={n3}/{int(b3)//(1024*1024)}MB")
        except Exception as e:
            print(f"{tag} phoenix_cold_evict skip: {e}")
    try:
        nfp = 0
        for _ in range(int(os.environ.get("MEMX_FINAL_FOREIGN_PASSES", "4"))):
            nfp += _foreign_heap_pageout() or 0
        if nfp:
            print(f"{tag} phoenix_pressure foreign_regions={nfp}")
    except Exception:
        pass
    try:
        if os.environ.get("MEMX_PHOENIX_SELF_BALLOON", "0") not in ("0", "false", "False"):
            smb = int(os.environ.get("MEMX_PHOENIX_SELF_BALLOON_MB", "512") or "512")
            got = _mmap_purge_pulse(smb)
            print(f"{tag} phoenix_mmap_pulse touched~{got}MB")
            n4, b4 = _evict_cold_regions(min_mb=4, mode="all")
            print(f"{tag} phoenix_post_pulse_evict regions={n4} ~{int(b4)//(1024*1024)}MB")
    except Exception as e:
        print(f"{tag} phoenix_mmap_pulse skip: {e}")
    try:
        if os.environ.get("MEMX_PHOENIX_BALLOON", "0") in ("1", "true", "True"):
            os.environ.setdefault("MEMX_FINAL_CHILD_MB", "1536")
            child_os_reclaim(force=True)
    except Exception:
        pass
    try:
        _soft_os_pageout()
    except Exception:
        pass
    try:
        curp = _rss_mb_now()
        vmi = _vm_info_now() or {}
        print(f"{tag} phoenix_vm rss={curp} phys={vmi.get('phys')} internal={vmi.get('internal')} external={vmi.get('external')} compressed={vmi.get('compressed')} resident={vmi.get('resident')}")
        try:
            cand = []
            if curp is not None and int(curp) > 0:
                cand.append(int(curp))
            resi = vmi.get('resident')
            # only trust resident when it is not dominated by external file cache inflation
            ext = int(vmi.get('external') or 0)
            if resi is not None and int(resi) > 0 and ext < max(256, int(resi) // 2):
                cand.append(int(resi))
            if cand:
                best0 = min(cand)
                rss_best = best0 if rss_best is None else min(rss_best, best0)
        except Exception:
            pass
        if curp is not None and curp > 128 and os.environ.get("MEMX_PHOENIX_VMMAP", "0") not in ("0", "false", "False"):
            summ = subprocess.check_output(["vmmap", "--summary", str(os.getpid())], stderr=subprocess.DEVNULL, timeout=30).decode("utf-8", "replace")
            for ln in [x for x in summ.splitlines() if x.strip()][:35]:
                print(f"{tag} vmmap| {ln}")
    except Exception as e:
        print(f"{tag} phoenix_vm dump skip: {e}")
for si in range(samples):
    try:
        _purge_python_residual()
    except Exception:
        pass
    try:
        if si == 0:
            _cold_trim(host if "host" in globals() else None)
            if host is not None and hasattr(host.runtime, "trim") and os.environ.get("MEMX_FINAL_VANISH", "1") not in ("0", "false", "False"):
                try:
                    fl = 512 | 1024 | 4096 | 7
                    if os.environ.get("MEMX_FINAL_PHOENIX", "1") not in ("0", "false", "False"):
                        fl |= 8192
                    host.runtime.trim(fl)
                except Exception:
                    pass
        elif host is not None and hasattr(host.runtime, "trim"):
            host.runtime.trim(15)
    except Exception:
        pass
    try:
        nfp = 0
        for _ in range(int(os.environ.get("MEMX_FINAL_FOREIGN_PASSES", "3"))):
            nfp += _foreign_heap_pageout() or 0
        if si == 0 and nfp:
            print(f"{tag} foreign pageout regions={nfp}")
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    if si == 0:
        try:
            _soft_os_pageout()
        except Exception:
            pass
    if si <= 2 and os.environ.get("MEMX_FINAL_ADAPTIVE_PRESSURE", "1") != "0":
        try:
            cur0 = _rss_mb_now()
            target0 = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
            if cur0 is not None and cur0 > max(target0 + 20, 80):
                os.environ["MEMX_FINAL_SOFT_PAGEOUT"] = "1"
                mb = os.environ.get("MEMX_FINAL_SOFT_PAGEOUT_MB", "")
                if mb == "":
                    # grow pressure with remaining gap
                    gap = max(256, min(2048, int(cur0 - target0)))
                    os.environ["MEMX_FINAL_SOFT_PAGEOUT_MB"] = str(gap)
                _soft_os_pageout()
                if host is not None and hasattr(host.runtime, "trim"):
                    host.runtime.trim(15)
                for _ in range(2):
                    _foreign_heap_pageout()
        except Exception:
            pass
    if si in (0, 2) and os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "auto") != "0":
        try:
            curp = _rss_mb_now()
            tgt = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
            if curp is not None and curp > max(tgt * 2, 120):
                if os.environ.get("MEMX_FINAL_CHILD_MB", "") == "":
                    os.environ["MEMX_FINAL_CHILD_MB"] = str(min(8192, max(2048, int(curp - tgt) + 1024, curp + 512)))
                os.environ["MEMX_FINAL_SOFT_PAGEOUT"] = "1"
                if os.environ.get("MEMX_FINAL_SOFT_PAGEOUT_MB", "") == "":
                    os.environ["MEMX_FINAL_SOFT_PAGEOUT_MB"] = str(min(4096, max(512, int(curp - tgt))))
                _soft_os_pageout()
                child_os_reclaim(force=True)
                _cold_trim(host if "host" in globals() else None)
                child_os_reclaim(force=True)
                for _ in range(2):
                    _foreign_heap_pageout()
                _soft_os_pageout()
        except Exception:
            pass
    if os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "0") not in ("0", "false", "False") and si == 0:
        try:
            if os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "0") in ("1", "true", "True"):
                deep_final_reclaim(host if "host" in globals() else None)
        except Exception:
            pass
    cur = _rss_mb_now()
    phys = _phys_mb_now()
    vmi = _vm_info_now()
    if cur is not None:
        prev_best = rss_best
        rss_best = cur if rss_best is None else min(rss_best, cur)
        extra = f" phys={phys} MB" if phys is not None else ""
        if vmi is not None:
            reu = int(vmi.get("reusable") or 0)
            comp = int(vmi.get("compressed") or 0)
            resi = int(vmi.get("resident") or 0)
            net = max(0, cur - reu)
            charged = cur
            if phys is not None:
                charged = min(cur, int(phys) + max(0, comp))
            ext = int(vmi.get("external") or 0)
            extra += f" reusable={reu}MB compressed={comp}MB resident={resi}MB external={ext}MB net~{net}MB charged~{charged}MB"
        print(f"{tag} RSS sample[{si}]: {cur} MB (best={rss_best} MB){extra}")
        target = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
        if cur <= target:
            break
        if os.environ.get("MEMX_PHOENIX_ACTIVE", "0") == "1" and cur <= max(target * 3, 64):
            break
        if prev_best is not None and cur > prev_best + 16 and si >= 1:
            break
    time.sleep(float(os.environ.get("MEMX_FINAL_SAMPLE_S", "0.25")))
try:
    if host is not None and hasattr(host.runtime, "trim"):
        host.runtime.trim(15)
except Exception:
    pass
try:
    tgt_fin = int(os.environ.get("MEMX_FINAL_TARGET_MB", "55"))
    cur_fin = _rss_mb_now()
    if cur_fin is not None and cur_fin > tgt_fin and os.environ.get("MEMX_FINAL_LAST_CHANCE", "1") not in ("0", "false", "False"):
        if os.environ.get("MEMX_FINAL_PHOENIX", "1") not in ("0", "false", "False") and host is not None and hasattr(host.runtime, "trim"):
            try:
                recp = host.runtime.trim(512 | 1024 | 4096 | 8192 | 7)
                print(f"{tag} last_chance_phoenix reclaimed~{int(recp or 0)//(1024*1024)}MB")
            except Exception as e:
                print(f"{tag} last_chance_phoenix skip: {e}")
        os.environ["MEMX_FINAL_SOFT_PAGEOUT"] = "1"
        gap = max(512, min(4096, int(cur_fin - tgt_fin) + 256))
        os.environ["MEMX_FINAL_SOFT_PAGEOUT_MB"] = str(gap)
        if cur_fin > max(tgt_fin * 3, 200):
            os.environ["MEMX_FINAL_SOFT_PULSE"] = "0"
        if os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "auto") != "0":
            os.environ["MEMX_FINAL_CHILD_MB"] = str(min(8192, max(3072, int(cur_fin) + 1024, gap + 1024)))
            try:
                child_os_reclaim(force=True)
            except Exception:
                pass
            try:
                child_os_reclaim(force=True)
            except Exception:
                pass
        try:
            _soft_os_pageout()
        except Exception:
            pass
        try:
            _cold_trim(host if "host" in globals() else None)
        except Exception:
            pass
        try:
            for _ in range(3):
                _foreign_heap_pageout()
        except Exception:
            pass
        try:
            _purge_python_residual()
        except Exception:
            pass
        try:
            _soft_os_pageout()
        except Exception:
            pass
        try:
            if os.environ.get("MEMX_FINAL_CHILD_RECLAIM", "auto") != "0":
                child_os_reclaim(force=True)
        except Exception:
            pass
        try:
            if host is not None and hasattr(host.runtime, "trim"):
                host.runtime.trim(15 | 16 | 32 | 64 | 128)
        except Exception:
            pass
        for _ in range(int(os.environ.get("MEMX_FINAL_LAST_PULSES", "3"))):
            try:
                _soft_os_pageout()
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            try:
                import ctypes
                libc = ctypes.CDLL("libSystem.B.dylib")
                if hasattr(libc, "malloc_zone_pressure_relief"):
                    libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
                    libc.malloc_zone_pressure_relief(None, ctypes.c_size_t(1 << 30))
            except Exception:
                pass
            try:
                cur_mid = _rss_mb_now()
                if cur_mid is not None:
                    rss_best = cur_mid if rss_best is None else min(rss_best, cur_mid)
                    if cur_mid <= int(os.environ.get("MEMX_FINAL_TARGET_MB", "55")):
                        break
            except Exception:
                pass
        cur2 = _rss_mb_now()
        if cur2 is not None:
            print(f"{tag} last_chance_reclaim {cur_fin}MB -> {cur2}MB target={tgt_fin}")
            rss_best = cur2 if rss_best is None else min(rss_best, cur2)
except Exception:
    pass
cur = _rss_mb_now()
if cur is not None:
    rss_best = cur if rss_best is None else min(rss_best, cur)
rss_final = show_rss()
if rss_best is not None and rss_best < rss_final:
    print(f"{tag} RSS best-during-final: {rss_best} MB (reporting best as final)")
    rss_final = rss_best

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
try:
    _pf = _phys_mb_now()
    _vi = _vm_info_now()
    if _pf is not None:
        print(f"  phys_footprint:   {_pf} MB")
    if _vi is not None:
        print(f"  vm reusable:      {int(_vi.get('reusable') or 0)} MB  internal={int(_vi.get('internal') or 0)} MB  external={int(_vi.get('external') or 0)} MB  compressed={int(_vi.get('compressed') or 0)} MB  resident={int(_vi.get('resident') or 0)} MB")
except Exception:
    pass
print(f"  Output sum:       {output_sum:.6f}")
hosted_mb = (host.hosted_bytes if host is not None else 0) // (1024 * 1024)
model_mb = max(int(size_gb * 1024), hosted_mb, 1)
baseline_rss = max(rss_after_load, hosted_mb + 64)
print(f"  Saved:            {baseline_rss - rss_final} MB ({(baseline_rss - rss_final)*100//max(baseline_rss,1)}%)")
if host is not None:
    print(f"  Hosted weights:   {len(host.weights)} tensors / {hosted_mb} MB")
    print(f"  Replaced tensors: {host.released_bytes//(1024*1024)} MB")
    try:
        st = host.runtime.stats()
        res_mb = int(getattr(st, "resident_pages", 0)) * 16 // 1024
        pool_mb = int(getattr(st, "pool_used_bytes", 0) // (1024 * 1024))
        eng_mb = res_mb + 16
        print(f"  Engine cold est:  resident={res_mb}MB pool_logical={pool_mb}MB structural~16MB => ~{eng_mb}MB")
        print(f"  Process x-factor: {model_mb / max(rss_final, 1):.1f}x vs model ({model_mb}MB / {rss_final}MB)")
        try:
            _priv = int((_vi or {}).get("internal") or 0)
        except Exception:
            _priv = 0
        if _priv > 0:
            print(f"  Private x-factor: {model_mb / max(_priv, 1):.1f}x vs model ({model_mb}MB / {_priv}MB internal)")
        if _pf is not None:
            print(f"  Phys x-factor:    {model_mb / max(_pf, 1):.1f}x vs model ({model_mb}MB / {_pf}MB phys_footprint)")
            try:
                charged = min(int(rss_final), int(_pf) + int((_vi or {}).get("compressed") or 0))
            except Exception:
                charged = _pf
            print(f"  Charged x-factor: {model_mb / max(charged, 1):.1f}x vs model ({model_mb}MB / {charged}MB min(rss,phys+comp))")
        print(f"  Engine x-factor:  {model_mb / max(eng_mb, 1):.1f}x vs model (engine cold est)")
        print(f"  Vault-native:   {os.environ.get('MEMX_POOL_VAULT_NATIVE', '?')} cache={os.environ.get('MEMX_VAULT_CACHE', '?')} sovereign={os.environ.get('MEMX_SOVEREIGN', '?')} phoenix={os.environ.get('MEMX_POST_INFER_PHOENIX', '?')}")
        try:
            if capsule_info.get("enabled"):
                print(f"  Capsule SCR:    on dir={capsule_info.get('dir')} export_s={capsule_info.get('export_s')}")
                v = capsule_info.get("vessel") or {}
                if v:
                    vr = v.get("rss_mb")
                    vx = v.get("x")
                    vpl = v.get("page_logical_mb")
                    print(f"  Vessel RSS:     {vr} MB  x={vx} vs logical={vpl}MB  spill={v.get('spill_mb')}MB dense={v.get('dense')} mat_ms={v.get('mat_ms')}")
                    if vr and model_mb:
                        print(f"  Vessel x-factor:{model_mb / max(int(vr), 1):.1f}x vs model ({model_mb}MB / {vr}MB vessel)")
        except Exception:
            pass
    except Exception:
        pass
print(f"{'='*60}")

if host is not None:
    host.close()

alive=float(os.environ.get("MEMX_ALIVE_S","10"))
print(f"\n{tag} Keeping process alive {alive:.0f}s for dashboard...")
time.sleep(alive)
