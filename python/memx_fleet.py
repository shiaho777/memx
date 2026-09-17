"""Multi-model fleet orchestrator for local LLM capsules (SCR plane).

One process, one live capsule attach at a time (global in libmemx_runtime);
the fleet keeps bookkeeping of resident models, rebinds the live attach on
demand, and enforces a max-resident policy with LRU eviction.
"""
import os
import re
import subprocess
import time
from pathlib import Path

import memx_runtime as _memx

REPO_ROOT = Path(__file__).resolve().parents[1]


class FleetError(Exception):
    pass


class ModelCapsule:
    def __init__(self, name, capsule_dir, approx_logical_bytes=0, hot_budget=0):
        self.name = name
        self.capsule_dir = str(Path(capsule_dir))
        self.approx_logical_bytes = int(approx_logical_bytes)
        self.hot_budget = int(hot_budget)
        self.registered_at = time.monotonic()
        self.last_attached_at = None
        self.resident = False
        self.attach_count = 0

    def touch(self):
        self.last_attached_at = time.monotonic()
        self.attach_count += 1


class FleetModel:
    def __init__(self, fleet, capsule):
        self._fleet = fleet
        self._capsule = capsule

    @property
    def name(self):
        return self._capsule.name

    def materialize_segment(self, segment, buf):
        return self._fleet._materialize_segment(self._capsule.name, segment, buf)

    def segment(self, segment):
        return self._fleet._segment(self._capsule.name, segment)

    def stats(self):
        return self._fleet._stats(self._capsule.name)

    def verify(self):
        return self._fleet._verify(self._capsule.name)


class ModelFleet:
    def __init__(self, runtime=None, max_resident=2):
        self._runtime = runtime
        self._capsules = {}
        self._live = None
        self.max_resident = int(max_resident)

    @property
    def runtime(self):
        if self._runtime is None:
            self._runtime = _memx.Runtime()
        return self._runtime

    def register(self, name, capsule_dir, approx_logical_bytes=0, hot_budget=0):
        if name in self._capsules:
            raise FleetError(f"model already registered: {name}")
        d = Path(capsule_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"capsule dir missing: {d}")
        if not (d / "ledger.bin").exists() or not (d / "spill.bin").exists():
            raise FleetError(f"not a capsule dir (no spill.bin/ledger.bin): {d}")
        cap = ModelCapsule(name, d, approx_logical_bytes, hot_budget)
        self._capsules[name] = cap
        return cap

    def attach(self, name):
        cap = self._require(name)
        self._enforce_budget(exclude=name)
        if self._live != name:
            if self._live is not None:
                try:
                    self.runtime.capsule_detach()
                except OSError:
                    pass
                self._live = None
            self.runtime.capsule_attach(cap.capsule_dir)
            self._live = name
        cap.touch()
        cap.resident = True
        return FleetModel(self, cap)

    def detach(self, name):
        cap = self._require(name)
        if self._live == name:
            try:
                self.runtime.capsule_detach()
            except OSError:
                pass
            self._live = None
        cap.resident = False

    def resident(self):
        names = [n for n, c in self._capsules.items() if c.resident]
        names.sort(key=lambda n: (self._capsules[n].last_attached_at or 0))
        return names

    def ensure_only(self, ids):
        keep = set(ids)
        for name in list(self.resident()):
            if name not in keep:
                self.detach(name)

    def plan_swap(self, from_name, to_name):
        frm = self._require(from_name)
        to = self._require(to_name)
        steps = [
            {"op": "detach", "model": from_name, "why": "seal+release capability"},
            {"op": "attach", "model": to_name, "dir": to.capsule_dir},
            {"op": "materialize_hot", "model": to_name, "hot_bytes": to.hot_budget},
        ]
        return steps

    def _require(self, name):
        cap = self._capsules.get(name)
        if cap is None:
            raise KeyError(f"model not registered: {name}")
        return cap

    def _enforce_budget(self, exclude=None):
        resident = [n for n in self.resident() if n != exclude]
        while len(resident) >= self.max_resident:
            victim = resident.pop(0)
            self.detach(victim)

    def _bound(self, name):
        cap = self._require(name)
        if self._live != name:
            self.attach(name)
        return cap

    def _materialize_segment(self, name, segment, buf):
        self._bound(name)
        return self.runtime.capsule_materialize_segment(segment, buf)

    def _segment(self, name, segment):
        self._bound(name)
        return self.runtime.capsule_segment(segment)

    def _stats(self, name):
        self._bound(name)
        st = self.runtime.capsule_stats()
        return {
            "attached": bool(st.attached),
            "ent_count": int(st.ent_count),
            "spill_bytes": int(st.spill_bytes),
            "page_bytes": int(st.page_bytes),
            "ledger_bytes": int(st.ledger_bytes),
            "materialize_pages": int(st.materialize_pages),
            "materialize_bytes": int(st.materialize_bytes),
            "dense": int(st.dense),
        }

    def _verify(self, name):
        self._bound(name)
        bad, pages, rc = self.runtime.capsule_verify()
        return {"bad": bad, "pages": pages, "rc": rc}


_VESSEL_KEYS = {
    "VESSEL_OK": ("ok", int),
    "VESSEL_RSS_MB": ("rss_mb", float),
    "VESSEL_PHYS_MB": ("phys_mb", float),
    "VESSEL_X": ("x", float),
    "VESSEL_PAGE_LOGICAL_MB": ("page_logical_mb", float),
    "VESSEL_SPILL_MB": ("spill_mb", float),
    "VESSEL_LEDGER_KB": ("ledger_kb", float),
    "VESSEL_DENSE": ("dense", int),
    "VESSEL_SPANS": ("spans", int),
    "VESSEL_MAT_MS": ("mat_ms", float),
    "VESSEL_PAGES_OK": ("pages_ok", int),
}


def fleet_cli(capsule_dir, pages=32, batch=1, vessel_bin=None, timeout=180):
    vessel_bin = vessel_bin or (REPO_ROOT / "build" / "memx_capsule_vessel")
    out = {"ok": 0, "rss_mb": None, "phys_mb": None, "x": None, "raw": ""}
    if not Path(vessel_bin).exists():
        out["raw"] = f"vessel missing: {vessel_bin}"
        return out
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(REPO_ROOT / "build") + (
        os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
    )
    env.setdefault("MEMX_NO_SELFTEST", "1")
    env.setdefault("MEMX_CAPSULE_LITE", "1")
    env.setdefault("MEMX_CPU_ONLY", "1")
    try:
        cp = subprocess.run(
            [str(vessel_bin), "--dir", str(capsule_dir), "--pages", str(int(pages)), "--batch", str(int(batch))],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    except Exception as e:
        out["raw"] = f"vessel_exc={e}"
        return out
    text = (cp.stdout or "") + "\n" + (cp.stderr or "")
    out["raw"] = text
    for m in re.finditer(r"(VESSEL_[A-Z_]+)=([-\w.]+)", text):
        key = _VESSEL_KEYS.get(m.group(1))
        if key:
            name, conv = key
            try:
                out[name] = conv(m.group(2))
            except ValueError:
                pass
    return out
