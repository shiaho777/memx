#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ["DYLD_LIBRARY_PATH"] = str(ROOT / "build") + (
    os.pathsep + os.environ["DYLD_LIBRARY_PATH"] if os.environ.get("DYLD_LIBRARY_PATH") else ""
)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pages", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    os.environ.setdefault("MEMX_CAPSULE_LITE", "1")
import memx_runtime as m
    rt = m.Runtime()
    rt.capsule_attach(args.dir)
    st = rt.capsule_stats()
    pages = max(1, min(int(args.pages), 512))
    buf = bytearray(pages * 16384)
    ents = max(int(st.ent_count), 1)
    pidxs = []
    for i in range(pages):
        rank = (i * 97) % ents
        try:
            pidxs.append(rt.capsule_pidx_at(rank))
        except Exception:
            pidxs.append(rank)
    t0 = time.time()
    ok = 0
    if args.batch and pages > 1:
        try:
            rt.capsule_materialize_v(pidxs, buf, 16384)
            ok = pages
        except Exception:
            for i, p in enumerate(pidxs):
                try:
                    mv = memoryview(buf)[i * 16384:(i + 1) * 16384]
                    rt.capsule_materialize(p, mv)
                    ok += 1
                except Exception:
                    pass
    else:
        for i, p in enumerate(pidxs):
            try:
                mv = memoryview(buf)[i * 16384:(i + 1) * 16384]
                rt.capsule_materialize(p, mv)
                ok += 1
            except Exception:
                pass
    ms = (time.time() - t0) * 1000.0
    st = rt.capsule_stats()
    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()) // 1024
    print("VESSEL_OK=1")
    print(f"VESSEL_RSS_MB={rss}")
    print(f"VESSEL_PAGES_OK={ok}")
    print(f"VESSEL_PAGES_REQ={pages}")
    print(f"VESSEL_MAT_MS={ms:.3f}")
    print(f"VESSEL_ENTS={int(st.ent_count)}")
    print(f"VESSEL_PAGE_LOGICAL_MB={int(st.page_bytes) // (1024 * 1024)}")
    print(f"VESSEL_SPILL_MB={int(st.spill_bytes) // (1024 * 1024)}")
    print(f"VESSEL_LEDGER_KB={int(st.ledger_bytes) // 1024}")
    print(f"VESSEL_DENSE={int(getattr(st, 'dense', 0))}")
    print(f"VESSEL_SPANS={int(getattr(st, 'materialize_spans', 0))}")
    print(f"VESSEL_BATCH_PAGES={int(getattr(st, 'materialize_batch_pages', 0))}")
    pl = max(int(st.page_bytes) // (1024 * 1024), 1)
    print(f"VESSEL_X={pl / max(rss, 1):.1f}")
    rt.capsule_detach()
    return 0 if ok > 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
