#!/usr/bin/env python3
"""Fleet orchestrator over SCR capsules: register / attach / LRU / plan_swap.

Synthesizes tiny capsule dirs with the runtime itself (no real 0.8B model),
then exercises a max-resident=2 fleet across 3 models.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))
os.environ.setdefault("MEMX_CAPSULE_HOST_BIND", "0")

import memx_runtime as memx
from memx_fleet import FleetError, ModelFleet, fleet_cli


def fill(buf, nbytes, seed):
    for i in range(nbytes):
        buf[i] = (i * seed + (i >> 7) + seed) & 0xFF


def make_segment(ctx, name, pages, seed):
    nbytes = pages * 16384
    desc = memx.tensor_desc(
        memx.MEMX_TENSOR_ROLE_DATA, memx.MEMX_TENSOR_DTYPE_INT32,
        memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
        shape=(pages, nbytes // 4), stride=(nbytes // 4, 1),
    )
    a = ctx.malloc_tensor(nbytes, desc, name=name)
    fill(a.buffer(), nbytes, seed)
    golden = bytes(a.buffer())
    ctx.update_tensor_flags_range(a, 0, nbytes,
                                  memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD)
    try:
        ctx.force_compress_range(a, 0, nbytes)
    except Exception:
        pass
    ctx.name_segment(a, name)
    return a, golden


def synthesize_capsule(rt, tag, golden_out):
    ctx = rt.create_context(f"fleet-{tag}")
    ctx.set_quota(256 * memx.MB)
    a, golden = make_segment(ctx, "weights", 48, 31 + len(golden_out) * 7)
    td = tempfile.mkdtemp(prefix=f"memx_fleet_{tag}_")
    rt.capsule_export(td)
    rt.capsule_detach()
    a.free()
    ctx.destroy()
    golden_out[tag] = golden
    return Path(td)


def main():
    dylib = ROOT / "build" / "libmemx_runtime.dylib"
    if not dylib.exists():
        print("SKIP: dylib missing")
        return 0
    rt = memx.Runtime(dylib)

    golden = {}
    dirs = {}
    fleet_dirs = []
    try:
        for tag in ("m0", "m1", "m2"):
            dirs[tag] = synthesize_capsule(rt, tag, golden)
            fleet_dirs.append(dirs[tag])

        fleet = ModelFleet(runtime=rt, max_resident=2)

        missing = False
        try:
            fleet.register("ghost", "/nonexistent/memx_capsule_nowhere")
        except FileNotFoundError:
            missing = True
        assert missing, "register on missing dir must raise FileNotFoundError"

        bad_dir = tempfile.mkdtemp(prefix="memx_fleet_notacapsule_")
        not_capsule = False
        try:
            fleet.register("junk", bad_dir)
        except FleetError:
            not_capsule = True
        assert not_capsule, "register on non-capsule dir must raise FleetError"

        for tag in ("m0", "m1", "m2"):
            cap = fleet.register(tag, dirs[tag],
                                 approx_logical_bytes=48 * 16384, hot_budget=4 * 16384)
            assert cap.name == tag

        h0 = fleet.attach("m0")
        assert fleet.resident() == ["m0"], fleet.resident()
        buf = bytearray(48 * 16384)
        h0.materialize_segment("weights", buf)
        assert bytes(buf) == golden["m0"], "m0 materialize not bitexact"
        rank, pages, nbytes = h0.segment("weights")
        assert nbytes == 48 * 16384, (rank, pages, nbytes)
        st = h0.stats()
        assert st["attached"], st
        vr = h0.verify()
        assert vr["bad"] == 0 and vr["rc"] == 0, vr

        h1 = fleet.attach("m1")
        assert sorted(fleet.resident()) == ["m0", "m1"], fleet.resident()
        h2 = fleet.attach("m2")
        assert fleet.resident() == ["m1", "m2"], f"LRU must evict m0: {fleet.resident()}"
        assert fleet._live == "m2"

        buf2 = bytearray(48 * 16384)
        h2.materialize_segment("weights", buf2)
        assert bytes(buf2) == golden["m2"], "m2 materialize not bitexact"

        h1b = fleet.attach("m1")
        assert fleet.resident() == ["m2", "m1"], f"m1 must re-resident without reattach: {fleet.resident()}"
        assert fleet._live == "m1", "attach of live model must not rebind"

        h2b = fleet.attach("m2")
        assert fleet.resident() == ["m1", "m2"], f"LRU evicts m1: {fleet.resident()}"

        fleet.ensure_only(["m2"])
        assert fleet.resident() == ["m2"], fleet.resident()

        dup = False
        try:
            fleet.register("m2", dirs["m2"])
        except FleetError:
            dup = True
        assert dup, "double register must raise FleetError"
        try:
            fleet.attach("nosuch")
        except KeyError:
            pass
        else:
            raise AssertionError("attach of unknown model must raise KeyError")

        steps = fleet.plan_swap("m2", "m0")
        assert [s["op"] for s in steps] == ["detach", "attach", "materialize_hot"], steps
        assert steps[0]["model"] == "m2" and steps[1]["model"] == "m0"
        assert steps[1]["dir"] == str(dirs["m0"])
        assert steps[2]["hot_bytes"] == 4 * 16384

        vessel = fleet_cli(dirs["m0"])
        if vessel["ok"] == 1 and vessel.get("rss_mb") is not None:
            print(f"vessel rss_mb={vessel['rss_mb']} phys_mb={vessel.get('phys_mb')} x={vessel.get('x')}")
        else:
            print(f"vessel best-effort skip: ok={vessel['ok']} raw={vessel['raw'][:80]!r}")

        for tag in ("m0", "m1", "m2"):
            fleet.detach(tag)
        assert fleet.resident() == [], fleet.resident()
        print("OK memx_fleet register/attach/materialize/LRU/plan_swap/vessel_cli")
        return 0
    finally:
        shutil.rmtree(bad_dir, ignore_errors=True)
        for d in fleet_dirs:
            shutil.rmtree(d, ignore_errors=True)
        rt.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
