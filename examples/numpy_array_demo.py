"""Demonstrate copy-only NumPy access to compressed MemX storage."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("MEMX_CPU_ONLY", "1")
os.environ.setdefault("MEMX_NO_SELFTEST", "1")

import numpy as np

from memx_array import MemXArray


def main():
    pattern = np.tile(np.arange(1024, dtype=np.float32), (256, 1))
    with MemXArray.zeros(pattern.shape) as array:
        assert array.numpy().tobytes() == bytes(array.nbytes)
        array[:] = pattern
        sealed = array.seal()
        before = array.stats()
        snapshot = array.materialize()
        assert snapshot.tobytes() == pattern.tobytes()
        assert array.stats()["compressed_pages"] == before["compressed_pages"]
        print(f"seal processed {sealed} pages; residency: {before}")
        array[10, 10] = -1.5
        pattern[10, 10] = -1.5
        array[20:24, 20:24] = 2.5
        pattern[20:24, 20:24] = 2.5
        assert array.numpy().tobytes() == pattern.tobytes()
        row = np.asarray(array)[123]
    assert row.tobytes() == pattern[123].tobytes()
    assert snapshot[10, 10] == 10.0
    print("Independent NumPy snapshots and slices survive close.")
    print("Write storage through array[key] = value, not array.numpy()[key].")
    print("numpy_array_demo: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
