#!/bin/sh
# Build the MemX core + POSIX adapter + end-to-end test for Android arm64.
# Requires ANDROID_NDK (or ANDROID_NDK_HOME) pointing at an NDK r26+ root.
# Output: build/android/{libmemx_core.a, test_posix_adapter}
# Push test_posix_adapter to a device (adb push) and run it there; the
# adapter uses sigaction/mprotect/pthreads, all available on bionic.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NDK="${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}"
if [ -z "$NDK" ]; then
    echo "ANDROID_NDK not set; cannot build" >&2
    exit 1
fi
HOST_TAG="$(ls "$NDK/toolchains/llvm/prebuilt" | head -1)"
CC="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android24-clang"
AR="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/llvm-ar"
OUT="$ROOT/build/android"
mkdir -p "$OUT"
"$CC" -O2 -ffreestanding -fno-builtin -I"$ROOT/core" -c "$ROOT/core/memx_core.c" -o "$OUT/memx_core.o"
"$AR" rcs "$OUT/libmemx_core.a" "$OUT/memx_core.o"
"$CC" -O2 -I"$ROOT/core" -I"$ROOT/platform/posix" \
    "$ROOT/tests/test_posix_adapter.c" "$ROOT/platform/posix/memx_posix.c" \
    "$OUT/libmemx_core.a" -o "$OUT/test_posix_adapter"
echo "built: $OUT/test_posix_adapter (arm64 Android, 16K logical pages)"
echo "run on device: adb push $OUT/test_posix_adapter /data/local/tmp/ && adb shell /data/local/tmp/test_posix_adapter"