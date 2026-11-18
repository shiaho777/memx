#!/bin/sh
# Build the MemX core + POSIX adapter + end-to-end test for iOS arm64.
# Requires Xcode with the iPhoneOS SDK. Compile/link is verified on macOS;
# running the test requires a device or a host-app embedding (iOS forbids
# bare executable launch outside an app bundle).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
TARGET="arm64-apple-ios15.0"
OUT="$ROOT/build/ios"
mkdir -p "$OUT"
xcrun --sdk iphoneos clang -target "$TARGET" -isysroot "$SDK" \
    -O2 -ffreestanding -fno-builtin -I"$ROOT/core" \
    -c "$ROOT/core/memx_core.c" -o "$OUT/memx_core.o"
xcrun --sdk iphoneos clang -target "$TARGET" -isysroot "$SDK" \
    -O2 -I"$ROOT/core" -I"$ROOT/platform/posix" \
    "$ROOT/tests/test_posix_adapter.c" "$ROOT/platform/posix/memx_posix.c" \
    "$OUT/memx_core.o" -o "$OUT/test_posix_adapter"
echo "built: $OUT/test_posix_adapter (arm64 iOS, 16K logical pages)"
echo "note: run inside an app bundle on device; bare exec is not permitted by iOS"