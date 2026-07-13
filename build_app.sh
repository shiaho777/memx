#!/bin/bash
# Build MemXApp and bundle runtime dylib
set -e

cd "$(dirname "$0")"
BUILD_DIR="$PWD/build"
XCODE_BUILD_DIR="$BUILD_DIR/xcode"
APP_OUTPUT="$BUILD_DIR/MemXApp.app"
DYLIB_PATH="$BUILD_DIR/libmemx_runtime.dylib"

mkdir -p "$BUILD_DIR"

# Build dylib
echo "Building libmemx_runtime.dylib..."
make explicit-runtime

# Build app
echo "Building MemXApp..."
cd MemXApp
xcodebuild -project MemXApp.xcodeproj -scheme MemXApp -configuration Release build \
    SYMROOT="$XCODE_BUILD_DIR" \
    2>&1 | tail -3

# Copy dylib into app bundle
APP_PATH="$XCODE_BUILD_DIR/Release/MemXApp.app"

if [ -n "$APP_PATH" ] && [ -d "$APP_PATH" ]; then
    cp "$DYLIB_PATH" "$APP_PATH/Contents/MacOS/"
    rm -rf "$APP_OUTPUT"
    cp -R "$APP_PATH" "$APP_OUTPUT"
    echo "✅ App ready: $APP_OUTPUT"
    echo "   Dylib: $(ls -la "$APP_PATH/Contents/MacOS/libmemx_runtime.dylib" | awk '{print $5}') bytes"
else
    echo "❌ Could not find built app"
    exit 1
fi
