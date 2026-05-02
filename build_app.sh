#!/bin/bash
# Build MemXApp and bundle dylib
set -e

cd "$(dirname "$0")"

# Build dylib
echo "Building libmemx3.dylib..."
clang -dynamiclib -O2 -framework Metal -framework Foundation -lz -o libmemx3.dylib libmemx3.m

# Build app
echo "Building MemXApp..."
cd MemXApp
xcodebuild -project MemXApp.xcodeproj -scheme MemXApp -configuration Release build \
    SYMROOT="$PWD/../build" \
    2>&1 | tail -3

# Copy dylib into app bundle
APP_PATH="../build/Release/MemXApp.app"
if [ ! -d "$APP_PATH" ]; then
    APP_PATH=$(find /Users/shiaho/Library/Developer/Xcode/DerivedData/MemXApp-*/Build/Products/Release -name "MemXApp.app" -type d 2>/dev/null | head -1)
fi
if [ -z "$APP_PATH" ] || [ ! -d "$APP_PATH" ]; then
    APP_PATH=$(find /Users/shiaho/Library/Developer/Xcode/DerivedData/MemXApp-*/Build/Products/Debug -name "MemXApp.app" -type d 2>/dev/null | head -1)
fi

if [ -n "$APP_PATH" ] && [ -d "$APP_PATH" ]; then
    cp ../libmemx3.dylib "$APP_PATH/Contents/MacOS/"
    cp -R "$APP_PATH" ../MemXApp.app
    echo "✅ App ready: $(pwd)/../MemXApp.app"
    echo "   Dylib: $(ls -la "$APP_PATH/Contents/MacOS/libmemx3.dylib" | awk '{print $5}') bytes"
else
    echo "❌ Could not find built app"
    exit 1
fi
