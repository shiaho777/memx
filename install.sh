#!/bin/bash
# MemX installer - enables GPU memory compression for all shell commands

set -e

MEMX_DIR="$(cd "$(dirname "$0")" && pwd)"
DYLIB="$MEMX_DIR/libmemx3.dylib"
SHELL_RC=""

# Detect shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

echo "╔══════════════════════════════════════════════╗"
echo "║        MemX GPU Memory Compression            ║"
echo "║        Installation Script                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Build if needed
if [ ! -f "$DYLIB" ]; then
    echo "🔧 Building libmemx3.dylib..."
    clang -dynamiclib -O2 -framework Metal -framework Foundation -lz \
        -o "$DYLIB" "$MEMX_DIR/libmemx3.m"
    echo "✅ Built successfully"
else
    echo "✅ libmemx3.dylib found"
fi

# Build launcher
if [ ! -f "$MEMX_DIR/memx" ]; then
    echo "🔧 Building memx launcher..."
    clang -O2 -o "$MEMX_DIR/memx" "$MEMX_DIR/memx.m" -framework Foundation
    echo "✅ Built successfully"
else
    echo "✅ memx launcher found"
fi

echo ""
echo "📋 Installation options:"
echo ""
echo "  1) Shell integration (recommended)"
echo "     Adds DYLD_INSERT_LIBRARIES to $SHELL_RC"
echo "     All new shell sessions auto-enable MemX"
echo "     Small processes (ls, cat) are NOT affected (lazy init)"
echo "     Apple-signed binaries are NOT affected (SIP protection)"
echo "     Only processes with large allocations (>64KB) activate GPU compression"
echo ""
echo "  2) Manual usage only"
echo "     Use: ./memx ./your_program"
echo "     Or:  DYLD_INSERT_LIBRARIES=./libmemx3.dylib ./your_program"
echo ""
echo -n "Choose [1/2]: "
read -r choice

if [ "$choice" = "1" ]; then
    # Check if already installed
    if grep -q "libmemx3.dylib" "$SHELL_RC" 2>/dev/null; then
        echo "⚠️  MemX already installed in $SHELL_RC"
        echo -n "Reinstall? [y/N]: "
        read -r reinstall
        if [ "$reinstall" != "y" ]; then
            echo "Cancelled."
            exit 0
        fi
        # Remove old entry
        sed -i.bak '/libmemx3.dylib/d' "$SHELL_RC"
        rm -f "$SHELL_RC.bak"
    fi

    echo "" >> "$SHELL_RC"
    echo "# MemX: GPU-accelerated memory compression" >> "$SHELL_RC"
    echo "export DYLD_INSERT_LIBRARIES=\"$DYLIB\"" >> "$SHELL_RC"

    echo ""
    echo "✅ MemX installed! Added to $SHELL_RC"
    echo ""
    echo "🔄 Open a new terminal to activate, or run:"
    echo "   source $SHELL_RC"
    echo ""
    echo "📊 Verify with:"
    echo "   DYLD_INSERT_LIBRARIES=$DYLIB /tmp/test_memx"
    echo "   (or just run any memory-heavy program)"
    echo ""
    echo "🗑️  To uninstall: run ./uninstall.sh"
    echo ""
    echo "⚠️  Note: macOS SIP prevents injection into Apple-signed binaries."
    echo "   This is a feature — ls, cat, etc. are unaffected."
    echo "   Only your own programs (and Homebrew binaries) will use MemX."
else
    echo ""
    echo "✅ Manual mode. Use: ./memx ./your_program"
fi
