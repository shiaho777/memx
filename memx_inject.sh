#!/bin/bash
# MemX Persistent Injection Setup
# Sets DYLD_INSERT_LIBRARIES in shell profile for automatic injection
# of all terminal-launched processes. Persists across reboots.
#
# NOTE: macOS 15+ launchctl silently drops DYLD_* variables (SIP),
# so LaunchAgent-based injection is NOT possible. Shell profile is
# the only reliable persistent method for non-SIP-protected binaries.
#
# Usage:
#   ./memx_inject.sh install   - Install persistent injection
#   ./memx_inject.sh status    - Check current injection status
#   ./memx_inject.sh uninstall - Remove persistent injection

MEMX_DYLIB="/Users/shiaho/Desktop/memx/libmemx3.dylib"
SHELL_RC="$HOME/.zshrc"
MARKER="# MemX: GPU-accelerated memory compression"

install() {
    echo "═══ MemX Persistent Injection Setup ═══"
    echo ""

    # 1. Verify dylib exists
    if [ ! -f "$MEMX_DYLIB" ]; then
        echo "❌ Error: $MEMX_DYLIB not found"
        exit 1
    fi
    echo "✅ Dylib found: $MEMX_DYLIB"

    # 2. Add to shell profile
    if grep -q "DYLD_INSERT_LIBRARIES.*libmemx3" "$SHELL_RC" 2>/dev/null; then
        echo "✅ Shell profile already configured"
    else
        echo "" >> "$SHELL_RC"
        echo "$MARKER" >> "$SHELL_RC"
        echo "export DYLD_INSERT_LIBRARIES=$MEMX_DYLIB" >> "$SHELL_RC"
        echo "✅ Shell profile updated: $SHELL_RC"
    fi

    # 3. Set for current shell
    export DYLD_INSERT_LIBRARIES="$MEMX_DYLIB"
    echo "✅ Current shell: DYLD_INSERT_LIBRARIES set"

    # 4. Install memx wrapper to /usr/local/bin
    MEMX_WRAPPER="/Users/shiaho/Desktop/memx/memx"
    if [ -f "$MEMX_WRAPPER" ]; then
        if [ -w "/usr/local/bin" ]; then
            cp "$MEMX_WRAPPER" /usr/local/bin/memx
            echo "✅ memx wrapper installed to /usr/local/bin/memx"
        else
            echo "⚠️  Cannot write to /usr/local/bin (try: sudo make install)"
        fi
    fi

    echo ""
    echo "═══ Coverage ═══"
    echo "  ✅ All programs launched from terminal (after new shell session)"
    echo "  ✅ Persists across reboots (via ~/.zshrc)"
    echo "  ✅ User-compiled programs, Homebrew packages, third-party tools"
    echo "  ⚠️  SIP-protected system binaries CANNOT be injected"
    echo "     (/usr/bin/*, /System/*, Apple-signed apps like Safari, Xcode)"
    echo "  💡 Use 'memx your_program' for guaranteed injection of any binary"
    echo ""
    echo "═══ Next Steps ═══"
    echo "  Open a NEW terminal, then test:"
    echo "    echo \$DYLD_INSERT_LIBRARIES"
    echo "    ./your_large_memory_program"
}

uninstall() {
    echo "═══ Removing MemX Persistent Injection ═══"

    # 1. Remove from shell profile
    if grep -q "DYLD_INSERT_LIBRARIES.*libmemx3" "$SHELL_RC" 2>/dev/null; then
        sed -i '' '/DYLD_INSERT_LIBRARIES.*libmemx3/d' "$SHELL_RC"
        sed -i '' "/$(echo $MARKER | sed 's/[\/&]/\\&/g')/d" "$SHELL_RC"
        echo "✅ Shell profile cleaned"
    else
        echo "ℹ️  Shell profile not configured"
    fi

    # 2. Unset current session
    unset DYLD_INSERT_LIBRARIES
    echo "✅ Current shell: DYLD_INSERT_LIBRARIES unset"

    # 3. Remove wrapper
    if [ -f "/usr/local/bin/memx" ]; then
        rm /usr/local/bin/memx 2>/dev/null && echo "✅ /usr/local/bin/memx removed"
    fi

    echo "✅ MemX injection removed. Open a new terminal for full effect."
}

status() {
    echo "═══ MemX Injection Status ═══"

    if grep -q "DYLD_INSERT_LIBRARIES.*libmemx3" "$SHELL_RC" 2>/dev/null; then
        echo "  Shell profile: ✅ configured"
    else
        echo "  Shell profile: ❌ not configured"
    fi

    if [ -n "$DYLD_INSERT_LIBRARIES" ]; then
        echo "  Current env: ✅ DYLD_INSERT_LIBRARIES=$DYLD_INSERT_LIBRARIES"
    else
        echo "  Current env: ❌ (not set — open a new terminal or source ~/.zshrc)"
    fi

    if [ -f "/usr/local/bin/memx" ]; then
        echo "  memx wrapper: ✅ /usr/local/bin/memx"
    else
        echo "  memx wrapper: ❌ not installed"
    fi

    if [ -f "$MEMX_DYLIB" ]; then
        echo "  Dylib: ✅ $MEMX_DYLIB"
    else
        echo "  Dylib: ❌ $MEMX_DYLIB not found"
    fi
}

case "${1:-status}" in
    install)   install ;;
    uninstall) uninstall ;;
    status)    status ;;
    *)         echo "Usage: $0 {install|uninstall|status}" ;;
esac
