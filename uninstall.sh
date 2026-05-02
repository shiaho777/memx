#!/bin/bash
# MemX uninstaller - removes GPU memory compression from shell

set -e

SHELL_RC=""

if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

echo "╔══════════════════════════════════════════════╗"
echo "║        MemX Uninstall Script                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

if ! grep -q "libmemx3.dylib" "$SHELL_RC" 2>/dev/null; then
    echo "⚠️  MemX is not installed in $SHELL_RC"
    exit 0
fi

# Remove MemX lines
sed -i.bak '/libmemx3.dylib/d' "$SHELL_RC"
sed -i.bak '/# MemX:/d' "$SHELL_RC"
rm -f "$SHELL_RC.bak"

echo "✅ MemX removed from $SHELL_RC"
echo ""
echo "🔄 Open a new terminal for changes to take effect."
