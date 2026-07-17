#!/bin/bash
# Web Agent Launcher — Run this to start Fiverr automation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  WEB AGENT — Fiverr Automation"
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
MISSING=""
for cmd in xdg-open xdotool xclip; do
    if command -v $cmd &>/dev/null; then
        echo "  ✓ $cmd"
    else
        echo "  ✗ $cmd (MISSING)"
        MISSING="$MISSING $cmd"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "Missing tools:$MISSING"
    echo "Install with: sudo apt install xdotool xclip"
    echo ""
    read -p "Install now? (y/n): " INSTALL
    if [ "$INSTALL" = "y" ]; then
        sudo apt install -y xdotool xclip
    else
        echo "Please install manually and try again."
        exit 1
    fi
fi

echo ""
echo "Starting Fiverr Agent..."
echo ""

# Run the agent
python3 "$SCRIPT_DIR/fiverr_agent.py" "$@"
