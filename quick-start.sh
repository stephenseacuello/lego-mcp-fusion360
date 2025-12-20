#!/bin/bash
# LEGO MCP Studio - Quick Start Script
# Version 7.0

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           🧱 LEGO MCP Studio - Quick Start                ║"
echo "║                     Version 7.0                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python $PY_VERSION found${NC}"

# Check/Install Flask
echo -e "${YELLOW}Checking Flask...${NC}"
if $PYTHON -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓ Flask installed${NC}"
else
    echo -e "${YELLOW}Installing Flask...${NC}"
    pip install flask requests
    echo -e "${GREEN}✓ Flask installed${NC}"
fi

# Verify brick catalog
echo -e "${YELLOW}Verifying brick catalog...${NC}"
cd "$(dirname "$0")"
$PYTHON -c "
import sys
sys.path.insert(0, 'shared')
from brick_catalog_extended import EXTENDED_BRICK_CATALOG, stats
s = stats()
print(f'✓ {s[\"total\"]} bricks in {s[\"categories\"]} categories')
"

# Check vision (optional)
echo -e "${YELLOW}Checking vision system (optional)...${NC}"
if $PYTHON -c "import cv2; import numpy" 2>/dev/null; then
    echo -e "${GREEN}✓ OpenCV available${NC}"
    VISION="opencv"
else
    echo -e "${YELLOW}○ OpenCV not installed (camera features disabled)${NC}"
    VISION="none"
fi

if $PYTHON -c "from ultralytics import YOLO" 2>/dev/null; then
    echo -e "${GREEN}✓ YOLO11 available${NC}"
    VISION="yolo"
else
    echo -e "${YELLOW}○ YOLO11 not installed (using mock detection)${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}System Ready!${NC}"
echo ""
echo "Start the dashboard:"
echo -e "  ${YELLOW}cd dashboard && python app.py${NC}"
echo ""
echo "Then open:"
echo -e "  ${BLUE}http://localhost:5000${NC}"
echo ""
echo "Keyboard shortcuts:"
echo "  W = Workspace    C = Collection    B = Builds"
echo "  S = Scan         I = Insights      ? = Help"
echo ""

# Optional: Start automatically
if [[ "$1" == "--start" ]]; then
    echo -e "${YELLOW}Starting dashboard...${NC}"
    cd dashboard
    $PYTHON app.py
fi
