# Quick Start Guide

Get LEGO MCP Studio running in 5 minutes.

---

## Prerequisites

- **Python 3.9+** - [Download Python](https://python.org)
- **Git** - [Download Git](https://git-scm.com)
- **Web Browser** - Chrome, Firefox, Safari, or Edge

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360
```

---

## Step 2: Install Dependencies

### Minimal (Dashboard Only)
```bash
pip install flask requests
```

### With Vision (Camera Detection)
```bash
pip install flask requests ultralytics opencv-python numpy pillow
```

### Full Stack
```bash
pip install -r dashboard/requirements.txt
```

---

## Step 3: Start the Dashboard

```bash
cd dashboard
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * LEGO Brick Catalog loaded: 323 elements
```

---

## Step 4: Open in Browser

Go to: **http://localhost:5000**

You'll see the LEGO MCP Dashboard home page.

---

## Step 5: Explore!

### Browse the Catalog
1. Click **📚 Brick Catalog** in the sidebar
2. Browse 323 brick types across 33 categories
3. Click any brick to see details

### Add to Collection
1. Click **📦 My Collection**
2. Click **➕ Add Manually**
3. Select brick type, color, quantity
4. Click **Add**

### Plan a Build
1. Click **🏗️ Builds**
2. Click **➕ New Build**
3. Enter name and parts list
4. See what you have vs. what you need

### Try Vision (Optional)
1. Click **🎯 Workspace**
2. Click **Start Camera**
3. Place LEGO bricks in view
4. Click **Detect** to identify them

---

## What's Next?

### Enable Claude Integration
See [MCP Setup Guide](MCP_TOOLS.md) to connect with Claude Desktop.

### Set Up Vision
See [Vision Setup Guide](VISION_SETUP.md) for camera and detection configuration.

### Full User Guide
See [User Guide](USER_GUIDE.md) for complete documentation.

---

## Quick Commands

```bash
# Start dashboard
cd dashboard && python app.py

# Run tests
python -m pytest tests/ -v

# Check system
python -c "from shared.brick_catalog_extended import stats; print(stats())"
```

---

## Common Issues

### "Module not found" Error
```bash
pip install flask requests
```

### Port Already in Use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
FLASK_PORT=5001 python app.py
```

### Camera Not Working
```bash
# Use mock detector (no camera needed)
export DETECTION_BACKEND=mock
python app.py
```

---

## Getting Help

- 📖 [Full Documentation](USER_GUIDE.md)
- 🐛 [Troubleshooting](USER_GUIDE.md#troubleshooting)
- 💬 [GitHub Discussions](https://github.com/yourusername/lego-mcp-fusion360/discussions)

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `S` | Go to Scan page |
| `C` | Go to Collection |
| `B` | Go to Builds |
| `W` | Go to Workspace |
| `/` | Focus search box |
| `?` | Show keyboard shortcuts |
| `Esc` | Close current modal |

---

You're all set! 🧱
