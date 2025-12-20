# User Guide

Complete documentation for LEGO MCP Studio.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Workspace (Digital Twin)](#workspace-digital-twin)
4. [Scanning Bricks](#scanning-bricks)
5. [Managing Your Collection](#managing-your-collection)
6. [Planning Builds](#planning-builds)
7. [Insights & Analytics](#insights--analytics)
8. [Brick Catalog](#brick-catalog)
9. [Creating Custom Bricks](#creating-custom-bricks)
10. [Using with Claude](#using-with-claude)
11. [Keyboard Shortcuts](#keyboard-shortcuts)
12. [Troubleshooting](#troubleshooting)

---

## Getting Started

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9 or higher |
| Browser | Chrome, Firefox, Safari, Edge |
| RAM | 4GB minimum, 8GB recommended |
| Disk | 500MB for installation |
| Camera | Optional (for vision features) |

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Install dependencies
pip install flask requests

# Start dashboard
cd dashboard
python app.py

# Open browser to http://localhost:5000
```

### First Run

On first launch, the system will:
1. Load the 323-brick catalog
2. Initialize empty inventory
3. Start the web server on port 5000
4. Initialize mock camera (if no real camera)

---

## Dashboard Overview

### Navigation

The sidebar contains all main sections:

| Icon | Page | Description |
|------|------|-------------|
| 🏠 | Dashboard | Home page with overview |
| 🎯 | Workspace | Live camera + digital twin |
| 📷 | Scan | Bulk brick scanning |
| 📦 | My Collection | Inventory management |
| 🏗️ | Builds | Build planning |
| 📊 | Insights | Analytics & charts |
| 📚 | Brick Catalog | Browse all bricks |
| 🔨 | Builder | Create custom bricks |
| 📁 | Files | STL and G-code files |
| 📜 | History | Operation history |
| 🧪 | Tools | MCP tool testing |
| ⚡ | Status | Service status |
| ⚙️ | Settings | Configuration |

### Header

- **Search Box**: Search across bricks and collection
- **Status Indicators**: Fusion 360 and Slicer connection status
- **Theme Toggle**: Switch between light/dark mode

---

## Workspace (Digital Twin)

The Workspace page provides a real-time view of bricks on your baseplate.

### Camera Feed

The left panel shows the live camera feed with detection overlays.

**Controls:**
- **Start/Stop Camera**: Toggle camera feed
- **Capture**: Save current frame as image
- **Camera Selector**: Choose between multiple cameras

### Digital Twin

The right panel shows an 8x8 grid representing your baseplate.

- Each cell corresponds to a position (A1-H8)
- Detected bricks appear as colored markers
- Click a brick to select it

### Detection Workflow

1. Place bricks on your baseplate
2. Click **Detect Bricks**
3. View detections overlaid on camera feed
4. See bricks appear in digital twin grid
5. Click **Add to Collection** to save

### Calibration

For accurate grid mapping:

1. Click **Calibrate**
2. Click the 4 corners of your baseplate
3. Save calibration

---

## Scanning Bricks

The Scan page is optimized for bulk inventory input.

### Workflow

1. **Place bricks** on the scanning area
2. **Click Detect** to identify them
3. **Review** the detections:
   - ✓ Green = High confidence (≥80%)
   - ? Yellow = Needs review (<80%)
4. **Confirm** correct detections
5. **Edit** misidentified bricks
6. **Add to Collection**

### Quick Add

For known bricks, use Quick Add:

1. Select brick type from dropdown
2. Choose color
3. Enter quantity
4. Click Add

### Session Tracking

The Scan page tracks your session:
- Number of batches scanned
- Total bricks detected
- Total bricks added

Export your session as JSON for records.

---

## Managing Your Collection

### Viewing Collection

The Collection page shows all your inventoried bricks.

**Filters:**
- **Category**: Bricks, plates, tiles, slopes, etc.
- **Color**: Filter by LEGO color
- **Search**: Find by name

**Sorting:**
- By quantity (most to least)
- By name (alphabetical)
- By color
- By date added

### Adding Bricks

**Method 1: Scanning**
1. Go to Scan page
2. Detect and confirm bricks
3. Add to collection

**Method 2: Manual**
1. Click **➕ Add Manually**
2. Select brick type
3. Choose color
4. Enter quantity
5. Click Add

### Adjusting Quantity

Each brick card has +/- buttons:
- Click **+** to increase quantity
- Click **−** to decrease
- Removing all deletes the entry

### Import/Export

**Import:**
- CSV files
- BrickLink XML
- Rebrickable CSV

**Export:**
- CSV (spreadsheet-friendly)
- JSON (for backup/programming)

---

## Planning Builds

### Creating a Build

1. Click **➕ New Build**
2. Enter build name and description
3. Add parts:
   - Select brick type
   - Choose color
   - Enter quantity needed
4. Click **Create Build**

### Importing Builds

Import from popular sources:

**BrickLink:**
1. Export wanted list as XML
2. Click **Import** → **BrickLink**
3. Upload file

**Rebrickable:**
1. Export set inventory as CSV
2. Click **Import** → **Rebrickable**
3. Upload file

### Checking a Build

The build detail page shows:

- **Completion %**: How much you can build
- **Have**: Parts you own
- **Missing**: Parts you need
- **Substitutes**: Alternative parts

### Shopping List

For missing parts:

1. Click **Shopping List**
2. View all needed parts with quantities
3. Copy to clipboard or export
4. Links to BrickLink for purchasing

---

## Insights & Analytics

### Collection Statistics

View at-a-glance stats:
- Total pieces
- Unique brick types
- Number of colors
- Estimated value

### Charts

**Color Distribution:**
Doughnut chart showing brick counts by color.

**Category Distribution:**
Bar chart showing counts by category.

### Top Bricks

See your most common bricks ranked by quantity.

### Recommendations

AI-generated suggestions:
- "Your collection is heavy on red—consider adding more blue"
- "You're 3 parts away from completing Build X"
- "Add more slope pieces for variety"

### Build Suggestions

See builds you can make:
- **Ready**: You have all parts
- **Almost**: Missing just a few pieces

---

## Brick Catalog

### Browsing

Browse 323 brick types across 33 categories:

| Category | Examples |
|----------|----------|
| Bricks | 1x1, 1x2, 2x4, etc. |
| Plates | Standard plates |
| Tiles | Smooth top (no studs) |
| Slopes | 18°, 33°, 45°, 65°, 75° |
| Technic | Beams, connectors, axles |
| Round | Cylinders, cones, domes |
| Arches | Curved tops |
| Wedges | Angled pieces |
| Modified | Special shapes |
| Hinges | Articulated joints |
| Clips | Connection points |

### Brick Details

Click any brick to see:
- Dimensions (studs × studs × plates)
- Physical size in millimeters
- Category and features
- Related bricks

---

## Creating Custom Bricks

### Builder Page

Create bricks with custom specifications:

1. **Basic Dimensions**
   - Width (1-48 studs)
   - Depth (1-48 studs)
   - Height (1-36 plates)

2. **Type**
   - Brick (standard)
   - Plate (⅓ height)
   - Tile (no studs)
   - Slope (angled top)

3. **Advanced Features**
   - Ball joints
   - Sockets
   - Clips
   - Hinges
   - Patterns (grille, vent, etc.)
   - Chamfered edges
   - Hollow interior

### Using with Claude

With Claude Desktop connected, just describe what you want:

> "Create a 2x2 brick with a ball joint on top"

> "Make a 1x4 tile with a grille pattern"

---

## Using with Claude

### Setup

1. Install [Claude Desktop](https://claude.ai/download)

2. Edit `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lego-mcp": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/lego-mcp-fusion360/mcp-server"
    }
  }
}
```

3. Restart Claude Desktop

### Example Conversations

**Creating Bricks:**
> "Create a 2x4 brick"
> "Make a 45-degree slope, 2 wide and 3 deep"
> "Build a Technic beam with 8 holes"

**Browsing Catalog:**
> "Show me all the slope bricks"
> "What Technic pieces are available?"
> "List round bricks"

**Exporting:**
> "Export the brick to STL"
> "Generate G-code for my Prusa MK3S"
> "Create CNC milling operations"

**Batch Operations:**
> "Create the basic brick set (1x1 through 2x8)"
> "Export all bricks as STL files"

---

## Keyboard Shortcuts

### Global

| Key | Action |
|-----|--------|
| `/` | Focus search |
| `?` | Show help |
| `Esc` | Close modal |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |

### Navigation

| Key | Page |
|-----|------|
| `W` | Workspace |
| `S` | Scan |
| `C` | Collection |
| `B` | Builds |
| `I` | Insights |
| `K` | Catalog |

### Workspace

| Key | Action |
|-----|--------|
| `Space` | Detect bricks |
| `Enter` | Add to collection |
| `R` | Refresh camera |

---

## Troubleshooting

### Dashboard Won't Start

**Error: "Module not found"**
```bash
pip install flask requests
```

**Error: "Port already in use"**
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or use different port
FLASK_PORT=5001 python app.py
```

### Camera Issues

**No cameras detected:**
```bash
# List available cameras
python -c "import cv2; [print(i) for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

**Use mock camera:**
```bash
export DETECTION_BACKEND=mock
python app.py
```

### Detection Not Working

**Check backend:**
```python
from dashboard.services.vision import get_detector
print(get_detector().backend)  # Should show: yolo, roboflow, or mock
```

**Install YOLO:**
```bash
pip install ultralytics
```

### Import Errors

**BrickLink XML:**
- Ensure file is valid XML
- Check encoding is UTF-8
- Export from BrickLink as "Wanted List XML"

**Rebrickable CSV:**
- Use "Part out a set" export
- Include headers in file

### Slow Performance

- Reduce camera resolution in Settings
- Use mock detector for testing
- Close other browser tabs

### Data Not Saving

- Check write permissions in dashboard/services/data/
- Ensure disk has free space
- Restart the server

---

## Getting Help

- **Documentation**: This guide and other docs in `/docs`
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas

---

## Appendix: LEGO Color Reference

| Color | Hex | RGB |
|-------|-----|-----|
| Red | #C91A09 | 201, 26, 9 |
| Blue | #0057A8 | 0, 87, 168 |
| Yellow | #F7D117 | 247, 209, 23 |
| Green | #00852B | 0, 133, 43 |
| Black | #1B2A34 | 27, 42, 52 |
| White | #FFFFFF | 255, 255, 255 |
| Orange | #FE8A18 | 254, 138, 24 |
| Light Gray | #A0A5A9 | 160, 165, 169 |
| Dark Gray | #5B6770 | 91, 103, 112 |
| Brown | #583927 | 88, 57, 39 |
| Tan | #E4CD9E | 228, 205, 158 |

See full color list in [Color Reference](COLORS.md).
