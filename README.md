# LEGO MCP Fusion 360

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/stephenseacuello/lego-mcp-fusion360)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)

<p align="center">
  <b>🧱 Create LEGO bricks with Claude AI + Fusion 360</b>
</p>

Design parametric LEGO bricks through natural language, export to STL/STEP/3MF, generate G-code for 3D printing, CNC milling, or laser engraving—all powered by Claude AI through the Model Context Protocol and Autodesk Fusion 360.

---

## ✨ What Can You Do?

| Feature | Description |
|---------|-------------|
| 💬 **Natural Language** | "Create a 2x4 brick and slice for my Bambu P1S" |
| 🧱 **Brick Types** | Standard, plates, tiles, slopes, technic, round, arch |
| 📤 **Export Formats** | STL, STEP, 3MF for any CAD/printing workflow |
| 🖨️ **3D Print** | Bambu Lab, Prusa, Ender + LEGO-optimized settings |
| 🔧 **CNC Mill** | GRBL, TinyG/Bantam, Haas + auto toolpaths |
| ✨ **Laser Engrave** | Custom text, logos, QR codes on bricks |
| 🖼️ **Dashboard** | Real-time WebSocket UI with 3D previews |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Desktop │────▶│   MCP Server    │────▶│  Fusion 360     │
│  (Natural Lang) │     │  (Port 8000)    │     │  (Port 8767)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │ Slicer Service  │     │  Web Dashboard  │
                        │ (Port 8766)     │     │  (Port 5000)    │
                        └─────────────────┘     └─────────────────┘
```

### Port Configuration

| Service | Port | Description |
|---------|------|-------------|
| Fusion 360 Add-in | 8767 | HTTP API for brick creation/export |
| Slicer Service | 8766 | Docker container for G-code generation |
| Web Dashboard | 5000 | Flask UI (optional, with `--profile full`) |

---

## 🚀 Quick Start

### Prerequisites

- **Autodesk Fusion 360** (free for personal use)
- **Python 3.9+**
- **Docker** (for slicer service)
- **Claude Desktop** (for MCP integration)

### Step 1: Clone and Setup

```bash
git clone https://github.com/stephenseacuello/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Setup export paths (creates symlinks for Docker)
chmod +x scripts/setup-paths.sh
./scripts/setup-paths.sh

# Install MCP server dependencies
cd mcp-server
pip install -r requirements.txt
```

### Step 2: Install the Fusion 360 Add-in

1. Open Fusion 360
2. Go to **Tools → Add-Ins → Scripts and Add-Ins**
3. Click **Add-Ins** tab → **+** button
4. Select the `fusion360-addin/LegoMCP` folder
5. Check **Run on Startup** and click **Run**

The add-in starts an HTTP server on `http://127.0.0.1:8767`.

### Step 3: Start Docker Services

```bash
# Start slicer service only
docker-compose up -d

# OR start with dashboard
docker-compose --profile full up -d
```

### Step 4: Configure Claude Desktop

Add to your Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "lego-mcp": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/lego-mcp-fusion360/mcp-server",
      "env": {
        "FUSION_API_URL": "http://127.0.0.1:8767",
        "SLICER_API_URL": "http://localhost:8766"
      }
    }
  }
}
```

### Step 5: Start Using!

Restart Claude Desktop and try:
- "Create a 2x4 LEGO brick"
- "Export the brick as STL and slice for my Bambu P1S"
- "Create a Technic beam with pin holes"
- "Make a tile with custom laser engraving"

---

## 🛠️ End-to-End Workflows

The system supports complete manufacturing workflows:

### create_and_print
```
Create brick → Export STL → Slice with printer profile → G-code
```

### create_and_mill
```
Create brick → Setup CAM → Generate toolpaths → NC code for CNC
```

### create_and_engrave
```
Create brick → Generate laser G-code → Custom text/logos
```

---

## 🧱 Available Brick Types

| Type | Command | Description |
|------|---------|-------------|
| **Standard** | `create_brick` | Classic LEGO bricks (2x4, 1x6, etc.) |
| **Plate** | `create_plate` | 1/3 height bricks |
| **Tile** | `create_tile` | Flat plates without studs |
| **Slope** | `create_slope` | Angled bricks (33°, 45°, 65°) |
| **Technic** | `create_technic` | Beams with 4.8mm pin holes |
| **Round** | `create_round` | Cylindrical bricks |
| **Arch** | `create_arch` | Curved opening bricks |

---

## 🖨️ Supported Printers

| Printer | Profile | Notes |
|---------|---------|-------|
| **Bambu Lab P1S** | `bambu_p1s` | Recommended, AMS support |
| **Bambu Lab X1C** | `bambu_x1c` | High-speed, lidar |
| **Bambu Lab A1** | `bambu_a1` | Budget-friendly |
| Prusa MK3S/MK4 | `prusa_mk3s`, `prusa_mk4` | Reliable workhorses |
| Creality Ender 3 | `ender3`, `ender3_v2` | Budget options |
| Voron 2.4 | `voron_24` | High-performance |

### LEGO-Optimized Print Settings

```json
{
  "layer_height": 0.12,
  "wall_count": 3,
  "infill_percent": 20,
  "xy_compensation": -0.08,
  "elephant_foot_compensation": 0.15,
  "support_enable": false
}
```

---

## 🔧 Supported CNC Machines

| Machine | Profile | Controller |
|---------|---------|------------|
| **Bantam Tools Desktop** | `bantam_explorer` | TinyG |
| **Shapeoko 4** | `shapeoko_4` | GRBL |
| **Generic GRBL Router** | `generic_grbl` | GRBL |
| **Haas VF-2** | `haas_vf2` | Haas NGC |
| **GRBL Laser Engraver** | `laser_grbl` | GRBL Laser |
| **CO2 Laser Cutter** | `laser_co2` | Ruida |

### CNC Tool Library

- 1mm, 2mm, 3mm, 6mm flat endmills
- 1mm, 2mm ball endmills
- 4.8mm drill (stud holes), 2.4mm drill (Technic pins)
- 30° engraving tool
- 45° chamfer mill

---

## ✨ Laser Engraving Presets

| Preset | Power | Speed | Use Case |
|--------|-------|-------|----------|
| `abs_engrave_light` | 15% | 1000mm/min | Subtle surface marks |
| `abs_engrave_medium` | 20% | 800mm/min | Standard engraving |
| `abs_engrave_deep` | 25% | 500mm/min | Tactile text/logos |
| `abs_vector_light` | 12% | 1500mm/min | Line drawings |
| `abs_cut_thin` | 80% | 200mm/min | Cut thin ABS |

⚠️ **Safety**: ABS releases toxic fumes when laser engraved. Use proper ventilation and air assist!

---

## 📐 LEGO Dimensions

| Dimension | Value (mm) |
|-----------|------------|
| Stud pitch | 8.0 |
| Stud diameter | 4.8 |
| Stud height | 1.7 |
| Plate height | 3.2 |
| Brick height | 9.6 |
| Wall thickness | 1.5 |
| Technic hole Ø | 4.8 |

---

## 🐳 Docker Commands

```bash
# Start slicer only
docker-compose up -d

# Start slicer + dashboard
docker-compose --profile full up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Health check
curl http://localhost:8766/health
curl http://localhost:5000/api/health  # (if dashboard running)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Fusion 360 not connecting | Check: `curl http://127.0.0.1:8767/health` |
| Slicer not responding | Check: `docker-compose logs slicer` |
| Port conflict | Check `.env` file for port settings |
| STL not exporting | Run `./scripts/setup-paths.sh` |
| IPv6 conflict | We use `127.0.0.1` instead of `localhost` |

---

## 📁 Project Structure

```
lego-mcp-fusion360/
├── mcp-server/              # MCP server for Claude Desktop
│   └── src/
│       ├── server.py        # Main MCP server
│       ├── fusion_client.py # HTTP client for Fusion 360
│       ├── slicer_client.py # HTTP client for slicer
│       └── tools/
│           ├── brick_tools.py
│           ├── export_tools.py
│           ├── milling_tools.py
│           ├── printing_tools.py
│           └── workflow_tools.py  # End-to-end workflows
├── fusion360-addin/         # Fusion 360 add-in
│   └── LegoMCP/
│       ├── LegoMCP.py       # Add-in entry point (port 8767)
│       ├── core/
│       │   ├── brick_modeler.py  # Parametric brick creation
│       │   └── cam_processor.py  # CAM + laser toolpaths
│       └── resources/
│           ├── tool_library.json   # CNC tools
│           ├── machines.json       # Machine configs
│           └── laser_presets.json  # Laser settings
├── dashboard/               # Flask web dashboard
├── slicer-service/          # Docker slicer container
│   └── profiles/
│       ├── bambu_p1s.json   # Bambu Lab P1S
│       ├── bambu_x1c.json   # Bambu Lab X1C
│       └── lego_bambu.json  # LEGO-optimized
├── scripts/
│   └── setup-paths.sh       # Export path symlinks
├── docker-compose.yml       # Container orchestration
└── .env                     # Environment config
```

---

## 🧪 Testing

```bash
# Test Fusion 360 connection
curl http://127.0.0.1:8767/health

# Test slicer service
curl http://localhost:8766/health

# Run unit tests
cd mcp-server && pytest tests/

# Integration tests (requires Fusion 360 running)
pytest tests/test_integration.py -v
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Credits

- LEGO is a trademark of the LEGO Group (not affiliated)
- Built with [Claude](https://anthropic.com) and MCP
- Powered by [Autodesk Fusion 360](https://www.autodesk.com/products/fusion-360)

---

<p align="center">Made with ❤️ using Claude AI + Fusion 360</p>
