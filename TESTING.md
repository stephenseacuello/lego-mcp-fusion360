# LEGO MCP Fusion 360 - Testing Guide

This document provides manual commands to test every component of the LEGO MCP system.

---

## Prerequisites

Before testing, ensure you have:
1. **Fusion 360** running with the LegoMCP add-in enabled
2. **Docker** installed and running
3. **Python 3.9+** with MCP server dependencies installed

---

## Phase 1: Initial Setup

```bash
# Navigate to project directory
cd /Users/stepheneacuello/Documents/lego_mcp_fusion360

# Setup export paths (symlinks for Docker)
chmod +x scripts/setup-paths.sh
./scripts/setup-paths.sh
```

Expected output:
```
LEGO MCP Path Setup
===================
Project directory: /Users/stepheneacuello/Documents/lego_mcp_fusion360
Export directory:  /Users/stepheneacuello/Documents/LegoMCP/exports
...
Path setup complete!
```

---

## Phase 2: Test Service Health

### Test Fusion 360 Add-in (Port 8767)

```bash
curl -s http://127.0.0.1:8767/health | jq .
```

Expected output:
```json
{
  "status": "ok",
  "service": "LegoMCP",
  "version": "1.0.0"
}
```

If this fails, ensure Fusion 360 is open and the add-in is running.

### Start and Test Slicer Service (Port 8766)

```bash
# Start slicer container
docker-compose up -d

# Wait a few seconds, then test
sleep 5
curl -s http://localhost:8766/health | jq .
```

Expected output:
```json
{
  "status": "healthy",
  "service": "lego-slicer"
}
```

### Start and Test Dashboard (Port 5000) - Optional

```bash
# Start dashboard (with slicer)
docker-compose --profile full up -d

# Test dashboard health
curl -s http://localhost:5000/api/health | jq .
```

Expected output:
```json
{
  "status": "healthy",
  "services": {...}
}
```

---

## Phase 3: Test Brick Creation in Fusion 360

### Create a 2x4 Brick

```bash
curl -X POST http://127.0.0.1:8767/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "create_brick",
    "params": {
      "studs_x": 2,
      "studs_y": 4,
      "height_units": 1,
      "hollow": true,
      "name": "test_2x4"
    }
  }' | jq .
```

Expected: Brick appears in Fusion 360 with correct dimensions.

### Create a Technic Brick

```bash
curl -X POST http://127.0.0.1:8767/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "create_technic_brick",
    "params": {
      "studs_x": 8,
      "studs_y": 1,
      "hole_axis": "x",
      "name": "test_technic"
    }
  }' | jq .
```

Expected: Technic beam with pin holes appears.

### Create a Slope Brick

```bash
curl -X POST http://127.0.0.1:8767/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "create_slope_brick",
    "params": {
      "studs_x": 2,
      "studs_y": 2,
      "height_units": 1,
      "slope_angle": 45,
      "direction": "front",
      "name": "test_slope"
    }
  }' | jq .
```

Expected: 45° slope brick appears.

---

## Phase 4: Test STL Export

```bash
# Export the test brick to STL
curl -X POST http://127.0.0.1:8767/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "export_stl",
    "params": {
      "component_name": "test_2x4",
      "output_path": "/Users/stepheneacuello/Documents/LegoMCP/exports/stl/test_2x4.stl",
      "refinement": "high"
    }
  }' | jq .
```

Verify export:
```bash
ls -la ~/Documents/LegoMCP/exports/stl/
```

Expected: `test_2x4.stl` file exists.

---

## Phase 5: Test Slicer Service

### List Available Printers

```bash
curl -s http://localhost:8766/printers | jq .
```

### List Available Materials

```bash
curl -s http://localhost:8766/materials | jq .
```

### Slice an STL for Bambu P1S

```bash
curl -X POST http://localhost:8766/slice/lego \
  -H "Content-Type: application/json" \
  -d '{
    "stl_path": "/input/stl/test_2x4.stl",
    "printer": "bambu_p1s",
    "material": "pla",
    "brick_type": "standard"
  }' | jq .
```

Expected: G-code file created in `/output/gcode/3dprint/`.

---

## Phase 6: Test MCP Server (if Claude Desktop not available)

### Start MCP Server in Test Mode

```bash
cd mcp-server
python -m src.server
```

The server will start listening on stdio for MCP messages.

---

## Phase 7: Test Dashboard UI

Open browser to:
```
http://localhost:5000
```

Check these pages:
- **Dashboard** (`/`): Status cards and quick actions
- **Catalog** (`/catalog`): Browse 323+ brick types
- **Builder** (`/builder`): Create custom bricks
- **Files** (`/files`): View exported files
- **Settings** (`/settings`): Configure preferences

---

## Phase 8: Verify Port Configuration

```bash
# Check all services are on correct ports
echo "=== Port Check ==="
echo "Fusion 360 (8767):"
curl -s http://127.0.0.1:8767/health | head -1

echo ""
echo "Slicer (8766):"
curl -s http://localhost:8766/health | head -1

echo ""
echo "Dashboard (5000):"
curl -s http://localhost:5000/api/health | head -1
```

---

## Phase 9: End-to-End Workflow Test

### Complete 3D Print Workflow

Test the full workflow: Create → Export → Slice → G-code

```bash
# 1. Create brick
curl -X POST http://127.0.0.1:8767/ \
  -H "Content-Type: application/json" \
  -d '{"command":"create_brick","params":{"studs_x":2,"studs_y":4,"height_units":1,"name":"workflow_test"}}' | jq .

# 2. Export STL
curl -X POST http://127.0.0.1:8767/ \
  -H "Content-Type: application/json" \
  -d '{"command":"export_stl","params":{"component_name":"workflow_test","output_path":"/Users/stepheneacuello/Documents/LegoMCP/exports/stl/workflow_test.stl"}}' | jq .

# 3. Slice for printing
curl -X POST http://localhost:8766/slice/lego \
  -H "Content-Type: application/json" \
  -d '{"stl_path":"/input/stl/workflow_test.stl","printer":"bambu_p1s","material":"pla"}' | jq .

# 4. Verify outputs
ls -la ~/Documents/LegoMCP/exports/stl/
ls -la output/gcode/3dprint/
```

---

## Troubleshooting

### Fusion 360 Not Responding

```bash
# Check if add-in is running
curl -s http://127.0.0.1:8767/health

# If not responding:
# 1. Open Fusion 360
# 2. Tools → Add-Ins → Scripts and Add-Ins
# 3. Find LegoMCP and click "Run"
```

### Docker Containers Not Starting

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f slicer

# Restart containers
docker-compose down && docker-compose up -d
```

### Port Already in Use

```bash
# Find what's using a port
lsof -i :8767
lsof -i :8766
lsof -i :5000

# Kill the process if needed
kill -9 <PID>
```

### STL Not Appearing in Slicer

```bash
# Check symlinks are correct
ls -la ~/Documents/LegoMCP/exports/stl/

# Re-run path setup
./scripts/setup-paths.sh
```

---

## Quick Reference

| Service | URL | Port |
|---------|-----|------|
| Fusion 360 | http://127.0.0.1:8767 | 8767 |
| Slicer | http://localhost:8766 | 8766 |
| Dashboard | http://localhost:5000 | 5000 |

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start slicer only |
| `docker-compose --profile full up -d` | Start slicer + dashboard |
| `docker-compose down` | Stop all containers |
| `docker-compose logs -f` | View container logs |

---

## Success Checklist

- [ ] Fusion 360 add-in responds on port 8767
- [ ] Slicer service healthy on port 8766
- [ ] Dashboard accessible on port 5000
- [ ] Brick creation works in Fusion 360
- [ ] STL export creates files in expected location
- [ ] Slicer generates G-code from STL
- [ ] Dashboard shows connected status
- [ ] End-to-end workflow completes successfully
