# API Reference

Complete reference for all LEGO MCP Fusion 360 API endpoints.

---

## Services Overview

| Service | Base URL | Description |
|---------|----------|-------------|
| **Fusion 360 Add-in** | `http://127.0.0.1:8765` | Core CAD operations |
| **Dashboard** | `http://localhost:5000` | Web UI (optional) |
| **Slicer Service** | `http://localhost:8081` | 3D print slicing |

---

# Fusion 360 Add-in API

The core API for brick creation, export, and CAM operations.

## Base URL

```
http://127.0.0.1:8765
```

## Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

## Execute Command

```
POST /
```

All commands are sent as JSON to the root endpoint.

**Request Format:**
```json
{
  "command": "<command_name>",
  "params": { ... }
}
```

**Response Format:**
```json
{
  "success": true,
  "brick_id": "brick_12345",
  "component_name": "Brick_2x4",
  "dimensions": { ... },
  "volume_mm3": 1234.5
}
```

---

## Brick Creation Commands

### create_brick
Create a standard LEGO brick.

```json
{
  "command": "create_brick",
  "params": {
    "studs_x": 2,
    "studs_y": 4,
    "height_units": 1,
    "hollow": true,
    "name": "MyBrick"
  }
}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `studs_x` | int | Yes | Width in studs (1-16) |
| `studs_y` | int | Yes | Depth in studs (1-16) |
| `height_units` | float | No | Height in brick units (default: 1) |
| `hollow` | bool | No | Hollow interior (default: true) |
| `name` | string | No | Custom component name |

### create_plate
Create a plate (1/3 brick height).

```json
{
  "command": "create_plate",
  "params": {
    "studs_x": 4,
    "studs_y": 4,
    "name": "MyPlate"
  }
}
```

### create_tile
Create a tile (no studs on top).

```json
{
  "command": "create_tile",
  "params": {
    "studs_x": 2,
    "studs_y": 2,
    "name": "MyTile"
  }
}
```

### create_slope
Create a slope brick.

```json
{
  "command": "create_slope",
  "params": {
    "studs_x": 2,
    "studs_y": 3,
    "slope_angle": 45,
    "slope_direction": "front",
    "name": "MySlope"
  }
}
```

| Param | Type | Values |
|-------|------|--------|
| `slope_angle` | int | 33, 45, 65 |
| `slope_direction` | string | front, back, left, right |

### create_technic
Create a Technic brick with pin holes.

```json
{
  "command": "create_technic",
  "params": {
    "studs_x": 1,
    "studs_y": 6,
    "hole_axis": "y",
    "name": "TechnicBeam"
  }
}
```

| Param | Type | Description |
|-------|------|-------------|
| `hole_axis` | string | "x" or "y" - axis for 4.8mm pin holes |

### create_round
Create a cylindrical round brick.

```json
{
  "command": "create_round",
  "params": {
    "diameter_studs": 2,
    "height_units": 1.0,
    "name": "RoundBrick"
  }
}
```

### create_arch
Create an arch brick with curved cutout.

```json
{
  "command": "create_arch",
  "params": {
    "studs_x": 4,
    "studs_y": 1,
    "arch_height": 1,
    "name": "ArchBrick"
  }
}
```

---

## Export Commands

### export_stl
Export component to STL format.

```json
{
  "command": "export_stl",
  "params": {
    "component_name": "Brick_2x4",
    "output_path": "/output/brick.stl",
    "resolution": "high"
  }
}
```

| Param | Type | Values |
|-------|------|--------|
| `resolution` | string | low, medium, high |

**Response:**
```json
{
  "success": true,
  "path": "/output/brick.stl",
  "size_kb": 45.2,
  "triangle_count": 1248
}
```

### export_step
Export component to STEP format (CAD exchange).

```json
{
  "command": "export_step",
  "params": {
    "component_name": "Brick_2x4",
    "output_path": "/output/brick.step"
  }
}
```

### export_3mf
Export component to 3MF format (3D printing).

```json
{
  "command": "export_3mf",
  "params": {
    "component_name": "Brick_2x4",
    "output_path": "/output/brick.3mf"
  }
}
```

---

## CAM Commands

### setup_cam
Set up CAM machining for a component.

```json
{
  "command": "setup_cam",
  "params": {
    "component_name": "Brick_2x4",
    "machine": "grbl",
    "material": "abs"
  }
}
```

### generate_gcode
Generate G-code for CNC milling.

```json
{
  "command": "generate_gcode",
  "params": {
    "component_name": "Brick_2x4",
    "output_path": "/output/brick.nc",
    "machine": "grbl"
  }
}
```

---

## Preview Commands

### generate_preview
Generate a preview image of a component.

```json
{
  "command": "generate_preview",
  "params": {
    "component_name": "Brick_2x4",
    "output_path": "/output/preview.png",
    "view": "isometric",
    "width": 800,
    "height": 600
  }
}
```

| Param | Type | Values |
|-------|------|--------|
| `view` | string | front, top, isometric, isometric_bottom |

### generate_thumbnail
Generate a small thumbnail image.

```json
{
  "command": "generate_thumbnail",
  "params": {
    "component_name": "Brick_2x4",
    "output_path": "/output/thumb.png",
    "size": 256
  }
}
```

---

## Example: Python Client

```python
import requests

FUSION_URL = "http://127.0.0.1:8765"

def create_brick(width, depth, height=1):
    response = requests.post(FUSION_URL, json={
        "command": "create_brick",
        "params": {
            "studs_x": width,
            "studs_y": depth,
            "height_units": height
        }
    })
    return response.json()

def export_stl(component_name, output_path):
    response = requests.post(FUSION_URL, json={
        "command": "export_stl",
        "params": {
            "component_name": component_name,
            "output_path": output_path,
            "resolution": "high"
        }
    })
    return response.json()

# Create a 2x4 brick
result = create_brick(2, 4)
print(f"Created: {result['component_name']}")

# Export to STL
export_result = export_stl(result['component_name'], "/output/brick.stl")
print(f"Exported: {export_result['path']}")
```

---

# Dashboard API

Web dashboard for managing bricks (optional component).

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, no authentication is required. All endpoints are open.

## Response Format

All API responses return JSON:

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

On error:
```json
{
  "success": false,
  "data": null,
  "error": "Error message"
}
```

---

## Workspace API

### Get Workspace Page
```
GET /workspace/
```
Returns HTML page for the digital twin workspace.

### Get Camera Frame
```
GET /workspace/frame
```
Returns single JPEG frame from camera.

**Response:** `image/jpeg`

### Get Camera Stream
```
GET /workspace/stream
```
Returns MJPEG video stream.

**Response:** `multipart/x-mixed-replace`

### Run Detection
```
POST /workspace/detect
```
Runs brick detection on current camera frame.

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "brick_id": "brick_2x4",
      "brick_name": "Brick 2x4",
      "color": "red",
      "color_rgb": [201, 26, 9],
      "confidence": 0.95,
      "bbox": [100, 100, 200, 200],
      "center": [150, 150],
      "grid_position": "A1"
    }
  ],
  "detection_time_ms": 45.2,
  "count": 1
}
```

### Get Workspace State
```
GET /workspace/state
```
Returns current bricks on workspace.

**Response:**
```json
{
  "bricks": [...],
  "count": 5,
  "summary": {
    "by_color": {"red": 3, "blue": 2},
    "by_type": {"brick_2x4": 2, "plate_4x4": 3}
  }
}
```

### Clear Workspace
```
POST /workspace/clear
```
Removes all bricks from workspace tracking.

### Add All to Collection
```
POST /workspace/add-all-to-collection
```
Adds all stable workspace bricks to inventory.

### List Cameras
```
GET /workspace/cameras
```
Returns available cameras.

**Response:**
```json
{
  "cameras": [
    {"id": 0, "name": "Webcam", "available": true},
    {"id": 1, "name": "USB Camera", "available": true}
  ],
  "current": 0
}
```

### Open Camera
```
POST /workspace/camera/<id>/open
```
Opens specified camera by index.

### Close Camera
```
POST /workspace/camera/close
```
Closes current camera.

### Get/Set Configuration
```
GET /workspace/config
POST /workspace/config
```

**POST Body:**
```json
{
  "detection_threshold": 0.5,
  "roi": [0, 0, 1280, 720],
  "grid_size": 8
}
```

### Calibrate Baseplate
```
POST /workspace/calibrate
```

**Body:**
```json
{
  "corners": [
    [100, 100],
    [1100, 100],
    [1100, 600],
    [100, 600]
  ]
}
```

---

## Scan API

### Get Scan Page
```
GET /scan/
```
Returns HTML page for bulk scanning.

### Detect Bricks
```
POST /scan/detect
```
Runs detection and categorizes by confidence.

**Response:**
```json
{
  "success": true,
  "detections": [...],
  "high_confidence": [...],
  "low_confidence": [...],
  "detection_time_ms": 52.1,
  "total": 8
}
```

### Confirm All
```
POST /scan/confirm
```
Adds all pending detections to collection.

**Response:**
```json
{
  "success": true,
  "added": 5,
  "session": {
    "total_scanned": 20,
    "total_added": 15
  }
}
```

### Confirm Single Item
```
POST /scan/confirm-item
```

**Body:**
```json
{
  "brick_id": "brick_2x4",
  "color": "red"
}
```

### Reject Item
```
POST /scan/reject-item
```

**Body:**
```json
{
  "brick_id": "brick_2x4",
  "color": "red"
}
```

### Update Item
```
POST /scan/update-item
```

**Body:**
```json
{
  "old_brick_id": "brick_2x4",
  "old_color": "red",
  "brick_id": "brick_2x2",
  "color": "blue"
}
```

### Quick Add
```
POST /scan/quick-add
```

**Body:**
```json
{
  "brick_id": "brick_2x4",
  "color": "red",
  "quantity": 5
}
```

### Upload Image
```
POST /scan/upload
```

**Body:** `multipart/form-data` with `image` file

### Get Session
```
GET /scan/session
```
Returns current scan session data.

### Clear Session
```
POST /scan/session/clear
```
Resets scan session.

### Export Session
```
GET /scan/session/export
```
Returns session as JSON file.

---

## Collection API

### Get Collection Page
```
GET /collection/
```

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `q` | string | Search query |
| `category` | string | Filter by category |
| `color` | string | Filter by color |
| `sort` | string | Sort field (quantity, name, color, added_date) |
| `page` | int | Page number |
| `per_page` | int | Items per page (default 50) |

### Add Brick
```
POST /collection/add
```

**Body:**
```json
{
  "brick_id": "brick_2x4",
  "color": "red",
  "quantity": 5,
  "category": "brick",
  "notes": "From bulk lot"
}
```

### Remove Brick
```
POST /collection/remove
```

**Body:**
```json
{
  "brick_id": "brick_2x4",
  "color": "red",
  "quantity": 2
}
```

### Update Quantity
```
POST /collection/update
```

**Body:**
```json
{
  "brick_id": "brick_2x4",
  "color": "red",
  "quantity": 10
}
```

### Get Item
```
GET /collection/item/<brick_id>/<color>
```

### Get Statistics
```
GET /collection/stats
```

**Response:**
```json
{
  "total_pieces": 500,
  "unique_types": 45,
  "unique_colors": 12,
  "categories": {"brick": 200, "plate": 150},
  "estimated_value": 125.50
}
```

### Check Parts
```
POST /collection/check-parts
```

**Body:**
```json
{
  "parts": [
    {"brick_id": "brick_2x4", "color": "red", "quantity": 10},
    {"brick_id": "plate_4x4", "color": "blue", "quantity": 5}
  ]
}
```

**Response:**
```json
{
  "can_build": false,
  "have": [...],
  "missing": [...],
  "completion_percent": 75
}
```

### Export Collection
```
GET /collection/export?format=csv
GET /collection/export?format=json
```

### Import Collection
```
POST /collection/import
```

**Body:** `multipart/form-data`
| Field | Description |
|-------|-------------|
| `file` | File to import |
| `format` | csv, bricklink, rebrickable |
| `merge` | true/false (merge with existing) |

### Clear Collection
```
POST /collection/clear
```

**Body:**
```json
{
  "confirm": true
}
```

### Search Autocomplete
```
GET /collection/search?q=brick
```

### Bulk Add
```
POST /collection/bulk-add
```

**Body:**
```json
{
  "items": [
    {"brick_id": "brick_2x4", "color": "red", "quantity": 5},
    {"brick_id": "brick_2x2", "color": "blue", "quantity": 10}
  ]
}
```

---

## Builds API

### Get Builds Page
```
GET /builds/
```

### Get Build Detail
```
GET /builds/<build_id>
```

### Create Build
```
POST /builds/create
```

**Body:**
```json
{
  "name": "My Castle",
  "description": "A simple castle",
  "parts": [
    {"brick_id": "brick_2x4", "color": "red", "quantity": 20},
    {"brick_id": "slope_45_2x2", "color": "red", "quantity": 8}
  ]
}
```

### Delete Build
```
POST /builds/<build_id>/delete
```

### Check Build
```
GET /builds/<build_id>/check
```

**Response:**
```json
{
  "can_build": false,
  "parts_owned": 15,
  "parts_missing": 5,
  "completion_percent": 75,
  "have": [...],
  "missing": [...]
}
```

### Get Substitutes
```
GET /builds/<build_id>/substitutes
```

### Get Shopping List
```
GET /builds/<build_id>/shopping
```

**Response:**
```json
{
  "build_name": "My Castle",
  "items": [
    {"brick_id": "slope_45_2x2", "brick_name": "Slope 45° 2x2", "color": "red", "quantity": 4}
  ],
  "total_items": 1,
  "total_pieces": 4
}
```

### Export Build
```
GET /builds/<build_id>/export?format=csv
GET /builds/<build_id>/export?format=json
```

### Import Build
```
POST /builds/import
```

**Body:** `multipart/form-data`
| Field | Description |
|-------|-------------|
| `file` | File to import |
| `format` | rebrickable, bricklink |
| `name` | Optional name override |

### Get Suggestions
```
GET /builds/suggest?max_missing=5
```

---

## Insights API

### Get Insights Page
```
GET /insights/
```

### Get Statistics
```
GET /insights/stats
```

### Get Color Distribution
```
GET /insights/colors
```

**Response:**
```json
{
  "colors": [
    {"color": "red", "count": 150},
    {"color": "blue", "count": 120}
  ]
}
```

### Get Category Distribution
```
GET /insights/categories
```

### Get Top Bricks
```
GET /insights/top-bricks?limit=10
```

### Get Recommendations
```
GET /insights/recommendations
```

**Response:**
```json
{
  "recommendations": [
    {
      "type": "build",
      "icon": "🏗️",
      "title": "Ready to Build",
      "message": "You can build 'Simple House' with your current parts",
      "priority": "high",
      "action": "/builds/123"
    }
  ]
}
```

### What Can I Build?
```
GET /insights/what-can-i-build
```

### Gap Analysis
```
GET /insights/gaps
```
Returns parts that would unlock the most builds.

### Find Duplicates
```
GET /insights/duplicates
```
Returns bricks with high quantities.

### Value Estimate
```
GET /insights/value-estimate
```

---

## Catalog API

### List Bricks
```
GET /catalog/
```

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `category` | string | Filter by category |
| `search` | string | Search query |
| `page` | int | Page number |

### Get Brick Detail
```
GET /catalog/<brick_id>
```

### List Categories
```
GET /catalog/categories
```

### Search Bricks
```
GET /catalog/search?q=slope
```

---

## MCP Bridge API

### Execute Tool
```
POST /api/mcp/execute
```

**Body:**
```json
{
  "tool": "create_brick",
  "params": {
    "width": 2,
    "depth": 4,
    "height": 3
  }
}
```

### List Tools
```
GET /api/mcp/tools
```

### Tool Info
```
GET /api/mcp/tools/<tool_name>
```

### Health Check
```
GET /api/health
```

---

## Status API

### Get Status
```
GET /status/
GET /api/status
```

### Check Service
```
GET /status/check/<service>
```
Services: fusion360, slicer, mcp

---

## WebSocket Events

Connect to WebSocket at `/socket.io/`

### Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Server→Client | Connection established |
| `workspace:subscribe` | Client→Server | Subscribe to workspace updates |
| `workspace:update` | Server→Client | Workspace state changed |
| `collection:update` | Server→Client | Collection changed |
| `detection:result` | Server→Client | Detection completed |
| `status_update` | Server→Client | Service status changed |

### Example (JavaScript)

```javascript
const socket = io();

socket.on('connect', () => {
  socket.emit('workspace:subscribe');
});

socket.on('workspace:update', (data) => {
  console.log('Bricks on workspace:', data.count);
});
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request - invalid parameters |
| 404 | Not found - resource doesn't exist |
| 500 | Server error - check logs |

---

## Rate Limits

Currently no rate limits are enforced.

---

## Examples

### Add Brick to Collection (Python)

```python
import requests

response = requests.post('http://localhost:5000/collection/add', json={
    'brick_id': 'brick_2x4',
    'color': 'red',
    'quantity': 10
})

print(response.json())
```

### Check Build Parts (JavaScript)

```javascript
fetch('/builds/123/check')
  .then(r => r.json())
  .then(data => {
    if (data.can_build) {
      console.log('Ready to build!');
    } else {
      console.log(`Missing ${data.parts_missing} parts`);
    }
  });
```

### Stream Camera (HTML)

```html
<img src="/workspace/stream" alt="Camera Feed">
```
