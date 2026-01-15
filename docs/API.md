# API Reference - LEGO MCP Fusion 360 v7.0

Complete API reference for all system endpoints and services.

---

## Table of Contents

1. [Services Overview](#services-overview)
2. [Fusion 360 Add-in API](#fusion-360-add-in-api)
3. [Dashboard Core API](#dashboard-core-api)
4. [Manufacturing API](#manufacturing-api)
5. [Quality API](#quality-api)
6. [Scheduling API](#scheduling-api)
7. [ERP API](#erp-api)
8. [MRP API](#mrp-api)
9. [Digital Twin API](#digital-twin-api)
10. [AI/MCP API](#aimcp-api)
11. [Sustainability API](#sustainability-api)
12. [Compliance API](#compliance-api)
13. [ROS2 Bridge API](#ros2-bridge-api)
14. [WebSocket Events](#websocket-events)

---

## Services Overview

| Service | Base URL | Description |
|---------|----------|-------------|
| **Dashboard** | `http://localhost:5000` | Main web application |
| **Fusion 360 Add-in** | `http://127.0.0.1:8765` | CAD operations |
| **Slicer Service** | `http://localhost:8081` | 3D print slicing |
| **OPC UA Server** | `opc.tcp://localhost:4840` | SCADA integration |
| **MTConnect Agent** | `http://localhost:5000/mtconnect` | CNC data streaming |

### Authentication

Most endpoints are currently open. For production:
- API Key: `X-API-Key: <key>` header
- JWT: `Authorization: Bearer <token>` header

### Response Format

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

# Fusion 360 Add-in API

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
  "version": "7.0.0"
}
```

## Execute Command
```
POST /
```

**Request Format:**
```json
{
  "command": "<command_name>",
  "params": { ... }
}
```

## Brick Creation Commands

### create_brick
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
| `studs_x` | int | Yes | Width (1-16) |
| `studs_y` | int | Yes | Depth (1-16) |
| `height_units` | float | No | Height (default: 1) |
| `hollow` | bool | No | Hollow interior (default: true) |

### create_plate
```json
{
  "command": "create_plate",
  "params": { "studs_x": 4, "studs_y": 4 }
}
```

### create_tile
```json
{
  "command": "create_tile",
  "params": { "studs_x": 2, "studs_y": 2 }
}
```

### create_slope
```json
{
  "command": "create_slope",
  "params": {
    "studs_x": 2,
    "studs_y": 3,
    "slope_angle": 45,
    "slope_direction": "front"
  }
}
```

### create_technic
```json
{
  "command": "create_technic",
  "params": { "studs_x": 1, "studs_y": 6, "hole_axis": "y" }
}
```

## Export Commands

### export_stl
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

### export_step
```json
{
  "command": "export_step",
  "params": { "component_name": "Brick_2x4", "output_path": "/output/brick.step" }
}
```

### export_3mf
```json
{
  "command": "export_3mf",
  "params": { "component_name": "Brick_2x4", "output_path": "/output/brick.3mf" }
}
```

## CAM Commands

### setup_cam
```json
{
  "command": "setup_cam",
  "params": { "component_name": "Brick_2x4", "machine": "grbl", "material": "abs" }
}
```

### generate_gcode
```json
{
  "command": "generate_gcode",
  "params": { "component_name": "Brick_2x4", "output_path": "/output/brick.nc" }
}
```

---

# Dashboard Core API

## Base URL
```
http://localhost:5000/api
```

## Workspace API

### GET /workspace/
Get workspace page

### GET /workspace/frame
Get camera frame (JPEG)

### GET /workspace/stream
Get camera stream (MJPEG)

### POST /workspace/detect
Run brick detection

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "brick_id": "brick_2x4",
      "color": "red",
      "confidence": 0.95,
      "bbox": [100, 100, 200, 200]
    }
  ],
  "count": 1
}
```

### GET /workspace/state
Get workspace state

### POST /workspace/clear
Clear workspace

---

# Manufacturing API

## Base URL
```
http://localhost:5000/api/manufacturing
```

## Work Orders

### GET /work-orders
List work orders

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `status` | string | Filter by status |
| `part_id` | string | Filter by part |
| `page` | int | Page number |
| `per_page` | int | Items per page |

**Response:**
```json
{
  "success": true,
  "data": {
    "work_orders": [...],
    "total": 100,
    "page": 1,
    "pages": 10
  }
}
```

### GET /work-orders/{id}
Get work order details

### POST /work-orders
Create work order

**Request:**
```json
{
  "part_id": "brick-2x4",
  "quantity": 100,
  "priority": "high",
  "due_date": "2024-12-31"
}
```

### PUT /work-orders/{id}
Update work order

### DELETE /work-orders/{id}
Delete work order

### POST /work-orders/{id}/release
Release work order to production

### POST /work-orders/{id}/complete
Complete work order

## Operations

### GET /operations
List operations

### GET /operations/{id}
Get operation details

### POST /operations/{id}/start
Start operation

### POST /operations/{id}/complete
Complete operation

**Request:**
```json
{
  "good_qty": 10,
  "scrap_qty": 1,
  "scrap_reason": "layer_shift"
}
```

## Shop Floor

### GET /shop-floor/page
Get shop floor dashboard page

### GET /shop-floor/work-centers
List work centers

**Response:**
```json
{
  "success": true,
  "data": {
    "work_centers": [
      {
        "id": "WC-PRINT-01",
        "name": "FDM Printer 1",
        "type": "fdm",
        "status": "running",
        "current_job": "WO-2024-001",
        "oee": 0.82
      }
    ]
  }
}
```

### GET /shop-floor/oee
Get OEE metrics

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_oee": 0.78,
    "availability": 0.92,
    "performance": 0.88,
    "quality": 0.96
  }
}
```

---

# Quality API

## Base URL
```
http://localhost:5000/api/quality
```

## SPC (Statistical Process Control)

### GET /spc/charts
List control charts

### GET /spc/charts/{id}
Get chart data

### POST /spc/charts
Create control chart

**Request:**
```json
{
  "name": "Stud Diameter",
  "chart_type": "xbar_r",
  "measurement_id": "stud_diameter",
  "ucl": 5.02,
  "lcl": 4.78,
  "target": 4.90
}
```

### POST /spc/data
Record SPC data point

**Request:**
```json
{
  "chart_id": "chart-001",
  "value": 4.92,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /spc/analysis/{id}
Get SPC analysis

**Response:**
```json
{
  "success": true,
  "data": {
    "cp": 1.33,
    "cpk": 1.25,
    "pp": 1.30,
    "ppk": 1.22,
    "violations": []
  }
}
```

## FMEA

### GET /fmea
List FMEAs

### GET /fmea/{id}
Get FMEA details

### POST /fmea
Create FMEA

**Request:**
```json
{
  "name": "Brick Printing FMEA",
  "process_id": "fdm-printing",
  "failure_modes": [
    {
      "mode": "Layer adhesion failure",
      "effect": "Weak structure",
      "cause": "Temperature too low",
      "severity": 7,
      "occurrence": 3,
      "detection": 4
    }
  ]
}
```

## QFD

### GET /qfd
List QFD matrices

### POST /qfd
Create QFD

**Request:**
```json
{
  "name": "Brick Quality Requirements",
  "customer_requirements": [
    {"name": "Tight fit", "weight": 9},
    {"name": "Smooth surface", "weight": 7}
  ],
  "engineering_characteristics": [
    {"name": "Stud diameter", "target": "4.90mm"},
    {"name": "Surface roughness", "target": "Ra 1.6"}
  ]
}
```

## Vision Inspection

### POST /vision/inspect
Run vision inspection

**Request:**
```json
{
  "image_source": "camera",
  "inspection_type": "defect_detection"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "defects": [],
    "quality_grade": "A",
    "confidence": 0.95
  }
}
```

---

# Scheduling API

## Base URL
```
http://localhost:5000/api/scheduling
```

## Scheduling

### GET /schedule
Get current schedule

### POST /schedule/optimize
Run schedule optimization

**Request:**
```json
{
  "algorithm": "cp_sat",
  "horizon_hours": 24,
  "objectives": ["minimize_makespan", "minimize_tardiness"],
  "constraints": {
    "max_concurrent_jobs": 5
  }
}
```

**Algorithms:**
- `fifo` - First In First Out
- `spt` - Shortest Processing Time
- `cp_sat` - Constraint Programming (OR-Tools)
- `nsga2` - Multi-objective genetic algorithm
- `rl_dqn` - Reinforcement learning
- `qaoa` - Quantum approximate optimization

### GET /schedule/gantt
Get Gantt chart data

**Response:**
```json
{
  "success": true,
  "data": {
    "operations": [
      {
        "id": "OP-001",
        "work_order_id": "WO-001",
        "work_center": "WC-PRINT-01",
        "start": "2024-01-15T08:00:00Z",
        "end": "2024-01-15T10:00:00Z"
      }
    ]
  }
}
```

### POST /schedule/manual
Manual schedule adjustment

**Request:**
```json
{
  "operation_id": "OP-001",
  "new_start": "2024-01-15T09:00:00Z",
  "new_work_center": "WC-PRINT-02"
}
```

---

# ERP API

## Base URL
```
http://localhost:5000/api/erp
```

## Vendors

### GET /vendors
List vendors

### GET /vendors/{id}
Get vendor details

### POST /vendors
Create vendor

### GET /vendors/{id}/scorecard
Get vendor scorecard

## Bill of Materials

### GET /bom
List BOMs

### GET /bom/{id}
Get BOM structure

### POST /bom
Create BOM

**Request:**
```json
{
  "part_id": "assembly-001",
  "revision": "A",
  "components": [
    {"part_id": "brick-2x4", "quantity": 4},
    {"part_id": "plate-4x4", "quantity": 2}
  ]
}
```

## Costing

### GET /costing/part/{id}
Get part cost breakdown

**Response:**
```json
{
  "success": true,
  "data": {
    "material_cost": 0.15,
    "labor_cost": 0.05,
    "overhead_cost": 0.03,
    "total_cost": 0.23
  }
}
```

## Financials

### GET /financials/ar
Get accounts receivable

### GET /financials/ap
Get accounts payable

### GET /financials/gl
Get general ledger

---

# MRP API

## Base URL
```
http://localhost:5000/api/mrp
```

## Planning

### POST /plan
Generate MRP plan

**Request:**
```json
{
  "planning_horizon_days": 30,
  "demand_forecast": true
}
```

### GET /plan/results
Get planning results

**Response:**
```json
{
  "success": true,
  "data": {
    "planned_orders": [...],
    "material_requirements": [...],
    "capacity_load": [...]
  }
}
```

## Materials

### GET /materials
List materials

### GET /materials/{id}/availability
Check material availability

### POST /materials/reserve
Reserve materials for work order

---

# Digital Twin API

## Base URL
```
http://localhost:5000/api/twin
```

## State

### GET /state/{entity_id}
Get twin state

**Response:**
```json
{
  "success": true,
  "data": {
    "entity_id": "printer-1",
    "state": {
      "status": "printing",
      "temperature": {"bed": 60, "nozzle": 210},
      "position": {"x": 100, "y": 50, "z": 10}
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### PUT /state/{entity_id}
Update twin state

## Simulation

### POST /simulate/thermal
Run thermal simulation (PINNs)

**Request:**
```json
{
  "bed_temp": 60,
  "nozzle_temp": 210,
  "ambient_temp": 25,
  "duration_sec": 60
}
```

### POST /simulate/stress
Run stress simulation

## Synchronization

### POST /sync
Force synchronization

### GET /sync/status
Get sync status

---

# AI/MCP API

## Base URL
```
http://localhost:5000/api/ai
```

## Copilot

### POST /copilot/ask
Ask AI copilot

**Request:**
```json
{
  "question": "Why did WC-PRINT-01 go down yesterday?",
  "context": {"include_history": true}
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "WC-PRINT-01 stopped at 14:32 due to a filament runout...",
    "sources": ["event_log", "equipment_status"],
    "tools_used": ["get_equipment_history", "analyze_downtime"]
  }
}
```

## MCP Tools

### GET /mcp/tools
List available MCP tools

**Response:**
```json
{
  "success": true,
  "data": {
    "tools": [
      {
        "name": "get_equipment_status",
        "description": "Get current equipment status",
        "parameters": [...]
      }
    ],
    "count": 150
  }
}
```

### POST /mcp/execute
Execute MCP tool

**Request:**
```json
{
  "tool": "get_equipment_status",
  "parameters": {"equipment_id": "WC-PRINT-01"}
}
```

## Agents

### GET /agents
List autonomous agents

### POST /agents/{id}/start
Start agent

### POST /agents/{id}/stop
Stop agent

---

# Sustainability API

## Base URL
```
http://localhost:5000/api/sustainability
```

## Carbon Footprint

### GET /carbon/footprint
Get factory carbon footprint

**Query Parameters:**
- `start_date` - Start of period
- `end_date` - End of period
- `scope` - 1, 2, or 3

**Response:**
```json
{
  "success": true,
  "data": {
    "total_kg_co2e": 1234.5,
    "scope_1": 100.0,
    "scope_2": 800.0,
    "scope_3": 334.5
  }
}
```

### GET /carbon/per-part/{id}
Get per-part carbon footprint

## LCA

### POST /lca/analyze
Run LCA analysis

**Request:**
```json
{
  "part_id": "brick-2x4",
  "quantity": 1000,
  "include_transport": true
}
```

## ESG

### GET /esg/metrics
Get ESG metrics

### GET /esg/report
Generate ESG report

---

# Compliance API

## Base URL
```
http://localhost:5000/api/compliance
```

## Audit Trail

### GET /audit
Get audit trail

**Query Parameters:**
- `entity_type` - Filter by entity
- `action` - Filter by action
- `user` - Filter by user
- `start_date` - Start of period
- `end_date` - End of period

### GET /audit/{id}
Get audit entry details

## ISO Tracking

### GET /iso/standards
List tracked standards

### GET /iso/{standard}/status
Get compliance status

**Response:**
```json
{
  "success": true,
  "data": {
    "standard": "ISO9001",
    "status": "compliant",
    "last_audit": "2024-01-01",
    "next_audit": "2025-01-01",
    "findings": []
  }
}
```

---

# ROS2 Bridge API

## Base URL
```
http://localhost:5000/api/ros2
```

## Lifecycle

### GET /lifecycle/states
Get all node states

**Response:**
```json
{
  "success": true,
  "data": {
    "nodes": {
      "grbl_node": {"state": "active", "available": true},
      "formlabs_node": {"state": "inactive", "available": true}
    }
  }
}
```

### POST /lifecycle/transition
Trigger lifecycle transition

**Request:**
```json
{
  "node_name": "grbl_node",
  "transition": "activate",
  "timeout_sec": 30
}
```

### POST /lifecycle/batch
Batch lifecycle transition

**Request:**
```json
{
  "transition": "activate",
  "nodes": ["grbl_node", "formlabs_node"]
}
```

## Topics

### GET /topics
List ROS2 topics

### GET /topics/{topic}/latest
Get latest message from topic

### POST /topics/{topic}/publish
Publish to topic

## Services

### GET /services
List ROS2 services

### POST /services/{service}/call
Call ROS2 service

**Request:**
```json
{
  "request": { ... }
}
```

## Equipment

### GET /equipment/status
Get all equipment status

### POST /equipment/{id}/command
Send equipment command

---

# WebSocket Events

## Connection

```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onopen = () => {
  // Subscribe to events
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['equipment', 'jobs', 'quality']
  }));
};
```

## Event Types

### equipment_status
```json
{
  "type": "equipment_status",
  "data": {
    "equipment_id": "WC-PRINT-01",
    "status": "running",
    "progress": 0.75
  }
}
```

### job_update
```json
{
  "type": "job_update",
  "data": {
    "job_id": "WO-2024-001",
    "status": "completed",
    "quantity_completed": 100
  }
}
```

### quality_alert
```json
{
  "type": "quality_alert",
  "data": {
    "alert_type": "spc_violation",
    "chart_id": "chart-001",
    "message": "Point outside control limits"
  }
}
```

### lifecycle_event
```json
{
  "type": "lifecycle_event",
  "data": {
    "node": "grbl_node",
    "transition": "activate",
    "new_state": "active",
    "success": true
  }
}
```

---

# Error Codes

| Code | Description |
|------|-------------|
| `400` | Bad Request - Invalid parameters |
| `401` | Unauthorized - Authentication required |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource doesn't exist |
| `409` | Conflict - Resource state conflict |
| `422` | Unprocessable Entity - Validation failed |
| `500` | Internal Server Error |
| `503` | Service Unavailable - ROS2 not connected |

---

# Rate Limiting

| Endpoint | Limit |
|----------|-------|
| `/api/ai/copilot/*` | 10 req/min |
| `/api/ros2/*` | 100 req/min |
| All other endpoints | 1000 req/min |

---

*LEGO MCP Fusion 360 v7.0 - Industry 4.0/5.0 Manufacturing Platform*
