# LEGO Brick Building Guide

## LegoMCP Industry 4.0 Digital Manufacturing Platform

Welcome to the complete guide for building LEGO-compatible bricks using the LegoMCP manufacturing system. This guide covers everything from brick design to production scheduling.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Creating Your First Brick](#creating-your-first-brick)
4. [Manufacturing Workflow](#manufacturing-workflow)
5. [Quality Control](#quality-control)
6. [Production Planning](#production-planning)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database
- 3D Printer (OctoPrint, Bambu Lab, or Prusa Connect)
- Optional: CNC Mill, Laser Engraver

### Start the System

```bash
# Start all services
docker compose up -d

# Verify services are running
docker compose ps
```

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:5000 | Main web interface |
| API | http://localhost:5000/api | REST API endpoints |
| Slicer | http://localhost:8766 | Slicing service |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LegoMCP Platform                          │
├─────────────────────────────────────────────────────────────────┤
│  Level 4: ERP                                                    │
│  ├── BOM Management (/api/erp/bom)                              │
│  ├── Costing (/api/erp/costing)                                 │
│  ├── Procurement (/api/erp/procurement)                         │
│  └── Demand Forecasting (/api/erp/demand)                       │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: MES/MOM                                                │
│  ├── Work Orders (/api/mes/work-orders)                         │
│  ├── Shop Floor (/api/mes/shop-floor)                           │
│  ├── OEE Dashboard (/api/mes/oee)                               │
│  ├── Quality (/api/quality)                                      │
│  └── Digital Twin (/api/twin)                                    │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: Equipment Control                                      │
│  ├── 3D Printers (FDM)                                          │
│  ├── CNC Mills                                                   │
│  └── Laser Engravers                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Creating Your First Brick

### Step 1: Design the Brick

Use Fusion 360 with the LegoMCP Add-in:

```python
# In Fusion 360, the MCP server provides these tools:

# Create a standard 2x4 brick
create_custom_brick(
    name="brick_2x4",
    width_studs=2,
    depth_studs=4,
    height_plates=3,
    brick_type="standard",
    hollow=True,
    studs=True,
    tubes=True
)

# Create a 1x2 slope brick
create_custom_brick(
    name="slope_1x2",
    width_studs=1,
    depth_studs=2,
    height_plates=3,
    brick_type="slope",
    slope_angle=45,
    slope_direction="front"
)
```

### Step 2: Export for Manufacturing

```bash
# Export to STL for 3D printing
curl -X POST http://localhost:5000/api/export/stl \
  -H "Content-Type: application/json" \
  -d '{
    "component": "brick_2x4",
    "output_path": "/output/brick_2x4.stl",
    "refinement": "high"
  }'
```

### Step 3: Generate Print Configuration

```bash
# Generate slicer configuration
curl -X POST http://localhost:5000/api/print/config \
  -H "Content-Type: application/json" \
  -d '{
    "stl_path": "/output/brick_2x4.stl",
    "printer": "prusa_mk3s",
    "material": "pla_generic",
    "quality": "lego"
  }'
```

---

## Manufacturing Workflow

### 1. Create a Part in the System

```bash
# Register a new part
curl -X POST http://localhost:5000/api/erp/bom \
  -H "Content-Type: application/json" \
  -d '{
    "part_number": "BRICK-2X4-RED",
    "name": "Standard 2x4 Brick - Red",
    "part_type": "finished_good",
    "uom": "each",
    "standard_cost": 0.15
  }'
```

### 2. Create a Work Order

```bash
# Create work order for 100 bricks
curl -X POST http://localhost:5000/api/mes/work-orders \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "<part-uuid>",
    "quantity": 100,
    "priority": "normal",
    "due_date": "2024-01-20T00:00:00Z"
  }'
```

### 3. Release to Production

```bash
# Release work order to shop floor
curl -X POST http://localhost:5000/api/mes/work-orders/<work-order-id>/release
```

### 4. Monitor Production

```bash
# View shop floor dashboard
curl http://localhost:5000/api/mes/shop-floor/dashboard

# View work queue for a printer
curl http://localhost:5000/api/mes/shop-floor/queue?work_center_id=<printer-uuid>

# View Andon display
curl http://localhost:5000/api/mes/shop-floor/andon
```

### 5. Record Production

```bash
# Start an operation
curl -X POST http://localhost:5000/api/mes/work-orders/<work-order-id>/operations/<op-id>/start

# Complete an operation
curl -X POST http://localhost:5000/api/mes/work-orders/<work-order-id>/operations/<op-id>/complete \
  -H "Content-Type: application/json" \
  -d '{
    "quantity_good": 98,
    "quantity_scrap": 2,
    "actual_time_minutes": 45
  }'
```

---

## Quality Control

### LEGO-Specific Quality Metrics

LEGO bricks must meet these critical specifications:

| Metric | Nominal | Tolerance |
|--------|---------|-----------|
| Stud Pitch | 8.0mm | ±0.02mm |
| Stud Diameter | 4.8mm | ±0.02mm |
| Wall Thickness | 1.6mm | ±0.05mm |
| Clutch Power | 1.0-3.0N | Optimal |

### Run Clutch Power Test

```bash
# Test clutch power for a batch
curl -X POST http://localhost:5000/api/quality/lego/clutch-power \
  -H "Content-Type: application/json" \
  -d '{
    "work_order_id": "<work-order-uuid>",
    "sample_size": 5,
    "forces_measured": [1.2, 1.5, 1.3, 1.8, 1.4]
  }'

# Response includes:
# - pass/fail status
# - average clutch power
# - individual measurements
# - recommended adjustments
```

### Run Full Compatibility Suite

```bash
# Run complete LEGO compatibility testing
curl -X POST http://localhost:5000/api/quality/lego/compatibility-suite \
  -H "Content-Type: application/json" \
  -d '{
    "work_order_id": "<work-order-uuid>",
    "sample_size": 5
  }'
```

### Statistical Process Control (SPC)

```bash
# Get control chart for stud diameter
curl http://localhost:5000/api/quality/spc/control-chart/stud_diameter

# Response includes:
# - UCL/LCL/Center Line
# - X-bar and R chart data
# - Out-of-control points
```

### Check for Out-of-Control Conditions

```bash
# Western Electric rules check
curl http://localhost:5000/api/quality/spc/out-of-control/stud_diameter

# Checks for:
# - Points beyond 3 sigma
# - 2 of 3 points beyond 2 sigma
# - 4 of 5 points beyond 1 sigma
# - 8 consecutive points on one side
```

---

## Production Planning

### Run MRP (Material Requirements Planning)

```bash
# Run MRP for specific parts
curl -X POST http://localhost:5000/api/mrp/planning/run \
  -H "Content-Type: application/json" \
  -d '{
    "part_ids": ["<part-uuid-1>", "<part-uuid-2>"],
    "horizon_days": 30,
    "lot_sizing_policy": "lot_for_lot"
  }'

# Returns:
# - Planned orders by period
# - Gross/net requirements
# - Projected inventory
```

### View Capacity Overview

```bash
# Get capacity for all work centers
curl http://localhost:5000/api/mrp/capacity/overview

# Identify bottlenecks
curl http://localhost:5000/api/mrp/capacity/bottlenecks
```

### Schedule a Work Order

```bash
# Finite capacity scheduling
curl -X POST http://localhost:5000/api/mrp/capacity/schedule/<work-order-id> \
  -H "Content-Type: application/json" \
  -d '{
    "direction": "forward",
    "start_date": "2024-01-15T08:00:00Z"
  }'
```

### Demand Forecasting

```bash
# Get demand forecast
curl http://localhost:5000/api/erp/demand/forecast/<part-id>?periods_ahead=6&method=exponential_smoothing

# Detect seasonality
curl http://localhost:5000/api/erp/demand/seasonality/<part-id>
```

---

## Digital Twin & Equipment Monitoring

### View Equipment Health

```bash
# Get health dashboard for all equipment
curl http://localhost:5000/api/twin/maintenance/health/dashboard

# Get specific equipment health
curl http://localhost:5000/api/twin/maintenance/health/<work-center-id>
```

### Predictive Maintenance

```bash
# Get maintenance recommendations
curl http://localhost:5000/api/twin/maintenance/recommendations

# Schedule maintenance
curl -X POST http://localhost:5000/api/twin/maintenance/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "work_center_id": "<work-center-uuid>",
    "maintenance_type": "preventive",
    "description": "Replace print bed surface",
    "scheduled_date": "2024-01-20T09:00:00Z",
    "estimated_hours": 1.0
  }'
```

### Real-Time Equipment State

```bash
# Get current digital twin state
curl http://localhost:5000/api/twin/state/<work-center-id>

# Get state history
curl http://localhost:5000/api/twin/state/<work-center-id>/history?hours=24

# Update equipment state
curl -X POST http://localhost:5000/api/twin/state/<work-center-id> \
  -H "Content-Type: application/json" \
  -d '{
    "state_type": "temperature",
    "state_data": {
      "bed": 60.0,
      "nozzle": 210.0
    }
  }'
```

### Production Simulation

```bash
# Simulate a production run
curl -X POST http://localhost:5000/api/twin/simulation/<work-center-id>/production \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "<part-uuid>",
    "quantity": 100,
    "parameters": {
      "speed_percent": 100,
      "quality_target": 0.98
    }
  }'
```

---

## API Reference

### Manufacturing (MES) - `/api/mes`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/shop-floor/dashboard` | GET | Shop floor overview |
| `/shop-floor/queue` | GET | Work queue by work center |
| `/shop-floor/andon` | GET | Andon display |
| `/work-orders` | GET/POST | Work order management |
| `/work-orders/<id>` | GET/PUT/DELETE | Single work order |
| `/work-orders/<id>/release` | POST | Release to production |
| `/work-centers` | GET/POST | Work center management |
| `/work-centers/<id>/status` | GET | Real-time status |
| `/oee/dashboard` | GET | OEE dashboard |
| `/oee/<work-center-id>` | GET | OEE for work center |
| `/oee/downtime/pareto` | GET | Downtime analysis |

### Quality - `/api/quality`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/inspections` | GET/POST | Inspection management |
| `/inspections/<id>` | GET | Single inspection |
| `/inspections/<id>/measure` | POST | Record measurement |
| `/measurements/cpk/<metric>` | GET | Process capability |
| `/lego/clutch-power` | POST | Clutch power test |
| `/lego/stud-fit` | POST | Stud fit test |
| `/lego/compatibility-suite` | POST | Full compatibility |
| `/spc/control-chart/<metric>` | GET | Control charts |
| `/spc/out-of-control/<metric>` | GET | OOC detection |

### ERP - `/api/erp`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/bom` | GET/POST | BOM management |
| `/bom/<part-id>/explode` | GET | Multi-level BOM |
| `/bom/<part-id>/where-used` | GET | Where-used analysis |
| `/costing/<part-id>` | GET | Cost breakdown |
| `/costing/variance/<wo-id>` | GET | Variance analysis |
| `/procurement/suppliers` | GET/POST | Supplier management |
| `/procurement/orders` | GET/POST | Purchase orders |
| `/demand/forecast/<part-id>` | GET | Demand forecast |
| `/demand/seasonality/<part-id>` | GET | Seasonality analysis |

### MRP - `/api/mrp`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/planning/run` | POST | Execute MRP |
| `/planning/planned-orders` | GET | View planned orders |
| `/planning/actions` | GET | MRP action messages |
| `/capacity/overview` | GET | Capacity overview |
| `/capacity/bottlenecks` | GET | Identify bottlenecks |
| `/capacity/schedule/<wo-id>` | POST | Schedule work order |
| `/capacity/gantt` | GET | Gantt chart data |

### Digital Twin - `/api/twin`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/state/<wc-id>` | GET/POST | Twin state |
| `/state/<wc-id>/history` | GET | State history |
| `/state/all` | GET | All twins |
| `/maintenance/health/<wc-id>` | GET | Equipment health |
| `/maintenance/dashboard` | GET | Health dashboard |
| `/maintenance/recommendations` | GET | Maintenance suggestions |
| `/maintenance/schedule` | POST | Schedule maintenance |
| `/simulation/production` | POST | Production simulation |
| `/simulation/what-if` | POST | What-if analysis |

---

## Troubleshooting

### Common Issues

#### Brick Doesn't Fit

1. Check clutch power: Should be 1.0-3.0N
2. Verify stud diameter: 4.8mm ±0.02mm
3. Check for warping: Ensure bed is level
4. Adjust print temperature and speed

```bash
# Run quality diagnostics
curl http://localhost:5000/api/quality/lego/adjustments/<work-order-id>
```

#### OEE is Low

```bash
# Check downtime pareto
curl http://localhost:5000/api/mes/oee/downtime/pareto

# Common causes:
# - Material changes (setup time)
# - Equipment issues (unplanned downtime)
# - Quality rejects (check SPC charts)
```

#### Equipment Shows CRITICAL Health

```bash
# Get detailed recommendations
curl http://localhost:5000/api/twin/maintenance/recommendations?work_center_id=<id>

# Schedule immediate maintenance
curl -X POST http://localhost:5000/api/twin/maintenance/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "work_center_id": "<id>",
    "maintenance_type": "emergency",
    "description": "Emergency maintenance - critical health",
    "scheduled_date": "2024-01-15T08:00:00Z",
    "estimated_hours": 4.0
  }'
```

### Logging

```bash
# View dashboard logs
docker compose logs dashboard

# View specific service logs
docker compose logs slicer
docker compose logs postgres
```

---

## LEGO Specifications Reference

### Critical Dimensions

| Dimension | Value | Tolerance |
|-----------|-------|-----------|
| Stud Pitch | 8.0mm | ±0.02mm |
| Stud Diameter | 4.8mm | ±0.02mm |
| Stud Height | 1.8mm | ±0.1mm |
| Wall Thickness | 1.6mm | ±0.05mm |
| Plate Height | 3.2mm | ±0.02mm |
| Brick Height (3 plates) | 9.6mm | ±0.05mm |
| Inter-brick Clearance | 0.1mm per side | - |
| Pin Hole Diameter | 4.9mm | ±0.02mm |
| Technic Hole Diameter | 4.85mm | ±0.02mm |
| Anti-stud ID | 6.51mm | ±0.02mm |

### Print Settings for PLA

| Setting | Value |
|---------|-------|
| Layer Height | 0.12mm |
| First Layer | 0.2mm |
| Nozzle Temp | 210°C |
| Bed Temp | 60°C |
| Infill | 100% (for clutch power) |
| Perimeters | 4 minimum |
| Print Speed | 40mm/s |

---

## Getting Help

- GitHub Issues: https://github.com/anthropics/claude-code/issues
- API Documentation: http://localhost:5000/api/docs
- Dashboard Help: http://localhost:5000/help

---

*LegoMCP Industry 4.0 - Building the future, one brick at a time.*
