# LEGO MCP Fusion 360 v6.0 - Complete Setup Guide

## World-Class Manufacturing Research Platform

This guide walks you through setting up and using every phase of the LEGO MCP manufacturing system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Starting the System](#3-starting-the-system)
4. [Phase 1: Brick Catalog & Design](#phase-1-brick-catalog--design)
5. [Phase 2: 3D Printing & Slicing](#phase-2-3d-printing--slicing)
6. [Phase 3: Quality Inspection](#phase-3-quality-inspection)
7. [Phase 4: Manufacturing Execution (MES)](#phase-4-manufacturing-execution-mes)
8. [Phase 5: ERP & Financials](#phase-5-erp--financials)
9. [Phase 6: MRP & Inventory](#phase-6-mrp--inventory)
10. [Phase 7: Scheduling & Optimization](#phase-7-scheduling--optimization)
11. [Phase 8: AI & Digital Twin](#phase-8-ai--digital-twin)
12. [Phase 9: Sustainability & Compliance](#phase-9-sustainability--compliance)
13. [Phase 10: Advanced Research Features](#phase-10-advanced-research-features)
14. [Troubleshooting](#troubleshooting)
15. [Quick Reference](#quick-reference)

---

## 1. Prerequisites

### Required Software

| Software | Version | Installation |
|----------|---------|--------------|
| **Python** | 3.9 - 3.12 | `brew install python@3.11` (macOS) |
| **Git** | Latest | `brew install git` |
| **pip** | Latest | Included with Python |

### Optional Software

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Containerized deployment |
| Node.js | 18+ | Frontend development |
| Fusion 360 | Latest | CAD integration |
| CUDA | 12+ | GPU acceleration for AI |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 5GB | 20GB+ |
| GPU | - | NVIDIA with CUDA 12+ |

---

## 2. Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install minimal dependencies for dashboard only
pip install flask flask-socketio requests sqlalchemy
```

### Step 4: Initialize Database (Optional)

```bash
cd dashboard
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
cd ..
```

### Step 5: Configure Environment

Create a `.env` file in the project root:

```bash
# Core Configuration
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-change-in-production
FLASK_PORT=5000

# Database
DATABASE_URL=sqlite:///lego_mcp.db

# Vision System (choose one)
DETECTION_BACKEND=mock  # Options: mock, yolo, roboflow
ROBOFLOW_API_KEY=your-api-key  # If using roboflow

# Printer Connection (configure your printer)
PRINTER_PROTOCOL=moonraker  # Options: moonraker, octoprint, bambu
PRINTER_HOST=localhost
PRINTER_PORT=7125

# For Bambu Lab printers
BAMBU_DEVICE_ID=your-device-id
BAMBU_ACCESS_TOKEN=your-token

# For OctoPrint
OCTOPRINT_API_KEY=your-api-key
```

---

## 3. Starting the System

### Quick Start (Dashboard Only)

```bash
cd dashboard
python app.py
```

Access at: **http://localhost:5000**

### Full Stack (Docker)

```bash
docker-compose up -d
```

Services:
- Dashboard: http://localhost:5000
- Slicer API: http://localhost:8081
- Fusion 360 API: http://localhost:8765

### Verify Installation

```bash
# Check system health
curl http://localhost:5000/api/health

# Expected response:
{
  "status": "healthy",
  "version": "6.0.0",
  "modules": {
    "ai": true,
    "quality": true,
    "erp": true,
    "mrp": true,
    "sustainability": true
  }
}
```

---

## Phase 1: Brick Catalog & Design

### Overview
Browse 323+ LEGO brick types and design custom bricks.

### Dashboard Access
- **URL**: http://localhost:5000/catalog
- **Keyboard Shortcut**: Press `C`

### API Endpoints

```bash
# List all brick categories
curl http://localhost:5000/api/catalog/categories

# Get bricks by category
curl http://localhost:5000/api/catalog/bricks?category=basic

# Get specific brick details
curl http://localhost:5000/api/catalog/bricks/3001

# Search bricks
curl "http://localhost:5000/api/catalog/search?q=2x4"
```

### Fusion 360 Integration

If using Fusion 360 add-in:

```bash
# Create a 2x4 brick
curl -X POST http://localhost:8765/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "create_brick",
    "params": {
      "studs_x": 2,
      "studs_y": 4,
      "height_units": 1,
      "hollow": true
    }
  }'

# Create a plate
curl -X POST http://localhost:8765/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "create_plate",
    "params": {"studs_x": 4, "studs_y": 4}
  }'

# Create a slope brick
curl -X POST http://localhost:8765/ \
  -H "Content-Type: application/json" \
  -d '{
    "command": "create_slope",
    "params": {
      "studs_x": 2,
      "studs_y": 3,
      "slope_angle": 45,
      "slope_direction": "front"
    }
  }'
```

### Collection Management

```bash
# Add to your collection
curl -X POST http://localhost:5000/api/collection \
  -H "Content-Type: application/json" \
  -d '{"part_number": "3001", "color": "red", "quantity": 50}'

# View collection
curl http://localhost:5000/api/collection

# Update quantity
curl -X PUT http://localhost:5000/api/collection/3001 \
  -H "Content-Type: application/json" \
  -d '{"quantity": 75}'
```

---

## Phase 2: 3D Printing & Slicing

### Overview
Generate G-code and send to 3D printers for LEGO brick production.

### Dashboard Access
- **URL**: http://localhost:5000/api/manufacturing/shop-floor/page

### Slicer API

```bash
# Slice STL file
curl -X POST http://localhost:8081/slice \
  -H "Content-Type: application/json" \
  -d '{
    "stl_path": "/path/to/brick.stl",
    "profile": "lego_quality",
    "settings": {
      "layer_height": 0.12,
      "infill": 20,
      "supports": false
    }
  }'

# Available profiles
curl http://localhost:8081/profiles

# Estimate print time
curl -X POST http://localhost:8081/estimate \
  -H "Content-Type: application/json" \
  -d '{"stl_path": "/path/to/brick.stl"}'
```

### Printer Connection

**Moonraker (Klipper)**:
```bash
# Check printer status
curl http://localhost:5000/api/manufacturing/printer/status

# Start print
curl -X POST http://localhost:5000/api/manufacturing/print \
  -H "Content-Type: application/json" \
  -d '{
    "gcode_path": "/path/to/brick.gcode",
    "printer_id": "printer-1"
  }'

# Pause print
curl -X POST http://localhost:5000/api/manufacturing/printer/pause

# Resume print
curl -X POST http://localhost:5000/api/manufacturing/printer/resume
```

**Bambu Lab**:
```bash
# Configure Bambu printer in .env:
PRINTER_PROTOCOL=bambu
BAMBU_DEVICE_ID=your-device-id
BAMBU_ACCESS_TOKEN=your-token

# Send to Bambu printer
curl -X POST http://localhost:5000/api/manufacturing/bambu/print \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/brick.3mf",
    "plate": 1,
    "ams_mapping": [0, 1, 2, 3]
  }'
```

### LEGO Parameter Optimization

```bash
# Get optimized parameters for material
curl -X POST http://localhost:5000/api/manufacturing/optimize-params \
  -H "Content-Type: application/json" \
  -d '{
    "material": "PLA",
    "brick_type": "2x4",
    "printer": "Bambu A1"
  }'

# Response includes:
# - Optimal layer height
# - Temperature settings
# - Speed adjustments
# - Dimensional compensation
```

---

## Phase 3: Quality Inspection

### Overview
Vision-based defect detection, SPC, FMEA, and QFD.

### Dashboard Access
- **Quality Dashboard**: http://localhost:5000/api/quality/dashboard/page
- **SPC Dashboard**: http://localhost:5000/api/quality/spc/page
- **FMEA Dashboard**: http://localhost:5000/api/quality/fmea/page
- **House of Quality**: http://localhost:5000/api/quality/qfd/hoq/page

### Vision Inspection

```bash
# Run defect detection on image
curl -X POST http://localhost:5000/api/quality/inspect \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/brick_photo.jpg"}'

# Response:
{
  "defects": [
    {"type": "surface_scratch", "confidence": 0.92, "location": [120, 45]},
    {"type": "warping", "confidence": 0.87, "severity": "minor"}
  ],
  "overall_quality": "acceptable",
  "score": 85.5
}

# Batch inspection
curl -X POST http://localhost:5000/api/quality/inspect/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
    "part_id": "brick-2x4"
  }'
```

### Statistical Process Control (SPC)

```bash
# Get control chart data
curl "http://localhost:5000/api/quality/spc/brick-2x4?parameter=stud_diameter"

# Record measurement
curl -X POST http://localhost:5000/api/quality/spc/measurements \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "parameter": "stud_diameter",
    "value": 4.81,
    "unit": "mm"
  }'

# Get process capability (Cpk)
curl http://localhost:5000/api/quality/spc/brick-2x4/capability

# Response:
{
  "cp": 1.45,
  "cpk": 1.33,
  "within_spec": true,
  "recommendation": "Process is capable"
}
```

### FMEA (Failure Mode Analysis)

```bash
# Create FMEA for a component
curl -X POST http://localhost:5000/api/quality/fmea \
  -H "Content-Type: application/json" \
  -d '{
    "component": "2x4 Brick Stud",
    "failure_modes": [
      {
        "mode": "Stud too short",
        "effect": "Poor clutch power",
        "cause": "Under-extrusion",
        "severity": 7,
        "occurrence": 4,
        "detection": 3
      }
    ]
  }'

# Get RPN rankings
curl http://localhost:5000/api/quality/fmea/brick-2x4/rpn

# Get AI-suggested mitigations
curl http://localhost:5000/api/quality/fmea/brick-2x4/mitigations
```

### QFD / House of Quality

```bash
# Create House of Quality
curl -X POST http://localhost:5000/api/quality/qfd/hoq \
  -H "Content-Type: application/json" \
  -d '{
    "project": "LEGO 2x4 Brick",
    "customer_requirements": [
      {"requirement": "Connects firmly", "importance": 9},
      {"requirement": "Easy to separate", "importance": 8},
      {"requirement": "Compatible with LEGO", "importance": 10}
    ],
    "technical_requirements": [
      {"requirement": "Stud diameter", "target": 4.8, "unit": "mm"},
      {"requirement": "Clutch force", "target": 2.0, "unit": "N"}
    ]
  }'

# Get relationship matrix
curl http://localhost:5000/api/quality/qfd/hoq/brick-2x4/matrix
```

---

## Phase 4: Manufacturing Execution (MES)

### Overview
Work orders, shop floor control, OEE tracking.

### Dashboard Access
- **Shop Floor**: http://localhost:5000/api/manufacturing/shop-floor/page
- **Work Orders**: http://localhost:5000/api/mes/work-orders/page
- **WIP Tracking**: http://localhost:5000/api/mes/work-orders/wip/page
- **OEE Dashboard**: http://localhost:5000/api/manufacturing/oee/page

### Work Order Management

```bash
# Create work order
curl -X POST http://localhost:5000/api/manufacturing/work-orders \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "quantity": 100,
    "priority": "high",
    "scheduled_start": "2025-01-06T08:00:00"
  }'

# List work orders
curl "http://localhost:5000/api/manufacturing/work-orders?status=released"

# Get work order details
curl http://localhost:5000/api/manufacturing/work-orders/WO-2025-00001

# Release work order
curl -X POST http://localhost:5000/api/manufacturing/work-orders/WO-2025-00001/release

# Start work order
curl -X POST http://localhost:5000/api/manufacturing/work-orders/WO-2025-00001/start

# Report production
curl -X POST http://localhost:5000/api/manufacturing/work-orders/WO-2025-00001/complete \
  -H "Content-Type: application/json" \
  -d '{
    "quantity_completed": 98,
    "quantity_scrapped": 2
  }'
```

### Operation Tracking

```bash
# Start operation
curl -X POST http://localhost:5000/api/manufacturing/work-orders/operations/OP-001/start \
  -H "Content-Type: application/json" \
  -d '{"work_center_id": "WC-PRINTER-01"}'

# Complete operation
curl -X POST http://localhost:5000/api/manufacturing/work-orders/operations/OP-001/complete \
  -H "Content-Type: application/json" \
  -d '{
    "quantity_completed": 98,
    "quantity_scrapped": 2,
    "scrap_reason": "Surface defects"
  }'

# Report scrap
curl -X POST http://localhost:5000/api/manufacturing/work-orders/operations/OP-001/report-scrap \
  -H "Content-Type: application/json" \
  -d '{
    "quantity": 2,
    "reason_code": "DEFECT",
    "notes": "Layer adhesion failure"
  }'
```

### OEE Tracking

```bash
# Get OEE for work center
curl http://localhost:5000/api/manufacturing/oee/WC-PRINTER-01

# Response:
{
  "availability": 92.5,
  "performance": 88.0,
  "quality": 98.0,
  "oee": 79.8,
  "target": 85.0,
  "status": "below_target"
}

# Record downtime
curl -X POST http://localhost:5000/api/manufacturing/oee/downtime \
  -H "Content-Type: application/json" \
  -d '{
    "work_center_id": "WC-PRINTER-01",
    "start_time": "2025-01-06T10:00:00",
    "end_time": "2025-01-06T10:30:00",
    "reason": "Filament change",
    "planned": true
  }'
```

---

## Phase 5: ERP & Financials

### Overview
Vendor management, AR, AP, GL, costing, and BOM.

### Dashboard Access
- **Vendors**: http://localhost:5000/api/erp/vendors/page
- **Accounts Receivable**: http://localhost:5000/api/erp/financials/ar/page
- **Accounts Payable**: http://localhost:5000/api/erp/financials/ap/page
- **General Ledger**: http://localhost:5000/api/erp/financials/gl/page
- **Customer Orders**: http://localhost:5000/api/erp/orders/page
- **BOM**: http://localhost:5000/api/erp/bom/page
- **Costing**: http://localhost:5000/api/erp/costing/page

### Vendor Management

```bash
# Create vendor
curl -X POST http://localhost:5000/api/erp/vendors \
  -H "Content-Type: application/json" \
  -d '{
    "code": "SUP001",
    "name": "Acme Plastics",
    "vendor_type": "raw_material",
    "payment_terms": "net_30",
    "lead_time_days": 14
  }'

# List vendors
curl http://localhost:5000/api/erp/vendors

# Get vendor scorecard
curl http://localhost:5000/api/erp/vendors/SUP001/scorecard

# Add certification
curl -X POST http://localhost:5000/api/erp/vendors/SUP001/certifications \
  -H "Content-Type: application/json" \
  -d '{
    "certification_type": "ISO9001",
    "certificate_number": "QMS-2024-12345",
    "issue_date": "2024-01-15",
    "expiry_date": "2027-01-14",
    "issuing_body": "TUV Rheinland"
  }'
```

### Accounts Receivable

```bash
# Create customer
curl -X POST http://localhost:5000/api/erp/financials/ar/customers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LEGO Builders Inc",
    "credit_limit": 50000,
    "payment_terms": "net_30"
  }'

# Create invoice
curl -X POST http://localhost:5000/api/erp/financials/ar/invoices \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001",
    "line_items": [
      {"description": "2x4 Red Bricks", "quantity": 1000, "unit_price": 0.25},
      {"description": "1x2 Blue Plates", "quantity": 500, "unit_price": 0.15}
    ]
  }'

# Record payment
curl -X POST http://localhost:5000/api/erp/financials/ar/payments \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001",
    "amount": 325.00,
    "payment_method": "ACH",
    "invoices": ["INV-2025-00001"]
  }'

# Get AR aging
curl http://localhost:5000/api/erp/financials/ar/aging
```

### Accounts Payable

```bash
# Create vendor bill
curl -X POST http://localhost:5000/api/erp/financials/ap/bills \
  -H "Content-Type: application/json" \
  -d '{
    "vendor_id": "SUP001",
    "bill_number": "INV-12345",
    "purchase_order_ref": "PO-2025-00100",
    "line_items": [
      {"description": "ABS Pellets", "quantity": 100, "unit": "kg", "unit_price": 2.50}
    ]
  }'

# Approve bill
curl -X POST http://localhost:5000/api/erp/financials/ap/bills/BILL-001/approve

# Create payment
curl -X POST http://localhost:5000/api/erp/financials/ap/payments \
  -H "Content-Type: application/json" \
  -d '{
    "vendor_id": "SUP001",
    "amount": 250.00,
    "payment_method": "check",
    "bills": ["BILL-001"]
  }'

# Get AP aging
curl http://localhost:5000/api/erp/financials/ap/aging

# Get 1099 summary
curl http://localhost:5000/api/erp/financials/ap/1099/summary
```

### General Ledger

```bash
# Create journal entry
curl -X POST http://localhost:5000/api/erp/financials/gl/journal-entries \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Monthly depreciation",
    "lines": [
      {"account": "6300", "debit": 1500, "description": "Depreciation expense"},
      {"account": "1510", "credit": 1500, "description": "Accumulated depreciation"}
    ]
  }'

# Get trial balance
curl http://localhost:5000/api/erp/financials/gl/trial-balance

# Get income statement
curl "http://localhost:5000/api/erp/financials/gl/income-statement?period=2025-01"

# Get balance sheet
curl http://localhost:5000/api/erp/financials/gl/balance-sheet
```

### BOM Management

```bash
# Create BOM
curl -X POST http://localhost:5000/api/erp/bom \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "components": [
      {"material": "ABS", "quantity": 2.5, "unit": "g"},
      {"material": "Pigment-Red", "quantity": 0.05, "unit": "g"}
    ]
  }'

# Get BOM
curl http://localhost:5000/api/erp/bom/brick-2x4

# Explode multi-level BOM
curl http://localhost:5000/api/erp/bom/brick-2x4/explode

# Get standard cost
curl http://localhost:5000/api/erp/costing/brick-2x4/standard
```

---

## Phase 6: MRP & Inventory

### Overview
Material planning, inventory management, filament tracking.

### Dashboard Access
- **Materials**: http://localhost:5000/api/mrp/materials/page
- **MRP Planning**: http://localhost:5000/api/mrp/planning/page

### Material Master

```bash
# Add filament spool
curl -X POST http://localhost:5000/api/mrp/materials/spools \
  -H "Content-Type: application/json" \
  -d '{
    "material_type": "pla",
    "brand": "Prusament",
    "color": "Galaxy Black",
    "initial_weight_g": 1000,
    "diameter_mm": 1.75
  }'

# Get inventory summary
curl http://localhost:5000/api/mrp/materials/summary

# Record consumption
curl -X POST http://localhost:5000/api/mrp/materials/consumption \
  -H "Content-Type: application/json" \
  -d '{
    "spool_id": "SPOOL-001",
    "weight_used_g": 50,
    "work_order": "WO-2025-00001"
  }'

# Check reorder points
curl http://localhost:5000/api/mrp/materials/reorder-alerts
```

### MRP Planning

```bash
# Run MRP
curl -X POST http://localhost:5000/api/mrp/planning/run \
  -H "Content-Type: application/json" \
  -d '{"horizon_days": 30}'

# Get planned orders
curl http://localhost:5000/api/mrp/planning/planned-orders

# Get net requirements
curl http://localhost:5000/api/mrp/requirements/brick-2x4

# Capacity planning (RCCP)
curl http://localhost:5000/api/mrp/capacity/rough-cut
```

---

## Phase 7: Scheduling & Optimization

### Overview
Production scheduling with QAOA, NSGA-II, and RL optimization.

### Dashboard Access
- **Scheduling**: http://localhost:5000/api/scheduling/dashboard/page

### Schedule Optimization

```bash
# CP-SAT optimization
curl -X POST http://localhost:5000/api/scheduling/optimize/cp-sat \
  -H "Content-Type: application/json" \
  -d '{
    "work_orders": ["WO-001", "WO-002", "WO-003"],
    "objective": "minimize_makespan"
  }'

# NSGA-II multi-objective
curl -X POST http://localhost:5000/api/scheduling/optimize/nsga2 \
  -H "Content-Type: application/json" \
  -d '{
    "work_orders": ["WO-001", "WO-002", "WO-003"],
    "objectives": ["makespan", "tardiness", "setup_time"]
  }'

# Get Pareto front
curl http://localhost:5000/api/scheduling/optimize/pareto-front

# What-if analysis
curl -X POST http://localhost:5000/api/scheduling/optimize/what-if \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "add_printer",
    "parameters": {"printer_id": "PRINTER-03"}
  }'
```

---

## Phase 8: AI & Digital Twin

### Overview
AI copilot, autonomous agents, digital twin with PINNs.

### Dashboard Access
- **AI Copilot**: http://localhost:5000/api/ai/copilot/page
- **Digital Twin**: http://localhost:5000/api/twin/dashboard/page
- **Agent Orchestration**: http://localhost:5000/api/ai/orchestration/page

### AI Copilot

```bash
# Natural language query
curl -X POST http://localhost:5000/api/ai/copilot/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the optimal print parameters for a 2x4 brick?"}'

# Get recommendations
curl -X POST http://localhost:5000/api/ai/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "context": "production",
    "part_id": "brick-2x4"
  }'

# Explain anomaly
curl -X POST http://localhost:5000/api/ai/explain \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "ANO-001",
    "type": "quality_drop"
  }'
```

### Autonomous Agents

```bash
# Quality Agent analysis
curl -X POST http://localhost:5000/api/ai/agents/quality/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "measurements": [4.78, 4.82, 4.80, 4.79, 4.83]
  }'

# Scheduling Agent optimization
curl -X POST http://localhost:5000/api/ai/agents/scheduling/optimize \
  -H "Content-Type: application/json" \
  -d '{"work_orders": ["WO-001", "WO-002"]}'

# Maintenance Agent prediction
curl -X POST http://localhost:5000/api/ai/agents/maintenance/predict \
  -H "Content-Type: application/json" \
  -d '{"printer_id": "PRINTER-01"}'
```

### Digital Twin

```bash
# Get twin state
curl http://localhost:5000/api/twin/state/printer-1

# Run thermal simulation (PINN)
curl -X POST http://localhost:5000/api/twin/simulate/thermal \
  -H "Content-Type: application/json" \
  -d '{
    "bed_temp": 60,
    "nozzle_temp": 210,
    "ambient_temp": 25
  }'

# Sync with physical printer
curl -X POST http://localhost:5000/api/twin/sync/printer-1

# Get prediction vs actual
curl http://localhost:5000/api/twin/accuracy/printer-1
```

### Causal AI

```bash
# Root cause analysis
curl -X POST http://localhost:5000/api/ai/causal/root-cause \
  -H "Content-Type: application/json" \
  -d '{
    "defect_type": "warping",
    "part_id": "brick-2x4"
  }'

# Counterfactual query
curl -X POST http://localhost:5000/api/ai/causal/counterfactual \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {"temp": 200, "speed": 60, "defect_rate": 0.05},
    "intervention": {"temp": 210},
    "outcome": "defect_rate"
  }'
```

---

## Phase 9: Sustainability & Compliance

### Overview
LCA, carbon tracking, ESG reporting, compliance management.

### Dashboard Access
- **Sustainability**: http://localhost:5000/api/sustainability/dashboard/page
- **Compliance**: http://localhost:5000/api/compliance/dashboard/page

### Life Cycle Assessment

```bash
# Calculate LCA for part
curl -X POST http://localhost:5000/api/sustainability/lca/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "quantity": 1000
  }'

# Get carbon footprint
curl http://localhost:5000/api/sustainability/carbon/brick-2x4

# Compare materials
curl -X POST http://localhost:5000/api/sustainability/compare \
  -H "Content-Type: application/json" \
  -d '{
    "materials": ["PLA", "ABS", "PETG"],
    "metrics": ["carbon", "energy", "water"]
  }'
```

### ESG Reporting

```bash
# Get ESG summary
curl http://localhost:5000/api/sustainability/esg/summary

# Generate sustainability report
curl -X POST http://localhost:5000/api/sustainability/reports/generate \
  -H "Content-Type: application/json" \
  -d '{"period": "2025-Q1", "format": "pdf"}'
```

### Compliance

```bash
# Check compliance status
curl http://localhost:5000/api/compliance/status

# Medical device compliance (ISO 13485)
curl http://localhost:5000/api/compliance/medical/iso13485

# Audit preparation
curl http://localhost:5000/api/compliance/audit/prepare \
  -H "Content-Type: application/json" \
  -d '{"standard": "ISO9001", "scope": "production"}'
```

---

## Phase 10: Advanced Research Features

### Overview
Experiment tracking, hypothesis testing, generative design.

### Experiment Tracking

```bash
# Create experiment
curl -X POST http://localhost:5000/api/research/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Temperature Impact Study",
    "hypothesis": "Higher nozzle temperature reduces warping",
    "parameters": {
      "temp_range": [200, 210, 220],
      "sample_size": 30
    }
  }'

# Log run
curl -X POST http://localhost:5000/api/research/experiments/EXP-001/runs \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {"nozzle_temp": 210},
    "metrics": {"warp_rate": 0.02, "surface_quality": 92}
  }'

# Compare runs
curl http://localhost:5000/api/research/experiments/EXP-001/compare
```

### Statistical Analysis

```bash
# A/B test analysis
curl -X POST http://localhost:5000/api/research/statistics/ab-test \
  -H "Content-Type: application/json" \
  -d '{
    "control": {"n": 100, "successes": 85},
    "treatment": {"n": 100, "successes": 92}
  }'

# Power analysis
curl -X POST http://localhost:5000/api/research/statistics/power-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "effect_size": 0.3,
    "alpha": 0.05,
    "power": 0.8
  }'
```

### Generative Design

```bash
# Topology optimization
curl -X POST http://localhost:5000/api/generative/topology-optimize \
  -H "Content-Type: application/json" \
  -d '{
    "design_space": {"x": 16, "y": 32, "z": 9.6},
    "loads": [{"point": [8, 16, 9.6], "force": [0, 0, -10]}],
    "constraints": {"volume_fraction": 0.3}
  }'

# Clutch power optimization
curl -X POST http://localhost:5000/api/generative/lego/clutch-optimize \
  -H "Content-Type: application/json" \
  -d '{
    "target_force": 2.0,
    "material": "PLA",
    "printer": "Bambu A1"
  }'
```

---

## Troubleshooting

### Common Issues

#### "Module not found" Error
```bash
pip install -r requirements.txt
```

#### Database Errors
```bash
cd dashboard
rm -f lego_mcp.db  # Reset database
flask db upgrade
```

#### Port Already in Use
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9

# Or use different port
FLASK_PORT=5001 python app.py
```

#### Vision System Not Working
```bash
# Use mock detector (no dependencies)
export DETECTION_BACKEND=mock
python app.py
```

#### Printer Connection Failed
```bash
# Test Moonraker
curl http://YOUR_PRINTER_IP:7125/printer/info

# Test OctoPrint
curl -H "X-Api-Key: YOUR_KEY" http://YOUR_PRINTER_IP/api/version
```

### Logs

```bash
# View application logs
tail -f dashboard/logs/app.log

# Docker logs
docker-compose logs -f dashboard
```

---

## Quick Reference

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `S` | Scan page |
| `C` | Collection |
| `B` | Builds |
| `W` | Workspace |
| `/` | Search |
| `?` | Help |
| `Esc` | Close modal |

### API Base URLs

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5000 |
| Fusion 360 | http://localhost:8765 |
| Slicer | http://localhost:8081 |

### Dashboard Quick Links

| Page | URL |
|------|-----|
| Home | http://localhost:5000/ |
| Catalog | http://localhost:5000/catalog |
| Shop Floor | http://localhost:5000/api/manufacturing/shop-floor/page |
| Quality | http://localhost:5000/api/quality/dashboard/page |
| Vendors | http://localhost:5000/api/erp/vendors/page |
| Financials | http://localhost:5000/api/erp/financials/dashboard/page |
| Materials | http://localhost:5000/api/mrp/materials/page |
| Scheduling | http://localhost:5000/api/scheduling/dashboard/page |
| AI Copilot | http://localhost:5000/api/ai/copilot/page |
| Digital Twin | http://localhost:5000/api/twin/dashboard/page |
| Sustainability | http://localhost:5000/api/sustainability/dashboard/page |

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# With coverage
python -m pytest tests/ --cov=dashboard --cov-report=html
```

### Docker Commands

```bash
docker-compose up -d          # Start all services
docker-compose down           # Stop all services
docker-compose logs -f        # View logs
docker-compose restart        # Restart services
docker-compose exec dashboard bash  # Shell access
```

---

## Next Steps

1. **Start the dashboard**: `cd dashboard && python app.py`
2. **Open browser**: http://localhost:5000
3. **Explore the catalog**: Browse brick types
4. **Create a work order**: Start manufacturing
5. **Run quality inspection**: Check your parts
6. **Set up vendors**: Manage your supply chain
7. **Configure printers**: Connect your 3D printers
8. **Use AI copilot**: Get intelligent recommendations

---

*LEGO MCP Fusion 360 v6.0 - World-Class Manufacturing Research Platform*

For more information:
- [API Reference](docs/API.md)
- [User Guide](docs/USER_GUIDE.md)
- [Developer Guide](docs/DEVELOPER.md)
