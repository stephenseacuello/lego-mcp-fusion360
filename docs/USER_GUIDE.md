# User Guide - LEGO MCP Fusion 360 v7.0

Complete documentation for the LEGO MCP Industry 4.0/5.0 Manufacturing Platform.

---

## Table of Contents

### Part I: Getting Started
1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Dashboard Navigation](#dashboard-navigation)
4. [ROS2 Operations](#ros2-operations)

### Part II: Manufacturing Operations
5. [Work Orders & MES](#work-orders--mes)
6. [Shop Floor Control](#shop-floor-control)
7. [Equipment Management](#equipment-management)
8. [Production Scheduling](#production-scheduling)
9. [Quality Management](#quality-management)

### Part III: Advanced Features
10. [AI Manufacturing Copilot](#ai-manufacturing-copilot)
11. [Digital Twin](#digital-twin)
12. [SCADA/MES Integration](#scadames-integration)
13. [Robotics Integration](#robotics-integration)
14. [AGV Fleet Management](#agv-fleet-management)

### Part IV: Enterprise Functions
15. [ERP Integration](#erp-integration)
16. [Supply Chain Management](#supply-chain-management)
17. [Sustainability & Carbon Tracking](#sustainability--carbon-tracking)
18. [Compliance & Audit Trail](#compliance--audit-trail)

### Part V: Administration
19. [System Configuration](#system-configuration)
20. [Security Management](#security-management)
21. [Monitoring & Diagnostics](#monitoring--diagnostics)
22. [Troubleshooting](#troubleshooting)

---

# Part I: Getting Started

## System Overview

LEGO MCP Fusion 360 v7.0 is a world-class Industry 4.0/5.0 manufacturing research platform that combines:

- **19 ROS2 packages** for real-time equipment control
- **Flask dashboard** for enterprise applications
- **ISA-95 compliant** architecture
- **OTP-style supervision** for fault tolerance
- **SCADA/MES bridges** for enterprise integration

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LEGO MCP v7.0 Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L5 ENTERPRISE    Cloud Services │ Analytics │ AI Copilot              │
│                                  │                                      │
│  L4 ERP/BUSINESS  ERP │ Supply Chain │ MRP │ Compliance                │
│                                  │                                      │
│  ══════════════════════════════════════════════════════════════════════│
│  SCADA/MES         OPC UA (40501) │ MTConnect │ Sparkplug B            │
│  ══════════════════════════════════════════════════════════════════════│
│                                  │                                      │
│  L3 MES/MOM       Scheduling │ Quality │ Digital Twin │ Traceability  │
│                                  │                                      │
│  L2 SUPERVISORY   ┌──────────────────────────────────────────────────┐ │
│  (ROS2 DDS)       │       OTP-STYLE SUPERVISION TREE                 │ │
│                   │  Root → Safety → Equipment → Robotics → AGV      │ │
│                   └──────────────────────────────────────────────────┘ │
│                   lego_mcp_orchestrator │ lego_mcp_agv │ MoveIt2       │
│                                  │                                      │
│  L1 CONTROL       lego_mcp_safety │ lego_mcp_calibration               │
│                                  │                                      │
│  L0 FIELD         grbl_ros2 │ formlabs_ros2 │ lego_mcp_microros        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Lifecycle Nodes** | Graceful state management (configure/activate/deactivate) |
| **Supervision Tree** | OTP-style fault tolerance with automatic recovery |
| **SCADA Bridges** | OPC UA, MTConnect, Sparkplug B protocol support |
| **Digital Twin** | Real-time synchronization with CRDTs and PINNs |
| **AI Copilot** | 150+ MCP tools with autonomous agents |
| **Quality System** | SPC, FMEA, QFD, vision inspection |

---

## Installation

See [QUICKSTART.md](QUICKSTART.md) for detailed installation instructions.

### Quick Install

```bash
# Clone and setup
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Install dependencies
pip install -r requirements.txt

# Build ROS2 workspace
cd ros2_ws && colcon build --symlink-install
source install/setup.bash

# Start system
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true
```

---

## Dashboard Navigation

### Main Navigation

| Icon | Page | Description |
|------|------|-------------|
| Home | Dashboard | System overview and KPIs |
| Factory | Manufacturing | Work orders, shop floor |
| Quality | Quality | SPC, FMEA, inspections |
| Calendar | Scheduling | Production scheduling |
| Truck | Supply Chain | Suppliers, procurement |
| Leaf | Sustainability | LCA, carbon tracking |
| Shield | Compliance | ISO tracking, audit trail |
| Brain | AI Copilot | Manufacturing assistant |

### Dashboard Pages

| Page | URL | Purpose |
|------|-----|---------|
| Shop Floor | `/api/manufacturing/shop-floor/page` | Real-time production |
| Quality Dashboard | `/api/quality/dashboard/page` | Quality metrics |
| OEE Analytics | `/api/analytics/oee/page` | OEE analysis |
| Scheduling | `/api/scheduling/page` | Gantt chart, optimization |
| Vendors | `/api/erp/vendors/page` | Supplier management |
| Financials | `/api/erp/financials/dashboard/page` | AR/AP/GL |

### Real-Time Updates

Dashboards update automatically via WebSocket:
- Work order status: every 30 seconds
- OEE metrics: every 60 seconds
- Quality alerts: instant (WebSocket)
- AI insights: every 45 seconds

---

## ROS2 Operations

### Basic Commands

```bash
# Source workspace (required for each terminal)
source ros2_ws/install/setup.bash

# List running nodes
ros2 node list

# List topics
ros2 topic list

# Monitor equipment status
ros2 topic echo /lego_mcp/equipment/status

# Check supervision health
ros2 topic echo /lego_mcp/supervision/health
```

### Lifecycle Management

```bash
# Get node state
ros2 lifecycle get /lego_mcp/grbl_node

# State transitions
ros2 lifecycle set /lego_mcp/grbl_node configure
ros2 lifecycle set /lego_mcp/grbl_node activate
ros2 lifecycle set /lego_mcp/grbl_node deactivate
ros2 lifecycle set /lego_mcp/grbl_node cleanup

# Get all lifecycle states (via bridge)
ros2 service call /lego_mcp/bridge/lifecycle/get_all_states std_srvs/srv/Trigger
```

### Launch Options

```bash
# Simulation mode (no hardware)
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true

# Production mode
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=false

# With SCADA bridges
ros2 launch lego_mcp_bringup full_system.launch.py enable_scada:=true

# With robotics
ros2 launch lego_mcp_bringup full_system.launch.py enable_robotics:=true

# With security (SROS2)
ros2 launch lego_mcp_bringup full_system.launch.py enable_security:=true
```

---

# Part II: Manufacturing Operations

## Work Orders & MES

### Creating Work Orders

**Via Dashboard:**
1. Navigate to Manufacturing > Work Orders
2. Click **+ New Work Order**
3. Fill required fields:
   - Part Number: Select from part master
   - Quantity: Units to produce
   - Due Date: Required completion date
   - Priority: 1 (highest) to 5 (lowest)
4. Click **Create**

**Via API:**
```bash
curl -X POST http://localhost:5000/api/manufacturing/work-orders \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "quantity": 100,
    "priority": "high",
    "due_date": "2024-12-31"
  }'
```

### Work Order Lifecycle

```
CREATED → RELEASED → IN_PROGRESS → COMPLETED
                  ↓
               ON_HOLD (if quality issue)
```

### Operations Tracking

Each work order has operations based on routing:

| Field | Description |
|-------|-------------|
| Operation Code | PRINT, INSPECT, PACK |
| Work Center | Assigned machine |
| Status | pending, in_progress, completed |
| Start/End Time | Actual timestamps |
| Operator | Assigned operator ID |

---

## Shop Floor Control

### Work Center Management

View work centers at Manufacturing > Shop Floor.

Each work center displays:
- **Status**: RUNNING, IDLE, SETUP, DOWN, MAINTENANCE
- **Current Job**: Active work order
- **OEE**: Real-time calculation
- **Operator**: Currently assigned

### Starting Operations

1. Select work center
2. Click **Start Operation**
3. Scan or select work order
4. Confirm material availability
5. Begin production

### Recording Production

```bash
# Record good and scrap units
curl -X POST http://localhost:5000/api/mes/operations/{id}/record \
  -H "Content-Type: application/json" \
  -d '{
    "good_qty": 10,
    "scrap_qty": 1,
    "scrap_reason": "layer_shift"
  }'
```

---

## Equipment Management

### ROS2 Equipment Nodes

| Node | Equipment | Protocol |
|------|-----------|----------|
| `/lego_mcp/grbl_node` | CNC/Laser | Serial (GRBL) |
| `/lego_mcp/formlabs_node` | SLA Printer | HTTP API |
| `/lego_mcp/bambu_node` | FDM Printer | MQTT |

### Lifecycle States

All equipment nodes are lifecycle-managed:

| State | Description |
|-------|-------------|
| `unconfigured` | Initial state |
| `inactive` | Configured but not active |
| `active` | Ready for operations |
| `finalized` | Shutdown complete |

### Equipment Topics

```bash
# Monitor all equipment status
ros2 topic echo /lego_mcp/equipment/status

# GRBL-specific
ros2 topic echo /lego_mcp/grbl/position
ros2 topic echo /lego_mcp/grbl/state

# Printer-specific
ros2 topic echo /lego_mcp/formlabs/status
ros2 topic echo /lego_mcp/bambu/print_progress
```

### Equipment Services

```bash
# Home GRBL machine
ros2 service call /lego_mcp/grbl/home std_srvs/srv/Trigger

# Start print job
ros2 service call /lego_mcp/formlabs/start_print lego_mcp_msgs/srv/StartPrint
```

---

## Production Scheduling

### Scheduling Algorithms

| Algorithm | Best For | Speed |
|-----------|----------|-------|
| FIFO | Simple priority | Instant |
| SPT | Minimize flow time | Instant |
| CP-SAT | Optimal makespan | 1-10 sec |
| NSGA-II | Multi-objective | 10-60 sec |
| RL (DQN) | Dynamic dispatching | Instant |
| QAOA | Quantum optimization | Variable |

### Running Optimizer

1. Navigate to Scheduling > Optimizer
2. Select time horizon (8h, 24h, 7d)
3. Choose algorithm
4. Set objectives (makespan, tardiness, energy)
5. Click **Optimize**

### Multi-Objective Optimization

NSGA-II provides Pareto-optimal solutions balancing:
- **Makespan**: Total completion time
- **Tardiness**: Lateness penalties
- **Energy**: kWh consumption
- **Quality Risk**: FMEA risk exposure

### Gantt Chart

The interactive Gantt shows:
- Operations by work center
- Dependencies between operations
- Drag-and-drop rescheduling
- Conflict highlighting

---

## Quality Management

### Statistical Process Control (SPC)

Supported chart types:

| Chart | Use Case |
|-------|----------|
| X-bar/R | Variable data, subgroups |
| I-MR | Single measurements |
| EWMA | Detecting small shifts |
| CUSUM | Detecting persistent shifts |
| Hotelling T2 | Multivariate control |

**Setting up SPC:**
1. Navigate to Quality > SPC
2. Click **+ New Control Chart**
3. Select measurement and chart type
4. Enter control limits or auto-calculate
5. Save and start monitoring

### FMEA (Failure Mode Effects Analysis)

**Creating FMEA:**
1. Navigate to Quality > FMEA
2. Click **+ New FMEA**
3. Add failure modes with:
   - Severity (1-10)
   - Occurrence (1-10)
   - Detection (1-10)
4. RPN = S x O x D calculated automatically

### QFD (Quality Function Deployment)

Build House of Quality:
1. Navigate to Quality > QFD
2. Enter customer requirements with weights
3. Add engineering characteristics
4. Fill relationship matrix (0, 1, 3, 9)
5. View calculated importance scores

### Vision Inspection

The CV system provides:
- Real-time defect detection (YOLO)
- Layer-by-layer inspection
- Surface quality grading (A-D)
- Dimensional verification

```bash
# Monitor vision detections
ros2 topic echo /lego_mcp/vision/detections
```

---

# Part III: Advanced Features

## AI Manufacturing Copilot

### Overview

The AI Copilot uses Claude to provide natural language manufacturing intelligence with 150+ MCP tools.

### Asking Questions

Navigate to AI > Copilot and type queries:
- "Why did WC-PRINT-01 go down yesterday?"
- "What's causing the quality drift?"
- "Recommend a schedule to minimize energy"
- "Explain the SPC violation on stud diameter"

### Capabilities

| Capability | Description |
|------------|-------------|
| Anomaly Explanation | Plain-English SPC/quality explanations |
| Root Cause Analysis | Trace issues to sources |
| Schedule Recommendations | Trade-off analysis |
| Process Optimization | Parameter adjustments |

### Autonomous Agents

Three autonomous agents:
1. **Quality Agent**: Monitors SPC, triggers inspections
2. **Scheduling Agent**: Optimizes production schedule
3. **Maintenance Agent**: Predicts equipment failures

---

## Digital Twin

### Real-Time Synchronization

The digital twin uses CRDTs (Conflict-free Replicated Data Types) for:
- Distributed state synchronization
- Offline operation support
- Conflict resolution

### Physics Simulation

PINNs (Physics-Informed Neural Networks) simulate:
- Thermal dynamics
- Material flow
- Stress analysis

### API Usage

```bash
# Get twin state
curl http://localhost:5000/api/twin/state/printer-1

# Run thermal simulation
curl -X POST http://localhost:5000/api/twin/simulate/thermal \
  -H "Content-Type: application/json" \
  -d '{"bed_temp": 60, "nozzle_temp": 210}'
```

---

## SCADA/MES Integration

### Supported Protocols

| Protocol | Standard | Port |
|----------|----------|------|
| OPC UA | OPC 40501 CNC | 4840 |
| MTConnect | ANSI/MTC1.4-2018 | 5000 |
| Sparkplug B | Eclipse 3.0 | 1883 |

### OPC UA Server

Access at: `opc.tcp://localhost:4840`

Namespace: `http://legomcp.dev/cnc`

Nodes available:
- CncInterface (OPC 40501)
- CncAxisList
- CncSpindleList
- CncAlarmList

### MTConnect Agent

Access at: `http://localhost:5000/mtconnect`

Endpoints:
- `/probe` - Device capability
- `/current` - Current state
- `/sample` - Historical data

### Sparkplug B

Topics:
- `spBv1.0/lego_mcp/NBIRTH/factory_floor`
- `spBv1.0/lego_mcp/NDATA/factory_floor`
- `spBv1.0/lego_mcp/DBIRTH/factory_floor/{device}`

### Launching SCADA Bridges

```bash
# All SCADA bridges
ros2 launch lego_mcp_bringup scada_bridges.launch.py

# OPC UA only
ros2 launch lego_mcp_bringup scada_bridges.launch.py \
    enable_mtconnect:=false enable_sparkplug:=false

# With custom ports
ros2 launch lego_mcp_bringup scada_bridges.launch.py \
    opcua_port:=4841 mtconnect_port:=5001
```

---

## Robotics Integration

### Supported Robots

| Robot | Package | Features |
|-------|---------|----------|
| Niryo Ned2 | `niryo_ned2_ros2` | 6-axis, gripper |
| xArm 6 Lite | `xarm_ros2` | 6-DOF, 5kg payload |

### MoveIt2 Integration

MoveIt2 provides:
- Motion planning (OMPL)
- Collision detection
- Pick and place operations
- Trajectory execution

### Launching Robotics

```bash
# Full robotics with MoveIt2
ros2 launch lego_mcp_bringup robotics.launch.py use_sim:=true

# Ned2 only
ros2 launch lego_mcp_bringup robotics.launch.py \
    enable_ned2:=true enable_xarm:=false
```

### Assembly Coordinator

The assembly coordinator manages:
- Pick approach distance
- Place approach distance
- Retry attempts
- Assembly timeout

---

## AGV Fleet Management

### Fleet Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fleet_size` | 4 | Number of AGVs |
| `map_file` | factory.yaml | Nav2 map |

### Navigation

AGVs use Nav2 for:
- Path planning
- Obstacle avoidance
- Traffic management
- Charging scheduling

### Launching AGV Fleet

```bash
ros2 launch lego_mcp_bringup full_system.launch.py \
    use_sim:=true \
    enable_agv:=true
```

### Topics

```bash
# Fleet status
ros2 topic echo /lego_mcp/agv/fleet_status

# Individual AGV
ros2 topic echo /lego_mcp/agv/agv_01/pose
ros2 topic echo /lego_mcp/agv/agv_01/battery
```

---

# Part IV: Enterprise Functions

## ERP Integration

### Vendor Management

Navigate to ERP > Vendors:
- Create/manage suppliers
- Track certifications
- Monitor scorecards

### Financial Modules

| Module | Features |
|--------|----------|
| **AR** | Customer invoices, payments, aging |
| **AP** | Vendor bills, payment scheduling |
| **GL** | Chart of accounts, journal entries |
| **Costing** | Standard/actual costing |

### BOM Management

- Multi-level BOMs
- Revision control
- Where-used analysis

---

## Supply Chain Management

### Supplier Scorecard

Each supplier scored on:
- **Quality**: PPM, rejections
- **Delivery**: OTD percentage
- **Cost**: Competitiveness
- **Responsiveness**: Communication

### Auto-Replenishment

1. Navigate to Supply Chain > Materials
2. Set reorder point and EOQ
3. Enable auto-replenishment
4. System generates POs automatically

### EDI Integration

Supported documents:
- 850: Purchase Order
- 856: ASN (Advance Ship Notice)
- 810: Invoice

---

## Sustainability & Carbon Tracking

### Carbon Footprint

Track emissions across Scope 1/2/3:

| Scope | Description |
|-------|-------------|
| Scope 1 | Direct (on-site fuel) |
| Scope 2 | Indirect (electricity) |
| Scope 3 | Value chain (materials) |

### Per-Unit Calculation

```bash
curl http://localhost:5000/api/sustainability/carbon/footprint?part_id=BRICK-2X4
```

### LCA Analysis

Life Cycle Assessment covers:
- Raw material extraction
- Manufacturing
- Transportation
- Use phase
- End of life

### Circular Economy

Metrics tracked:
- Material Circularity Index (MCI)
- Recycling rate
- Waste diversion rate

---

## Compliance & Audit Trail

### FDA 21 CFR Part 11

Key features:
- Complete audit trail
- Electronic signatures
- Access control
- Secure retention

### ISO Standards

| Standard | Coverage |
|----------|----------|
| ISO 9001 | Quality Management |
| ISO 14001 | Environmental |
| ISO 45001 | Safety |
| ISO 13485 | Medical Devices |
| IEC 62443 | Industrial Security |

### Viewing Audit Trail

Navigate to Compliance > Audit Trail

Filter by:
- Date range
- User
- Entity type
- Action type

---

# Part V: Administration

## System Configuration

### Environment Variables

```bash
# Core
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export DATABASE_URL=postgresql://...

# ROS2
export ROS_DOMAIN_ID=42
export ROS_SECURITY_ENABLE=true

# SCADA
export OPCUA_PORT=4840
export MTCONNECT_PORT=5000
export MQTT_HOST=localhost
```

### Configuration Files

| File | Purpose |
|------|---------|
| `config/supervision_tree.yaml` | Supervision configuration |
| `config/security_policy.yaml` | SROS2 security policies |
| `config/equipment.yaml` | Equipment parameters |

---

## Security Management

### SROS2 Security

IEC 62443 security zones:
- Zone 0: Safety (highest)
- Zone 1: Control
- Zone 2: Supervisory
- Zone 3: MES/SCADA
- Zone 4: Enterprise

### Enabling Security

```bash
# Generate keystore
ros2 security generate_keystore /etc/lego_mcp/keystore

# Enable security
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_KEYSTORE=/etc/lego_mcp/keystore

# Launch with security
ros2 launch lego_mcp_bringup full_system.launch.py enable_security:=true
```

---

## Monitoring & Diagnostics

### Supervision Health

```bash
# Monitor supervision tree
ros2 topic echo /lego_mcp/supervision/health

# Check node heartbeats
ros2 topic echo /lego_mcp/heartbeat/grbl_node
```

### Diagnostics

```bash
# ROS2 diagnostics
ros2 topic echo /diagnostics

# System report
ros2 doctor --report
```

### Logging

Logs location:
- ROS2: `~/.ros/log/`
- Dashboard: `dashboard/logs/`
- Docker: `docker-compose logs`

---

## Troubleshooting

### Common Issues

#### ROS2 nodes not starting
```bash
source ros2_ws/install/setup.bash
colcon build --symlink-install
```

#### Lifecycle transitions failing
```bash
ros2 lifecycle get /lego_mcp/grbl_node
ros2 topic echo /rosout | grep grbl
```

#### Dashboard connection refused
```bash
# Check if running
lsof -i :5000

# Restart
cd dashboard && python app.py
```

#### Equipment not responding
```bash
# Check serial port
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
```

### Getting Help

- Quick Start: [QUICKSTART.md](QUICKSTART.md)
- Developer Guide: [DEVELOPER.md](DEVELOPER.md)
- API Reference: [API.md](API.md)
- ROS2 Guide: [ROS2_GUIDE.md](ROS2_GUIDE.md)
- GitHub Issues: https://github.com/yourusername/lego-mcp-fusion360/issues

---

*LEGO MCP Fusion 360 v7.0 - Industry 4.0/5.0 Manufacturing Platform*
