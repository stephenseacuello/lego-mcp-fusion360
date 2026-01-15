# Quick Start Guide - LEGO MCP Fusion 360 v7.0

**Industry 4.0/5.0 Manufacturing Research Platform**

Get the complete LEGO MCP manufacturing system running in minutes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Starting the System](#starting-the-system)
4. [ROS2 Launch Options](#ros2-launch-options)
5. [Core Features Overview](#core-features-overview)
6. [Your First Manufacturing Job](#your-first-manufacturing-job)
7. [Common Commands](#common-commands)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Ubuntu | 22.04 LTS | Operating system (ROS2 Humble) |
| ROS2 | Humble | Robot Operating System |
| Python | 3.10+ | Runtime |
| Docker | 20.10+ | Containerized deployment |
| Git | Latest | Version control |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| Storage | 20GB | 100GB+ (SSD) |
| GPU | - | NVIDIA CUDA 12+ (for AI/Vision) |
| Network | 1Gbps | 10Gbps (for multi-machine) |

### Optional Hardware

- **3D Printers**: Bambu Lab, Prusa, Ender (supported via ROS2 nodes)
- **CNC Machines**: GRBL-compatible controllers
- **Cameras**: USB webcams for vision inspection
- **ESP32**: Micro-ROS sensor nodes

---

## Installation

### Option 1: Full ROS2 Stack (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Install ROS2 Humble (if not already installed)
sudo apt update && sudo apt install -y ros-humble-desktop

# Install additional ROS2 packages
sudo apt install -y \
    ros-humble-lifecycle \
    ros-humble-launch-testing \
    ros-humble-diagnostic-updater \
    ros-humble-nav2-msgs \
    ros-humble-moveit

# Install Python dependencies
pip install -r requirements.txt

# Build ROS2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Return to project root
cd ..
```

### Option 2: Dashboard Only (No ROS2)

```bash
# Clone repository
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Install minimal dependencies
pip install flask requests sqlalchemy

# Start dashboard
cd dashboard
python app.py
```

### Option 3: Docker Deployment

```bash
# Clone and start with Docker Compose
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Full system with ROS2
docker-compose -f docker-compose.yml up -d

# Dashboard only
docker-compose -f docker-compose.yml up -d dashboard
```

---

## Starting the System

### Quick Start (Simulation Mode)

```bash
# Terminal 1: Start ROS2 system
cd ros2_ws
source install/setup.bash
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true

# Terminal 2: Start Dashboard
cd dashboard
python app.py
```

Access dashboard at: **http://localhost:5000**

### Production Mode (Real Hardware)

```bash
# Terminal 1: Start ROS2 with real equipment
cd ros2_ws
source install/setup.bash
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=false

# Terminal 2: Dashboard
cd dashboard
python app.py
```

---

## ROS2 Launch Options

### Full System Launch

```bash
# All features enabled (simulation)
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true

# Production with real hardware
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=false

# With SCADA/MES bridges (OPC UA, MTConnect, Sparkplug B)
ros2 launch lego_mcp_bringup full_system.launch.py \
    use_sim:=false \
    enable_scada:=true

# With robotics (MoveIt2 + robot arms)
ros2 launch lego_mcp_bringup full_system.launch.py \
    use_sim:=true \
    enable_robotics:=true

# With AGV fleet management
ros2 launch lego_mcp_bringup full_system.launch.py \
    use_sim:=true \
    enable_agv:=true

# With SROS2 security (IEC 62443)
ros2 launch lego_mcp_bringup full_system.launch.py \
    use_sim:=false \
    enable_security:=true
```

### Subsystem Launches

```bash
# Robotics only (MoveIt2 + Ned2 + xArm)
ros2 launch lego_mcp_bringup robotics.launch.py use_sim:=true

# SCADA bridges only (OPC UA, MTConnect, Sparkplug B)
ros2 launch lego_mcp_bringup scada_bridges.launch.py use_sim:=true

# Equipment nodes only
ros2 launch lego_mcp_bringup equipment.launch.py use_sim:=true

# Safety systems only
ros2 launch lego_mcp_bringup safety.launch.py use_sim:=true
```

### Launch Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim` | `false` | Simulation mode (no real hardware) |
| `enable_safety` | `true` | Safety subsystem (e-stop, watchdog) |
| `enable_equipment` | `true` | Equipment nodes (CNC, printers) |
| `enable_robotics` | `false` | Robot arms and MoveIt2 |
| `enable_agv` | `false` | AGV fleet management |
| `enable_vision` | `true` | Computer vision subsystem |
| `enable_supervision` | `true` | OTP-style supervision tree |
| `enable_security` | `false` | SROS2 DDS security |
| `enable_scada` | `false` | SCADA/MES protocol bridges |
| `heartbeat_timeout_ms` | `500` | Supervision heartbeat timeout |
| `max_restarts` | `5` | Max restart attempts before escalation |

---

## Core Features Overview

### System Architecture

```
ISA-95 COMPLIANT ARCHITECTURE
=============================

L4 ERP/Business    Dashboard: ERP, Supply Chain, MRP, Compliance
                              ↓
─────────────────────────────────────────────────────────────────
L3 MES/MOM         Dashboard: Scheduling, Quality, Digital Twin
                   SCADA Bridges: OPC UA, MTConnect, Sparkplug B
                              ↓
─────────────────────────────────────────────────────────────────
L2 Supervisory     ROS2: lego_mcp_orchestrator, lego_mcp_agv
                   Supervision Tree (OTP-style)
                              ↓
─────────────────────────────────────────────────────────────────
L1 Control         ROS2: lego_mcp_safety, lego_mcp_calibration
                              ↓
─────────────────────────────────────────────────────────────────
L0 Field           ROS2: grbl_ros2, formlabs_ros2, lego_mcp_microros
```

### ROS2 Packages (19 Total)

| Package | Purpose |
|---------|---------|
| `lego_mcp_msgs` | Custom messages, services, actions |
| `lego_mcp_bringup` | Launch files and configs |
| `lego_mcp_orchestrator` | Job coordination, lifecycle management |
| `lego_mcp_safety` | Safety PLC interface, e-stop |
| `lego_mcp_vision` | Computer vision, defect detection |
| `lego_mcp_calibration` | Camera/printer calibration |
| `lego_mcp_agv` | AGV fleet management (Nav2) |
| `lego_mcp_supervisor` | OTP-style supervision tree |
| `lego_mcp_security` | SROS2 security (IEC 62443) |
| `lego_mcp_edge` | SCADA protocol bridges |
| `grbl_ros2` | GRBL CNC/laser controller |
| `formlabs_ros2` | Formlabs SLA printer |
| `lego_mcp_moveit_config` | MoveIt2 configuration |
| `lego_mcp_simulation` | Gazebo simulation |
| `lego_mcp_microros` | ESP32/Micro-ROS nodes |

### Dashboard Modules

| Module | Route | Description |
|--------|-------|-------------|
| AI/MCP | `/api/ai` | 150+ tools, 3 autonomous agents |
| Digital Twin | `/api/twin` | PINNs physics, CRDTs, OWL ontology |
| Quality | `/api/quality` | Vision, SPC, FMEA, QFD |
| ERP | `/api/erp` | BOM, Costing, Orders, Procurement |
| MRP | `/api/mrp` | Planning, Capacity, Materials |
| Scheduling | `/api/scheduling` | CP-SAT, NSGA-II, RL, QAOA |
| Manufacturing | `/api/manufacturing` | Work orders, shop floor |
| Sustainability | `/api/sustainability` | LCA, carbon tracking, ESG |

---

## Your First Manufacturing Job

### Step 1: Verify System Status

```bash
# Check ROS2 nodes are running
ros2 node list

# Expected output:
# /lego_mcp/orchestrator
# /lego_mcp/safety_node
# /lego_mcp/grbl_node
# /lego_mcp/formlabs_node
# /lego_mcp/vision_node
# ...

# Check lifecycle states
ros2 service call /lego_mcp/bridge/lifecycle/get_all_states std_srvs/srv/Trigger
```

### Step 2: Create a Work Order (API)

```bash
curl -X POST http://localhost:5000/api/manufacturing/work-orders \
  -H "Content-Type: application/json" \
  -d '{
    "part_id": "brick-2x4",
    "quantity": 10,
    "priority": "high"
  }'
```

### Step 3: Monitor Production

```bash
# Watch equipment status
ros2 topic echo /lego_mcp/equipment/status

# Watch job progress
ros2 topic echo /lego_mcp/jobs/status

# Watch OEE metrics
ros2 topic echo /lego_mcp/analytics/oee
```

### Step 4: View in Dashboard

1. Open http://localhost:5000
2. Navigate to **Manufacturing** > **Shop Floor**
3. View real-time work order status
4. Click any work order for details

---

## Common Commands

### ROS2 Commands

```bash
# Source workspace
source ros2_ws/install/setup.bash

# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo a topic
ros2 topic echo /lego_mcp/equipment/status

# List all services
ros2 service list

# Call a service
ros2 service call /lego_mcp/orchestrator/get_job_status lego_mcp_msgs/srv/GetJobStatus

# Lifecycle management
ros2 lifecycle get /lego_mcp/grbl_node
ros2 lifecycle set /lego_mcp/grbl_node configure
ros2 lifecycle set /lego_mcp/grbl_node activate

# Check supervision tree health
ros2 topic echo /lego_mcp/supervision/health
```

### Dashboard Commands

```bash
# Start dashboard
cd dashboard && python app.py

# Run tests
python -m pytest tests/ -v

# Database operations
flask db upgrade

# Background worker (for async tasks)
python worker.py
```

### Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose build
docker-compose up -d
```

---

## Troubleshooting

### ROS2 Issues

#### Nodes not starting
```bash
# Check if ROS2 is sourced
echo $ROS_DISTRO  # Should show: humble

# Source workspace
source ros2_ws/install/setup.bash

# Rebuild workspace
cd ros2_ws && colcon build --symlink-install
```

#### Lifecycle transition failed
```bash
# Check node state
ros2 lifecycle get /lego_mcp/grbl_node

# View node logs
ros2 topic echo /rosout | grep grbl_node

# Check available transitions
ros2 lifecycle list /lego_mcp/grbl_node
```

#### DDS communication issues
```bash
# Check network
ros2 doctor --report

# Set DDS domain
export ROS_DOMAIN_ID=42
```

### Dashboard Issues

#### Port already in use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
FLASK_PORT=5001 python app.py
```

#### Database errors
```bash
cd dashboard
flask db upgrade
```

### Hardware Issues

#### GRBL not connecting
```bash
# Check serial port
ls /dev/ttyUSB*

# Set permissions
sudo chmod 666 /dev/ttyUSB0
sudo usermod -a -G dialout $USER
```

#### Camera not detected
```bash
# List video devices
ls /dev/video*

# Test with OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

---

## Quick Reference

### Important URLs

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5000 |
| OPC UA Server | opc.tcp://localhost:4840 |
| MTConnect Agent | http://localhost:5000 |
| Sparkplug B | mqtt://localhost:1883 |

### Important Topics

| Topic | Description |
|-------|-------------|
| `/lego_mcp/equipment/status` | Equipment state |
| `/lego_mcp/jobs/status` | Job progress |
| `/lego_mcp/safety/estop` | Emergency stop |
| `/lego_mcp/analytics/oee` | OEE metrics |
| `/lego_mcp/supervision/health` | Supervision tree health |

### Keyboard Shortcuts (Dashboard)

| Key | Action |
|-----|--------|
| `S` | Go to Scan page |
| `C` | Go to Collection |
| `W` | Go to Workspace |
| `/` | Focus search box |
| `?` | Show keyboard shortcuts |
| `Esc` | Close modal |

---

## Getting Help

- Full Documentation: [USER_GUIDE.md](USER_GUIDE.md)
- Developer Guide: [DEVELOPER.md](DEVELOPER.md)
- API Reference: [API.md](API.md)
- ROS2 Guide: [ROS2_GUIDE.md](ROS2_GUIDE.md)
- GitHub Issues: https://github.com/yourusername/lego-mcp-fusion360/issues

---

## What's New in v7.0

### ROS2 Industry 4.0 Architecture
- **19 ROS2 packages** with full ISA-95 compliance
- **OTP-style supervision tree** for fault tolerance
- **Lifecycle nodes** for graceful state management
- **SROS2 security** with IEC 62443 zones

### SCADA/MES Integration
- **OPC UA Server** (OPC 40501 CNC compliant)
- **MTConnect Agent** (ANSI/MTC1.4-2018)
- **Sparkplug B Edge Node** (Eclipse Sparkplug 3.0)

### Advanced Manufacturing
- **Robotics integration** (MoveIt2 + Niryo Ned2 + xArm)
- **AGV fleet management** (Nav2-based)
- **Chaos engineering** framework for resilience testing

### Enhanced Dashboard
- **Digital Thread** with tamper-evident audit trail
- **Real-time WebSocket** updates
- **PhD-level research** modules

---

You're all set! Start with simulation mode to explore the system:

```bash
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true
```

*LEGO MCP Fusion 360 v7.0 - Industry 4.0/5.0 Manufacturing Platform*
