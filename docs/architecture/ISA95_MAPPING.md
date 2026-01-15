# ISA-95 Architecture Mapping

## LEGO MCP Fusion 360 - Industry 4.0/5.0 Architecture

This document maps the LEGO MCP system components to the ISA-95 (IEC 62264) automation pyramid model.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LEGO MCP ISA-95 Architecture Mapping                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LEVEL 5          ┌──────────────────────────────────────────────────────────┐  │
│  ENTERPRISE       │  Cloud Services │ Analytics │ AI Copilot │ ERP API       │  │
│                   │  dashboard/services/cloud, ai, analytics                  │  │
│                   └──────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│  LEVEL 4          ┌──────────────────────▼──────────────────────────────────┐  │
│  BUSINESS         │  ERP Integration │ Supply Chain │ MRP │ Compliance      │  │
│  PLANNING         │  dashboard/services/erp, supply_chain, mrp, compliance   │  │
│                   └──────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│  ════════════════════════════════════════╪═════════════════════════════════════ │
│  SCADA/MES BRIDGE   OPC UA │ MTConnect │ Sparkplug B │ Rosbridge               │
│  ════════════════════════════════════════╪═════════════════════════════════════ │
│                                          │                                       │
│  LEVEL 3          ┌──────────────────────▼──────────────────────────────────┐  │
│  MES/MOM          │  Scheduling (CP-SAT, NSGA2, RL, QAOA)                    │  │
│                   │  Quality Management (SPC, FMEA, QFD)                     │  │
│                   │  Digital Twin (CRDT, PINN)                               │  │
│                   │  Traceability & Audit Trail                              │  │
│                   │  dashboard/services/manufacturing, quality, scheduling   │  │
│                   └──────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│  LEVEL 2          ┌──────────────────────▼──────────────────────────────────┐  │
│  SUPERVISORY      │  lego_mcp_orchestrator (Lifecycle Node)                  │  │
│  CONTROL          │  lego_mcp_supervisor (OTP-style supervision)             │  │
│                   │  lego_mcp_agv (Fleet Management)                         │  │
│                   │  lego_mcp_vision (Quality Inspection)                    │  │
│                   └──────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│  LEVEL 1          ┌──────────────────────▼──────────────────────────────────┐  │
│  BASIC CONTROL    │  lego_mcp_safety (ISO 10218 E-stop)                      │  │
│                   │  lego_mcp_calibration (Robot Calibration)                │  │
│                   │  ros2_control Hardware Interfaces                        │  │
│                   └──────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│  LEVEL 0          ┌──────────────────────▼──────────────────────────────────┐  │
│  FIELD DEVICES    │  grbl_ros2 (CNC, Laser - TinyG, GRBL)                    │  │
│                   │  formlabs_ros2 (SLA Printer)                             │  │
│                   │  bambu_ros2 (FDM Printer - Bambu Lab A1)                 │  │
│                   │  lego_mcp_microros (ESP32 Sensors - Alvik)               │  │
│                   │  Robot Arms (Niryo Ned2, xArm 6 Lite)                    │  │
│                   └──────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 0 - Field Devices

### Physical Equipment
| Equipment | Package | Protocol | Status |
|-----------|---------|----------|--------|
| TinyG Bantam CNC | grbl_ros2 | G-code over Serial | Implemented |
| Coastrunner CR-1 | grbl_ros2 | GRBL Serial | Implemented |
| MKS Laser Engraver | grbl_ros2 | GRBL Serial | Implemented |
| Formlabs SLA | formlabs_ros2 | REST API | Implemented |
| Bambu Lab A1 | bambu_ros2 | MQTT | Implemented |
| Niryo Ned2 | ROS2 native | ROS2 Topics | Implemented |
| xArm 6 Lite | xarm_ros2 | ROS2 Topics | Implemented |
| Arduino Alvik | lego_mcp_microros | micro-ROS/DDS | Implemented |

### Sensors
| Sensor | Connection | Data |
|--------|------------|------|
| E-stop Button | GPIO | Digital input |
| Limit Switches | CNC Controller | G-code feedback |
| Encoders | Robot Controllers | Joint positions |
| Temperature | ESP32 | MQTT/micro-ROS |

---

## Level 1 - Basic Control

### Safety System
**Package:** `lego_mcp_safety`

- **SafetyNode**: Standard ROS2 node for backward compatibility
- **SafetyLifecycleNode**: Lifecycle-managed for deterministic startup
- ISO 10218 compliant emergency stop
- Hardware watchdog timer (500ms default)
- Equipment interlocks

### Calibration System
**Package:** `lego_mcp_calibration`

- Hand-eye calibration
- TCP calibration
- Robot workspace calibration

### Hardware Interfaces
- ros2_control compliant interfaces
- Joint state publishers/subscribers
- Gripper control interfaces

---

## Level 2 - Supervisory Control

### Orchestrator
**Package:** `lego_mcp_orchestrator`

| Node | Type | Function |
|------|------|----------|
| orchestrator_node | Regular | Legacy job coordination |
| orchestrator_lifecycle_node | Lifecycle | Deterministic startup, graceful shutdown |
| twin_sync_node | Regular | Digital twin synchronization |
| failure_detector_node | Regular | Anomaly detection |
| recovery_engine_node | Regular | Automated recovery |

### Supervisor (OTP-style)
**Package:** `lego_mcp_supervisor`

- Restart strategies: ONE_FOR_ONE, ONE_FOR_ALL, REST_FOR_ONE
- Heartbeat monitoring
- Max restart limits
- Dependency management

### AGV Fleet Management
**Package:** `lego_mcp_agv`

- Path planning (A*, RRT)
- Traffic management
- Task allocation

### Vision System
**Package:** `lego_mcp_vision`

- Quality inspection
- Defect detection
- Layer analysis

---

## Level 3 - Manufacturing Execution

### Scheduling Engine
**Location:** `dashboard/services/scheduling/`

| Algorithm | Use Case | Optimization |
|-----------|----------|--------------|
| CP-SAT | Constraint-based scheduling | Makespan, utilization |
| NSGA-II | Multi-objective | Pareto optimal |
| RL (PPO) | Dynamic rescheduling | Reward-based |
| QAOA | Quantum optimization | Combinatorial |

### Quality Management
**Location:** `dashboard/services/quality/`

- Statistical Process Control (SPC)
  - Control charts (X-bar, R, p, c)
  - Cpk/Ppk calculation
- FMEA (Failure Mode and Effects Analysis)
- QFD (Quality Function Deployment)

### Digital Twin
**Location:** `dashboard/services/digital_twin/`

- CRDT-based state synchronization
- PINN (Physics-Informed Neural Network)
- Bidirectional sync with ROS2

### Traceability
**Location:** `dashboard/services/traceability/`

- Tamper-evident audit trail
- Digital thread from design to production
- Merkle tree integrity verification

---

## Level 4 - Business Planning

### ERP Integration
**Location:** `dashboard/services/erp/`

- SAP S/4HANA API integration
- Material management
- Production orders

### Supply Chain
**Location:** `dashboard/services/supply_chain/`

- Inventory optimization
- Demand forecasting
- Supplier management

### MRP (Material Requirements Planning)
**Location:** `dashboard/services/mrp/`

- BOM explosion
- Lead time calculation
- Purchase requisitions

### Compliance
**Location:** `dashboard/services/compliance/`

- ISO 9001 quality management
- ISO 14001 environmental
- Regulatory reporting

---

## Level 5 - Enterprise

### Cloud Services
**Location:** `dashboard/services/cloud/`

- Azure IoT Hub integration
- AWS IoT Core support
- Multi-cloud abstraction

### Analytics
**Location:** `dashboard/services/analytics/`

- OEE calculation
- Predictive maintenance
- KPI dashboards

### AI Services
**Location:** `dashboard/services/ai/`

- Claude API integration
- Manufacturing copilot
- Natural language queries

---

## Security Architecture (IEC 62443)

### Security Zones
| Zone | Level | Components | SL Target |
|------|-------|------------|-----------|
| Safety | L1 | E-stop, watchdog | SL-4 |
| Control | L0-L1 | Equipment nodes | SL-3 |
| Supervisory | L2 | Orchestrator, AGV | SL-2 |
| MES | L3 | Dashboard, scheduling | SL-2 |
| DMZ | L3-L4 | API gateway, Rosbridge | SL-2 |
| Enterprise | L4-L5 | Cloud, ERP | SL-1 |

### Security Implementation
**Package:** `lego_mcp_security`

- SROS2 key/certificate management
- IEC 62443 zone enforcement
- Security audit pipeline
- Intrusion detection

---

## Protocol Bridges

### OPC UA
**Location:** `dashboard/services/edge/protocol_adapters/opcua_adapter.py`

- OPC 40501 CNC Systems compliance
- Information model for manufacturing

### MTConnect
**Location:** `dashboard/services/edge/protocol_adapters/mtconnect_adapter.py`

- ANSI/MTC1.4-2018 compliant
- Agent and adapter modes
- SHDR protocol support

### Sparkplug B
**Location:** `dashboard/services/edge/protocol_adapters/sparkplug_b.py`

- Birth/death certificates
- Metric aliasing
- Host application support

### MQTT
**Location:** `dashboard/services/edge/protocol_adapters/mqtt_adapter.py`

- Equipment telemetry
- Event streaming
- Command/response

---

## Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Equipment│────▶│  ROS2    │────▶│ Dashboard│────▶│   ERP    │
│  (L0)    │     │  (L1-L2) │     │  (L3)    │     │  (L4-L5) │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                │                │                │
     │ G-code/Serial  │ DDS Topics     │ REST/WS        │ API
     │ micro-ROS      │ Services       │ OPC UA         │ SAP RFC
     │                │ Actions        │ MTConnect      │
     └────────────────┴────────────────┴────────────────┘
```

---

## ROS2 Packages Summary

| Package | ISA-95 Level | Type | Purpose |
|---------|--------------|------|---------|
| lego_mcp_msgs | All | Messages | Custom msg/srv/action |
| lego_mcp_bringup | L0-L2 | Launch | System startup |
| lego_mcp_orchestrator | L2 | Nodes | Job coordination |
| lego_mcp_supervisor | L2 | Nodes | OTP supervision |
| lego_mcp_safety | L1 | Nodes | E-stop, watchdog |
| lego_mcp_agv | L2 | Nodes | Fleet management |
| lego_mcp_vision | L2 | Nodes | Quality inspection |
| lego_mcp_calibration | L1 | Nodes | Robot calibration |
| lego_mcp_simulation | L2 | Nodes | Gazebo simulation |
| lego_mcp_security | L1-L3 | Nodes | SROS2 security |
| lego_mcp_chaos | L2 | Test | Chaos engineering |
| lego_mcp_discovery | L2 | Nodes | Equipment discovery |
| lego_mcp_microros | L0 | Nodes | ESP32 integration |
| lego_mcp_moveit_config | L1-L2 | Config | Motion planning |
| grbl_ros2 | L0 | Nodes | CNC/Laser control |
| formlabs_ros2 | L0 | Nodes | SLA printer |
| bambu_ros2 | L0 | Nodes | FDM printer |

---

## Lifecycle Management

### Node States
```
                    ┌─────────────┐
                    │ Unconfigured│
                    └──────┬──────┘
                           │ configure()
                    ┌──────▼──────┐
                    │   Inactive  │
                    └──────┬──────┘
                           │ activate()
                    ┌──────▼──────┐
                    │    Active   │◀─────┐
                    └──────┬──────┘      │ error recovery
                           │             │
                    ┌──────▼──────┐      │
                    │   Error     │──────┘
                    └─────────────┘
```

### Startup Order (Deterministic)
1. **Safety nodes** (L1) - Must be active before equipment
2. **Equipment nodes** (L0) - Hardware interfaces
3. **Supervisory nodes** (L2) - Orchestrator, supervisor
4. **MES bridges** (L3) - Optional SCADA integration

---

## Compliance Matrix

| Standard | Coverage | Implementation |
|----------|----------|----------------|
| ISA-95 | Full | 5-level architecture |
| IEC 62443 | Partial | Security zones, access control |
| ISO 10218 | Full | Safety node |
| ISO 23247 | Partial | Digital twin framework |
| OPC 40501 | Partial | CNC information model |
| MTConnect | Full | Agent/adapter implementation |
