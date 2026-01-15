# ROS2 Guide - LEGO MCP Fusion 360 v7.0

Complete ROS2 documentation for the Industry 4.0/5.0 Manufacturing Platform.

---

## Table of Contents

1. [ROS2 Architecture](#ros2-architecture)
2. [Packages](#packages)
3. [Messages, Services & Actions](#messages-services--actions)
4. [Lifecycle Nodes](#lifecycle-nodes)
5. [Supervision Tree](#supervision-tree)
6. [Launch Files](#launch-files)
7. [Topics & Services Reference](#topics--services-reference)
8. [SROS2 Security](#sros2-security)
9. [SCADA Protocol Bridges](#scada-protocol-bridges)
10. [Simulation](#simulation)
11. [Troubleshooting](#troubleshooting)

---

## ROS2 Architecture

### ISA-95 Layer Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROS2 ISA-95 Architecture                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L2 SUPERVISORY   ┌─────────────────────────────────────────────────┐  │
│                   │           SUPERVISION TREE                       │  │
│                   │  RootSupervisor                                  │  │
│                   │    ├── SafetySupervisor (one_for_all)           │  │
│                   │    ├── EquipmentSupervisor (one_for_one)        │  │
│                   │    ├── RoboticsSupervisor (rest_for_one)        │  │
│                   │    └── AGVSupervisor (one_for_one)              │  │
│                   └─────────────────────────────────────────────────┘  │
│                   lego_mcp_orchestrator                                 │
│                   lego_mcp_agv (Nav2)                                   │
│                   MoveIt2 (motion planning)                             │
│                                                                         │
│  L1 CONTROL       lego_mcp_safety (e-stop, watchdog)                   │
│                   lego_mcp_calibration (camera, printer)               │
│                   lego_mcp_vision (defect detection)                   │
│                                                                         │
│  L0 FIELD         grbl_ros2 (CNC/Laser - GRBL protocol)                │
│                   formlabs_ros2 (SLA - HTTP API)                       │
│                   bambu_ros2 (FDM - MQTT)                              │
│                   lego_mcp_microros (ESP32 sensors)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### DDS Middleware

- Default: FastDDS (ROS2 Humble default)
- Domain ID: 42 (configurable via `ROS_DOMAIN_ID`)
- QoS: Reliability, Durability configured per topic

---

## Packages

### Core Packages

| Package | Type | Description |
|---------|------|-------------|
| `lego_mcp_msgs` | ament_cmake | Custom messages, services, actions |
| `lego_mcp_bringup` | ament_cmake | Launch files and configuration |
| `lego_mcp_orchestrator` | ament_python | Job orchestration, lifecycle management |
| `lego_mcp_supervisor` | ament_python | OTP-style supervision tree |
| `lego_mcp_safety` | ament_python | Safety systems (e-stop, watchdog) |

### Equipment Packages

| Package | Type | Protocol | Hardware |
|---------|------|----------|----------|
| `grbl_ros2` | ament_python | Serial (115200) | GRBL CNC/Laser |
| `formlabs_ros2` | ament_python | HTTP REST | Formlabs SLA |
| `bambu_ros2` | ament_python | MQTT | Bambu Lab FDM |
| `lego_mcp_microros` | ament_cmake | Micro-ROS | ESP32 sensors |

### Advanced Packages

| Package | Type | Description |
|---------|------|-------------|
| `lego_mcp_vision` | ament_python | Computer vision, YOLO detection |
| `lego_mcp_calibration` | ament_python | Camera/printer calibration |
| `lego_mcp_agv` | ament_python | AGV fleet management (Nav2) |
| `lego_mcp_moveit_config` | ament_cmake | MoveIt2 robot configuration |
| `lego_mcp_simulation` | ament_cmake | Gazebo simulation |
| `lego_mcp_security` | ament_python | SROS2 security (IEC 62443) |
| `lego_mcp_edge` | ament_python | SCADA protocol bridges |

### Package Structure

```
lego_mcp_orchestrator/
├── lego_mcp_orchestrator/
│   ├── __init__.py
│   ├── orchestrator_lifecycle_node.py    # Main orchestrator
│   ├── lifecycle_manager.py              # Coordinated lifecycle
│   ├── lifecycle_monitor_node.py         # State monitoring
│   ├── lifecycle_service_bridge.py       # External bridge
│   └── moveit_assembly_node.py           # MoveIt2 assembly
├── config/
│   └── orchestrator.yaml
├── launch/
│   └── orchestrator.launch.py
├── test/
│   └── test_orchestrator.py
├── package.xml
├── setup.py
└── CMakeLists.txt
```

---

## Messages, Services & Actions

### Custom Messages

```bash
# List all messages
ros2 interface list | grep lego_mcp_msgs

# Show message definition
ros2 interface show lego_mcp_msgs/msg/EquipmentStatus
```

#### EquipmentStatus.msg
```
std_msgs/Header header
string equipment_id
string equipment_type     # cnc, sla, fdm, sensor
uint8 state              # 0=offline, 1=idle, 2=running, 3=error, 4=maintenance
float32 utilization
string current_job_id
float32 temperature
```

#### JobStatus.msg
```
std_msgs/Header header
string job_id
string part_id
uint32 quantity_target
uint32 quantity_completed
uint8 state              # 0=pending, 1=running, 2=completed, 3=failed, 4=paused
float32 progress_percent
string assigned_equipment
builtin_interfaces/Time start_time
builtin_interfaces/Time estimated_completion
```

#### SafetyStatus.msg
```
std_msgs/Header header
bool estop_active
bool watchdog_ok
string[] active_zones
uint8 safety_level       # 0=normal, 1=warning, 2=critical, 3=emergency
```

### Custom Services

#### StartJob.srv
```
string job_id
string part_id
int32 quantity
int32 priority
---
bool success
string message
string assigned_equipment
```

#### LifecycleTransition.srv
```
string node_name
uint8 transition_id      # 1=configure, 2=cleanup, 3=activate, 4=deactivate
float32 timeout_sec
---
bool success
string message
uint8 previous_state
uint8 current_state
float32 transition_duration_sec
string error_code
string error_detail
```

#### GetEquipmentStatus.srv
```
string equipment_id
---
bool success
lego_mcp_msgs/EquipmentStatus status
```

### Custom Actions

#### ExecuteJob.action
```
# Goal
string job_id
string part_id
int32 quantity

---
# Result
bool success
int32 completed_quantity
int32 failed_quantity
string[] defects

---
# Feedback
float32 progress_percent
int32 current_quantity
string current_operation
string status_message
```

#### AssemblePart.action
```
# Goal
string assembly_id
geometry_msgs/PoseStamped[] pick_poses
geometry_msgs/PoseStamped[] place_poses

---
# Result
bool success
int32 assembled_count
string[] errors

---
# Feedback
int32 current_step
int32 total_steps
string current_action
```

---

## Lifecycle Nodes

### Lifecycle States

```
                    ┌───────────────┐
                    │   UNKNOWN     │
                    └───────┬───────┘
                            │ create
                            ▼
                    ┌───────────────┐
       ┌───────────▶│ UNCONFIGURED  │◀───────────┐
       │            └───────┬───────┘            │
       │                    │ configure          │ cleanup
       │                    ▼                    │
       │            ┌───────────────┐            │
       │            │   INACTIVE    │────────────┘
       │            └───────┬───────┘
       │                    │ activate
       │                    ▼
       │            ┌───────────────┐
       │ shutdown   │    ACTIVE     │
       │            └───────┬───────┘
       │                    │ deactivate
       │                    ▼
       │            ┌───────────────┐
       └────────────│   INACTIVE    │
                    └───────────────┘
```

### Lifecycle Node Implementation

```python
#!/usr/bin/env python3
"""Example Lifecycle Node."""

from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn


class MyLifecycleNode(LifecycleNode):
    """Lifecycle-managed equipment node."""

    def __init__(self):
        super().__init__('my_node')
        self.declare_parameter('use_sim', False)
        self._publisher = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure: Initialize resources, connect to hardware."""
        self.get_logger().info('Configuring...')

        # Create lifecycle publisher (inactive until activated)
        self._publisher = self.create_lifecycle_publisher(
            String, 'output', 10
        )

        # Connect to hardware (simulation or real)
        use_sim = self.get_parameter('use_sim').value
        if not use_sim:
            # Connect to real hardware
            pass

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate: Start publishing, enable operations."""
        self.get_logger().info('Activating...')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate: Stop operations, maintain connection."""
        self.get_logger().info('Deactivating...')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup: Release resources, disconnect hardware."""
        self.get_logger().info('Cleaning up...')
        self._publisher = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown: Final cleanup before destruction."""
        self.get_logger().info('Shutting down...')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Error: Handle error recovery."""
        self.get_logger().error(f'Error in state: {state.label}')
        return TransitionCallbackReturn.SUCCESS  # Allow recovery
```

### Lifecycle Commands

```bash
# Get current state
ros2 lifecycle get /lego_mcp/grbl_node

# List available transitions
ros2 lifecycle list /lego_mcp/grbl_node

# Trigger transitions
ros2 lifecycle set /lego_mcp/grbl_node configure
ros2 lifecycle set /lego_mcp/grbl_node activate
ros2 lifecycle set /lego_mcp/grbl_node deactivate
ros2 lifecycle set /lego_mcp/grbl_node cleanup
ros2 lifecycle set /lego_mcp/grbl_node shutdown

# Batch operations via service bridge
ros2 service call /lego_mcp/bridge/lifecycle/get_all_states std_srvs/srv/Trigger
```

---

## Supervision Tree

### OTP-Style Supervision

The supervision tree provides fault tolerance using Erlang/OTP patterns:

```
RootSupervisor (one_for_all)
├── SafetySupervisor (one_for_all)
│   ├── safety_node [LIFECYCLE]
│   └── watchdog_node [LIFECYCLE]
├── EquipmentSupervisor (one_for_one)
│   ├── grbl_node [LIFECYCLE]
│   ├── formlabs_node [LIFECYCLE]
│   └── bambu_node [LIFECYCLE]
├── RoboticsSupervisor (rest_for_one)
│   ├── moveit_node
│   ├── ned2_node [LIFECYCLE]
│   └── xarm_node [LIFECYCLE]
└── AGVSupervisor (one_for_one)
    └── agv_fleet_node [LIFECYCLE]
```

### Restart Strategies

| Strategy | Behavior |
|----------|----------|
| `one_for_one` | Restart only failed child |
| `one_for_all` | Restart all children when one fails |
| `rest_for_one` | Restart failed + all started after it |

### Heartbeat Monitoring

```bash
# Monitor heartbeats
ros2 topic echo /lego_mcp/heartbeat/grbl_node

# Supervision health
ros2 topic echo /lego_mcp/supervision/health

# Recovery service
ros2 service call /lego_mcp/supervisor/recover std_srvs/srv/Trigger
```

### Configuration

```yaml
# config/supervision_tree.yaml
supervision:
  heartbeat_timeout_ms: 500
  max_restarts: 5
  restart_window_sec: 60

  supervisors:
    safety:
      strategy: one_for_all
      children:
        - safety_node
        - watchdog_node

    equipment:
      strategy: one_for_one
      children:
        - grbl_node
        - formlabs_node
        - bambu_node
```

---

## Launch Files

### Full System Launch

```bash
# Basic simulation
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true

# Production with all features
ros2 launch lego_mcp_bringup full_system.launch.py \
    use_sim:=false \
    enable_safety:=true \
    enable_equipment:=true \
    enable_robotics:=true \
    enable_agv:=true \
    enable_vision:=true \
    enable_supervision:=true \
    enable_security:=true \
    enable_scada:=true
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim` | `false` | Simulation mode |
| `enable_safety` | `true` | Safety subsystem |
| `enable_equipment` | `true` | Equipment nodes |
| `enable_robotics` | `false` | Robot arms + MoveIt2 |
| `enable_agv` | `false` | AGV fleet |
| `enable_vision` | `true` | Computer vision |
| `enable_supervision` | `true` | Supervision tree |
| `enable_security` | `false` | SROS2 security |
| `enable_scada` | `false` | SCADA bridges |
| `heartbeat_timeout_ms` | `500` | Heartbeat timeout |
| `max_restarts` | `5` | Max restart attempts |

### Subsystem Launches

```bash
# Robotics only
ros2 launch lego_mcp_bringup robotics.launch.py use_sim:=true

# SCADA bridges only
ros2 launch lego_mcp_bringup scada_bridges.launch.py use_sim:=true

# Supervision tree only
ros2 launch lego_mcp_supervisor supervision.launch.py
```

### Startup Sequence

```
Phase 1 (0s):    Supervision Tree
Phase 2 (2s):    Safety Systems (ISA-95 L1)
Phase 3 (3s):    Security (if enabled)
Phase 4 (5s):    Equipment Nodes (ISA-95 L0)
Phase 5 (8s):    Vision Systems
Phase 6 (12s):   Orchestration (ISA-95 L2)
Phase 7 (15s):   Robotics (if enabled)
Phase 8 (18s):   AGV Fleet (if enabled)
Phase 9 (22s):   SCADA Bridges (if enabled)
Phase 10 (25s):  Startup Complete
```

---

## Topics & Services Reference

### Core Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/lego_mcp/equipment/status` | `EquipmentStatus` | Equipment state |
| `/lego_mcp/jobs/status` | `JobStatus` | Job progress |
| `/lego_mcp/safety/estop` | `SafetyStatus` | Safety state |
| `/lego_mcp/supervision/health` | `DiagnosticArray` | Supervision health |
| `/lego_mcp/heartbeat/<node>` | `Header` | Node heartbeats |

### Equipment Topics

| Topic | Type | Equipment |
|-------|------|-----------|
| `/lego_mcp/grbl/position` | `Point` | CNC position |
| `/lego_mcp/grbl/state` | `String` | CNC state |
| `/lego_mcp/formlabs/status` | `EquipmentStatus` | SLA status |
| `/lego_mcp/formlabs/print_progress` | `Float32` | Print progress |
| `/lego_mcp/bambu/status` | `EquipmentStatus` | FDM status |
| `/lego_mcp/bambu/temperatures` | `Float32MultiArray` | Temperatures |

### Vision Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/lego_mcp/vision/image_raw` | `Image` | Raw camera image |
| `/lego_mcp/vision/detections` | `Detection2DArray` | YOLO detections |
| `/lego_mcp/vision/defects` | `String` | Defect classifications |

### Core Services

| Service | Type | Description |
|---------|------|-------------|
| `/lego_mcp/orchestrator/start_job` | `StartJob` | Start job |
| `/lego_mcp/orchestrator/get_job_status` | `GetJobStatus` | Get job status |
| `/lego_mcp/bridge/lifecycle/transition` | `LifecycleTransition` | Lifecycle control |
| `/lego_mcp/bridge/lifecycle/get_all_states` | `Trigger` | Get all states |

### Equipment Services

| Service | Type | Equipment |
|---------|------|-----------|
| `/lego_mcp/grbl/home` | `Trigger` | Home CNC |
| `/lego_mcp/grbl/send_gcode` | `String` | Send G-code |
| `/lego_mcp/formlabs/start_print` | `StartPrint` | Start SLA print |
| `/lego_mcp/formlabs/abort` | `Trigger` | Abort print |

---

## SROS2 Security

### IEC 62443 Security Zones

| Zone | Level | Nodes |
|------|-------|-------|
| Zone 0 | SL-4 | safety_node, watchdog_node |
| Zone 1 | SL-3 | grbl_node, formlabs_node, bambu_node |
| Zone 2 | SL-2 | orchestrator, agv_fleet |
| Zone 3 | SL-1 | scada_bridges, mcp_bridge |
| Zone 4 | SL-0 | cloud_connectors |

### Enabling Security

```bash
# Generate keystore
ros2 security generate_keystore /etc/lego_mcp/keystore

# Generate keys for all nodes
ros2 security generate_keys \
    /etc/lego_mcp/keystore \
    /lego_mcp/safety_node \
    /lego_mcp/grbl_node \
    /lego_mcp/orchestrator

# Enable security
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_KEYSTORE=/etc/lego_mcp/keystore

# Launch with security
ros2 launch lego_mcp_bringup full_system.launch.py enable_security:=true
```

### Permission Configuration

```xml
<!-- config/permissions/orchestrator.xml -->
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:noNamespaceSchemaLocation="http://www.omg.org/spec/DDS-SECURITY/20170901/dds_security.xsd">
  <permissions>
    <grant name="orchestrator_permissions">
      <subject_name>CN=/lego_mcp/orchestrator</subject_name>
      <validity>
        <not_before>2024-01-01T00:00:00</not_before>
        <not_after>2034-01-01T00:00:00</not_after>
      </validity>
      <allow_rule>
        <domains><id>42</id></domains>
        <publish>
          <topics><topic>/lego_mcp/jobs/*</topic></topics>
        </publish>
        <subscribe>
          <topics><topic>/lego_mcp/equipment/*</topic></topics>
        </subscribe>
      </allow_rule>
    </grant>
  </permissions>
</dds>
```

---

## SCADA Protocol Bridges

### OPC UA Server (OPC 40501)

```bash
# Launch OPC UA server
ros2 launch lego_mcp_bringup scada_bridges.launch.py enable_opcua:=true

# Server endpoint
opc.tcp://localhost:4840

# Namespace
http://legomcp.dev/cnc
```

Node structure:
```
Root
├── Objects
│   └── CncInterface (OPC 40501)
│       ├── CncAxisList
│       │   ├── X_Axis
│       │   ├── Y_Axis
│       │   └── Z_Axis
│       ├── CncSpindleList
│       │   └── MainSpindle
│       └── CncAlarmList
```

### MTConnect Agent

```bash
# Launch MTConnect agent
ros2 launch lego_mcp_bringup scada_bridges.launch.py enable_mtconnect:=true

# Endpoints
http://localhost:5000/probe    # Device capability
http://localhost:5000/current  # Current state
http://localhost:5000/sample   # Historical data
```

### Sparkplug B

```bash
# Launch Sparkplug B edge node
ros2 launch lego_mcp_bringup scada_bridges.launch.py enable_sparkplug:=true

# MQTT topics
spBv1.0/lego_mcp/NBIRTH/factory_floor     # Node birth
spBv1.0/lego_mcp/NDATA/factory_floor      # Node data
spBv1.0/lego_mcp/DBIRTH/factory_floor/*   # Device birth
spBv1.0/lego_mcp/DDATA/factory_floor/*    # Device data
```

---

## Simulation

### Gazebo Simulation

```bash
# Launch simulation
ros2 launch lego_mcp_simulation simulation.launch.py

# With specific world
ros2 launch lego_mcp_simulation simulation.launch.py \
    world:=factory_floor.sdf
```

### Simulation Mode

All equipment nodes support `use_sim:=true`:

```bash
# Simulated equipment
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true
```

In simulation mode:
- No real hardware connections
- Simulated responses with realistic timing
- State machines for equipment behavior
- Random defect generation for testing

---

## Troubleshooting

### Common Issues

#### Nodes not starting
```bash
# Check ROS2 is sourced
echo $ROS_DISTRO  # Should show: humble

# Source workspace
source ros2_ws/install/setup.bash

# Rebuild
cd ros2_ws && colcon build --symlink-install
```

#### Lifecycle transition failed
```bash
# Check current state
ros2 lifecycle get /lego_mcp/grbl_node

# View node logs
ros2 topic echo /rosout | grep grbl_node

# List available transitions
ros2 lifecycle list /lego_mcp/grbl_node
```

#### DDS communication issues
```bash
# Check domain
echo $ROS_DOMAIN_ID

# Run diagnostics
ros2 doctor --report

# Check multicast
ros2 multicast receive
```

#### Equipment not responding
```bash
# Check serial port
ls /dev/ttyUSB*

# Set permissions
sudo chmod 666 /dev/ttyUSB0
sudo usermod -a -G dialout $USER
```

### Debugging Tools

```bash
# Node graph
ros2 run rqt_graph rqt_graph

# Topic monitor
ros2 run rqt_topic rqt_topic

# Service caller
ros2 run rqt_service_caller rqt_service_caller

# Parameter editor
ros2 run rqt_reconfigure rqt_reconfigure

# Log viewer
ros2 run rqt_console rqt_console
```

### Logs

```bash
# View logs
cat ~/.ros/log/latest/launch.log

# Live log stream
ros2 topic echo /rosout

# Filter by node
ros2 topic echo /rosout | grep grbl_node
```

---

## Quick Reference

### Essential Commands

```bash
# Source workspace
source ros2_ws/install/setup.bash

# List nodes
ros2 node list

# List topics
ros2 topic list

# Echo topic
ros2 topic echo /lego_mcp/equipment/status

# Call service
ros2 service call /lego_mcp/grbl/home std_srvs/srv/Trigger

# Lifecycle
ros2 lifecycle set /lego_mcp/grbl_node activate

# Launch simulation
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true
```

### Environment Variables

```bash
export ROS_DOMAIN_ID=42
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_KEYSTORE=/etc/lego_mcp/keystore
export RCUTILS_COLORIZED_OUTPUT=1
```

---

*LEGO MCP Fusion 360 v7.0 - Industry 4.0/5.0 Manufacturing Platform*
