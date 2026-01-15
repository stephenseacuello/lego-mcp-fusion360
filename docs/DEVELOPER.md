# Developer Guide - LEGO MCP Fusion 360 v7.0

Contributing to and extending the LEGO MCP Industry 4.0/5.0 Manufacturing Platform.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Development Environment](#development-environment)
4. [ROS2 Development](#ros2-development)
5. [Dashboard Development](#dashboard-development)
6. [Adding New Features](#adding-new-features)
7. [Testing](#testing)
8. [Contributing](#contributing)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEGO MCP v7.0 System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FLASK DASHBOARD                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │   AI     │ │ Digital  │ │ Quality  │ │   ERP    │ │  MRP     │  │   │
│  │  │ Copilot  │ │   Twin   │ │  System  │ │  System  │ │  System  │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  │       └────────────┴────────────┴────────────┴────────────┘        │   │
│  │                              │                                       │   │
│  │  ┌───────────────────────────▼────────────────────────────────────┐ │   │
│  │  │                    ROS2 MCP Bridge                              │ │   │
│  │  │           (WebSocket + Service Clients)                         │ │   │
│  │  └───────────────────────────┬────────────────────────────────────┘ │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│  ════════════════════════════════════════════════════════════════════════  │
│                            ROS2 DDS MIDDLEWARE                              │
│  ════════════════════════════════════════════════════════════════════════  │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │                    SUPERVISION TREE (OTP-STYLE)                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ RootSupervisor (one_for_all)                                │    │   │
│  │  │   ├── SafetySupervisor (one_for_all)                        │    │   │
│  │  │   │     ├── safety_node [LIFECYCLE]                         │    │   │
│  │  │   │     └── watchdog_node [LIFECYCLE]                       │    │   │
│  │  │   ├── EquipmentSupervisor (one_for_one)                     │    │   │
│  │  │   │     ├── grbl_node [LIFECYCLE]                           │    │   │
│  │  │   │     ├── formlabs_node [LIFECYCLE]                       │    │   │
│  │  │   │     └── bambu_node [LIFECYCLE]                          │    │   │
│  │  │   └── RoboticsSupervisor (rest_for_one)                     │    │   │
│  │  │         ├── moveit_node                                     │    │   │
│  │  │         └── assembly_coordinator                            │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         ROS2 PACKAGES                                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │  │
│  │  │ orchestrator│ │   safety    │ │   vision    │ │    agv      │    │  │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │  │
│  │         └────────────────┴───────────────┴───────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      EQUIPMENT LAYER (L0/L1)                          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │  │
│  │  │  grbl_ros2  │ │formlabs_ros2│ │  bambu_ros2 │ │  microros   │    │  │
│  │  │ (CNC/Laser) │ │   (SLA)     │ │   (FDM)     │ │  (ESP32)    │    │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ISA-95 Layer Mapping

| Layer | Components | Responsibility |
|-------|------------|----------------|
| **L5** | Cloud Services, Analytics | Enterprise integration |
| **L4** | Dashboard ERP/MRP/Supply Chain | Business operations |
| **L3** | Dashboard MES, Scheduling, Quality | Manufacturing operations |
| **L2** | ROS2 Orchestrator, Supervision | Supervisory control |
| **L1** | ROS2 Safety, Calibration | Direct control |
| **L0** | ROS2 Equipment Nodes | Field devices |

---

## Project Structure

```
lego-mcp-fusion360/
├── ros2_ws/                          # ROS2 Workspace
│   └── src/
│       ├── lego_mcp_msgs/            # Custom messages/services/actions
│       │   ├── msg/
│       │   ├── srv/
│       │   └── action/
│       ├── lego_mcp_bringup/         # Launch files and configs
│       │   ├── launch/
│       │   │   ├── full_system.launch.py
│       │   │   ├── robotics.launch.py
│       │   │   └── scada_bridges.launch.py
│       │   └── config/
│       ├── lego_mcp_orchestrator/    # Job orchestration
│       │   └── lego_mcp_orchestrator/
│       │       ├── orchestrator_lifecycle_node.py
│       │       ├── lifecycle_manager.py
│       │       ├── lifecycle_monitor_node.py
│       │       └── lifecycle_service_bridge.py
│       ├── lego_mcp_supervisor/      # OTP-style supervision
│       │   └── lego_mcp_supervisor/
│       │       ├── supervisor_base.py
│       │       ├── strategies.py
│       │       ├── root_supervisor.py
│       │       └── heartbeat_monitor.py
│       ├── lego_mcp_safety/          # Safety systems
│       ├── lego_mcp_vision/          # Computer vision
│       ├── lego_mcp_calibration/     # Calibration
│       ├── lego_mcp_agv/             # AGV fleet (Nav2)
│       ├── lego_mcp_security/        # SROS2 security
│       ├── lego_mcp_edge/            # SCADA protocol bridges
│       ├── lego_mcp_simulation/      # Gazebo simulation
│       ├── lego_mcp_moveit_config/   # MoveIt2 configuration
│       ├── lego_mcp_microros/        # ESP32/Micro-ROS
│       ├── grbl_ros2/                # GRBL CNC controller
│       └── formlabs_ros2/            # Formlabs SLA printer
│
├── dashboard/                         # Flask Web Application
│   ├── app.py                        # Application factory
│   ├── routes/                       # Route blueprints
│   │   ├── api.py                    # Core API
│   │   ├── ai/                       # AI routes
│   │   ├── manufacturing/            # MES routes
│   │   ├── quality/                  # Quality routes
│   │   ├── erp/                      # ERP routes
│   │   └── scheduling/               # Scheduling routes
│   ├── services/                     # Business logic
│   │   ├── ai/                       # AI services
│   │   ├── digital_twin/             # Digital twin
│   │   ├── quality/                  # SPC, FMEA, QFD
│   │   ├── manufacturing/            # MES services
│   │   ├── erp/                      # ERP services
│   │   └── mcp_bridge.py             # ROS2 bridge
│   ├── templates/                    # Jinja2 templates
│   ├── static/                       # Static assets
│   └── websocket/                    # WebSocket handlers
│
├── mcp-server/                        # MCP Server (Claude tools)
│   └── src/
│       ├── server_enhanced.py
│       └── tools/
│
├── fusion360-addin/                   # Fusion 360 Add-in
│   └── LegoMCP/
│
├── slicer-service/                    # Slicing Service
│   └── src/slicer_api.py
│
├── shared/                            # Shared modules
│   ├── lego_specs.py
│   └── brick_catalog.py
│
├── config/                            # Configuration files
├── tests/                             # Test suite
├── docs/                              # Documentation
└── docker-compose.yml                 # Docker deployment
```

---

## Development Environment

### Prerequisites

```bash
# Ubuntu 22.04 LTS
sudo apt update
sudo apt install -y \
    git \
    python3-pip \
    python3-venv \
    build-essential \
    cmake

# ROS2 Humble
sudo apt install -y ros-humble-desktop

# Additional ROS2 packages
sudo apt install -y \
    ros-humble-lifecycle \
    ros-humble-diagnostic-updater \
    ros-humble-nav2-msgs \
    ros-humble-moveit \
    ros-humble-launch-testing
```

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r tests/requirements-test.txt

# Build ROS2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running in Development Mode

```bash
# Terminal 1: ROS2 (simulation)
cd ros2_ws
source install/setup.bash
ros2 launch lego_mcp_bringup full_system.launch.py use_sim:=true

# Terminal 2: Dashboard (development)
cd dashboard
FLASK_ENV=development python app.py
```

---

## ROS2 Development

### Creating a New Node

```python
#!/usr/bin/env python3
"""
Example ROS2 Lifecycle Node.

LEGO MCP Manufacturing System v7.0
"""

import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from std_msgs.msg import String


class MyLifecycleNode(LifecycleNode):
    """Example lifecycle node."""

    def __init__(self):
        super().__init__('my_node')

        # Declare parameters
        self.declare_parameter('use_sim', False)

        # Publishers/subscribers created in on_configure
        self._publisher = None
        self._subscription = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the node."""
        self.get_logger().info('Configuring...')

        # Create publisher
        self._publisher = self.create_lifecycle_publisher(
            String,
            'my_topic',
            10
        )

        # Create subscription
        self._subscription = self.create_subscription(
            String,
            'input_topic',
            self._callback,
            10
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the node."""
        self.get_logger().info('Activating...')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate the node."""
        self.get_logger().info('Deactivating...')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Clean up the node."""
        self.get_logger().info('Cleaning up...')
        self._publisher = None
        self._subscription = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown the node."""
        self.get_logger().info('Shutting down...')
        return TransitionCallbackReturn.SUCCESS

    def _callback(self, msg: String):
        """Handle incoming messages."""
        if self._publisher is not None:
            self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MyLifecycleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Creating Custom Messages

```
# lego_mcp_msgs/msg/EquipmentStatus.msg
std_msgs/Header header
string equipment_id
string equipment_type
uint8 state           # 0=offline, 1=idle, 2=running, 3=error
float32 utilization
string current_job_id
```

```
# lego_mcp_msgs/srv/StartJob.srv
string job_id
string part_id
int32 quantity
---
bool success
string message
string assigned_equipment
```

### Adding to CMakeLists.txt

```cmake
# lego_mcp_msgs/CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(lego_mcp_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EquipmentStatus.msg"
  "srv/StartJob.srv"
  DEPENDENCIES std_msgs
)

ament_package()
```

### Launch File Pattern

```python
#!/usr/bin/env python3
"""Example launch file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode


def generate_launch_description():
    """Generate launch description."""

    # Declare arguments
    declare_use_sim = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Use simulation mode'
    )

    # Lifecycle node (Phase 1)
    my_node = LifecycleNode(
        package='my_package',
        executable='my_node',
        name='my_node',
        namespace='lego_mcp',
        parameters=[{
            'use_sim': LaunchConfiguration('use_sim'),
        }],
        output='screen',
    )

    # Dependent node (Phase 2 - delayed start)
    dependent_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='my_package',
                executable='dependent_node',
                name='dependent_node',
                namespace='lego_mcp',
                output='screen',
            ),
        ]
    )

    return LaunchDescription([
        declare_use_sim,
        LogInfo(msg='Starting My System...'),
        my_node,
        dependent_node,
    ])
```

---

## Dashboard Development

### Adding a New Route

```python
# dashboard/routes/myfeature.py
from flask import Blueprint, render_template, jsonify, request

bp = Blueprint('myfeature', __name__, url_prefix='/api/myfeature')


@bp.route('/')
def index():
    """MyFeature dashboard page."""
    return render_template('pages/myfeature/dashboard.html')


@bp.route('/data')
def get_data():
    """Get myfeature data."""
    return jsonify({
        'success': True,
        'data': []
    })


@bp.route('/action', methods=['POST'])
def perform_action():
    """Perform an action."""
    data = request.json
    # Process data
    return jsonify({
        'success': True,
        'message': 'Action completed'
    })
```

### Registering Blueprint

```python
# dashboard/app.py
from routes.myfeature import bp as myfeature_bp

def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(myfeature_bp)

    return app
```

### Adding ROS2 Bridge Integration

```python
# dashboard/services/myfeature_service.py
from services.mcp_bridge import get_ros2_client


class MyFeatureService:
    """Service for myfeature functionality."""

    def __init__(self):
        self.ros2_client = get_ros2_client()

    async def get_equipment_status(self, equipment_id: str):
        """Get equipment status from ROS2."""
        topic = f'/lego_mcp/{equipment_id}/status'
        return await self.ros2_client.get_latest_message(topic)

    async def send_command(self, equipment_id: str, command: dict):
        """Send command to equipment via ROS2 service."""
        service = f'/lego_mcp/{equipment_id}/command'
        return await self.ros2_client.call_service(service, command)
```

### Template Pattern

```html
<!-- dashboard/templates/pages/myfeature/dashboard.html -->
{% extends "base.html" %}

{% block title %}MyFeature - LEGO MCP v7.0{% endblock %}

{% block extra_css %}
<style>
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .stat-card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1.25rem;
    }
</style>
{% endblock %}

{% block content %}
<div class="page-header">
    <h1>MyFeature Dashboard</h1>
</div>

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value" id="metric1">--</div>
        <div class="stat-label">Metric 1</div>
    </div>
</div>

<div class="toast-container" id="toastContainer"></div>
{% endblock %}

{% block extra_js %}
<script>
document.addEventListener('DOMContentLoaded', () => {
    loadData();
    setInterval(loadData, 30000);
});

async function loadData() {
    try {
        const response = await fetch('/api/myfeature/data');
        const result = await response.json();
        updateDisplay(result.data);
    } catch (error) {
        showToast('Error', 'Failed to load data', 'error');
    }
}

function updateDisplay(data) {
    document.getElementById('metric1').textContent = data.metric1 || '--';
}

function showToast(title, message, type = 'info') {
    // Toast implementation
}
</script>
{% endblock %}
```

---

## Adding New Features

### 1. Adding a New ROS2 Package

```bash
# Create package
cd ros2_ws/src
ros2 pkg create --build-type ament_python lego_mcp_myfeature \
    --dependencies rclpy std_msgs lego_mcp_msgs

# Add to package
cd lego_mcp_myfeature/lego_mcp_myfeature
touch my_node.py
```

### 2. Adding to Supervision Tree

```python
# In supervisor configuration
SUPERVISED_NODES = {
    'my_node': {
        'package': 'lego_mcp_myfeature',
        'executable': 'my_node',
        'lifecycle': True,
        'supervisor': 'equipment',  # or 'safety', 'robotics'
        'restart_strategy': 'one_for_one',
        'max_restarts': 5,
    }
}
```

### 3. Adding SCADA Integration

```python
# In SCADA adapter
class MyFeatureOPCUANode:
    """OPC UA node for myfeature."""

    def __init__(self, server):
        self.namespace = server.register_namespace('http://legomcp.dev/myfeature')

        # Create node structure
        self.root = server.nodes.objects.add_object(
            self.namespace,
            'MyFeature'
        )

        # Add variables
        self.status = self.root.add_variable(
            self.namespace,
            'Status',
            'Unknown'
        )
        self.status.set_writable()
```

---

## Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# ROS2 tests
cd ros2_ws
colcon test

# Specific package
colcon test --packages-select lego_mcp_orchestrator

# With coverage
python -m pytest tests/ --cov=dashboard --cov-report=html
```

### Writing Unit Tests

```python
# tests/test_myfeature.py
import pytest
from dashboard.services.myfeature_service import MyFeatureService


class TestMyFeatureService:
    """Tests for MyFeatureService."""

    @pytest.fixture
    def service(self):
        return MyFeatureService()

    def test_get_data(self, service):
        result = service.get_data()
        assert result is not None
        assert 'success' in result

    @pytest.mark.asyncio
    async def test_async_operation(self, service):
        result = await service.async_get_data()
        assert result['success'] is True
```

### Writing ROS2 Tests

```python
# ros2_ws/src/lego_mcp_myfeature/test/test_my_node.py
import pytest
import rclpy
from rclpy.node import Node
from lego_mcp_myfeature.my_node import MyNode


@pytest.fixture
def node():
    rclpy.init()
    node = MyNode()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_node_creates(node):
    assert node.get_name() == 'my_node'


def test_parameters(node):
    assert node.get_parameter('use_sim').value == False
```

### Integration Tests

```python
# tests/integration/test_ros2_bridge.py
import pytest
import asyncio
from dashboard.services.mcp_bridge import ROS2Bridge


@pytest.mark.integration
class TestROS2Bridge:
    """Integration tests for ROS2 bridge."""

    @pytest.fixture
    async def bridge(self):
        bridge = ROS2Bridge()
        await bridge.connect()
        yield bridge
        await bridge.disconnect()

    @pytest.mark.asyncio
    async def test_get_node_list(self, bridge):
        nodes = await bridge.get_node_list()
        assert isinstance(nodes, list)

    @pytest.mark.asyncio
    async def test_lifecycle_transition(self, bridge):
        result = await bridge.lifecycle_transition(
            'grbl_node',
            'configure'
        )
        assert result['success'] is True
```

---

## Contributing

### Git Workflow

```bash
# Fork repository
# Clone your fork
git clone https://github.com/yourusername/lego-mcp-fusion360.git

# Create feature branch
git checkout -b feature/my-feature

# Make changes
# ...

# Run tests
python -m pytest tests/ -v
cd ros2_ws && colcon test

# Commit
git add .
git commit -m "feat: Add my feature

- Added new ROS2 node
- Added dashboard integration
- Added tests"

# Push
git push origin feature/my-feature

# Create Pull Request
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructure
- `test`: Tests
- `chore`: Maintenance

### Code Review Checklist

- [ ] Tests pass (Python and ROS2)
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Error handling in place
- [ ] Lifecycle nodes used where appropriate
- [ ] ISA-95 layer compliance

---

## Resources

### Documentation
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [ROS2 Lifecycle](https://design.ros2.org/articles/node_lifecycle.html)
- [MoveIt2 Documentation](https://moveit.picknik.ai/main/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Standards
- ISA-95 (IEC 62264)
- OPC UA (IEC 62541)
- MTConnect (ANSI/MTC1.4)
- IEC 62443 (Industrial Security)

---

*LEGO MCP Fusion 360 v7.0 - Industry 4.0/5.0 Manufacturing Platform*
