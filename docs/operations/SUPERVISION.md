# Supervision Tree Operations Guide

## LEGO MCP Fusion 360 - Industry 4.0/5.0 Architecture

This guide covers the OTP-style supervision system for fault-tolerant operations.

---

## Overview

The LEGO MCP supervision system implements Erlang/OTP-inspired fault tolerance patterns for ROS2. It provides automatic recovery from node failures with configurable restart strategies.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LEGO MCP Supervision Tree                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                    ┌───────────────────────┐                             │
│                    │   RootSupervisor      │                             │
│                    │   (one_for_all)       │                             │
│                    └───────────┬───────────┘                             │
│                                │                                         │
│         ┌──────────────────────┼──────────────────────┐                 │
│         │                      │                      │                 │
│         ▼                      ▼                      ▼                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │   Safety     │      │  Equipment   │      │  Robotics    │          │
│  │  Supervisor  │      │  Supervisor  │      │  Supervisor  │          │
│  │(one_for_all) │      │(one_for_one) │      │(rest_for_one)│          │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘          │
│         │                      │                      │                 │
│    ┌────┴────┐          ┌──────┼──────┐        ┌─────┼─────┐           │
│    │         │          │      │      │        │     │     │           │
│    ▼         ▼          ▼      ▼      ▼        ▼     ▼     ▼           │
│ [safety] [watchdog]  [grbl][formlabs][bambu][moveit][ned2][xarm]       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Supervision Strategies

### one_for_one

**Behavior**: Restart only the failed child node.

**Use Case**: Independent nodes where failure is isolated.

```yaml
# Example: Equipment nodes
equipment_supervisor:
  strategy: one_for_one
  children:
    - grbl_node      # If fails, only grbl_node restarts
    - formlabs_node  # Other nodes unaffected
    - bambu_node
```

**When to use**:
- Equipment nodes (each machine is independent)
- Stateless services
- Nodes with no dependencies on each other

### one_for_all

**Behavior**: Restart ALL children when ANY child fails.

**Use Case**: Tightly coupled nodes that share state.

```yaml
# Example: Safety system
safety_supervisor:
  strategy: one_for_all
  children:
    - safety_node    # If fails, ALL restart
    - watchdog_node  # Ensures consistent state
```

**When to use**:
- Safety-critical systems (fail-safe)
- Nodes with shared state
- Distributed transactions
- When partial failure is worse than full restart

### rest_for_one

**Behavior**: Restart the failed child AND all children started AFTER it.

**Use Case**: Nodes with ordering dependencies.

```yaml
# Example: Robotics chain
robotics_supervisor:
  strategy: rest_for_one
  children:
    - moveit_node    # 1. Motion planning
    - ned2_node      # 2. Robot 1 (depends on moveit)
    - xarm_node      # 3. Robot 2 (depends on moveit)
    # If ned2 fails: ned2 + xarm restart
    # If moveit fails: ALL restart
```

**When to use**:
- Initialization order matters
- Pipeline-style processing
- Master/slave relationships

---

## Configuration

### supervision_tree.yaml

```yaml
# Root supervisor configuration
root_supervisor:
  strategy: one_for_all
  max_restarts: 3
  restart_window_sec: 60
  children:
    - safety_supervisor
    - equipment_supervisor
    - robotics_supervisor
    - agv_supervisor

# Safety supervisor (highest priority)
safety_supervisor:
  strategy: one_for_all
  max_restarts: 5
  restart_window_sec: 30
  children:
    - node: safety_node
      package: lego_mcp_safety
      restart_delay_sec: 0.1  # Fast restart
      critical: true
    - node: watchdog_node
      package: lego_mcp_safety
      restart_delay_sec: 0.1

# Equipment supervisor
equipment_supervisor:
  strategy: one_for_one
  max_restarts: 3
  restart_window_sec: 60
  children:
    - node: grbl_node
      package: grbl_ros2
      restart_delay_sec: 2.0
    - node: formlabs_node
      package: formlabs_ros2
      restart_delay_sec: 5.0  # Longer for printer warmup
    - node: bambu_node
      package: bambu_ros2
      restart_delay_sec: 3.0

# Robotics supervisor
robotics_supervisor:
  strategy: rest_for_one
  max_restarts: 3
  restart_window_sec: 120
  children:
    - node: moveit_node
      package: lego_mcp_moveit_config
      restart_delay_sec: 5.0
    - node: ned2_node
      package: niryo_ned2
      restart_delay_sec: 3.0
      depends_on: [moveit_node]
    - node: xarm_node
      package: xarm_ros2
      restart_delay_sec: 3.0
      depends_on: [moveit_node]
```

---

## Heartbeat Monitoring

### Topics

Each supervised node publishes heartbeats:

```
/lego_mcp/heartbeat/<node_name>
```

### Message Format

```python
# std_msgs/String
{
    "node": "grbl_node",
    "timestamp": "2024-01-15T10:30:00.123Z",
    "state": "active",
    "health": {
        "cpu_percent": 15.2,
        "memory_mb": 128,
        "errors": 0
    }
}
```

### Monitoring Commands

```bash
# View all heartbeats
ros2 topic echo /lego_mcp/heartbeat/+

# Check specific node
ros2 topic echo /lego_mcp/heartbeat/grbl_node

# Health check service
ros2 service call /lego_mcp/supervisor/health std_srvs/srv/Trigger
```

---

## Recovery Procedures

### Automatic Recovery

The supervisor automatically handles:

1. **Node Crash**: Restarts according to strategy
2. **Heartbeat Timeout**: Kills and restarts unresponsive node
3. **Error State**: Attempts recovery or restart

### Manual Recovery

```bash
# Request manual recovery
ros2 service call /lego_mcp/supervisor/recover std_srvs/srv/Trigger

# Force restart specific node
ros2 service call /lego_mcp/supervisor/restart_node \
  lego_mcp_msgs/srv/RestartNode "{node_name: 'grbl_node'}"

# View supervisor status
ros2 topic echo /lego_mcp/supervisor/status
```

### Escalation

If max restarts exceeded within restart window:

1. Supervisor logs CRITICAL error
2. Parent supervisor is notified
3. If root supervisor, system enters DEGRADED mode
4. Alert sent to dashboard

---

## State Checkpointing

### Checkpoint Creation

```python
from lego_mcp_supervisor import CheckpointManager

checkpoint = CheckpointManager()

# Save state before risky operation
checkpoint.save('job_123_start', {
    'job_id': 'job_123',
    'position': [100, 50, 20],
    'tool': 'end_mill_3mm',
    'line_number': 150,
})
```

### Checkpoint Restoration

```python
# Restore after recovery
state = checkpoint.load('job_123_start')
if state:
    resume_job(state['job_id'], state['line_number'])
```

### Automatic Checkpointing

Configure in `supervision_tree.yaml`:

```yaml
checkpointing:
  enabled: true
  interval_sec: 30
  max_checkpoints: 10
  storage_path: /var/lib/lego_mcp/checkpoints
  nodes:
    - orchestrator
    - agv_fleet
```

---

## Monitoring Dashboard

### Status Indicators

| Status | Color | Description |
|--------|-------|-------------|
| HEALTHY | Green | All nodes active, no errors |
| DEGRADED | Yellow | Some nodes restarting |
| CRITICAL | Red | Max restarts exceeded |
| OFFLINE | Gray | Supervisor not running |

### Metrics

- Restart count per node
- Mean time between failures (MTBF)
- Mean time to recovery (MTTR)
- Heartbeat latency
- CPU/memory per node

---

## Best Practices

### 1. Design for Failure

- Assume nodes WILL crash
- Keep nodes stateless where possible
- Use checkpointing for stateful operations

### 2. Choose Strategy Carefully

- Start with `one_for_one` (least impact)
- Use `one_for_all` only when necessary
- Test failure scenarios

### 3. Set Appropriate Timeouts

```yaml
# Fast heartbeat for critical nodes
safety_node:
  heartbeat_interval_ms: 100
  timeout_ms: 500

# Slower for non-critical
analytics_node:
  heartbeat_interval_ms: 5000
  timeout_ms: 15000
```

### 4. Limit Restart Cascades

```yaml
# Prevent infinite restarts
max_restarts: 3
restart_window_sec: 60

# Add delay between restarts
restart_delay_sec: 1.0
```

### 5. Monitor and Alert

- Set up alerts for DEGRADED state
- Track restart patterns
- Review logs after incidents

---

## Troubleshooting

### Node Won't Start

```bash
# Check supervisor logs
ros2 topic echo /lego_mcp/supervisor/logs

# Check if dependencies are ready
ros2 service call /lego_mcp/supervisor/check_dependencies \
  std_srvs/srv/Trigger
```

### Constant Restarts

1. Check max_restarts setting
2. Review node logs for root cause
3. Increase restart_delay_sec
4. Consider `one_for_all` if state corruption suspected

### Supervisor Not Responding

```bash
# Check supervisor health
ros2 node info /lego_mcp/supervisor

# Force supervisor restart (last resort)
ros2 lifecycle set /lego_mcp/supervisor shutdown
ros2 run lego_mcp_supervisor supervisor_node
```

---

## References

- [Erlang/OTP Supervision](https://www.erlang.org/doc/design_principles/sup_princ.html)
- [ROS2 Lifecycle Nodes](https://design.ros2.org/articles/node_lifecycle.html)
- [ISA-95 Standards](https://www.isa.org/standards-and-publications/isa-standards)
