# Chaos Testing Guide

## LEGO MCP Fusion 360 - Industry 4.0/5.0 Architecture

This guide covers chaos engineering practices for validating system resilience.

---

## Overview

Chaos engineering is the discipline of experimenting on a system to build confidence in its capability to withstand turbulent conditions in production.

### Why Chaos Testing?

- **Validate Resilience**: Ensure system recovers from failures
- **Find Weaknesses**: Discover hidden failure modes
- **Improve Recovery**: Reduce Mean Time to Recovery (MTTR)
- **Build Confidence**: Know your system handles failures gracefully

### Safety First

> **WARNING**: Chaos testing should ONLY be performed in isolated test environments. Never inject faults in production without explicit approval and safety controls.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Chaos Testing Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Chaos Controller                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │  │
│  │  │   Fault     │  │  Scenario   │  │    Resilience           │   │  │
│  │  │  Injector   │  │   Runner    │  │    Validator            │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Target System                                  │  │
│  │                                                                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │ Safety   │  │Equipment │  │Orchestr- │  │ AGV Fleet        │  │  │
│  │  │ (Zone 0) │  │ (Zone 1) │  │ator (2)  │  │ (Zone 2)         │  │  │
│  │  │ PROTECTED│  │ ✓ Target │  │ ✓ Target │  │ ✓ Target         │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Launch Chaos Controller

```bash
# Source workspace
source install/setup.bash

# Launch with safety confirmation
ros2 launch lego_mcp_chaos chaos.launch.py \
  i_understand_this_is_for_testing:=true
```

### 2. Run a Scenario

```bash
# Trigger equipment failure scenario
ros2 service call /lego_mcp/chaos/run_scenario std_srvs/srv/Trigger
```

### 3. Validate Resilience

```bash
# Run resilience validation
ros2 service call /lego_mcp/chaos/validate_resilience std_srvs/srv/Trigger
```

---

## Fault Types

### Available Fault Injections

| Fault Type | Description | Use Case |
|------------|-------------|----------|
| NODE_CRASH | Kill a node process | Test supervisor recovery |
| NODE_HANG | SIGSTOP/SIGCONT | Test watchdog detection |
| MESSAGE_DELAY | Add latency | Test timeout handling |
| MESSAGE_DROP | Drop messages | Test reliability |
| NETWORK_PARTITION | Isolate nodes | Test split-brain handling |
| RESOURCE_EXHAUSTION | CPU/memory pressure | Test performance degradation |

### Injection Example

```python
from lego_mcp_chaos import FaultInjector, FaultType

injector = FaultInjector()

# Inject message delay
injection = injector.inject_message_delay(
    topic='/lego_mcp/equipment/status',
    delay_ms=500,
    duration_seconds=30.0
)

# Stop injection
injector.stop_injection(injection.injection_id)
```

---

## Predefined Scenarios

### Equipment Failure

**Purpose**: Test supervisor recovery when equipment node crashes

```yaml
scenario_id: equipment_failure_grbl_node
steps:
  1. Validate initial state
  2. Inject equipment crash
  3. Wait for detection (5s)
  4. Validate supervisor detected failure
  5. Wait for recovery (10s)
  6. Validate equipment recovered
```

**Expected Outcome**:
- Supervisor detects crash within 5 seconds
- Equipment restarts within 15 seconds
- No data loss

### Safety E-Stop

**Purpose**: Test e-stop triggered by orchestrator heartbeat timeout

```yaml
scenario_id: safety_estop_test
steps:
  1. Verify safety node active
  2. Inject orchestrator heartbeat timeout (3s)
  3. Wait for watchdog timeout (2s)
  4. Validate e-stop triggered
  5. Wait for node resume
```

**Expected Outcome**:
- E-stop triggers within 2 seconds
- All equipment stops safely
- System requires manual reset

### Cascade Failure Prevention

**Purpose**: Verify single failure doesn't cascade

```yaml
scenario_id: cascade_failure_test
steps:
  1. Inject supervisor crash
  2. Wait (5s)
  3. Validate safety still active
  4. Validate orchestrator still running
```

**Expected Outcome**:
- Safety zone unaffected
- Other systems continue operating
- Supervisor restarts independently

### Network Partition

**Purpose**: Test handling of network partition between zones

```yaml
scenario_id: network_partition_test
steps:
  1. Inject network partition (control <-> supervisory)
  2. Wait during partition (10s)
  3. Validate equipment in safe state
  4. Stop partition
  5. Wait for recovery (10s)
  6. Validate reconnection
```

**Expected Outcome**:
- Equipment enters safe state during partition
- No data corruption
- Full recovery after partition heals

---

## Resilience Validation

### Validation Criteria

| Criteria | Description | Timeout |
|----------|-------------|---------|
| safety_available | Safety system responsive | 5s |
| orchestrator_responsive | Orchestrator responds | 10s |
| equipment_connected | All equipment online | 30s |
| state_consistent | State matches across nodes | 15s |

### Recovery Time Objectives (RTO)

| Component | Target RTO | Criticality |
|-----------|-----------|-------------|
| Safety Node | 1 second | Critical |
| Equipment Node | 10 seconds | High |
| Orchestrator | 15 seconds | High |
| AGV Fleet | 30 seconds | Medium |

### Running Validation

```python
from lego_mcp_chaos import ResilienceValidator, ValidationLevel

validator = ResilienceValidator(level=ValidationLevel.STRICT)

# Run all validations
report = validator.validate_resilience(scenario_id='my_test')

print(f"Passed: {report.overall_passed}")
print(f"Availability: {report.availability_percentage}%")
print(f"Recovery Time: {report.recovery_time_seconds}s")

# Check recommendations
for rec in report.recommendations:
    print(f"  - {rec}")
```

---

## Configuration

### chaos_scenarios.yaml

```yaml
defaults:
  timeout_seconds: 300
  cleanup_on_failure: true
  validation_level: normal

scenarios:
  custom_scenario:
    name: "Custom Test"
    description: "My custom chaos scenario"
    steps:
      - name: "Inject fault"
        action: inject
        target: target_node
        fault_type: node_crash
      - name: "Wait"
        action: wait
        duration_seconds: 5.0
      - name: "Validate"
        action: validate

recovery_targets:
  safety_node: 1.0
  equipment_node: 10.0
  orchestrator: 15.0
```

### Safety Zone Protection

```yaml
security_zones:
  zone_0_safety:
    chaos_allowed: false  # NEVER inject faults in safety zone
```

---

## Best Practices

### 1. Start Small
- Begin with single fault injections
- Gradually increase complexity
- Monitor system behavior closely

### 2. Isolate Tests
- Use dedicated test environment
- Never test on production systems
- Backup state before testing

### 3. Define Hypothesis
- What do you expect to happen?
- What constitutes success?
- What is the blast radius?

### 4. Automate Cleanup
- Always enable `cleanup_on_failure`
- Use lifecycle nodes for graceful shutdown
- Monitor for lingering faults

### 5. Document Results
- Record all test outcomes
- Track improvements over time
- Share findings with team

---

## Monitoring

### Status Topic

```bash
ros2 topic echo /lego_mcp/chaos/status
```

Output:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "state": "active",
  "active_injections": 1,
  "scenario_running": true,
  "loaded_scenarios": ["equipment_failure_grbl_node", "safety_estop_test"]
}
```

### Results Topic

```bash
ros2 topic echo /lego_mcp/chaos/results
```

---

## Troubleshooting

### Scenario Won't Start

```bash
# Check if another scenario is running
ros2 service call /lego_mcp/chaos/health std_srvs/srv/Trigger

# Stop all active tests
ros2 service call /lego_mcp/chaos/stop_all std_srvs/srv/Trigger
```

### Faults Not Cleaning Up

```bash
# Force cleanup
ros2 service call /lego_mcp/chaos/stop_all std_srvs/srv/Trigger

# Check for orphaned processes
ps aux | grep ros2
```

### Validation Always Fails

- Check if target nodes are running
- Verify network connectivity
- Increase timeout values
- Check validation function implementation

---

## References

- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Netflix Chaos Monkey](https://netflix.github.io/chaosmonkey/)
- [AWS Fault Injection Simulator](https://aws.amazon.com/fis/)
- [Google DiRT (Disaster Recovery Testing)](https://cloud.google.com/blog/products/management-tools/shrinking-the-time-to-mitigate-production-incidents)
