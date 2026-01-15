# ADR-005: Unified Command Center Architecture

## Status

Accepted

## Date

2026-01-07

## Context

LEGO MCP v7 had multiple independent dashboards and monitoring systems:
- Equipment status in one dashboard
- Quality metrics in another
- Alerts scattered across systems
- Actions required manual coordination

This led to:
1. **Context Switching**: Operators must check multiple screens
2. **Delayed Response**: No unified view of system health
3. **Manual Correlation**: Events across systems not linked
4. **Action Fragmentation**: No single place to manage actions

World-class manufacturing systems (Siemens MindSphere, PTC ThingWorx) provide unified command centers for holistic visibility.

## Decision

We will implement a **Unified Command Center (UCC)** as the single pane of glass for all operations:

### 1. Core Components

**System Health Panel**
- Real-time status of all services (32 ROS2 nodes, microservices)
- Equipment health aggregation
- Network and infrastructure status
- Automatic dependency mapping

**KPI Aggregator**
- Real-time metrics from all sources
- Hierarchical rollup (machine -> cell -> line -> plant)
- Configurable thresholds and targets
- Trend analysis and forecasting

**Alert Manager**
- Unified alert ingestion from all systems
- Severity-based prioritization
- Alert correlation and deduplication
- Escalation workflows
- On-call integration

**Action Console**
- Algorithm-to-action pipeline
- Human-in-the-loop approval workflow
- Execution tracking
- Audit trail

**Service Registry**
- Dynamic service discovery
- Health monitoring
- Version tracking
- Dependency management

### 2. Data Flow

```
Equipment/Sensors -> ROS2 Bridge -> Message Bus -> KPI Aggregator
                                              \-> Alert Manager
                                               \-> Action Engine

AI/ML Models -> Insight Generator -> Decision Engine -> Action Console
                                                    \-> Approval Workflow
                                                     \-> Execution Engine
```

### 3. Architecture

```
+----------------------------------------------------------+
|                 UNIFIED COMMAND CENTER                    |
|  +------------+  +------------+  +------------+          |
|  | System     |  | KPI        |  | Alert      |          |
|  | Health     |  | Dashboard  |  | Center     |          |
|  +------------+  +------------+  +------------+          |
|  +------------+  +------------+                          |
|  | Action     |  | Service    |                          |
|  | Console    |  | Registry   |                          |
|  +------------+  +------------+                          |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|                 ORCHESTRATION LAYER                       |
|  +----------------+  +----------------+  +-------------+  |
|  | Algorithm-to-  |  | Co-Simulation  |  | Event       |  |
|  | Action Engine  |  | Coordinator    |  | Correlator  |  |
|  +----------------+  +----------------+  +-------------+  |
+----------------------------------------------------------+
                         |
+----------------------------------------------------------+
|                 SERVICE LAYER                             |
|  AI/ML | Simulation | Equipment | Quality | Scheduling   |
+----------------------------------------------------------+
```

### Implementation

- `dashboard/services/command_center/__init__.py`
- `dashboard/routes/command_center/`
- `dashboard/templates/pages/command_center/`

## Consequences

### Positive

- **Single Pane of Glass**: All information in one place
- **Faster Response**: Immediate visibility to issues
- **Coordinated Actions**: Unified action management
- **Correlation**: Cross-system event correlation
- **Audit**: Complete operational audit trail

### Negative

- **Complexity**: More components to maintain
- **Single Point of Failure**: Command center becomes critical
- **Data Volume**: Must handle high event throughput
- **Learning Curve**: Operators need training

### Risks

- Performance degradation under high load
- Information overload for operators
- Alert fatigue from too many notifications

### Mitigations

- Horizontal scaling with load balancing
- Progressive disclosure UI design
- Smart alert correlation and deduplication
- Configurable notification preferences
- Regular UX testing with operators

## KPI Dashboard Metrics

| Category | Metric | Update Frequency |
|----------|--------|------------------|
| Production | OEE | Real-time |
| Production | Throughput | 1 minute |
| Production | Cycle Time | Real-time |
| Quality | First Pass Yield | 1 minute |
| Quality | Defect Rate | 5 minutes |
| Equipment | Availability | Real-time |
| Equipment | MTBF | Hourly |
| Sustainability | Energy/Unit | 5 minutes |
| Sustainability | Carbon/Unit | Hourly |

## Performance Requirements

| Metric | Requirement |
|--------|-------------|
| Dashboard Load Time | < 1 second |
| KPI Update Latency | < 1 second |
| Alert Notification | < 500ms |
| Action Execution | < 2 seconds |
| Concurrent Users | 100+ |
| Events/Second | 1000+ |

## References

- [ISA-95 Operations Management](https://www.isa.org/standards-and-publications/isa-standards/isa-95)
- [Siemens MindSphere](https://siemens.mindsphere.io/)
- [PTC ThingWorx](https://www.ptc.com/en/products/thingworx)
