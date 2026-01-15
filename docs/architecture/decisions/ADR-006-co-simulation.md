# ADR-006: Co-Simulation Engine Architecture

## Status

Accepted

## Date

2026-01-07

## Context

Manufacturing simulation traditionally uses isolated tools:
- **Discrete Event Simulation (DES)**: Material flow, queuing, scheduling
- **Physics Simulation**: FEA, CFD, thermal analysis
- **Digital Twin**: Real-time state mirroring
- **Monte Carlo**: Stochastic analysis, risk assessment

Each tool operates independently, leading to:
1. **Inconsistent Results**: Different assumptions across simulators
2. **Manual Integration**: Engineers must transfer data between tools
3. **Limited Scenarios**: Can't combine stochastic + physics + flow
4. **No Real-Time**: Can't run simulations alongside production

## Decision

We will implement a **Co-Simulation Engine** that orchestrates multiple simulation engines in a unified framework:

### 1. Simulation Engines

| Engine | Purpose | Interface |
|--------|---------|-----------|
| **DES Engine** | Discrete event flow, queuing | SimPy-based |
| **PINN Twin** | Physics-informed predictions | PyTorch |
| **Monte Carlo** | Stochastic analysis | NumPy-based |
| **What-If Engine** | Scenario comparison | Custom |

### 2. Synchronization Strategy

**Conservative Synchronization** with Lookahead:
- Each engine advances to its next event
- Coordinator collects all next-event times
- Global advance to minimum time
- Repeat until end condition

```
Time: t=0     t=5     t=10    t=15    t=20
DES:  [E1]----[E2]----------[E3]----[E4]
PINN: [----P1----][----P2----][----P3----]
MC:   [-------M1-------][-------M2-------]
      ↑                 ↑
      Sync Points
```

### 3. Operating Modes

**Real-Time Shadow Mode**
- Digital twin mirrors physical factory at 1:1 speed
- Detects drift between prediction and reality
- Triggers alerts on divergence

**Accelerated Mode**
- Run simulations at 100-1000x real-time
- Used for planning and optimization
- Trades accuracy for speed

**Scenario Analysis Mode**
- Multiple parallel simulations
- Compare what-if scenarios
- Statistical analysis of outcomes

**Optimization Loop Mode**
- Automated parameter tuning
- Genetic algorithms, Bayesian optimization
- Find optimal operating points

### 4. Data Exchange

**FMI 2.0 Standard** (Functional Mock-up Interface):
- Standard interface for model exchange
- Import/export FMUs from other tools
- Connect to Simulink, Modelica, etc.

### Implementation

- `dashboard/services/cosimulation/coordinator.py`
- `dashboard/services/cosimulation/scenario_manager.py`
- `dashboard/services/cosimulation/sync_engine.py`
- `dashboard/services/cosimulation/optimization_loop.py`

## Consequences

### Positive

- **Unified Results**: Single source of truth
- **Combined Analysis**: Physics + stochastic + flow together
- **Real-Time Capable**: Shadow mode during production
- **Standard Interface**: FMI 2.0 compatibility
- **Optimization**: Automated parameter tuning

### Negative

- **Complexity**: Coordination logic is intricate
- **Performance**: Sync overhead limits speedup
- **Coupling**: Tight integration between engines
- **Debugging**: Multi-engine issues hard to trace

### Risks

- Synchronization deadlocks
- Numerical instability at sync points
- Performance bottlenecks in coordinator

### Mitigations

- Timeout handling for all engine calls
- Interpolation for time alignment
- Parallel execution where possible
- Comprehensive logging for debugging

## Architecture Diagram

```
                    +--------------------+
                    |  Scenario Manager  |
                    |  (What-If Config)  |
                    +---------+----------+
                              |
                    +---------v----------+
                    |    Coordinator     |
                    | (Sync & Orchestrate)|
                    +---------+----------+
                              |
        +----------+----------+----------+----------+
        |          |          |          |          |
   +----v----+ +---v----+ +---v----+ +---v----+
   |   DES   | |  PINN  | | Monte  | | What-If|
   | Engine  | |  Twin  | | Carlo  | | Engine |
   +---------+ +--------+ +--------+ +--------+
        |          |          |          |
        +----------+----------+----------+
                              |
                    +---------v----------+
                    |   Result Aggregator |
                    |   (Statistics, KPIs)|
                    +--------------------+
```

## Implementation Notes

```python
# Create co-simulation scenario
from dashboard.services.cosimulation import CoSimCoordinator, Scenario

coordinator = CoSimCoordinator()

scenario = Scenario(
    name="Demand Surge Analysis",
    duration_hours=24,
    parameters={
        "demand_multiplier": 2.0,
        "failure_rate": 0.02,
    }
)

# Configure engines
coordinator.add_engine("des", DESEngine(scenario))
coordinator.add_engine("pinn", PINNEngine(thermal_model))
coordinator.add_engine("monte_carlo", MonteCarloEngine(iterations=1000))

# Run simulation
result = coordinator.run(
    mode=SimulationMode.ACCELERATED,
    speed_factor=100,
)

# Analyze results
print(f"Throughput: {result.kpis['throughput']}")
print(f"OEE: {result.kpis['oee']}")
print(f"95th percentile lead time: {result.percentile(95, 'lead_time')}")
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Real-time sync latency | < 100ms |
| Accelerated mode speedup | 100x+ |
| Scenario comparison | 5 parallel |
| Monte Carlo iterations | 10,000/min |

## References

- [FMI Standard](https://fmi-standard.org/)
- [High Level Architecture (HLA) IEEE 1516](https://standards.ieee.org/standard/1516-2010.html)
- [SimPy Discrete Event Simulation](https://simpy.readthedocs.io/)
