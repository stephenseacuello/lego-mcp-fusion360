# LEGO MCP Fusion 360

[![Version](https://img.shields.io/badge/version-6.0.0-blue.svg)](https://github.com/stephenseacuello/lego-mcp-fusion360)
[![Python](https://img.shields.io/badge/python-3.9--3.12-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![ISA-95](https://img.shields.io/badge/ISA--95-compliant-green.svg)](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard)
[![Industry 4.0](https://img.shields.io/badge/Industry-4.0%2F5.0-orange.svg)](https://www.plattform-i40.de)
[![RAMI 4.0](https://img.shields.io/badge/RAMI-4.0-orange.svg)](https://www.plattform-i40.de)
[![Research Platform](https://img.shields.io/badge/Research-Platform-purple.svg)](docs/DEVELOPER.md)

<p align="center">
  <b>World-Class Cyber-Physical Production System for LEGO-Compatible Bricks</b><br>
  <i>v6.0: Algorithm-to-Action Research Platform</i>
</p>

A **world-class, best-in-class** digital manufacturing platform featuring:
- **Industry 4.0/5.0 Foundation**: ISA-95/IEC 62264, RAMI 4.0, OPC-UA ready
- **AI-Native Operations**: Claude-powered manufacturing copilot, autonomous optimization
- **Zero-Defect Manufacturing**: Six Sigma quality, predictive quality, in-process control
- **Advanced Scheduling**: CP-SAT optimization, NSGA-II multi-objective, RL dispatching
- **Sustainability**: Carbon footprint tracking, Scope 1/2/3 emissions, circular economy
- **Enterprise Integration**: ERP/MES/MRP, full traceability, FDA 21 CFR Part 11 ready

**NEW in v6.0:**
- **Multi-Agent Orchestration**: Coordinated Quality, Scheduling, and Maintenance agents
- **Causal AI Engine**: Structural causal models, counterfactual analysis, root cause identification
- **Generative Design**: Topology optimization, LEGO-specific clutch power optimization
- **Closed-Loop Learning**: Production feedback to model retraining, drift detection
- **Algorithm-to-Action Bridge**: Direct equipment control from AI decisions
- **Research Platform**: Experiment tracking, A/B testing, multi-armed bandits, publication export

Design parametric LEGO bricks through natural language, manage production end-to-end, and manufacture with 3D printing, CNC milling, or laser engraving—all powered by Claude AI through the Model Context Protocol and Autodesk Fusion 360.

---

## World-Class Benchmarks

| Metric | Industry Average | World-Class | Our Target |
|--------|------------------|-------------|------------|
| OEE | 60% | 85%+ | **90%** |
| First Pass Yield | 95% | 99.5%+ | **99.7%** |
| DPMO | 6,210 | <3.4 (6 Sigma) | **<10** |
| Schedule Adherence | 85% | 98%+ | **99%** |
| Inventory Turns | 8-12 | 20+ | **24** |
| Order-to-Ship | 5-10 days | <24 hours | **Same Day** |
| Carbon per Unit | N/A | Tracked | **Net Zero Ready** |

---

## What Can You Do?

### Design & Manufacturing
| Feature | Description |
|---------|-------------|
| Natural Language | "Create a 2x4 brick and slice for my Bambu P1S" |
| Brick Types | Standard, plates, tiles, slopes, technic, round, arch |
| Export Formats | STL, STEP, 3MF for any CAD/printing workflow |
| 3D Print | Bambu Lab, Prusa, Ender + LEGO-optimized settings |
| CNC Mill | GRBL, TinyG/Bantam, Haas + auto toolpaths |
| Laser Engrave | Custom text, logos, QR codes on bricks |

### Industry 4.0/5.0 Manufacturing (v5.0)
| Feature | Description |
|---------|-------------|
| **MES** | Work orders, shop floor display, Andon boards, real-time events |
| **OEE Tracking** | Availability x Performance x Quality, real-time dashboard |
| **Quality Control** | SPC (EWMA, CUSUM, T-squared), Zero-defect, FMEA, QFD |
| **ERP** | BOM, costing (ABC, target), procurement, demand forecasting |
| **MRP** | Material planning, finite capacity scheduling, ATP/CTP |
| **Digital Twin** | Real-time state, predictive maintenance, DES simulation |
| **AI Copilot** | Claude-powered decision support, anomaly explanation |
| **Sustainability** | Carbon tracking (Scope 1/2/3), energy optimization |
| **Compliance** | FDA 21 CFR Part 11 ready, complete audit trail |

---

## Architecture (v5.0 World-Class)

```
                          WORLD-CLASS CPPS ARCHITECTURE

                      AI LAYER (Cross-Cutting Intelligence)
   +---------------+ +---------------+ +---------------+ +---------------+
   |    Claude     | |     RAG       | |  Autonomous   | |  Predictive   |
   |    Copilot    | |   Knowledge   | |    Agents     | |    Models     |
   |     (LLM)     | |     Base      | |    (Multi)    | |    (ML/RL)    |
   +---------------+ +---------------+ +---------------+ +---------------+

                 Level 4: Business Planning (ERP) + Sustainability
   +------------+ +------------+ +------------+ +------------+ +------------+
   |  Customer  | |  ATP/CTP   | |    QFD     | |   Carbon   | |   Supply   |
   |   Orders   | |   Promise  | |   (VoC)    | |   Tracker  | |   Chain    |
   +------------+ +------------+ +------------+ +------------+ +------------+
   +------------+ +------------+ +------------+ +------------+ +------------+
   |  Dynamic   | |   Target   | |    ABC     | |   Demand   | |  Supplier  |
   |  Pricing   | |   Costing  | |   Costing  | |  Forecast  | |   Portal   |
   +------------+ +------------+ +------------+ +------------+ +------------+
               | CONSTRAINTS & OBJECTIVES (not commands) |

             Level 3: Manufacturing Operations (MES/MOM) + Simulation
   +------------+ +------------+ +------------+ +------------+ +------------+
   |  NSGA-II   | |   CP-SAT   | |    DES     | |    MRP     | |Alternative |
   | Multi-Obj  | | Scheduler  | | Simulation | |  Enhanced  | |  Routings  |
   +------------+ +------------+ +------------+ +------------+ +------------+
   +------------+ +------------+ +------------+ +------------+ +------------+
   |  Dynamic   | |Zero-Defect | |  Digital   | | Compliance | |    KPI     |
   |   FMEA     | |  Quality   | |   Thread   | |  21CFR11   | |   Engine   |
   +------------+ +------------+ +------------+ +------------+ +------------+
                    | CQRS/Event Sourcing via Redis Streams |

                Level 2: MCP/Supervisory Control + Edge Computing
   +------------+ +------------+ +------------+ +------------+ +------------+
   |  Rolling   | |     RL     | |    MPC     | |    Edge    | |    IIoT    |
   |  Horizon   | | Dispatcher | | Controller | |  Runtime   | |  Gateway   |
   +------------+ +------------+ +------------+ +------------+ +------------+
   +------------+ +------------+ +------------+ +------------+ +------------+
   | Predictive | |  Process   | | In-Process | |   Voice    | |     AR     |
   |  Quality   | |Fingerprint | |  Control   | | Interface  | |  Overlays  |
   +------------+ +------------+ +------------+ +------------+ +------------+
                       | <10ms Event Loop |

               Level 1: Cell/PLC Control (Reactive) + Vision
   +------------+ +------------+ +------------+ +------------+ +------------+
   |  Printer   | |    Mill    | |   Laser    | |     CV     | |  Virtual   |
   | Controller | | Controller | | Controller | | Inspector  | | Metrology  |
   | (Multi-API)| |   (GRBL)   | |(LightBurn) | |  (YOLO11)  | |    (ML)    |
   +------------+ +------------+ +------------+ +------------+ +------------+

                   Level 0: Sensors/Actuators + Energy Monitoring
   +------------+ +------------+ +------------+ +------------+ +------------+
   |Temperature | |  Position  | |   Camera   | |   Energy   | |   Layer    |
   |  Sensors   | |  Encoders  | |  Streams   | |   Meters   | |  Scanner   |
   +------------+ +------------+ +------------+ +------------+ +------------+
             | OPC-UA / MQTT / MTConnect / Modbus |
```

### Port Configuration

| Service | Port | Description |
|---------|------|-------------|
| Fusion 360 Add-in | 8767 | HTTP API for brick creation/export |
| Slicer Service | 8766 | Docker container for G-code generation |
| Web Dashboard | 5000 | Flask UI + REST APIs |
| PostgreSQL | 5432 | Manufacturing database |
| Redis | 6379 | Event streaming & cache |

### API Endpoints

| API | Base URL | Description |
|-----|----------|-------------|
| Manufacturing | `/api/mes` | Work orders, shop floor, OEE |
| Quality | `/api/quality` | Inspections, SPC, FMEA, QFD |
| ERP | `/api/erp` | BOM, costing, orders, ATP/CTP |
| **ERP - Vendors** | `/api/erp/vendors` | Supplier lifecycle, scorecarding, certifications |
| **ERP - Financials** | `/api/erp/financials` | AR/AP/GL, aging, cash management |
| MRP | `/api/mrp` | Planning, capacity, scheduling |
| **MRP - Materials** | `/api/mrp/materials` | Filament spool tracking, consumption |
| Digital Twin | `/api/twin` | Equipment state, maintenance |
| Analytics | `/api/analytics` | KPIs, OEE, real-time dashboards |
| Sustainability | `/api/sustainability` | Carbon, energy, materials |
| Simulation | `/api/simulation` | DES, what-if scenarios |

---

## Quick Start

### Prerequisites

- **Autodesk Fusion 360** (free for personal use)
- **Python 3.9-3.12** (Roboflow/vision deps require <3.13)
- **Docker** (for slicer service)
- **Claude Desktop** (for MCP integration)

### Step 1: Clone and Setup

```bash
git clone https://github.com/stephenseacuello/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Setup export paths
chmod +x scripts/setup-paths.sh
./scripts/setup-paths.sh

# Install dependencies
cd mcp-server && pip install -r requirements.txt
cd ../dashboard && pip install -r requirements.txt
```

### Step 2: Install Fusion 360 Add-in

1. Open Fusion 360
2. Go to **Tools -> Add-Ins -> Scripts and Add-Ins**
3. Click **Add-Ins** tab -> **+** button
4. Select the `fusion360-addin/LegoMCP` folder
5. Check **Run on Startup** and click **Run**

### Step 3: Start Docker Services

```bash
# Start all services
docker-compose --profile full up -d

# Verify services
curl http://localhost:8766/health  # Slicer
curl http://localhost:5000/api/health  # Dashboard
```

### Step 4: Configure Claude Desktop

Add to your Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "lego-mcp": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/lego-mcp-fusion360/mcp-server",
      "env": {
        "FUSION_API_URL": "http://127.0.0.1:8767",
        "SLICER_API_URL": "http://localhost:8766"
      }
    }
  }
}
```

### Step 5: Start Using!

Restart Claude Desktop and try:
- "Create a 2x4 LEGO brick"
- "Export the brick as STL and slice for my Bambu P1S"
- "Create a work order for 100 2x4 bricks"
- "What's the current OEE for printer-1?"

---

## v5.0 System Modules

### Phase 7: Event-Driven Architecture
Real-time event streaming with Redis Streams, CQRS pattern, <10ms latency.

```python
from dashboard.services.events import EventBus, ManufacturingEvent

event_bus = EventBus()
event_bus.publish(ManufacturingEvent(
    event_type="work_order_started",
    work_center_id="printer-1",
    payload={"work_order_id": "WO-001", "part_id": "2x4-brick"}
))
```

### Phase 8: Customer Orders & ATP/CTP
Complete order management with Available-to-Promise and Capable-to-Promise.

```python
from dashboard.services.erp import OrderService, ATPService, CTPService

order_service = OrderService(atp_service=ATPService(), ctp_service=CTPService())
order = order_service.create_order(OrderCreateRequest(
    customer_id="CUST-001",
    customer_name="LEGO Factory",
    requested_delivery_date=date.today() + timedelta(days=7)
))
```

### Phase 9: Alternative Routings & Enhanced BOM
Multiple production routings with quality-aware BOM components.

```python
from dashboard.services.manufacturing import RoutingSelector

selector = RoutingSelector()
best_routing = selector.select_optimal_routing(
    part_id="2x4-brick",
    strategy=SelectionStrategy.LOWEST_COST
)
```

### Phase 10: FMEA Engine (Dynamic)
Dynamic RPN calculation with real-time risk factors.

```python
from dashboard.services.quality import FMEAService

fmea = FMEAService()
analysis = fmea.analyze_part("2x4-brick")
# Returns failure modes with dynamic RPN based on machine health, operator skill
```

### Phase 11: QFD House of Quality
Translate customer requirements to engineering characteristics.

```python
from dashboard.models.qfd import HouseOfQuality

hoq = HouseOfQuality(
    name="LEGO Brick QFD",
    customer_requirements=LEGO_REQUIREMENTS,
    engineering_characteristics=LEGO_CHARACTERISTICS
)
```

### Phase 12: Advanced Scheduling
OR-Tools CP-SAT, NSGA-II multi-objective, and RL dispatching.

```python
from dashboard.services.scheduling import CPScheduler, NSGA2Scheduler

# Constraint programming
scheduler = CPScheduler()
schedule = scheduler.schedule(jobs)

# Multi-objective optimization
nsga = NSGA2Scheduler()
pareto_front = nsga.schedule(jobs, objectives=["makespan", "tardiness", "energy"])
```

### Phase 13: Computer Vision Quality
Defect detection with multi-class classification.

```python
from dashboard.services.vision import DefectDetector

detector = DefectDetector()
result = detector.inspect(image, part_id="2x4-brick")
# Detects: LAYER_SHIFT, STRINGING, WARPING, UNDER_EXTRUSION, etc.
```

### Phase 14: Advanced SPC
EWMA, CUSUM, and Multivariate T-squared charts.

```python
from dashboard.services.quality import AdvancedSPCService

spc = AdvancedSPCService()
ewma = spc.add_ewma_chart("stud_diameter", target=4.8, ucl=4.82, lcl=4.78)
signal = ewma.add_point(4.79)  # Returns signal if out of control
```

### Phase 15: Digital Thread & Genealogy
Complete traceability from order to delivery.

```python
from dashboard.services.traceability import DigitalThreadService

thread = DigitalThreadService()
genealogy = thread.build_genealogy(serial_number="LEGO-2024-001")
root_cause = thread.trace_defect_root_cause(defect_id="DEF-001")
```

### Phase 17: AI Manufacturing Copilot
Claude-powered intelligent assistant for manufacturing decisions.

```python
from dashboard.services.ai import ManufacturingCopilot

copilot = ManufacturingCopilot()
explanation = await copilot.explain_anomaly(spc_signal)
recommendation = await copilot.recommend_schedule(constraints)
```

### Phase 18: Discrete Event Simulation
Factory simulation for capacity planning and what-if analysis.

```python
from dashboard.services.simulation import DESEngine

engine = DESEngine()
engine.add_machine("printer-1", cycle_time=30, mtbf=100, mttr=5)
results = engine.run(duration=480)  # 8-hour shift
```

### Phase 19: Sustainability & Carbon Tracking
Scope 1/2/3 carbon footprint with energy optimization.

```python
from dashboard.services.sustainability import CarbonTracker

tracker = CarbonTracker()
footprint = tracker.calculate_footprint(
    part_id="2x4-brick",
    process_energy_kwh=0.05,
    material_weight_kg=0.005
)
```

### Phase 20: Human-Machine Interface
Digital work instructions with AR overlay support.

```python
from dashboard.services.hmi import WorkInstructionService

service = WorkInstructionService()
instruction = service.create_instruction(
    operation_id="OP-001",
    steps=[InstructionStep(description="Load filament", duration=30)]
)
```

### Phase 21: Zero-Defect Quality
Predictive quality, process fingerprinting, virtual metrology.

```python
from dashboard.services.quality.zero_defect import VirtualMetrology

vm = VirtualMetrology()
prediction = vm.predict_dimensions(process_data={"nozzle_temp": 210, "speed": 60})
# Predicts stud_diameter, height, clutch_power without physical measurement
```

### Phase 22: Supply Chain Integration
Supplier portal with automated procurement.

```python
from dashboard.services.supply_chain import SupplierPortalService

portal = SupplierPortalService()
risk = portal.assess_supply_risk("SUPP-001")
scorecard = portal.generate_scorecard("SUPP-001")
```

### Phase 23: Real-Time Analytics
100+ KPIs with real-time dashboards.

```python
from dashboard.services.analytics import KPIEngine

kpi = KPIEngine()
oee = kpi.calculate_oee("printer-1", date.today())
snapshot = kpi.get_kpi_snapshot()  # All 100+ KPIs
```

### Phase 24: Compliance & Audit Trail
FDA 21 CFR Part 11 ready with electronic signatures.

```python
from dashboard.services.compliance import AuditTrailService

audit = AuditTrailService()
audit.log_action(AuditAction(
    user_id="operator-1",
    action_type=ActionType.UPDATE,
    entity_type="WorkOrder",
    entity_id="WO-001"
))
```

### Phase 25: Edge Computing & IIoT Gateway
Multi-protocol gateway with offline operation.

```python
from dashboard.services.edge import IIoTGateway, Protocol

gateway = IIoTGateway()
gateway.register_device("printer-1", "Bambu P1S", Protocol.MQTT, "192.168.1.100", 1883)
gateway.connect("printer-1")
```

---

## PhD-Level AI/ML Research Modules (v2.0)

The platform includes cutting-edge AI/ML research implementations for academic publication and world-class manufacturing excellence.

### Uncertainty Quantification
Bayesian approaches to prediction uncertainty for risk-aware decisions.

```python
from dashboard.services.ai.uncertainty import MCDropout, DeepEnsemble, ConformalPredictor

# Monte Carlo Dropout for Bayesian uncertainty
mc_dropout = MCDropout(model, n_samples=100)
mean, std, entropy = mc_dropout.predict_with_uncertainty(inputs)

# Conformal Prediction for valid coverage guarantees
cp = ConformalPredictor(model, alpha=0.1)
cp.calibrate(X_cal, y_cal)
prediction_sets = cp.predict(X_test)  # Valid 90% coverage
```

### Causal Inference
Pearl's do-calculus for manufacturing root cause analysis.

```python
from dashboard.services.ai.causal import CausalGraph, InterventionEngine

graph = CausalGraph()
graph.add_edge("temperature", "quality")
graph.add_edge("humidity", "quality")

engine = InterventionEngine(graph, data)
ate = engine.estimate_ate("temperature", "quality", method="backdoor")
```

### Continual Learning
Prevent catastrophic forgetting when learning new tasks.

```python
from dashboard.services.ai.continual import ContinualLearner, EWC

learner = ContinualLearner(model, method=EWC(lambda_importance=1000))
learner.learn_task(task1_data, task_id=1)
learner.learn_task(task2_data, task_id=2)  # Doesn't forget task 1
```

### Distributed Training
Scale training across GPUs with DDP, FSDP, and DeepSpeed.

```python
from dashboard.services.ai.distributed import DistributedTrainer, FSDPStrategy

trainer = DistributedTrainer(model, strategy=FSDPStrategy(sharding_strategy="FULL_SHARD"))
trainer.fit(train_loader, epochs=100)
```

### Explainability (XAI)
SHAP, LIME, and GradCAM for interpretable predictions.

```python
from dashboard.services.ai.explainability import SHAPExplainer

explainer = SHAPExplainer(model, background_data)
shap_values = explainer.explain(instance)
explainer.plot_waterfall(shap_values)
```

### Federated Learning
Privacy-preserving learning across factories.

```python
from dashboard.services.ai.federated import FederatedServer, FedAvg, SecureAggregation

server = FederatedServer(model, aggregation=FedAvg(), secure_agg=SecureAggregation())
```

### Research References

| Module | Key Papers |
|--------|-----------|
| Uncertainty | Gal & Ghahramani (2016), Lakshminarayanan et al. (2017) |
| Causality | Pearl (2009), Peters et al. (2017) |
| Continual | Kirkpatrick et al. (2017), Aljundi et al. (2018) |
| XAI | Lundberg & Lee (2017) - SHAP |
| Federated | McMahan et al. (2017) - FedAvg |

---

## v6.0 World-Class Research Platform

Version 6.0 transforms LEGO MCP into a **complete research platform** that bridges the gap from **Algorithm to Action**.

### Phase 1: Multi-Agent Orchestration

Coordinated decision-making across Quality, Scheduling, and Maintenance agents.

```python
from dashboard.services.agents import AgentOrchestrator, MessageBus
from dashboard.services.agents.consensus import ContractNetProtocol

# Initialize orchestrator with consensus protocol
orchestrator = AgentOrchestrator()
orchestrator.set_consensus_protocol(ContractNetProtocol())

# Coordinate multi-agent decision
context = ManufacturingContext(work_order_id="WO-001", equipment_id="printer-1")
decision = await orchestrator.coordinate_decision(context)
# Agents negotiate and reach consensus on optimal action
```

### Phase 2: Causal AI & Explainability

True causal inference with counterfactual analysis for root cause identification.

```python
from dashboard.services.causal import SCMBuilder, CounterfactualEngine
from dashboard.services.explainability import SHAPExplainer, ConceptActivationTester

# Build structural causal model from domain knowledge
scm = SCMBuilder()
scm.add_node("temperature", parents=[])
scm.add_node("quality", parents=["temperature", "speed"])
scm.add_mechanism("quality", lambda t, s: 0.9 - 0.01 * abs(t - 200) - 0.005 * s)

# Counterfactual analysis: "What if temperature was 210°C?"
engine = CounterfactualEngine(scm)
result = engine.query(
    observation={"temperature": 200, "speed": 60, "quality": 0.85},
    intervention={"temperature": 210},
    outcome="quality"
)
print(f"Counterfactual quality: {result.mean:.3f}")

# TCAV concept testing for explainability
tcav = ConceptActivationTester(model)
tcav.add_concept("warping", positive_examples=warped_images)
score = tcav.compute_tcav_score("warping", "defect", layer="conv4")
```

### Phase 3: Generative Design

AI-driven part geometry optimization for 3D printing.

```python
from dashboard.services.generative import TopologyOptimizer, ClutchOptimizer
from dashboard.services.generative.lego import FDMCompensator

# Topology optimization for optimal LEGO geometry
optimizer = TopologyOptimizer()
result = optimizer.optimize(
    design_space=bounding_box,
    loads=[Load(position=(0,0,5), force=(0,0,-10))],
    constraints=PrintabilityConstraints(max_overhang=45)
)

# LEGO-specific clutch power optimization
clutch = ClutchOptimizer()
optimal_stud = clutch.optimize_stud_geometry(
    target_clutch_force=2.0,  # Newtons
    compatibility_tolerance=0.02  # mm
)

# FDM shrinkage compensation
compensator = FDMCompensator(material="PLA", printer="BambuP1S")
compensated_model = compensator.apply_compensation(model)
```

### Phase 4: Closed-Loop Learning

Automatic production feedback to model improvement.

```python
from dashboard.services.closed_loop import FeedbackCollector, ModelUpdater, DriftDetector
from dashboard.services.digital_twin import RealtimeTwinSync, MultiTwinSyncManager

# Real-time digital twin synchronization
sync = RealtimeTwinSync(twin_id="printer-1", sync_interval_ms=100)
await sync.start()

# Collect production outcomes for model training
collector = FeedbackCollector()
collector.register_stream("quality", QualityStream())
await collector.collect_and_update(production_event)

# Detect model drift and trigger retraining
drift = DriftDetector()
if drift.detect(current_features, baseline_features):
    updater = ModelUpdater()
    await updater.retrain(model_id, new_data)
```

### Phase 5: Algorithm-to-Action Bridge

Direct equipment control from AI decisions with safety interlocks.

```python
from dashboard.services.equipment import (
    PrinterController, CNCController, ConveyorController,
    create_cnc_controller, create_conveyor_controller
)
from dashboard.services.action import ActionPipeline, SafetyValidator

# CNC machine control
cnc = create_cnc_controller(protocol_type="grbl", port="/dev/ttyUSB0")
await cnc.connect()

# Execute AI decision on equipment
decision = AIDecision(type="speed_adjustment", spindle_speed=12000)
result = await cnc.execute_ai_decision(decision)

# Action pipeline with safety validation
pipeline = ActionPipeline(safety=SafetyValidator())
await pipeline.execute(
    decision=ai_recommendation,
    equipment=cnc,
    require_approval=decision.risk_level > 0.7
)

# Conveyor system with product tracking
conveyor = create_conveyor_controller(protocol_type="modbus", host="192.168.1.100")
product = conveyor.track_product("2x4-brick", entry_zone_id="zone-1")
```

### Phase 6: Research Platform Infrastructure

Complete experiment tracking with statistical rigor.

```python
from dashboard.services.research import ExperimentTracker, ModelRegistry
from dashboard.services.research.statistics import (
    ABTestAnalyzer, MultiArmedBandit, CausalInferenceEngine,
    HypothesisTester, BayesianTester
)

# Experiment tracking
tracker = ExperimentTracker()
with tracker.start_run(name="quality_model_v2") as run:
    run.log_params({"learning_rate": 0.001, "epochs": 50})
    run.log_metrics({"accuracy": 0.968, "f1": 0.954})
    run.log_artifact("model.pt", artifact_type="model")

# A/B testing with sequential analysis
ab = ABTestAnalyzer()
test = ab.create_test(
    name="Temperature Optimization",
    metric_type=MetricType.CONTINUOUS,
    minimum_detectable_effect=0.05
)
result = ab.analyze_sequential(test.test_id, control_data, treatment_data, analysis_number=3)

# Multi-armed bandit for adaptive experimentation
bandit = MultiArmedBandit(config)
bandit.add_arm("200C", "Standard temperature")
bandit.add_arm("210C", "Higher temperature")
bandit.add_arm("205C", "Balanced temperature")

for _ in range(1000):
    arm = bandit.select_arm()
    reward = run_experiment(arm)
    bandit.update(arm.arm_id, reward)

# Causal effect estimation
causal = CausalInferenceEngine()
ate = causal.estimate_ate(
    treated_outcomes, control_outcomes,
    method=EstimationMethod.DOUBLY_ROBUST,
    covariates=X, treatment_indicators=T
)
```

### ISO 23247 Digital Twin Framework

Full compliance with the ISO 23247 Digital Twin for Manufacturing standard.

```python
from dashboard.services.digital_twin.ome_registry import OMERegistry
from dashboard.services.digital_twin.twin_engine import TwinEngine
from dashboard.services.unity.bridge import UnityBridge

# Register Observable Manufacturing Element (OME)
registry = OMERegistry()
ome_id = registry.register_ome({
    "ome_type": "equipment",
    "name": "Prusa MK3S+ #1",
    "parent_id": "CELL-001",
    "static_attributes": {"model": "Prusa MK3S+", "serial": "PRS-001"}
})

# Create digital twin for real-time monitoring
engine = TwinEngine()
twin_id = engine.create_twin(ome_id, {
    "twin_type": "monitoring",
    "behavior_model": "pinn",  # Physics-Informed Neural Network
    "sync_interval_ms": 100
})

# Connect to Unity 3D visualization
bridge = UnityBridge()
await bridge.start_server(port=8770)
```

### Unity 3D Digital Twin Visualization

Real-time 3D visualization with WebGL, VR, and AR support.

```python
from dashboard.services.unity.scene_data import SceneDataService
from dashboard.websocket import emit_unity_scene_update, emit_unity_highlight

# Get full 3D scene for Unity initialization
scene_service = SceneDataService()
scene = scene_service.get_full_scene("FactoryFloor")

# Send real-time updates to Unity clients
emit_unity_scene_update(
    scene_name="FactoryFloor",
    equipment_updates=[
        {"equipment_id": "EQ-001", "state": "printing", "progress": 45.5}
    ],
    delta_only=True
)

# Highlight equipment on alert
emit_unity_highlight("EQ-001", highlight_type="alert", color="#FF0000", duration_ms=2000)
```

### Robotic Arms Control (ISO 10218 / ISO/TS 15066)

Safety-compliant robotic arm control with multi-arm coordination.

```python
from dashboard.services.robotics.arm_controller import ArmController
from dashboard.websocket import emit_robot_status, emit_robot_safety_violation

# Initialize arm controller
controller = ArmController()

# Queue pick-and-place task
task_id = controller.queue_task({
    "task_type": "pick_and_place",
    "arm_id": "ARM-001",
    "source_position": {"x": 100, "y": 0, "z": 200},
    "target_position": {"x": 300, "y": 0, "z": 200}
})

# Multi-arm synchronized motion
sync_id = controller.create_synchronized_motion({
    "sync_type": "barrier",
    "arms": ["ARM-001", "ARM-002"],
    "waypoints": [...]
})

# ISO 10218 safety zone monitoring
controller.define_safety_zone({
    "zone_id": "ZONE-RESTRICTED-01",
    "zone_type": "restricted",
    "max_speed_mm_s": 250
})
```

### VR Training System

Immersive VR training with performance tracking and certification.

```python
from dashboard.services.hmi.vr_training import VRTrainingService
from dashboard.websocket import emit_vr_session_started, emit_vr_step_progress

# Create training scenario
service = VRTrainingService()
scenario_id = service.create_scenario({
    "name": "Equipment Safety Fundamentals",
    "category": "safety",
    "difficulty": "beginner",
    "passing_score": 80,
    "steps": [
        {"step_id": 1, "name": "PPE Check", "duration_min": 3},
        {"step_id": 2, "name": "Emergency Procedures", "duration_min": 5}
    ]
})

# Start training session
session = service.start_session({
    "scenario_id": scenario_id,
    "trainee_id": "TRN-001"
})

# Track progress
emit_vr_step_progress(
    session_id=session["session_id"],
    step_number=1,
    total_steps=10,
    step_name="PPE Check",
    status="completed",
    score=95
)
```

### Supply Chain Digital Twin

End-to-end supply chain visualization with disruption simulation.

```python
from dashboard.services.digital_twin.supply_chain_twin import SupplyChainTwin
from dashboard.websocket import emit_supply_chain_disruption, emit_supply_chain_flow_update

# Build supply chain network
twin = SupplyChainTwin()
twin.add_node({"node_id": "SUP-001", "type": "supplier", "name": "ABS Pellet Supplier"})
twin.add_node({"node_id": "WH-001", "type": "warehouse", "name": "Raw Materials Warehouse"})
twin.add_node({"node_id": "FAC-001", "type": "factory", "name": "LEGO Production"})
twin.add_edge({"source": "SUP-001", "target": "WH-001", "lead_time_days": 3})
twin.add_edge({"source": "WH-001", "target": "FAC-001", "lead_time_days": 1})

# Simulate disruption
result = twin.simulate_disruption({
    "node_id": "SUP-001",
    "disruption_type": "supplier_shutdown",
    "duration_days": 7
})
# Returns: affected_nodes, production_impact_percent, mitigation_options

# Real-time flow updates
emit_supply_chain_flow_update(
    edge_id="E-001",
    source_node="SUP-001",
    target_node="WH-001",
    material_type="ABS_RED",
    flow_rate=5000,
    eta_hours=24
)
```

### v6.0 API Endpoints

| API | Base URL | Description |
|-----|----------|-------------|
| Orchestration | `/api/v6/orchestration` | Multi-agent coordination |
| Causal | `/api/v6/causal` | SCM, counterfactuals, root cause |
| Generative | `/api/v6/generative` | Topology optimization, design |
| Closed-Loop | `/api/v6/closed-loop` | Feedback, drift, retraining |
| Actions | `/api/v6/actions` | Equipment control, execution |
| Research | `/api/v6/research` | Experiments, models, statistics |
| Health | `/api/v6/health` | Platform status and capabilities |
| **Unity** | `/api/unity` | 3D scene, equipment, highlights |
| **Robotics** | `/api/robotics` | Arm control, tasks, safety zones |
| **VR Training** | `/api/hmi/vr/training` | Scenarios, sessions, leaderboards |
| **Supply Chain Twin** | `/api/supply-chain/twin` | Network, disruption, flow |
| **Quality Heatmap** | `/api/quality/heatmap` | 3D defect mapping, clusters |
| **ISO 23247 OME** | `/api/ome` | Observable Manufacturing Elements |

### v6.0 Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| **Financials Overview** | `/api/erp/financials/dashboard/page` | AR/AP/GL summary, cash flow |
| **Accounts Receivable** | `/api/erp/financials/ar/page` | Customer invoices, aging, DSO |
| **Accounts Payable** | `/api/erp/financials/ap/page` | Vendor bills, payments, DPO |
| **General Ledger** | `/api/erp/financials/gl/page` | Chart of accounts, journal entries |
| **Vendor Management** | `/api/erp/vendors/page` | Supplier lifecycle, scorecards |
| **WIP & Orders** | `/api/mes/work-orders/wip/page` | Work-in-progress, order tracking |
| **Material Master** | `/api/mrp/materials/page` | Filament tracking, consumption |
| QFD Cascade | `/api/quality/qfd/cascade/page` | 4-phase QFD deployment |
| Design FMEA | `/api/quality/fmea/dfmea/page` | AI-enhanced design FMEA |
| Process FMEA | `/api/quality/fmea/pfmea/page` | Process failure analysis |
| VOC Capture | `/api/quality/qfd/voc/page` | Voice of Customer with Kano |
| Model Comparison | `/api/ai/research/models/page` | Performance comparison |
| Publication Export | `/api/ai/research/publication/page` | LaTeX-ready figures |
| Agent Orchestration | `/api/ai/orchestration/page` | Multi-agent visualization |
| Closed-Loop | `/api/ai/closed-loop/page` | Feedback monitoring |
| Action Approval | `/api/ai/actions/page` | Human-in-the-loop |

---

## Interactive Dashboards (v6.0 Enhanced)

The web dashboard (`http://localhost:5000`) features **16+ world-class interactive dashboards** with real-time data, modal drill-downs, toast notifications, and live WebSocket updates.

### Dashboard Features

| Dashboard | Key Features |
|-----------|-------------|
| **MES Dashboard** | Real-time work order tracking, shop floor display, Andon alerts, WebSocket updates every 30s |
| **Work Centers** | Equipment status grid, utilization gauges, drill-down modals with sensor data and maintenance history |
| **OEE Analytics** | Interactive OEE waterfall, time-series trends, pareto charts, export to PDF/CSV |
| **Quality Control** | Vision AI defect gallery, SPC control charts (EWMA/CUSUM/T²), FMEA risk matrix |
| **Scheduling** | Drag-drop Gantt chart, algorithm comparison (CP-SAT vs NSGA-II vs RL), bottleneck analysis |
| **Sustainability** | LCA lifecycle impact, carbon scope breakdown, circular economy metrics, ESG scoring |
| **Supply Chain** | Interactive supplier map, risk heat matrix, inventory optimization, S&OP planning |
| **Compliance** | ISO 9001/14001/45001 clause tracking, CAPA workflow stepper, 21 CFR Part 11 e-signatures |
| **AI Copilot** | GPT-4/Claude model selection, XAI SHAP explanations, query history analytics, knowledge base |

### Enterprise ERP Dashboards (v6.0 NEW)

| Dashboard | Key Features |
|-----------|-------------|
| **Accounts Receivable** | Customer invoices, payment tracking, aging analysis (Current/31-60/61-90/91-120/120+), DSO calculation, cash forecast |
| **Accounts Payable** | Vendor bill management, payment processing, DPO tracking, payment scheduling, 1099 compliance tracking |
| **General Ledger** | Chart of accounts hierarchy, journal entry creation, trial balance, income statement, balance sheet |
| **Vendor Management** | Full supplier lifecycle (Prospect→Approved→Preferred→Strategic), performance scorecarding, certification tracking |
| **WIP & Orders** | Work-in-progress monitoring, Kanban board, customer order fulfillment, ATP/CTP integration, material batch tracking |
| **Material Master** | Filament spool tracking, consumption analytics, reorder points, lot/batch traceability |
| **BOM Dashboard** | Multi-level bill of materials, phantom assemblies, revision control, where-used analysis |

### Common Dashboard Patterns

All dashboards implement consistent UX patterns:

- **Modal System**: Click any metric card, table row, or chart element for detailed drill-down
- **Toast Notifications**: Real-time feedback for actions (success, warning, error, info)
- **Tab Navigation**: Organize complex information into manageable sections
- **Real-time Updates**: WebSocket + polling for live manufacturing data
- **Export Functionality**: PDF reports, CSV data, JSON API access
- **Keyboard Shortcuts**: Escape to close modals, Enter to submit forms
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Dashboard Access

```
http://localhost:5000/dashboard/mes          # MES Dashboard
http://localhost:5000/dashboard/work-centers # Work Centers
http://localhost:5000/dashboard/oee          # OEE Analytics
http://localhost:5000/dashboard/quality      # Quality Control
http://localhost:5000/dashboard/scheduling   # Scheduling
http://localhost:5000/dashboard/sustainability # Sustainability
http://localhost:5000/dashboard/supply-chain # Supply Chain
http://localhost:5000/dashboard/compliance   # Compliance & Audit
http://localhost:5000/dashboard/copilot      # AI Copilot
```

---

## LEGO Dimensions (v5.0 Corrected)

| Dimension | Value (mm) | Tolerance |
|-----------|------------|-----------|
| Stud pitch | 8.0 | +/-0.02 |
| Stud diameter | 4.8 | +/-0.02 |
| Stud height | 1.8 | +/-0.1 |
| Plate height | 3.2 | +/-0.02 |
| Brick height | 9.6 | +/-0.05 |
| **Wall thickness** | **1.6** | +/-0.05 |
| **Inter-brick clearance** | **0.1/side** | - |
| **Pin hole diameter** | **4.9** | +/-0.02 |
| Technic hole diameter | 4.85 | +/-0.02 |

---

## Docker Commands

```bash
# Start all services
docker-compose --profile full up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Health checks
curl http://localhost:8766/health  # Slicer
curl http://localhost:5000/api/health  # Dashboard
curl http://127.0.0.1:8767/health  # Fusion 360
```

---

## Testing

See [TESTING.md](TESTING.md) for comprehensive test commands for all 25 phases.

```bash
# Quick health check
./scripts/test-health.sh

# Run unit tests
cd mcp-server && pytest tests/
cd dashboard && pytest tests/

# Integration tests (requires Fusion 360)
pytest tests/test_integration.py -v
```

---

## Project Structure (v5.0)

```
lego-mcp-fusion360/
+-- mcp-server/              # MCP server for Claude Desktop
+-- fusion360-addin/         # Fusion 360 add-in
+-- dashboard/               # Flask web dashboard + Industry 4.0
|   +-- models/              # SQLAlchemy models
|   |   +-- manufacturing.py
|   |   +-- quality.py
|   |   +-- analytics.py
|   |   +-- customer_order.py  # Phase 8
|   |   +-- routing_extended.py  # Phase 9
|   |   +-- bom_extended.py  # Phase 9
|   |   +-- fmea.py  # Phase 10
|   |   +-- qfd.py  # Phase 11
|   +-- services/
|   |   +-- events/  # Phase 7: Event-Driven
|   |   +-- manufacturing/  # MES + Scheduling
|   |   |   +-- routing_selector.py  # Phase 9
|   |   +-- erp/  # ERP + Orders
|   |   |   +-- order_service.py  # Phase 8
|   |   |   +-- atp_service.py  # Phase 8
|   |   |   +-- ctp_service.py  # Phase 8
|   |   +-- quality/  # Quality Management
|   |   |   +-- fmea_service.py  # Phase 10
|   |   |   +-- advanced_spc.py  # Phase 14
|   |   |   +-- zero_defect/  # Phase 21
|   |   +-- scheduling/  # Phase 12: Advanced Scheduling
|   |   |   +-- cp_scheduler.py
|   |   |   +-- nsga2_scheduler.py
|   |   |   +-- rl_dispatcher.py
|   |   +-- vision/  # Phase 13: CV Quality
|   |   |   +-- defect_detector.py
|   |   +-- traceability/  # Phase 15: Digital Thread
|   |   |   +-- digital_thread.py
|   |   +-- ai/  # Phase 17: AI Copilot
|   |   |   +-- manufacturing_copilot.py
|   |   |   +-- agents/
|   |   +-- simulation/  # Phase 18: DES
|   |   |   +-- des_engine.py
|   |   +-- sustainability/  # Phase 19
|   |   |   +-- carbon_tracker.py
|   |   +-- hmi/  # Phase 20
|   |   |   +-- work_instructions.py
|   |   +-- supply_chain/  # Phase 22
|   |   |   +-- supplier_portal.py
|   |   +-- analytics/  # Phase 23
|   |   |   +-- kpi_engine.py
|   |   +-- compliance/  # Phase 24
|   |   |   +-- audit_trail.py
|   |   +-- edge/  # Phase 25
|   |       +-- iiot_gateway.py
+-- slicer-service/          # Docker slicer container
+-- shared/                  # Shared specifications
+-- scripts/                 # Utility scripts
+-- docker-compose.yml
+-- TESTING.md               # Comprehensive test commands
+-- CHANGELOG.md
```

---

## Technology Stack (v5.0)

| Category | Technology |
|----------|------------|
| **Database** | PostgreSQL 16 + TimescaleDB + pgvector |
| **Event Streaming** | Redis Streams |
| **Optimization** | OR-Tools CP-SAT, DEAP (NSGA-II) |
| **ML/RL** | PyTorch, Stable-Baselines3, scikit-learn |
| **LLM Integration** | Claude API, LangChain, vector embeddings |
| **Computer Vision** | YOLO11, torchvision |
| **Visualization** | Grafana, Apache ECharts |
| **Real-time** | WebSockets, SSE, MQTT |
| **Standards** | OPC-UA, MTConnect, ISA-95 XML |
| **Containers** | Docker, Kubernetes-ready |

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Credits

- LEGO is a trademark of the LEGO Group (not affiliated)
- Built with [Claude](https://anthropic.com) and MCP
- Powered by [Autodesk Fusion 360](https://www.autodesk.com/products/fusion-360)

---

<p align="center">
  <b>World-Class Cyber-Physical Production System</b><br>
  Made with Claude AI + Fusion 360
</p>
# flask_cnc_app
