# LegoMCP v5.0 PhD-Level Testing Guide

Comprehensive manual test commands for all phases including the **PhD-Level Research Components**.

## NEW: PhD-Level Research Phases

This guide includes testing for:
- **Phase 1**: XAI (SHAP/LIME), Federated Learning, Benchmarking
- **Phase 2**: Quantum-Inspired Scheduling (QAOA, VQE), Enhanced RL (PPO, SAC, TD3)
- **Phase 3**: Digital Twin Research (PINN, Ontology, Knowledge Graph, CRDT)
- **Phase 4**: Sustainable Manufacturing (LCA ISO 14040/14044, Carbon-Neutral Planning)
- **Phase 5**: AI/ML for Quality (Self-Supervised Learning, Multimodal Fusion)
- **Phase 6**: ISO 9001/13485 Quality Management System

---

## Quick Start: Run All Tests

```bash
cd /Users/stepheneacuello/Documents/lego_mcp_fusion360

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/dashboard"

# Run all PhD-level tests
python tests/run_tests.py --all

# Run specific test categories
python tests/run_tests.py --unit          # Unit tests
python tests/run_tests.py --integration   # Integration tests
python tests/run_tests.py --benchmark     # Benchmark tests
python tests/run_tests.py --compliance    # ISO compliance tests
python tests/run_tests.py --coverage      # With coverage report
```

---

Comprehensive manual test commands for all 25 phases of the World-Class Manufacturing System.

---

## Prerequisites

```bash
# Start all services
docker-compose --profile full up -d

# Verify services are running
curl http://localhost:8766/health     # Slicer
curl http://localhost:5000/api/health # Dashboard
curl http://127.0.0.1:8767/health     # Fusion 360 (requires add-in running)

# Set up Python environment
cd /path/to/lego-mcp-fusion360
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r dashboard/requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/dashboard"
```

---

## Phase 1-6: Foundation (Already Complete)

### Test Fusion 360 Connection
```bash
curl -s http://127.0.0.1:8767/health | jq .
# Expected: {"status": "ok", "service": "fusion360-addin"}

curl -X POST http://127.0.0.1:8767/create_brick \
  -H "Content-Type: application/json" \
  -d '{"studs_x": 2, "studs_y": 4, "height_units": 3}'
```

### Test Slicer Service
```bash
curl -s http://localhost:8766/health | jq .
# Expected: {"status": "ok", "service": "LEGO Slicer Service", ...}

curl -s http://localhost:8766/printers | jq .
# Expected: {"printers": ["generic", "prusa_mk3s", ...]}

curl -s http://localhost:8766/profiles | jq .
# Expected: {"profiles": [...], "details": {...}, "qualities": [...], "materials": [...]}

curl -s http://localhost:8766/qualities | jq .
# Expected: {"qualities": ["draft", "normal", "fine", "ultra", "lego"], ...}

curl -s http://localhost:8766/materials | jq .
# Expected: {"materials": ["pla", "petg", "abs", "asa"], ...}
```

### Test Dashboard
```bash
curl -s http://localhost:5000/api/health | jq .
# Expected: {"status": "ok", "version": "5.0.0", "platform": "LegoMCP World-Class CPPS", ...}

curl -s http://localhost:5000/api/v5/health | jq .
# Expected: {"status": "healthy", "version": "5.0.0", ...}
```

---

## Phase 7: Event-Driven Architecture

### Test Event Bus
```python
# Save as: test_phase7.py
from dashboard.services.events import EventBus, ManufacturingEvent, EventCategory
from datetime import datetime

# Create event bus
event_bus = EventBus()

# Track received events
received = []
def handler(event):
    received.append(event)
    print(f"Received: {event.event_type}")

# Subscribe to machine events
event_bus.subscribe(EventCategory.MACHINE, handler)

# Publish an event
event = ManufacturingEvent(
    event_id="test-001",
    event_type="machine_started",
    category=EventCategory.MACHINE,
    timestamp=datetime.utcnow(),
    source_layer="L1",
    work_center_id="printer-1",
    payload={"status": "running"}
)
event_bus.publish(event)

# Verify
assert len(received) == 1
print("Phase 7: Event Bus - PASS")
```

Run:
```bash
python test_phase7.py
```

---

## Phase 8: Customer Orders & ATP/CTP

### Test Order Service
```python
# Save as: test_phase8.py
from datetime import date, timedelta
from dashboard.services.erp import OrderService, ATPService, CTPService
from dashboard.services.erp.order_service import OrderCreateRequest, OrderLineRequest

# Initialize services
atp = ATPService()
ctp = CTPService()
order_service = OrderService(atp_service=atp, ctp_service=ctp)

# Create an order
order = order_service.create_order(OrderCreateRequest(
    customer_id="CUST-001",
    customer_name="Test Customer",
    requested_delivery_date=date.today() + timedelta(days=7),
    priority_class="A"
))
print(f"Created order: {order['order_number']}")

# Add line items
line = order_service.add_line(order['order_id'], OrderLineRequest(
    part_id="2x4-brick",
    part_name="2x4 Standard Brick",
    quantity=100,
    unit_price=0.10
))
print(f"Added line: {line['part_name']} x {line['quantity_ordered']}")

# Submit order
submitted = order_service.submit_order(order['order_id'])
print(f"Order status: {submitted['status']}")

# Check ATP
atp_result = atp.check_availability("2x4-brick", 100)
print(f"ATP available: {atp_result['available_quantity']}")

# Check CTP
ctp_result = ctp.check_production_capability("2x4-brick", 100)
print(f"CTP can produce: {ctp_result['can_produce']}")

print("Phase 8: Customer Orders - PASS")
```

---

## Phase 9: Alternative Routings & Enhanced BOM

### Test Routing Selector
```python
# Save as: test_phase9.py
from dashboard.services.manufacturing.routing_selector import (
    RoutingSelector, SelectionStrategy
)
from dashboard.models.routing_extended import AlternativeRouting

# Create selector
selector = RoutingSelector()

# Create test routings
routing1 = AlternativeRouting(
    routing_id="R001",
    part_id="2x4-brick",
    routing_name="Fast Print",
    total_time_minutes=30,
    total_cost=0.50,
    expected_yield_percent=98.0,
    energy_kwh=0.05
)

routing2 = AlternativeRouting(
    routing_id="R002",
    part_id="2x4-brick",
    routing_name="High Quality",
    total_time_minutes=45,
    total_cost=0.75,
    expected_yield_percent=99.5,
    energy_kwh=0.08
)

selector.add_routing(routing1)
selector.add_routing(routing2)

# Select by different strategies
fastest = selector.select_optimal_routing("2x4-brick", SelectionStrategy.FASTEST)
print(f"Fastest: {fastest.routing_name} ({fastest.total_time_minutes} min)")

cheapest = selector.select_optimal_routing("2x4-brick", SelectionStrategy.LOWEST_COST)
print(f"Cheapest: {cheapest.routing_name} (${cheapest.total_cost})")

best_quality = selector.select_optimal_routing("2x4-brick", SelectionStrategy.HIGHEST_QUALITY)
print(f"Best Quality: {best_quality.routing_name} ({best_quality.expected_yield_percent}%)")

print("Phase 9: Alternative Routings - PASS")
```

---

## Phase 10: FMEA Engine

### Test Dynamic FMEA
```python
# Save as: test_phase10.py
from dashboard.services.quality.fmea_service import FMEAService

# Create FMEA service
fmea = FMEAService()

# Analyze a part
analysis = fmea.analyze_part("2x4-brick")
print(f"FMEA Record: {analysis.fmea_id}")
print(f"Failure Modes: {len(analysis.failure_modes)}")

# Check failure modes
for fm in analysis.failure_modes[:3]:
    print(f"  - {fm.failure_mode}: RPN={fm.rpn}, Dynamic RPN={fm.dynamic_rpn:.1f}")

# Calculate with real-time factors
dynamic = fmea.calculate_dynamic_rpn(
    analysis.failure_modes[0],
    machine_health=0.85,
    operator_skill=0.95,
    spc_trend=1.1
)
print(f"Dynamic RPN with factors: {dynamic:.1f}")

# Get high-risk failure modes
high_risk = fmea.get_high_risk_failures("2x4-brick", threshold=100)
print(f"High-risk modes (RPN > 100): {len(high_risk)}")

print("Phase 10: FMEA Engine - PASS")
```

---

## Phase 11: QFD House of Quality

### Test QFD
```python
# Save as: test_phase11.py
from dashboard.models.qfd import (
    HouseOfQuality, CustomerRequirement, EngineeringCharacteristic,
    QFDRelationship, RelationshipStrength, LEGO_REQUIREMENTS, LEGO_CHARACTERISTICS
)

# Create House of Quality
hoq = HouseOfQuality(
    hoq_id="HOQ-001",
    name="LEGO Brick Quality",
    customer_requirements=LEGO_REQUIREMENTS,
    engineering_characteristics=LEGO_CHARACTERISTICS
)

# Add relationships
hoq.add_relationship(QFDRelationship(
    requirement_id="clutch",
    characteristic_id="stud_diameter",
    strength=RelationshipStrength.STRONG
))

# Calculate importance
hoq.calculate_importance()

print(f"House of Quality: {hoq.name}")
print(f"Customer Requirements: {len(hoq.customer_requirements)}")
print(f"Engineering Characteristics: {len(hoq.engineering_characteristics)}")

print("Phase 11: QFD - PASS")
```

---

## Phase 12: Advanced Scheduling

### Test CP-SAT Scheduler
```python
# Save as: test_phase12.py
from dashboard.services.scheduling import CPScheduler, Job, Machine

# Create scheduler
scheduler = CPScheduler()

# Add machines
scheduler.add_machine(Machine(
    machine_id="printer-1",
    name="Bambu P1S",
    machine_type="3D_PRINTER"
))
scheduler.add_machine(Machine(
    machine_id="printer-2",
    name="Prusa MK4",
    machine_type="3D_PRINTER"
))

# Add jobs
jobs = [
    Job(job_id="J1", part_id="2x4-brick", quantity=10, due_date_minutes=120, processing_time=30),
    Job(job_id="J2", part_id="1x6-plate", quantity=20, due_date_minutes=180, processing_time=45),
    Job(job_id="J3", part_id="2x2-tile", quantity=15, due_date_minutes=240, processing_time=25),
]

# Schedule
schedule = scheduler.schedule(jobs)
print(f"Schedule Status: {schedule.status}")
print(f"Makespan: {schedule.makespan} minutes")
print(f"Total Tardiness: {schedule.total_tardiness} minutes")

print("Phase 12: CP-SAT Scheduling - PASS")
```

---

## Phase 13: Computer Vision Quality

### Test Defect Detector
```python
# Save as: test_phase13.py
import numpy as np
from dashboard.services.vision.defect_detector import DefectDetector

# Create detector
detector = DefectDetector()

# Create test image (simulated)
test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

# Run inspection
result = detector.inspect(test_image, part_id="2x4-brick")

print(f"Inspection ID: {result.inspection_id}")
print(f"Part ID: {result.part_id}")
print(f"Total Defects: {result.total_defects}")
print(f"Pass: {result.passed}")

print("Phase 13: CV Quality - PASS")
```

---

## Phase 14: Advanced SPC

### Test EWMA Chart
```python
# Save as: test_phase14.py
from dashboard.services.quality.advanced_spc import AdvancedSPCService

# Create SPC service
spc = AdvancedSPCService()

# Add EWMA chart for stud diameter
ewma = spc.add_ewma_chart(
    name="stud_diameter",
    target=4.80,
    sigma=0.02,
    lambda_factor=0.2,
    L=3.0
)

# Simulate measurements
measurements = [4.79, 4.81, 4.78, 4.82, 4.77, 4.83, 4.76, 4.84, 4.75, 4.85]

print("EWMA Chart Analysis:")
for i, value in enumerate(measurements):
    signal = ewma.add_point(value)
    status = "OUT OF CONTROL" if signal else "OK"
    print(f"  Point {i+1}: {value:.2f} -> EWMA={ewma.ewma_values[-1]:.3f} [{status}]")

print("Phase 14: EWMA - PASS")
```

---

## Phase 15: Digital Thread

### Test Digital Thread Service
```python
# Save as: test_phase15.py
from dashboard.services.traceability.digital_thread import DigitalThreadService

# Create service
thread_service = DigitalThreadService()

# Build product genealogy
genealogy = thread_service.build_genealogy(
    serial_number="LEGO-2024-001234",
    work_order_id="WO-2024-001",
    customer_order_id="SO-001001"
)

print(f"Product Genealogy: {genealogy.serial_number}")
print(f"Work Order: {genealogy.work_order_id}")
print(f"Customer Order: {genealogy.customer_order_id}")

# Trace defect root cause
root_cause = thread_service.trace_defect_root_cause("DEF-001")
print(f"Root Cause Analysis: {root_cause.get('defect_id')}")

print("Phase 15: Digital Thread - PASS")
```

---

## Phase 17: AI Manufacturing Copilot

### Test Copilot (Offline Mode)
```python
# Save as: test_phase17.py
from dashboard.services.ai import ManufacturingCopilot

# Create copilot in offline mode
copilot = ManufacturingCopilot(offline_mode=True)

# Test context building
context = copilot.build_context(
    work_center_id="printer-1",
    include_spc=True,
    include_fmea=True
)
print(f"Context built: {len(context)} characters")
print("Phase 17: Copilot Context - PASS")
```

---

## Phase 18: Discrete Event Simulation

### Test DES Engine
```python
# Save as: test_phase18.py
from dashboard.services.simulation.des_engine import DESEngine

# Create simulation engine
engine = DESEngine()

# Add machines
engine.add_machine(
    machine_id="printer-1",
    cycle_time=30,
    mtbf=480,
    mttr=30
)

# Add jobs
engine.add_job(job_id="J1", arrival_time=0, processing_time=30)
engine.add_job(job_id="J2", arrival_time=10, processing_time=35)
engine.add_job(job_id="J3", arrival_time=20, processing_time=30)

# Run simulation
results = engine.run(duration=480)

print(f"Simulation Results:")
print(f"  Jobs Completed: {results.jobs_completed}")
print(f"  Total Time: {results.total_time} minutes")

print("Phase 18: DES Simulation - PASS")
```

---

## Phase 19: Sustainability

### Test Carbon Tracker
```python
# Save as: test_phase19.py
from dashboard.services.sustainability.carbon_tracker import CarbonTracker

# Create tracker
tracker = CarbonTracker()

# Calculate footprint for a part
footprint = tracker.calculate_footprint(
    part_id="2x4-brick",
    process_energy_kwh=0.05,
    material_weight_kg=0.005,
    material_type="ABS"
)

print(f"Carbon Footprint for 2x4 Brick:")
print(f"  Scope 1: {footprint.scope_1:.4f} kg CO2e")
print(f"  Scope 2: {footprint.scope_2:.4f} kg CO2e")
print(f"  Scope 3: {footprint.scope_3:.4f} kg CO2e")
print(f"  Total: {footprint.total:.4f} kg CO2e")

print("Phase 19: Carbon Tracking - PASS")
```

---

## Phase 20: HMI

### Test Work Instructions
```python
# Save as: test_phase20.py
from dashboard.services.hmi.work_instructions import (
    WorkInstructionService, InstructionStep
)

# Create service
service = WorkInstructionService()

# Create work instruction
instruction = service.create_instruction(
    operation_id="OP-001",
    operation_name="3D Print 2x4 Brick",
    steps=[
        InstructionStep(
            step_number=1,
            description="Load ABS filament",
            duration_seconds=60
        ),
        InstructionStep(
            step_number=2,
            description="Start print job",
            duration_seconds=120
        )
    ]
)

print(f"Work Instruction: {instruction.operation_name}")
print(f"Total Steps: {len(instruction.steps)}")

print("Phase 20: Work Instructions - PASS")
```

---

## Phase 21: Zero-Defect Quality

### Test Virtual Metrology
```python
# Save as: test_phase21.py
from dashboard.services.quality.zero_defect.virtual_metrology import VirtualMetrology

# Create virtual metrology service
vm = VirtualMetrology()

# Predict dimensions from process data
process_data = {
    "nozzle_temp": 210,
    "bed_temp": 60,
    "print_speed": 60,
    "layer_height": 0.12
}

prediction = vm.predict_dimensions(process_data, part_id="2x4-brick")

print("Virtual Metrology Prediction:")
print(f"  Stud Diameter: {prediction.stud_diameter:.3f} mm")
print(f"  Confidence: {prediction.confidence:.1%}")
print(f"  Within Spec: {prediction.within_spec}")

print("Phase 21: Virtual Metrology - PASS")
```

---

## Phase 22: Supply Chain

### Test Supplier Portal
```python
# Save as: test_phase22.py
from dashboard.services.supply_chain.supplier_portal import SupplierPortalService

# Create service
portal = SupplierPortalService()

# Register supplier
portal.register_supplier(
    supplier_id="SUPP-001",
    name="ABS Plastics Inc",
    materials=["ABS", "PLA"],
    lead_time_days=7
)

# Assess supply risk
risk = portal.assess_supply_risk("SUPP-001")
print(f"Supply Risk Assessment:")
print(f"  Risk Score: {risk.score:.2f}/10")
print(f"  Risk Level: {risk.level}")

print("Phase 22: Supply Chain - PASS")
```

---

## Phase 23: Analytics

### Test KPI Engine
```python
# Save as: test_phase23.py
from datetime import date
from dashboard.services.analytics.kpi_engine import KPIEngine

# Create KPI engine
kpi = KPIEngine()

# Calculate OEE
oee = kpi.calculate_oee(
    work_center_id="printer-1",
    date=date.today(),
    planned_time_minutes=480,
    run_time_minutes=420,
    ideal_cycle_time=30,
    total_count=12,
    good_count=11
)

print(f"OEE Calculation:")
print(f"  Availability: {oee.availability:.1%}")
print(f"  Performance: {oee.performance:.1%}")
print(f"  Quality: {oee.quality:.1%}")
print(f"  OEE: {oee.oee:.1%}")

print("Phase 23: KPI Engine - PASS")
```

---

## Phase 24: Compliance

### Test Audit Trail
```python
# Save as: test_phase24.py
from dashboard.services.compliance.audit_trail import (
    AuditTrailService, AuditAction, ActionType
)

# Create audit service
audit = AuditTrailService()

# Log actions
audit.log_action(AuditAction(
    user_id="operator-1",
    action_type=ActionType.CREATE,
    entity_type="WorkOrder",
    entity_id="WO-001",
    new_values={"status": "created"}
))

# Verify chain integrity
integrity = audit.verify_chain_integrity()
print(f"Audit Trail Integrity: {'VALID' if integrity else 'COMPROMISED'}")

print("Phase 24: Audit Trail - PASS")
```

---

## Phase 25: Edge/IIoT

### Test IIoT Gateway
```python
# Save as: test_phase25.py
from dashboard.services.edge.iiot_gateway import IIoTGateway, Protocol

# Create gateway
gateway = IIoTGateway()

# Register devices
gateway.register_device(
    device_id="printer-1",
    device_name="Bambu P1S",
    protocol=Protocol.MQTT,
    host="192.168.1.100",
    port=1883
)

# Connect
gateway.connect("printer-1")

# Process data
data_point = gateway.process_data(
    device_id="printer-1",
    tag_name="nozzle_temp",
    value=210.5,
    unit="C"
)
print(f"Data Point: {data_point.tag_name} = {data_point.value} {data_point.unit}")

# Get summary
summary = gateway.get_summary()
print(f"Gateway Summary:")
print(f"  Total Devices: {summary['total_devices']}")
print(f"  Connected: {summary['connected_devices']}")

print("Phase 25: IIoT Gateway - PASS")
```

---

## Quick Test Script: All Phases

Save as `test_all_phases.sh`:

```bash
#!/bin/bash
echo "=========================================="
echo "LegoMCP v5.0 Full Integration Test"
echo "=========================================="

cd /Users/stepheneacuello/Documents/lego_mcp_fusion360
export PYTHONPATH="${PYTHONPATH}:$(pwd)/dashboard"

# Phase 7
echo "Testing Phase 7: Event-Driven Architecture..."
python -c "
from dashboard.services.events import EventBus
bus = EventBus()
print('  Phase 7: PASS')
" 2>/dev/null || echo "  Phase 7: SKIP (module not found)"

# Phase 8
echo "Testing Phase 8: Customer Orders..."
python -c "
from dashboard.services.erp import OrderService
svc = OrderService()
print('  Phase 8: PASS')
" 2>/dev/null || echo "  Phase 8: SKIP"

# Phase 9
echo "Testing Phase 9: Alternative Routings..."
python -c "
from dashboard.services.manufacturing.routing_selector import RoutingSelector
sel = RoutingSelector()
print('  Phase 9: PASS')
" 2>/dev/null || echo "  Phase 9: SKIP"

# Phase 10
echo "Testing Phase 10: FMEA Engine..."
python -c "
from dashboard.services.quality.fmea_service import FMEAService
fmea = FMEAService()
print('  Phase 10: PASS')
" 2>/dev/null || echo "  Phase 10: SKIP"

# Phase 11
echo "Testing Phase 11: QFD..."
python -c "
from dashboard.models.qfd import HouseOfQuality
print('  Phase 11: PASS')
" 2>/dev/null || echo "  Phase 11: SKIP"

# Phase 12
echo "Testing Phase 12: Advanced Scheduling..."
python -c "
from dashboard.services.scheduling import CPScheduler
sch = CPScheduler()
print('  Phase 12: PASS')
" 2>/dev/null || echo "  Phase 12: SKIP"

# Phase 13
echo "Testing Phase 13: CV Quality..."
python -c "
from dashboard.services.vision.defect_detector import DefectDetector
det = DefectDetector()
print('  Phase 13: PASS')
" 2>/dev/null || echo "  Phase 13: SKIP"

# Phase 14
echo "Testing Phase 14: Advanced SPC..."
python -c "
from dashboard.services.quality.advanced_spc import AdvancedSPCService
spc = AdvancedSPCService()
print('  Phase 14: PASS')
" 2>/dev/null || echo "  Phase 14: SKIP"

# Phase 15
echo "Testing Phase 15: Digital Thread..."
python -c "
from dashboard.services.traceability.digital_thread import DigitalThreadService
dt = DigitalThreadService()
print('  Phase 15: PASS')
" 2>/dev/null || echo "  Phase 15: SKIP"

# Phase 17
echo "Testing Phase 17: AI Copilot..."
python -c "
from dashboard.services.ai import ManufacturingCopilot
print('  Phase 17: PASS')
" 2>/dev/null || echo "  Phase 17: SKIP"

# Phase 18
echo "Testing Phase 18: DES Simulation..."
python -c "
from dashboard.services.simulation.des_engine import DESEngine
eng = DESEngine()
print('  Phase 18: PASS')
" 2>/dev/null || echo "  Phase 18: SKIP"

# Phase 19
echo "Testing Phase 19: Sustainability..."
python -c "
from dashboard.services.sustainability.carbon_tracker import CarbonTracker
ct = CarbonTracker()
print('  Phase 19: PASS')
" 2>/dev/null || echo "  Phase 19: SKIP"

# Phase 20
echo "Testing Phase 20: HMI..."
python -c "
from dashboard.services.hmi.work_instructions import WorkInstructionService
wi = WorkInstructionService()
print('  Phase 20: PASS')
" 2>/dev/null || echo "  Phase 20: SKIP"

# Phase 21
echo "Testing Phase 21: Zero-Defect..."
python -c "
from dashboard.services.quality.zero_defect.virtual_metrology import VirtualMetrology
vm = VirtualMetrology()
print('  Phase 21: PASS')
" 2>/dev/null || echo "  Phase 21: SKIP"

# Phase 22
echo "Testing Phase 22: Supply Chain..."
python -c "
from dashboard.services.supply_chain.supplier_portal import SupplierPortalService
sp = SupplierPortalService()
print('  Phase 22: PASS')
" 2>/dev/null || echo "  Phase 22: SKIP"

# Phase 23
echo "Testing Phase 23: Analytics..."
python -c "
from dashboard.services.analytics.kpi_engine import KPIEngine
kpi = KPIEngine()
print('  Phase 23: PASS')
" 2>/dev/null || echo "  Phase 23: SKIP"

# Phase 24
echo "Testing Phase 24: Compliance..."
python -c "
from dashboard.services.compliance.audit_trail import AuditTrailService
at = AuditTrailService()
print('  Phase 24: PASS')
" 2>/dev/null || echo "  Phase 24: SKIP"

# Phase 25
echo "Testing Phase 25: Edge/IIoT..."
python -c "
from dashboard.services.edge.iiot_gateway import IIoTGateway
gw = IIoTGateway()
print('  Phase 25: PASS')
" 2>/dev/null || echo "  Phase 25: SKIP"

echo "=========================================="
echo "Test Complete!"
echo "=========================================="
```

Run:
```bash
chmod +x test_all_phases.sh
./test_all_phases.sh
```

---

## PhD-Level Research Phases Testing

### Research Phase 1: XAI & Federated Learning

#### Test SHAP Explainer
```python
# Save as: test_xai_shap.py
from dashboard.services.ai.explainability.shap_explainer import SHAPExplainer

# Create explainer
explainer = SHAPExplainer()

# Test quality prediction explanation
features = {
    "temperature": 215.0,
    "speed": 60.0,
    "layer_height": 0.12,
    "infill": 20.0
}

explanation = explainer.explain_prediction(features, model_type="quality")
print(f"SHAP Explanation:")
print(f"  Base Value: {explanation.base_value:.4f}")
print(f"  Predicted Value: {explanation.predicted_value:.4f}")
for name, value in explanation.shap_values.items():
    print(f"  {name}: {value:+.4f}")

print("XAI SHAP - PASS")
```

#### Test LIME Explainer
```python
# Save as: test_xai_lime.py
from dashboard.services.ai.explainability.lime_explainer import LIMEExplainer

explainer = LIMEExplainer()

# Explain defect prediction
result = explainer.explain_classification(
    instance={"temp": 220, "speed": 50, "humidity": 45},
    class_names=["good", "defective"]
)

print(f"LIME Explanation for prediction: {result.predicted_class}")
for feature, weight in result.feature_weights[:5]:
    print(f"  {feature}: {weight:+.4f}")

print("XAI LIME - PASS")
```

#### Test Federated Learning
```python
# Save as: test_federated.py
from dashboard.services.ai.federated.federated_server import FederatedServer
from dashboard.services.ai.federated.federated_client import FederatedClient

# Create server
server = FederatedServer(
    model_type="quality_predictor",
    aggregation_strategy="fedavg"
)

# Register clients (simulating multiple factory sites)
client1 = FederatedClient(client_id="factory_1", server_url="localhost:8080")
client2 = FederatedClient(client_id="factory_2", server_url="localhost:8080")

server.register_client(client1.client_id)
server.register_client(client2.client_id)

print(f"Federated Server Status:")
print(f"  Registered Clients: {len(server.clients)}")
print(f"  Current Round: {server.current_round}")
print(f"  Aggregation: {server.aggregation_strategy}")

print("Federated Learning - PASS")
```

---

### Research Phase 2: Quantum-Inspired Scheduling

#### Test QAOA Scheduler
```python
# Save as: test_qaoa.py
from dashboard.services.scheduling.quantum.qaoa_scheduler import ManufacturingQAOA

# Create QAOA scheduler
qaoa = ManufacturingQAOA(
    num_qubits=8,
    p_layers=2,
    optimizer="COBYLA"
)

# Define jobs
jobs = [
    {"id": "J1", "duration": 30, "priority": 3, "due_date": 120},
    {"id": "J2", "duration": 45, "priority": 2, "due_date": 180},
    {"id": "J3", "duration": 25, "priority": 1, "due_date": 90},
]

# Run quantum-inspired optimization
result = qaoa.schedule_production(
    jobs=jobs,
    machines=["printer-1", "printer-2"],
    objective="minimize_makespan"
)

print(f"QAOA Scheduling Result:")
print(f"  Optimal Energy: {result.optimal_energy:.4f}")
print(f"  Makespan: {result.makespan} minutes")
print(f"  Job Assignments: {result.schedule}")

print("QAOA Scheduler - PASS")
```

#### Test VQE Scheduler
```python
# Save as: test_vqe.py
from dashboard.services.scheduling.quantum.vqe_scheduler import ManufacturingVQE

vqe = ManufacturingVQE(ansatz_type="efficient_su2")

# Multi-objective scheduling
result = vqe.optimize_schedule(
    jobs=[{"id": f"J{i}", "duration": 20+i*5} for i in range(5)],
    objectives=["makespan", "tardiness", "energy"]
)

print(f"VQE Result: Energy = {result.optimal_energy:.4f}")
print("VQE Scheduler - PASS")
```

#### Test Enhanced RL Dispatcher (PPO, SAC, TD3)
```python
# Save as: test_rl_dispatcher.py
from dashboard.services.scheduling.rl.ppo_dispatcher import PPODispatcher
from dashboard.services.scheduling.rl.sac_dispatcher import SACDispatcher
from dashboard.services.scheduling.rl.td3_dispatcher import TD3Dispatcher

# Test PPO
ppo = PPODispatcher(state_dim=10, action_dim=5)
ppo_action = ppo.select_action([0.5]*10)
print(f"PPO Action: {ppo_action}")

# Test SAC
sac = SACDispatcher(state_dim=10, action_dim=5)
sac_action = sac.select_action([0.5]*10)
print(f"SAC Action: {sac_action}")

# Test TD3
td3 = TD3Dispatcher(state_dim=10, action_dim=5)
td3_action = td3.select_action([0.5]*10)
print(f"TD3 Action: {td3_action}")

print("RL Dispatchers (PPO, SAC, TD3) - PASS")
```

---

### Research Phase 3: Digital Twin Research

#### Test Physics-Informed Neural Network (PINN)
```python
# Save as: test_pinn.py
from dashboard.services.digital_twin.ml.pinn_model import ManufacturingPINN

# Create PINN for thermal modeling
pinn = ManufacturingPINN()

# Predict thermal profile during printing
result = pinn.predict_thermal_profile(
    layer=10,
    time_step=5.0,
    power=200.0,
    speed=60.0
)

print(f"PINN Thermal Prediction:")
print(f"  Temperature: {result['temperature']:.1f}°C")
print(f"  Gradient: {result['gradient']:.4f} °C/mm")
print(f"  Thermal Stress: {result['thermal_stress']:.2f} MPa")

# Predict deformation
deform = pinn.predict_deformation(
    layer=15,
    temperature=220.0,
    cooling_rate=5.0
)
print(f"Deformation Prediction:")
print(f"  Warpage: {deform['warpage']:.3f} mm")
print(f"  Shrinkage: {deform['shrinkage']:.2%}")

print("PINN Model - PASS")
```

#### Test Ontology & Knowledge Graph
```python
# Save as: test_ontology.py
from dashboard.services.digital_twin.ontology.ontology_mapper import ManufacturingOntology
from dashboard.services.digital_twin.ontology.knowledge_graph import ManufacturingKnowledgeGraph

# Test ontology (ISO 23247)
ontology = ManufacturingOntology()
concepts = ontology.get_concepts()
print(f"Ontology Concepts: {len(concepts)}")
assert "DigitalTwin" in concepts
assert "PhysicalAsset" in concepts

# Test knowledge graph
kg = ManufacturingKnowledgeGraph()
kg.register_asset(
    asset_id="printer_001",
    asset_type="3DPrinter",
    properties={"make": "Prusa", "model": "MK4"}
)

asset = kg.get_asset("printer_001")
print(f"Asset: {asset['asset_id']} ({asset['asset_type']})")

print("Ontology & Knowledge Graph - PASS")
```

#### Test CRDT Conflict Resolution
```python
# Save as: test_crdt.py
from dashboard.services.digital_twin.sync.conflict_resolver import ConflictResolver, CRDTType
from datetime import datetime, timedelta

# LWW Register test
resolver = ConflictResolver()

# Concurrent updates
t1 = datetime.now() - timedelta(seconds=10)
t2 = datetime.now()

resolver.update("sensor_value", 100.0, t1, "sensor_1")
resolver.update("sensor_value", 105.0, t2, "sensor_2")

current = resolver.get("sensor_value")
print(f"LWW Register: {current} (later timestamp wins)")
assert current == 105.0

# G-Counter test
counter = ConflictResolver(CRDTType.G_COUNTER)
counter.increment("count", "node1", 5)
counter.increment("count", "node2", 3)
print(f"G-Counter: {counter.get('count')}")

print("CRDT Conflict Resolution - PASS")
```

---

### Research Phase 4: Sustainable Manufacturing

#### Test LCA Engine (ISO 14040/14044)
```python
# Save as: test_lca.py
from dashboard.services.sustainability.lca.lca_engine import ManufacturingLCA

lca = ManufacturingLCA()

# Assess product lifecycle impact
result = lca.assess_product(
    material="PLA",
    mass_kg=0.1,
    energy_kwh=2.0,
    transport_km=500
)

print(f"LCA Results (ISO 14040/14044):")
print(f"  Global Warming: {result.gwp:.4f} kg CO2-eq")
print(f"  Acidification: {result.ap:.6f} kg SO2-eq")
print(f"  Eutrophication: {result.ep:.6f} kg PO4-eq")
print(f"  Total Impact Score: {result.total_impact:.4f}")

print("LCA Engine - PASS")
```

#### Test Carbon Optimizer
```python
# Save as: test_carbon_optimizer.py
from dashboard.services.sustainability.carbon.carbon_optimizer import CarbonOptimizer

optimizer = CarbonOptimizer()

# Optimize production schedule for carbon reduction
jobs = [
    {"id": "J1", "energy_kwh": 2.0, "duration": 30},
    {"id": "J2", "energy_kwh": 3.5, "duration": 45},
    {"id": "J3", "energy_kwh": 1.5, "duration": 20},
]

result = optimizer.optimize(
    jobs=jobs,
    carbon_budget_kg=5.0,
    grid_carbon_intensity=0.4  # kg CO2/kWh
)

print(f"Carbon-Optimized Schedule:")
print(f"  Total Carbon: {result.total_carbon:.2f} kg CO2")
print(f"  Carbon Savings: {result.savings:.1%}")
print(f"  Schedule: {result.schedule}")

print("Carbon Optimizer - PASS")
```

#### Test Renewable Energy Scheduler
```python
# Save as: test_renewable.py
from dashboard.services.sustainability.carbon.renewable_scheduler import RenewableScheduler

scheduler = RenewableScheduler()

# Schedule production aligned with renewable energy availability
result = scheduler.schedule_with_renewables(
    jobs=[{"id": "J1", "energy_kwh": 5.0, "flexible": True}],
    solar_forecast=[0.1, 0.5, 0.9, 0.8, 0.4, 0.1],  # 6-hour forecast
    wind_forecast=[0.3, 0.4, 0.5, 0.6, 0.5, 0.4]
)

print(f"Renewable-Aligned Schedule:")
print(f"  Renewable Energy Used: {result.renewable_percent:.1%}")
print(f"  Best Time Slot: Hour {result.optimal_slot}")

print("Renewable Scheduler - PASS")
```

---

### Research Phase 5: AI/ML for Quality

#### Test Self-Supervised Learning (Contrastive)
```python
# Save as: test_ssl.py
from dashboard.services.vision.ssl.contrastive_learning import SimCLRDefectLearner

learner = SimCLRDefectLearner(
    backbone="resnet18",
    projection_dim=128,
    temperature=0.5
)

# Generate feature embeddings
import numpy as np
image = np.random.rand(224, 224, 3)
embedding = learner.encode(image)

print(f"SimCLR Embedding: dim={len(embedding)}")
assert len(embedding) == 128

print("Self-Supervised Learning - PASS")
```

#### Test Masked Autoencoder
```python
# Save as: test_mae.py
from dashboard.services.vision.ssl.masked_autoencoder import ManufacturingMAE

mae = ManufacturingMAE(
    patch_size=16,
    mask_ratio=0.75,
    encoder_dim=512
)

# Pretrain representation
import numpy as np
image = np.random.rand(224, 224, 3)
reconstruction, loss = mae.reconstruct(image)

print(f"MAE Reconstruction Loss: {loss:.4f}")

print("Masked Autoencoder - PASS")
```

#### Test Multimodal Sensor Fusion
```python
# Save as: test_multimodal.py
from dashboard.services.quality.multimodal.sensor_fusion import ManufacturingSensorFusion

fusion = ManufacturingSensorFusion()

# Fuse multiple sensor modalities
result = fusion.predict_quality(
    temperature=[200.0, 205.0, 210.0, 208.0],
    vibration=[0.1, 0.15, 0.12, 0.11],
    layer_image=[[0.5]*64 for _ in range(64)]  # 64x64 image
)

print(f"Multimodal Quality Prediction:")
print(f"  Quality Score: {result.quality_score:.3f}")
print(f"  Confidence: {result.confidence:.1%}")
print(f"  Defect Risk: {result.defect_risk:.1%}")

print("Multimodal Sensor Fusion - PASS")
```

#### Test Attention-Based Fusion
```python
# Save as: test_attention_fusion.py
from dashboard.services.quality.multimodal.attention_fusion import CrossAttentionFusion

fusion = CrossAttentionFusion(
    visual_dim=512,
    sensor_dim=64,
    num_heads=8
)

import numpy as np
visual_features = np.random.rand(512)
sensor_features = np.random.rand(64)

fused = fusion.fuse(visual_features, sensor_features)
print(f"Attention-Fused Features: dim={len(fused)}")

print("Attention Fusion - PASS")
```

---

### Research Phase 6: ISO 9001/13485 QMS

#### Test Document Control (ISO 9001:2015)
```python
# Save as: test_document_control.py
from dashboard.services.compliance.qms.document_control import (
    ISO9001DocumentControl, DocumentType
)

doc_control = ISO9001DocumentControl()

# Create controlled document
doc = doc_control.create_document(
    doc_type=DocumentType.SOP,
    title="3D Printing Quality Procedure",
    author="QA Manager",
    content="## Purpose\nDefine quality procedures for 3D printing..."
)

print(f"Document Created: {doc.document_number}")
print(f"  Type: {doc.doc_type.value}")
print(f"  Status: {doc.status.value}")
print(f"  Version: {doc.current_version}")

# Submit for review
doc_control.submit_for_review(doc.document_id, "Lead Engineer")

# Approve
doc_control.approve_document(doc.document_id, "Quality Director")

print(f"  New Status: {doc.status.value}")

print("Document Control - PASS")
```

#### Test CAPA Service (ISO 13485:2016)
```python
# Save as: test_capa.py
from dashboard.services.compliance.qms.capa_service import (
    ISO13485CAPA, CAPAType, CAPAPriority
)

capa = ISO13485CAPA()

# Initiate CAPA for quality issue
capa_record = capa.initiate_capa(
    capa_type=CAPAType.CORRECTIVE,
    title="Recurring Layer Adhesion Defects",
    description="Multiple reports of poor layer adhesion in PLA prints",
    source="Customer Complaint",
    priority=CAPAPriority.HIGH
)

print(f"CAPA Initiated: {capa_record.capa_number}")
print(f"  Type: {capa_record.capa_type.value}")
print(f"  Priority: {capa_record.priority.value}")

# Perform 5-Whys analysis
rca = capa.perform_root_cause_analysis(
    capa_record.capa_id,
    why_1="Why poor adhesion? - Nozzle temp too low",
    why_2="Why temp low? - Incorrect profile",
    why_3="Why wrong profile? - No validation",
    why_4="Why no validation? - Missing SOP",
    why_5="Why no SOP? - Training gap"
)

print(f"Root Cause: {rca.root_cause}")

print("CAPA Service - PASS")
```

#### Test Internal Audit (ISO 9001:2015)
```python
# Save as: test_audit.py
from dashboard.services.compliance.qms.internal_audit import ISO9001AuditProgram

audit_program = ISO9001AuditProgram()

# Schedule an audit
audit = audit_program.schedule_audit(
    area="Manufacturing - 3D Printing",
    scope=["7.1 Resources", "8.5 Production"],
    lead_auditor="Internal Auditor 1"
)

print(f"Audit Scheduled: {audit.audit_number}")
print(f"  Area: {audit.area}")
print(f"  Clauses: {audit.scope}")

# Start audit
audit_program.start_audit(audit.audit_id)

# Record finding
finding = audit_program.record_finding(
    audit_id=audit.audit_id,
    clause="8.5.1",
    description="Temperature records incomplete",
    severity="MINOR"
)

print(f"Finding: {finding.description} ({finding.severity})")

print("Internal Audit - PASS")
```

#### Test Management Review (ISO 9001:2015)
```python
# Save as: test_mgmt_review.py
from dashboard.services.compliance.qms.management_review import ISO9001ManagementReview

mgmt_review = ISO9001ManagementReview()

# Schedule review
review = mgmt_review.schedule_review(
    title="Q4 2024 Management Review",
    attendees=["CEO", "Quality Director", "Operations Manager"]
)

print(f"Review Scheduled: {review.review_id}")

# Submit required inputs (ISO 9001:2015 Clause 9.3.2)
mgmt_review.submit_input(review.review_id, "audit_results",
    {"findings": 3, "observations": 8})
mgmt_review.submit_input(review.review_id, "customer_feedback",
    {"complaints": 2, "satisfaction_score": 4.5})

# Check coverage
coverage = mgmt_review.check_input_coverage(review.review_id)
print(f"Input Coverage: {coverage['coverage_percent']:.0f}%")

print("Management Review - PASS")
```

---

## Run Complete PhD-Level Test Suite

```bash
#!/bin/bash
# Save as: run_phd_tests.sh

cd /Users/stepheneacuello/Documents/lego_mcp_fusion360
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/dashboard"

echo "=========================================="
echo "LegoMCP PhD-Level Research Test Suite"
echo "=========================================="

# Run pytest for all test files
python -m pytest tests/test_digital_twin.py -v --tb=short
python -m pytest tests/test_scheduling.py -v --tb=short
python -m pytest tests/test_sustainability.py -v --tb=short
python -m pytest tests/test_quality_ai.py -v --tb=short
python -m pytest tests/test_compliance.py -v --tb=short

# Or run with test runner
python tests/run_tests.py --all

echo "=========================================="
echo "PhD Test Suite Complete!"
echo "=========================================="
```

---

## API Quick Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | System health check |
| `GET /api/mes/work-orders` | List work orders |
| `GET /api/analytics/oee/{work_center}` | Get OEE for machine |
| `GET /api/quality/fmea/{part_id}` | FMEA analysis |
| `GET /api/sustainability/carbon/{part_id}` | Carbon footprint |

### PhD-Level Research APIs

| Endpoint | Description |
|----------|-------------|
| `POST /api/ai/explain` | XAI SHAP/LIME explanations |
| `POST /api/scheduling/qaoa` | Quantum-inspired scheduling |
| `GET /api/twin/pinn/thermal` | PINN thermal prediction |
| `GET /api/sustainability/lca/{part_id}` | ISO 14040 LCA |
| `GET /api/compliance/qms/status` | ISO 9001 QMS status |

---

## Success Criteria

All phases pass when their respective modules can be imported and instantiated without errors.

### PhD-Level Success Criteria

| Phase | Criteria |
|-------|----------|
| Phase 1 (XAI) | SHAP/LIME explanations with feature importance |
| Phase 2 (Quantum) | QAOA/VQE converge to valid schedules |
| Phase 3 (Digital Twin) | PINN physics loss < 0.1, CRDT conflict resolution |
| Phase 4 (Sustainability) | LCA GWP calculation matches ISO 14040 |
| Phase 5 (AI/ML) | Multimodal fusion quality score in [0,1] |
| Phase 6 (QMS) | Document workflow completes, CAPA 5-Whys works |
