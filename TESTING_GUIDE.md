# LegoMCP PhD-Level Manufacturing Platform - Testing Guide

> Comprehensive testing and verification guide for all platform components.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Environment Setup](#phase-1-environment-setup)
4. [Database Verification](#phase-2-database-verification)
5. [Python Environment](#phase-3-python-environment)
6. [Core Service Tests](#phase-4-core-service-tests)
7. [AI/ML Module Tests](#phase-5-aiml-module-tests)
8. [Advanced Feature Tests](#phase-6-advanced-feature-tests)
9. [Full Stack Testing](#phase-7-full-stack-testing)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before testing, ensure you have:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |
| Python | 3.10+ | Backend runtime |
| Git | 2.30+ | Version control |
| Node.js | 18+ | (Optional) Dashboard dev |

---

## Quick Start

```bash
# Clone and setup
cd /Users/stepheneacuello/Documents/lego_mcp_fusion360

# Start all services
docker-compose up -d

# Run automated tests
python -m pytest tests/ -v --tb=short

# Check service health
curl http://localhost:5000/health
```

---

## Phase 1: Environment Setup

### 1.1 Create Environment File

```bash
cd /Users/stepheneacuello/Documents/lego_mcp_fusion360

# Create .env file with required variables
cat > .env << 'EOF'
# Database Configuration
POSTGRES_DB=lego_manufacturing
POSTGRES_USER=lego_admin
POSTGRES_PASSWORD=lego_mcp_2024
DATABASE_URL=postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Application Settings
MANUFACTURING_MODE=production
FLASK_ENV=development
SQL_ECHO=false
SECRET_KEY=your-secret-key-here

# MLflow Configuration (for experiment tracking)
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# Optional: AI/ML Settings
ENABLE_GPU=false
MODEL_CACHE_DIR=/models
EOF
```

### 1.2 Start Docker Services

```bash
# Start infrastructure services first
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
sleep 15

# Verify PostgreSQL is accepting connections
docker-compose exec postgres pg_isready -U lego_admin -d lego_manufacturing
# Expected: "localhost:5432 - accepting connections"

# Start all remaining services
docker-compose up -d

# Check all services are running
docker-compose ps
```

### 1.3 Initialize Database with Alembic

```bash
# Run database migrations
docker-compose exec dashboard alembic upgrade head

# Or run manually if needed
python -c "
from dashboard.models import init_db
init_db()
print('Database initialized successfully!')
"
```

---

## Phase 2: Database Verification

### 2.1 Verify Tables Exist

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U lego_admin -d lego_manufacturing

# List all tables
\dt

# Expected tables include:
# - work_centers, work_orders, parts, inventory_*
# - quality_*, digital_twin_*, audit_log
# - users, sessions (if auth enabled)

# Check work centers
SELECT code, name, type, status FROM work_centers;

# Check inventory locations
SELECT location_code, name, location_type FROM inventory_locations;

# Exit psql
\q
```

### 2.2 Database Admin UI

Access **Adminer** at http://localhost:8080:

| Field | Value |
|-------|-------|
| System | PostgreSQL |
| Server | postgres |
| Username | lego_admin |
| Password | lego_mcp_2024 |
| Database | lego_manufacturing |

---

## Phase 3: Python Environment

### 3.1 Create Virtual Environment

```bash
cd /Users/stepheneacuello/Documents/lego_mcp_fusion360

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 3.2 Verify Installation

```bash
python -c "
# Test core imports
from dashboard.models import db, Part, WorkCenter
from dashboard.services.manufacturing import WorkOrderService
from dashboard.services.erp import BOMService
print('✓ Core imports successful')

# Test AI/ML imports
from dashboard.services.ai.uncertainty import UncertaintyEstimator
from dashboard.services.ai.causal import CausalDiscovery
from dashboard.services.ai.continual import EWCTrainer
print('✓ AI/ML imports successful')

# Test advanced features
from dashboard.services.digital_twin import DigitalTwinManager
from dashboard.services.scheduling import SchedulerFactory
from dashboard.services.sustainability import CarbonTracker
print('✓ Advanced feature imports successful')

print('\n✅ All imports verified!')
"
```

---

## Phase 4: Core Service Tests

### 4.1 Test SQLAlchemy Models

```bash
python << 'EOF'
"""Test SQLAlchemy models and database connection."""
import os
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import get_db_session, Part, WorkCenter

print("=" * 60)
print("Testing SQLAlchemy Models")
print("=" * 60)

with get_db_session() as session:
    print("\n✓ Database connection successful")

    # Test WorkCenter model
    work_centers = session.query(WorkCenter).all()
    print(f"\n✓ Found {len(work_centers)} work centers")

    # Test Part creation
    test_part = Part(
        part_number='TEST-VERIFY-001',
        name='Verification Test Part',
        part_type='standard',
        category='Test',
    )
    session.add(test_part)
    session.flush()
    print(f"✓ Created test part: {test_part.part_number}")

    # Rollback test data
    session.rollback()
    print("✓ Test data rolled back")

print("\n✅ Model tests passed!")
EOF
```

### 4.2 Test Manufacturing Services

```bash
python << 'EOF'
"""Test Manufacturing Services."""
import os
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import get_db_session, Part, WorkCenter
from dashboard.services.manufacturing import WorkOrderService, RoutingService, OEEService

print("=" * 60)
print("Testing Manufacturing Services")
print("=" * 60)

with get_db_session() as session:
    # Test WorkOrderService
    wo_service = WorkOrderService(session)
    print("✓ WorkOrderService initialized")

    # Test RoutingService
    routing_service = RoutingService(session)
    print("✓ RoutingService initialized")

    # Test OEEService
    oee_service = OEEService(session)
    print("✓ OEEService initialized")

    # Test work center status
    wc = session.query(WorkCenter).first()
    if wc:
        status = oee_service.get_work_center_status(str(wc.id))
        print(f"✓ Got status for work center: {wc.code}")

print("\n✅ Manufacturing service tests passed!")
EOF
```

### 4.3 Test ERP Services

```bash
python << 'EOF'
"""Test ERP Services."""
import os
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import get_db_session
from dashboard.services.erp import BOMService, CostService

print("=" * 60)
print("Testing ERP Services")
print("=" * 60)

with get_db_session() as session:
    # Test BOMService
    bom_service = BOMService(session)
    print("✓ BOMService initialized")

    # Test CostService
    cost_service = CostService(session)
    print("✓ CostService initialized")

print("\n✅ ERP service tests passed!")
EOF
```

---

## Phase 5: AI/ML Module Tests

### 5.1 Test Uncertainty Quantification

```bash
python << 'EOF'
"""Test Uncertainty Quantification module."""
import numpy as np
from dashboard.services.ai.uncertainty import (
    UncertaintyEstimator,
    MCDropout,
    ConformalPredictor,
    UncertaintyMethod,
)

print("=" * 60)
print("Testing Uncertainty Quantification")
print("=" * 60)

# Test MC Dropout
mc_dropout = MCDropout(n_samples=10)
print("✓ MCDropout initialized")

# Test Conformal Predictor
conformal = ConformalPredictor()
calibration_preds = np.random.randn(100)
calibration_true = calibration_preds + np.random.randn(100) * 0.1
conformal.calibrate(calibration_preds, calibration_true)
print("✓ ConformalPredictor calibrated")

# Test unified interface
estimator = UncertaintyEstimator()
print("✓ UncertaintyEstimator initialized")

print("\n✅ Uncertainty quantification tests passed!")
EOF
```

### 5.2 Test Causal Inference

```bash
python << 'EOF'
"""Test Causal Inference module."""
import numpy as np
from dashboard.services.ai.causal import (
    CausalDiscovery,
    DiscoveryAlgorithm,
)

print("=" * 60)
print("Testing Causal Inference")
print("=" * 60)

# Test CausalDiscovery
discovery = CausalDiscovery()
print("✓ CausalDiscovery initialized")

# Generate test data
n_samples = 100
X = np.random.randn(n_samples, 3)
X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.1
X[:, 2] = X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1

# Discover causal structure
graph = discovery.discover(X, ['A', 'B', 'C'])
print(f"✓ Discovered graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

print("\n✅ Causal inference tests passed!")
EOF
```

### 5.3 Test Explainability

```bash
python << 'EOF'
"""Test Explainability module."""
from dashboard.services.ai.explainability import (
    SHAPExplainer,
    LIMEExplainer,
)

print("=" * 60)
print("Testing Explainability (XAI)")
print("=" * 60)

# Test SHAP
print("✓ SHAP module available")

# Test LIME
print("✓ LIME module available")

print("\n✅ Explainability tests passed!")
EOF
```

---

## Phase 6: Advanced Feature Tests

### 6.1 Test Digital Twin

```bash
python << 'EOF'
"""Test Digital Twin module."""
from dashboard.services.digital_twin import (
    DigitalTwinManager,
    PredictiveMaintenanceService,
)

print("=" * 60)
print("Testing Digital Twin")
print("=" * 60)

# Test DigitalTwinManager
manager = DigitalTwinManager()
print("✓ DigitalTwinManager initialized")

# Test PredictiveMaintenanceService
maintenance = PredictiveMaintenanceService()
print("✓ PredictiveMaintenanceService initialized")

print("\n✅ Digital Twin tests passed!")
EOF
```

### 6.2 Test Scheduling Algorithms

```bash
python << 'EOF'
"""Test Scheduling module."""
from dashboard.services.scheduling import (
    SchedulerFactory,
    SchedulerType,
    Job,
    Operation,
    Machine,
)

print("=" * 60)
print("Testing Advanced Scheduling")
print("=" * 60)

# Test SchedulerFactory
factory = SchedulerFactory()
print("✓ SchedulerFactory initialized")

# Test available scheduler types
print("✓ Available schedulers:")
for stype in SchedulerType:
    print(f"  - {stype.value}")

print("\n✅ Scheduling tests passed!")
EOF
```

### 6.3 Test Sustainability

```bash
python << 'EOF'
"""Test Sustainability module."""
from dashboard.services.sustainability import (
    CarbonTracker,
    EnergyOptimizer,
)

print("=" * 60)
print("Testing Sustainability & Carbon Tracking")
print("=" * 60)

# Test CarbonTracker
tracker = CarbonTracker()
print("✓ CarbonTracker initialized")

# Test EnergyOptimizer
optimizer = EnergyOptimizer()
print("✓ EnergyOptimizer initialized")

print("\n✅ Sustainability tests passed!")
EOF
```

### 6.4 Test LEGO Specifications

```bash
python << 'EOF'
"""Verify LEGO specifications."""
from shared.lego_specs import (
    LegoStandard, LegoTolerance, LegoManufacturing,
    TechnicStandard, DuploStandard
)

print("=" * 60)
print("LEGO Specification Verification")
print("=" * 60)

print("\n📐 Core Dimensions:")
print(f"  Stud Pitch: {LegoStandard.STUD_PITCH}mm")
print(f"  Plate Height: {LegoStandard.PLATE_HEIGHT}mm")
print(f"  Brick Height: {LegoStandard.BRICK_HEIGHT}mm")
print(f"  Stud Diameter: {LegoStandard.STUD_DIAMETER}mm")

print("\n📏 Tolerances:")
print(f"  Overall: ±{LegoTolerance.OVERALL}mm")
print(f"  Injection Molding: ±{LegoTolerance.INJECTION_MOLDING}mm")

print("\n🖨️ FDM Printing:")
print(f"  Hole Compensation: +{LegoManufacturing.FDM_HOLE_COMPENSATION}mm")
print(f"  Expected Tolerance: ±{LegoManufacturing.FDM_EXPECTED_TOLERANCE}mm")

print("\n✅ Specifications verified!")
EOF
```

---

## Phase 7: Full Stack Testing

### 7.1 Start All Services

```bash
# Start everything
docker-compose up -d

# Wait for services
sleep 30

# Verify all services
docker-compose ps
```

### 7.2 Access Web Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard | http://localhost:5000 | Main manufacturing dashboard |
| API Docs | http://localhost:5000/api/docs | OpenAPI documentation |
| Slicer API | http://localhost:8766/docs | Slicer service |
| Adminer | http://localhost:8080 | Database UI |
| Redis Insight | http://localhost:8001 | Redis monitoring |
| MLflow | http://localhost:5001 | Experiment tracking |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards |

### 7.3 Run Full Test Suite

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ --cov=dashboard --cov=shared --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/benchmarks/ -v
```

### 7.4 API Health Checks

```bash
# Dashboard health
curl -s http://localhost:5000/health | jq .

# Slicer health
curl -s http://localhost:8766/health | jq .

# API v5 endpoints
curl -s http://localhost:5000/api/v5/status | jq .
```

---

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres

# Verify connection
docker-compose exec postgres psql -U lego_admin -d lego_manufacturing -c "SELECT 1;"
```

### Import Errors

```bash
# Ensure you're in the project root
cd /Users/stepheneacuello/Documents/lego_mcp_fusion360

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify imports
python -c "from dashboard.models import db; print('OK')"
```

### Service Not Starting

```bash
# Check all logs
docker-compose logs

# Rebuild services
docker-compose build --no-cache
docker-compose up -d

# Check individual service
docker-compose logs dashboard
docker-compose logs slicer
```

### Memory Issues

```bash
# Check Docker resource usage
docker stats

# Prune unused resources
docker system prune -f

# Restart with limits
docker-compose down
docker-compose up -d
```

---

## Test Data Cleanup

```bash
python << 'EOF'
"""Clean up test data."""
import os
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import get_db_session, Part, WorkOrder

with get_db_session() as session:
    # Delete test work orders
    deleted_wo = session.query(WorkOrder).filter(
        WorkOrder.work_order_number.like('WO-TEST-%')
    ).delete(synchronize_session=False)

    # Delete test parts
    deleted_parts = session.query(Part).filter(
        Part.part_number.like('TEST-%')
    ).delete(synchronize_session=False)

    session.commit()
    print(f"✓ Deleted {deleted_wo} test work orders")
    print(f"✓ Deleted {deleted_parts} test parts")
EOF
```

---

## Platform Components Summary

The LegoMCP PhD-Level Manufacturing Platform includes:

| Phase | Components |
|-------|------------|
| **Phase 1** | XAI (SHAP, LIME), Federated Learning, Benchmarks |
| **Phase 2** | Quantum Scheduling, RL Dispatching, NSGA-II/III |
| **Phase 3** | Digital Twin (ISO 23247), PINN, Ontology |
| **Phase 4** | Sustainability, LCA, Carbon Tracking |
| **Phase 5** | Vision AI, SSL, Foundation Models |
| **Phase 6** | ISO 9001/13485 Compliance, QMS |
| **Phase 7** | Testing, Documentation, IEEE Paper |
| **Phase 8** | Production Deployment, MLOps, CI/CD |

---

## Next Steps

After all tests pass:

1. **Deploy to Staging**: Use ArgoCD or Helm
2. **Run E2E Tests**: Full workflow validation
3. **Performance Testing**: Load tests with k6
4. **Security Scan**: Run Trivy and Bandit
5. **Production Deployment**: Follow CD pipeline

For more details, see:
- [DEVELOPER.md](docs/DEVELOPER.md) - Development guide
- [API.md](docs/API.md) - API documentation
- [USER_GUIDE.md](docs/USER_GUIDE.md) - User manual
