"""
Pytest configuration and fixtures for LegoMCP test suite.
"""

import pytest
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Digital Twin Fixtures
# ============================================================================

@pytest.fixture
def pinn_config():
    """Create PINN configuration."""
    from dashboard.services.digital_twin.ml.pinn_model import PINNConfig
    return PINNConfig()


@pytest.fixture
def pinn_model(pinn_config):
    """Create PINN model instance."""
    from dashboard.services.digital_twin.ml.pinn_model import PINNModel
    return PINNModel(pinn_config)


@pytest.fixture
def knowledge_graph():
    """Create knowledge graph instance."""
    from dashboard.services.digital_twin.ontology.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph()


@pytest.fixture
def conflict_resolver():
    """Create conflict resolver instance."""
    from dashboard.services.digital_twin.sync.conflict_resolver import ConflictResolver
    return ConflictResolver()


# ============================================================================
# Scheduling Fixtures
# ============================================================================

@pytest.fixture
def qaoa_scheduler():
    """Create QAOA scheduler instance."""
    from dashboard.services.scheduling.quantum.qaoa_scheduler import QAOAScheduler
    return QAOAScheduler()


@pytest.fixture
def rl_dispatcher():
    """Create RL dispatcher instance."""
    from dashboard.services.scheduling.rl_dispatcher import RLDispatcher
    return RLDispatcher(algorithm="PPO")


@pytest.fixture
def nsga2_scheduler():
    """Create NSGA-II scheduler instance."""
    from dashboard.services.scheduling.nsga2_scheduler import NSGA2Scheduler
    return NSGA2Scheduler(population_size=20, num_generations=5)


@pytest.fixture
def sample_jobs():
    """Create sample jobs for scheduling tests."""
    return [
        {"id": "J1", "duration": 10, "priority": 1, "energy": 5},
        {"id": "J2", "duration": 15, "priority": 2, "energy": 8},
        {"id": "J3", "duration": 8, "priority": 1, "energy": 3},
        {"id": "J4", "duration": 12, "priority": 3, "energy": 6},
    ]


# ============================================================================
# Sustainability Fixtures
# ============================================================================

@pytest.fixture
def lca_engine():
    """Create LCA engine instance."""
    from dashboard.services.sustainability.lca.lca_engine import LCAEngine
    return LCAEngine()


@pytest.fixture
def carbon_optimizer():
    """Create carbon optimizer instance."""
    from dashboard.services.sustainability.carbon.carbon_optimizer import CarbonOptimizer
    return CarbonOptimizer()


@pytest.fixture
def scope3_tracker():
    """Create Scope 3 tracker instance."""
    from dashboard.services.sustainability.carbon.scope3_tracker import Scope3Tracker
    return Scope3Tracker()


# ============================================================================
# Quality AI Fixtures
# ============================================================================

@pytest.fixture
def contrastive_learner():
    """Create contrastive learner instance."""
    from dashboard.services.vision.ssl.contrastive_learning import ContrastiveLearner
    return ContrastiveLearner(backbone="resnet18", projection_dim=128)


@pytest.fixture
def anomaly_detector():
    """Create SSL anomaly detector instance."""
    from dashboard.services.vision.ssl.anomaly_ssl import SSLAnomalyDetector
    return SSLAnomalyDetector()


@pytest.fixture
def sensor_fusion():
    """Create sensor fusion instance."""
    from dashboard.services.quality.multimodal.sensor_fusion import SensorFusion
    return SensorFusion()


@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    return [[0.5] * 64 for _ in range(64)]


@pytest.fixture
def sample_image_batch():
    """Create batch of sample images."""
    return [[[0.5] * 64 for _ in range(64)] for _ in range(8)]


# ============================================================================
# Compliance Fixtures
# ============================================================================

@pytest.fixture
def document_controller():
    """Create document controller instance."""
    from dashboard.services.compliance.qms.document_control import DocumentController
    return DocumentController()


@pytest.fixture
def iso9001_doc_control():
    """Create ISO 9001 document controller instance."""
    from dashboard.services.compliance.qms.document_control import ISO9001DocumentControl
    return ISO9001DocumentControl()


@pytest.fixture
def capa_service():
    """Create CAPA service instance."""
    from dashboard.services.compliance.qms.capa_service import CAPAService
    return CAPAService()


@pytest.fixture
def audit_program():
    """Create ISO 9001 audit program instance."""
    from dashboard.services.compliance.qms.internal_audit import ISO9001AuditProgram
    return ISO9001AuditProgram()


@pytest.fixture
def management_review_service():
    """Create management review service instance."""
    from dashboard.services.compliance.qms.management_review import ManagementReviewService
    return ManagementReviewService()


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def full_manufacturing_context():
    """Create full manufacturing context with all services."""
    context = {
        "lca": None,
        "scheduling": None,
        "quality": None,
        "compliance": None,
    }

    try:
        from dashboard.services.sustainability.lca.lca_engine import ManufacturingLCA
        context["lca"] = ManufacturingLCA()
    except ImportError:
        pass

    try:
        from dashboard.services.scheduling.quantum.qaoa_scheduler import ManufacturingQAOA
        context["scheduling"] = ManufacturingQAOA()
    except ImportError:
        pass

    try:
        from dashboard.services.quality.multimodal.sensor_fusion import ManufacturingSensorFusion
        context["quality"] = ManufacturingSensorFusion()
    except ImportError:
        pass

    try:
        from dashboard.services.compliance.qms.document_control import ISO9001DocumentControl
        context["compliance"] = ISO9001DocumentControl()
    except ImportError:
        pass

    return context


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "compliance: Compliance tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")


# ============================================================================
# Test Utilities
# ============================================================================

def assert_valid_result(result: Dict[str, Any], required_keys: List[str]):
    """Assert that result contains required keys."""
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"


def create_temporal_sequence(length: int = 10, feature_dim: int = 64):
    """Create temporal sequence for testing."""
    from dashboard.services.quality.multimodal.temporal_fusion import (
        TemporalSequence, TemporalFeature
    )

    features = [
        TemporalFeature(
            features=[0.5] * feature_dim,
            timestamp=datetime.now(),
            timestep=i,
            modality="sensor"
        )
        for i in range(length)
    ]

    return TemporalSequence(
        features=features,
        modality="sensor",
        start_time=datetime.now(),
        end_time=datetime.now()
    )


# ============================================================================
# ERP Services Fixtures
# ============================================================================

@pytest.fixture
def gl_service():
    """Create General Ledger service."""
    from dashboard.services.erp.gl_integration import create_gl_service
    return create_gl_service()


@pytest.fixture
def ar_service():
    """Create Accounts Receivable service."""
    from dashboard.services.erp.ar_ap_service import create_ar_service
    return create_ar_service()


@pytest.fixture
def ap_service():
    """Create Accounts Payable service."""
    from dashboard.services.erp.ar_ap_service import create_ap_service
    return create_ap_service()


@pytest.fixture
def edi_processor():
    """Create EDI Processor service."""
    from dashboard.services.erp.edi_processor import create_edi_processor
    return create_edi_processor()


# ============================================================================
# Supply Chain Fixtures
# ============================================================================

@pytest.fixture
def risk_service():
    """Create Supply Chain Risk service."""
    from dashboard.services.supply_chain.risk_assessment import create_risk_service
    return create_risk_service()


@pytest.fixture
def inventory_service():
    """Create Inventory Optimizer service."""
    from dashboard.services.supply_chain.inventory_optimizer import create_inventory_optimizer
    return create_inventory_optimizer()


@pytest.fixture
def logistics_service():
    """Create Logistics Tracker service."""
    from dashboard.services.supply_chain.logistics_tracker import create_logistics_tracker
    return create_logistics_tracker()


@pytest.fixture
def sop_service():
    """Create S&OP Planner service."""
    from dashboard.services.supply_chain.sop_planner import create_sop_planner
    return create_sop_planner()


@pytest.fixture
def supplier_quality_service():
    """Create Supplier Quality service."""
    from dashboard.services.supply_chain.supplier_quality import create_supplier_quality_service
    return create_supplier_quality_service()


# ============================================================================
# Advanced Technology Fixtures
# ============================================================================

@pytest.fixture
def blockchain_service():
    """Create Blockchain Traceability service."""
    from dashboard.services.blockchain.traceability_ledger import create_traceability_service
    return create_traceability_service()


@pytest.fixture
def security_service():
    """Create IEC 62443 Security service."""
    from dashboard.services.security.iec62443_framework import create_security_service
    return create_security_service()


@pytest.fixture
def sync_service():
    """Create Cloud-Edge Sync service."""
    from dashboard.services.cloud.edge_sync import create_cloud_edge_service
    return create_cloud_edge_service()


@pytest.fixture
def ar_instructions_service():
    """Create AR Instructions service."""
    from dashboard.services.hmi.ar_instructions import create_ar_service
    return create_ar_service()


@pytest.fixture
def amr_service():
    """Create AMR Integration service."""
    from dashboard.services.robotics.amr_integration import create_amr_service
    return create_amr_service()


# ============================================================================
# QMS Compliance Fixtures (Extended)
# ============================================================================

@pytest.fixture
def deviation_service():
    """Create Deviation service."""
    from dashboard.services.compliance.qms.deviation_service import create_deviation_service
    return create_deviation_service()


@pytest.fixture
def batch_record_service():
    """Create Batch Record service."""
    from dashboard.services.compliance.qms.batch_record import create_batch_record_service
    return create_batch_record_service()


@pytest.fixture
def training_service():
    """Create Training Management service."""
    from dashboard.services.compliance.qms.training_service import create_training_service
    return create_training_service()


# ============================================================================
# Sample Test Data
# ============================================================================

@pytest.fixture
def sample_order_data():
    """Generate sample order data for testing."""
    return {
        "customer_id": "CUST-TEST-001",
        "customer_name": "Test Customer",
        "items": [
            {"sku": "LEGO-42100", "quantity": 10, "unit_price": 399.99},
            {"sku": "LEGO-42125", "quantity": 5, "unit_price": 179.99}
        ],
        "shipping_address": {"city": "New York", "country": "USA"},
        "payment_terms": "NET_30"
    }


@pytest.fixture
def sample_deviation_data():
    """Generate sample deviation data for testing."""
    return {
        "title": "Test Deviation",
        "deviation_type": "PRODUCT",
        "severity": "MINOR",
        "description": "Test deviation description",
        "area": "Assembly",
        "detected_by": "qa_inspector",
        "affected_batch": "BATCH-TEST-001"
    }


@pytest.fixture
def sample_capa_data():
    """Generate sample CAPA data for testing."""
    return {
        "title": "Test CAPA",
        "capa_type": "CORRECTIVE",
        "priority": "MEDIUM",
        "source": "Internal Audit",
        "description": "Test CAPA description",
        "affected_products": ["PROD-001"],
        "initiated_by": "quality_manager"
    }


# ============================================================================
# ISO 23247 Digital Twin Fixtures
# ============================================================================

@pytest.fixture
def ome_registry():
    """Create OME registry instance."""
    from dashboard.services.digital_twin import get_ome_registry
    return get_ome_registry()


@pytest.fixture
def twin_engine():
    """Create twin engine instance."""
    from dashboard.services.digital_twin import get_twin_engine
    return get_twin_engine()


@pytest.fixture
def sample_printer_ome():
    """Create sample printer OME."""
    from dashboard.services.digital_twin import create_printer_ome
    return create_printer_ome(
        name="Test Printer",
        manufacturer="Test Corp",
        model="TC-1000",
        serial_number="TC-TEST-001"
    )


@pytest.fixture
def sample_sensor_ome():
    """Create sample sensor OME."""
    from dashboard.services.digital_twin import create_sensor_ome
    return create_sensor_ome(
        name="Test Temperature Sensor",
        sensor_type="temperature",
        measurement_unit="celsius",
        sampling_rate_hz=10.0
    )


@pytest.fixture
def sample_robotic_arm_ome():
    """Create sample robotic arm OME."""
    from dashboard.services.digital_twin import create_robotic_arm_ome
    return create_robotic_arm_ome(
        name="Test Robot Arm",
        arm_model="generic",
        dof=6,
        reach_mm=600,
        payload_kg=2.0
    )


# ============================================================================
# Unity Integration Fixtures
# ============================================================================

@pytest.fixture
def unity_bridge():
    """Create Unity bridge instance."""
    from dashboard.services.unity import UnityBridge
    return UnityBridge()


@pytest.fixture
def unity_scene_service():
    """Create Unity scene data service."""
    from dashboard.services.unity import UnitySceneDataService
    return UnitySceneDataService()


# ============================================================================
# Anomaly Response Fixtures
# ============================================================================

@pytest.fixture
def anomaly_response_service():
    """Create anomaly response service."""
    from dashboard.services.digital_twin import get_anomaly_response_service
    return get_anomaly_response_service()


@pytest.fixture
def sample_anomaly():
    """Create sample anomaly for testing."""
    from dashboard.services.digital_twin import Anomaly, AnomalyType, SeverityLevel
    import uuid
    return Anomaly(
        id=str(uuid.uuid4()),
        ome_id="test-ome-001",
        anomaly_type=AnomalyType.TEMPERATURE,
        severity=SeverityLevel.MEDIUM,
        detected_at=datetime.now(),
        sensor_readings={"temperature": 85.0},
        context={"threshold": 80.0}
    )


@pytest.fixture
def sample_response_rule():
    """Create sample response rule for testing."""
    from dashboard.services.digital_twin import (
        ResponseRule, AnomalyType, SeverityLevel, ResponseType
    )
    return ResponseRule(
        rule_id="rule-test-001",
        name="Test Temperature Response",
        anomaly_type=AnomalyType.TEMPERATURE,
        severity_threshold=SeverityLevel.HIGH,
        response_type=ResponseType.AUTOMATIC,
        action="reduce_speed",
        is_active=True
    )


# ============================================================================
# Supply Chain Twin Fixtures
# ============================================================================

@pytest.fixture
def supply_chain_twin_service():
    """Create supply chain twin service."""
    from dashboard.services.digital_twin import get_supply_chain_twin_service
    return get_supply_chain_twin_service()


@pytest.fixture
def sample_supply_chain_node():
    """Create sample supply chain node."""
    from dashboard.services.digital_twin import (
        SupplyChainNode, NodeType, NodeStatus, GeoLocation
    )
    import uuid
    return SupplyChainNode(
        id=str(uuid.uuid4()),
        node_id="SUPPLIER-TEST-001",
        name="Test Supplier",
        node_type=NodeType.SUPPLIER,
        status=NodeStatus.ACTIVE,
        location=GeoLocation(latitude=40.7128, longitude=-74.0060),
        lead_time_days=14.0
    )


@pytest.fixture
def sample_supply_chain_edge():
    """Create sample supply chain edge."""
    from dashboard.services.digital_twin import SupplyChainEdge, TransportMode
    import uuid
    return SupplyChainEdge(
        id=str(uuid.uuid4()),
        source_id="node-001",
        target_id="node-002",
        transport_mode=TransportMode.ROAD,
        distance_km=500.0,
        transit_time_hours=8.0
    )


# ============================================================================
# VR Training Fixtures
# ============================================================================

@pytest.fixture
def vr_training_service():
    """Create VR training service."""
    from dashboard.services.hmi import get_vr_training_service
    return get_vr_training_service()


@pytest.fixture
def sample_training_scenario():
    """Create sample training scenario."""
    from dashboard.services.hmi import (
        TrainingScenario, TrainingCategory, DifficultyLevel, TrainingStep
    )
    return TrainingScenario(
        scenario_id="scenario-test-001",
        name="Test Scenario",
        description="A test training scenario",
        category=TrainingCategory.EQUIPMENT_OPERATION,
        difficulty=DifficultyLevel.BEGINNER,
        steps=[
            TrainingStep(
                step_id="step-1",
                title="Step 1",
                instructions="Do the first thing",
                success_criteria={"completed": True}
            )
        ],
        estimated_duration_minutes=10,
        max_score=100,
        passing_score=70
    )


# ============================================================================
# Quality Heatmap Fixtures
# ============================================================================

@pytest.fixture
def quality_heatmap_generator():
    """Create quality heatmap generator."""
    from dashboard.services.vision import get_quality_heatmap_generator
    return get_quality_heatmap_generator()


@pytest.fixture
def sample_quality_data_points():
    """Create sample quality data points."""
    from dashboard.services.vision import QualityDataPoint, Vector3
    return [
        QualityDataPoint(
            position=Vector3(x=float(i * 10), y=float(j * 10), z=0.0),
            value=0.8 + (i + j) * 0.01,
            timestamp=datetime.now(),
            metric_type="quality_score"
        )
        for i in range(5)
        for j in range(5)
    ]


@pytest.fixture
def sample_bounding_box():
    """Create sample bounding box."""
    from dashboard.services.vision import BoundingBox, Vector3
    return BoundingBox(
        min_point=Vector3(x=0.0, y=0.0, z=0.0),
        max_point=Vector3(x=100.0, y=100.0, z=50.0)
    )


# ============================================================================
# Defect Mapping Fixtures
# ============================================================================

@pytest.fixture
def defect_mapping_service():
    """Create defect mapping service."""
    from dashboard.services.vision import get_defect_mapping_service
    return get_defect_mapping_service()


@pytest.fixture
def sample_2d_defect():
    """Create sample 2D defect."""
    from dashboard.services.vision import Defect2D, DefectType, DefectSeverity
    import uuid
    return Defect2D(
        id=str(uuid.uuid4()),
        defect_type=DefectType.SCRATCH,
        severity=DefectSeverity.MINOR,
        x=100,
        y=200,
        width=50,
        height=10,
        confidence=0.95,
        detected_at=datetime.now()
    )


# ============================================================================
# Predictive Analytics Fixtures
# ============================================================================

@pytest.fixture
def predictive_analytics_service():
    """Create predictive analytics service."""
    from dashboard.services.digital_twin import get_predictive_analytics_service
    return get_predictive_analytics_service()


# ============================================================================
# Event Store Fixtures
# ============================================================================

@pytest.fixture
def event_store():
    """Create event store instance."""
    from dashboard.services.digital_twin import EventStore
    return EventStore()


@pytest.fixture
def sample_twin_event():
    """Create sample twin event."""
    from dashboard.services.digital_twin import (
        TwinEvent, EventCategory, EventPriority
    )
    import uuid
    return TwinEvent(
        event_id=str(uuid.uuid4()),
        twin_id="test-twin-001",
        event_type="sensor_update",
        category=EventCategory.SENSOR_UPDATE,
        priority=EventPriority.NORMAL,
        timestamp=datetime.now(),
        data={"temperature": 45.0}
    )
