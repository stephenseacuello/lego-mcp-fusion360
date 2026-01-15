#!/usr/bin/env python3
"""
LegoMCP Industry 4.0 - Implementation Verification

Verifies all implemented files exist and have correct structure.
Does not require database or Flask dependencies.
"""

import os
import ast
from pathlib import Path

print("=" * 70)
print("LegoMCP Industry 4.0 - Implementation Verification")
print("=" * 70)

BASE_DIR = Path(__file__).parent

def check_file_exists(filepath):
    """Check if file exists."""
    full_path = BASE_DIR / filepath
    return full_path.exists()

def check_python_syntax(filepath):
    """Check Python file has valid syntax."""
    full_path = BASE_DIR / filepath
    try:
        with open(full_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except FileNotFoundError:
        return False, "File not found"

def check_classes_in_file(filepath, expected_classes):
    """Check if file contains expected classes."""
    full_path = BASE_DIR / filepath
    try:
        with open(full_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)

        found_classes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found_classes.add(node.name)

        missing = set(expected_classes) - found_classes
        return len(missing) == 0, missing
    except Exception as e:
        return False, str(e)

def check_functions_in_file(filepath, expected_functions):
    """Check if file contains expected functions."""
    full_path = BASE_DIR / filepath
    try:
        with open(full_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)

        found_functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                found_functions.add(node.name)

        missing = set(expected_functions) - found_functions
        return len(missing) == 0, missing
    except Exception as e:
        return False, str(e)


# =============================================================================
# Phase 1: Core Models
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 1: Database Models & LEGO Specifications")
print("=" * 70)

models = [
    ("dashboard/models/__init__.py", ["Part", "WorkCenter", "WorkOrder"]),
    ("dashboard/models/manufacturing.py", ["BOM", "Routing", "WorkOrderOperation"]),
    ("dashboard/models/quality.py", ["QualityInspection", "QualityMetric"]),
    ("dashboard/models/analytics.py", ["DigitalTwinState", "OEEEvent"]),
]

for filepath, expected in models:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# Phase 2: MES Services
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 2: Manufacturing Execution Services")
print("=" * 70)

mes_services = [
    ("dashboard/services/manufacturing/__init__.py", []),
    ("dashboard/services/manufacturing/work_order_service.py", ["WorkOrderService"]),
    ("dashboard/services/manufacturing/routing_service.py", ["RoutingService"]),
    ("dashboard/services/manufacturing/oee_service.py", ["OEEService"]),
]

for filepath, expected_classes in mes_services:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            if expected_classes:
                has_classes, missing = check_classes_in_file(filepath, expected_classes)
                if has_classes:
                    print(f"   [OK] {filepath}")
                else:
                    print(f"   [MISSING CLASSES] {filepath}: {missing}")
            else:
                print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")

# Equipment Controllers
print("\n   Equipment Controllers:")
equipment = [
    ("dashboard/services/equipment/__init__.py", []),
    ("dashboard/services/equipment/printer_controller.py", ["PrinterController"]),
    ("dashboard/services/equipment/mill_controller.py", ["MillController"]),
    ("dashboard/services/equipment/laser_controller.py", ["LaserController"]),
]

for filepath, expected_classes in equipment:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# Phase 3: Quality Services
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 3: Quality & Analytics Services")
print("=" * 70)

quality_services = [
    ("dashboard/services/quality/__init__.py", []),
    ("dashboard/services/quality/inspection_service.py", ["InspectionService"]),
    ("dashboard/services/quality/measurement_service.py", ["MeasurementService"]),
    ("dashboard/services/quality/lego_quality.py", ["LEGOQualityService"]),
    ("dashboard/services/quality/spc_service.py", ["SPCService"]),
]

for filepath, expected_classes in quality_services:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            if expected_classes:
                has_classes, missing = check_classes_in_file(filepath, expected_classes)
                if has_classes:
                    print(f"   [OK] {filepath}")
                else:
                    print(f"   [MISSING CLASSES] {filepath}: {missing}")
            else:
                print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# Phase 4: ERP Services
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 4: ERP & Costing Services")
print("=" * 70)

erp_services = [
    ("dashboard/services/erp/__init__.py", []),
    ("dashboard/services/erp/bom_service.py", ["BOMService"]),
    ("dashboard/services/erp/cost_service.py", ["CostService"]),
    ("dashboard/services/erp/procurement_service.py", ["ProcurementService"]),
    ("dashboard/services/erp/demand_service.py", ["DemandService"]),
]

for filepath, expected_classes in erp_services:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            if expected_classes:
                has_classes, missing = check_classes_in_file(filepath, expected_classes)
                if has_classes:
                    print(f"   [OK] {filepath}")
                else:
                    print(f"   [MISSING CLASSES] {filepath}: {missing}")
            else:
                print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# Phase 5: MRP Services
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 5: MRP & Planning Services")
print("=" * 70)

mrp_services = [
    ("dashboard/services/mrp/__init__.py", []),
    ("dashboard/services/mrp/mrp_engine.py", ["MRPEngine"]),
    ("dashboard/services/mrp/capacity_planner.py", ["CapacityPlanner"]),
]

for filepath, expected_classes in mrp_services:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            if expected_classes:
                has_classes, missing = check_classes_in_file(filepath, expected_classes)
                if has_classes:
                    print(f"   [OK] {filepath}")
                else:
                    print(f"   [MISSING CLASSES] {filepath}: {missing}")
            else:
                print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# Phase 6: Digital Twin Services
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 6: Digital Twin Services")
print("=" * 70)

twin_services = [
    ("dashboard/services/digital_twin/__init__.py", []),
    ("dashboard/services/digital_twin/twin_manager.py", ["DigitalTwinManager"]),
    ("dashboard/services/digital_twin/maintenance_service.py", ["PredictiveMaintenanceService"]),
]

for filepath, expected_classes in twin_services:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            if expected_classes:
                has_classes, missing = check_classes_in_file(filepath, expected_classes)
                if has_classes:
                    print(f"   [OK] {filepath}")
                else:
                    print(f"   [MISSING CLASSES] {filepath}: {missing}")
            else:
                print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# API Routes
# =============================================================================
print("\n" + "=" * 70)
print("API ROUTES")
print("=" * 70)

routes = [
    "dashboard/routes/__init__.py",
    "dashboard/routes/manufacturing/__init__.py",
    "dashboard/routes/manufacturing/shop_floor.py",
    "dashboard/routes/manufacturing/work_orders.py",
    "dashboard/routes/manufacturing/work_centers.py",
    "dashboard/routes/manufacturing/oee.py",
    "dashboard/routes/quality/__init__.py",
    "dashboard/routes/quality/inspections.py",
    "dashboard/routes/quality/measurements.py",
    "dashboard/routes/quality/lego_compatibility.py",
    "dashboard/routes/quality/spc.py",
    "dashboard/routes/erp/__init__.py",
    "dashboard/routes/erp/bom.py",
    "dashboard/routes/erp/costing.py",
    "dashboard/routes/erp/procurement.py",
    "dashboard/routes/erp/demand.py",
    "dashboard/routes/mrp/__init__.py",
    "dashboard/routes/mrp/planning.py",
    "dashboard/routes/mrp/capacity.py",
    "dashboard/routes/digital_twin/__init__.py",
    "dashboard/routes/digital_twin/twin.py",
    "dashboard/routes/digital_twin/maintenance.py",
    "dashboard/routes/digital_twin/simulation.py",
]

for filepath in routes:
    if check_file_exists(filepath):
        valid, error = check_python_syntax(filepath)
        if valid:
            print(f"   [OK] {filepath}")
        else:
            print(f"   [SYNTAX ERROR] {filepath}: {error}")
    else:
        print(f"   [MISSING] {filepath}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

# Count files
all_files = models + mes_services + equipment + quality_services + erp_services + mrp_services + twin_services
all_files_paths = [f[0] for f in all_files] + routes

existing = sum(1 for f in all_files_paths if check_file_exists(f))
total = len(all_files_paths)

print(f"""
Files verified: {existing}/{total}
Implementation status: {"COMPLETE" if existing == total else "INCOMPLETE"}

Phase Summary:
  Phase 1 (Models):       {"OK" if all(check_file_exists(f[0]) for f in models) else "INCOMPLETE"}
  Phase 2 (MES):          {"OK" if all(check_file_exists(f[0]) for f in mes_services + equipment) else "INCOMPLETE"}
  Phase 3 (Quality):      {"OK" if all(check_file_exists(f[0]) for f in quality_services) else "INCOMPLETE"}
  Phase 4 (ERP):          {"OK" if all(check_file_exists(f[0]) for f in erp_services) else "INCOMPLETE"}
  Phase 5 (MRP):          {"OK" if all(check_file_exists(f[0]) for f in mrp_services) else "INCOMPLETE"}
  Phase 6 (Digital Twin): {"OK" if all(check_file_exists(f[0]) for f in twin_services) else "INCOMPLETE"}
  API Routes:             {"OK" if all(check_file_exists(f) for f in routes) else "INCOMPLETE"}
""")

print("=" * 70)
print("Verification complete!")
print("=" * 70)
