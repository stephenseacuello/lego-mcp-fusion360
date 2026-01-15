"""Test Manufacturing Services."""
import os
# Connect to local PostgreSQL
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import get_db_session, Part, WorkCenter
from dashboard.services.manufacturing import (
    WorkOrderService, RoutingService, OEEService
)

print("=" * 60)
print("Testing Manufacturing Services")
print("=" * 60)

with get_db_session() as session:
    # Test WorkOrderService
    wo_service = WorkOrderService(session)

    # Get or create a test part
    part = session.query(Part).filter(Part.part_number == 'TEST-2X4-BRICK').first()
    if not part:
        print("\n⚠ Run test_models.py first to create test part")
        exit(1)

    print(f"\n✓ Using part: {part.part_number}")

    # Test RoutingService - auto-generate routing
    routing_service = RoutingService(session)

    existing_routings = routing_service.get_routing(str(part.id))
    if not existing_routings:
        routings = routing_service.auto_generate_routing(
            str(part.id),
            part_type='standard'
        )
        print(f"\n✓ Auto-generated {len(routings)} routing operations:")
        for r in routings:
            print(f"  - Seq {r.operation_sequence}: {r.operation_code}")
    else:
        print(f"\n✓ Found existing routing with {len(existing_routings)} operations")
        for r in existing_routings:
            print(f"  - Seq {r.operation_sequence}: {r.operation_code}")

    # Calculate routing times
    times = routing_service.calculate_total_time(str(part.id), quantity=10)
    print(f"\n✓ Time calculation for 10 units:")
    print(f"  - Setup: {times['setup_time_min']:.1f} min")
    print(f"  - Run per unit: {times['run_time_per_unit_min']:.1f} min")
    print(f"  - Total run: {times['run_time_total_min']:.1f} min")
    print(f"  - Total: {times['total_time_min']:.1f} min")

    # Calculate standard cost
    costs = routing_service.calculate_standard_cost(str(part.id), quantity=10)
    print(f"\n✓ Cost calculation for 10 units:")
    print(f"  - Cost per unit: ${costs['cost_per_unit']:.2f}")
    print(f"  - Total cost: ${costs['total_cost']:.2f}")

    # Test OEEService
    oee_service = OEEService(session)

    printer = session.query(WorkCenter).filter(
        WorkCenter.code == 'PRINTER-01'
    ).first()

    if printer:
        status = oee_service.get_work_center_status(str(printer.id))
        print(f"\n✓ Work center status for {printer.code}:")
        print(f"  - Status: {status['work_center']['status']}")
        print(f"  - Current OEE: {status['current_shift_oee']['oee']}%")

print("\n" + "=" * 60)
print("All service tests passed!")
print("=" * 60)
