"""Test ERP Services - BOM and Costing."""
import os
# Connect to local PostgreSQL
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import get_db_session, Part
from dashboard.services.erp import BOMService, CostService

print("=" * 60)
print("Testing ERP Services")
print("=" * 60)

with get_db_session() as session:
    # Create test parts for BOM
    parent = session.query(Part).filter(Part.part_number == 'TEST-ASSEMBLY').first()
    child1 = session.query(Part).filter(Part.part_number == 'TEST-2X4-BRICK').first()

    if not parent:
        parent = Part(
            part_number='TEST-ASSEMBLY',
            name='Test Assembly',
            part_type='assembly',
            category='Assemblies',
            standard_cost=0
        )
        session.add(parent)
        session.flush()
        print(f"\n✓ Created parent part: {parent.part_number}")

    if not child1:
        child1 = Part(
            part_number='TEST-2X4-BRICK',
            name='Test 2x4 Brick',
            part_type='standard',
            standard_cost=0.15
        )
        session.add(child1)
        session.flush()

    child2 = session.query(Part).filter(Part.part_number == 'TEST-1X2-PLATE').first()
    if not child2:
        child2 = Part(
            part_number='TEST-1X2-PLATE',
            name='Test 1x2 Plate',
            part_type='standard',
            category='Plates',
            studs_x=2,
            studs_y=1,
            height_plates=1,
            standard_cost=0.05
        )
        session.add(child2)
        session.flush()
        print(f"\n✓ Created child part: {child2.part_number}")

    session.commit()

    # Test BOMService
    bom_service = BOMService(session)

    # Check existing BOM
    existing_bom = bom_service.get_bom(str(parent.id))
    if not existing_bom:
        # Create BOM lines
        bom_service.create_bom_line(
            parent_part_id=str(parent.id),
            child_part_id=str(child1.id),
            quantity=4,
            sequence=10
        )
        print(f"\n✓ Added 4x {child1.part_number} to BOM")

        bom_service.create_bom_line(
            parent_part_id=str(parent.id),
            child_part_id=str(child2.id),
            quantity=8,
            sequence=20
        )
        print(f"✓ Added 8x {child2.part_number} to BOM")
    else:
        print(f"\n✓ BOM already exists with {len(existing_bom)} components")

    # Get BOM
    bom = bom_service.get_bom(str(parent.id))
    print(f"\n✓ BOM for {parent.part_number}:")
    for line in bom:
        print(f"  - {line['quantity']}x {line['child_part_number']}")

    # Explode BOM
    explosion = bom_service.explode_bom(str(parent.id), quantity=2)
    print(f"\n✓ BOM explosion for 2 assemblies:")
    for item in explosion:
        print(f"  - Level {item['level']}: {item['extended_quantity']}x {item['part_number']}")

    # Test CostService
    cost_service = CostService(session)

    cost = cost_service.calculate_standard_cost(str(parent.id))
    print(f"\n✓ Standard cost for {parent.part_number}:")
    print(f"  - Material: ${cost['material_cost']:.2f}")
    print(f"  - Labor: ${cost['labor_cost']:.2f}")
    print(f"  - Machine: ${cost['machine_cost']:.2f}")
    print(f"  - Overhead: ${cost['overhead_cost']:.2f}")
    print(f"  - Total: ${cost['total_cost']:.2f}")

print("\n" + "=" * 60)
print("All ERP tests passed!")
print("=" * 60)
