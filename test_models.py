"""Test SQLAlchemy models and database connection."""
import os
# Connect to local PostgreSQL
os.environ['DATABASE_URL'] = 'postgresql://lego_admin:lego_mcp_2024@localhost:5432/lego_manufacturing'

from dashboard.models import (
    db, init_db, get_db_session,
    Part, WorkCenter, WorkOrder, InventoryLocation
)
from dashboard.models.manufacturing import WorkCenterStatus, WorkOrderStatus

print("=" * 60)
print("Testing SQLAlchemy Models")
print("=" * 60)

# Test database connection
with get_db_session() as session:
    print("\n✓ Database connection successful")

    # Test WorkCenter model
    work_centers = session.query(WorkCenter).all()
    print(f"\n✓ Found {len(work_centers)} work centers:")
    for wc in work_centers:
        print(f"  - {wc.code}: {wc.name} ({wc.type}) - {wc.status}")

    # Test InventoryLocation model
    locations = session.query(InventoryLocation).all()
    print(f"\n✓ Found {len(locations)} inventory locations:")
    for loc in locations:
        print(f"  - {loc.location_code}: {loc.name}")

    # Test creating a Part
    existing_part = session.query(Part).filter(
        Part.part_number == 'TEST-2X4-BRICK'
    ).first()

    if not existing_part:
        test_part = Part(
            part_number='TEST-2X4-BRICK',
            name='Test 2x4 Brick',
            part_type='standard',
            category='Bricks',
            studs_x=4,
            studs_y=2,
            height_plates=3,
            volume_mm3=2560,
            standard_cost=0.15
        )
        session.add(test_part)
        session.flush()
        print(f"\n✓ Created test part: {test_part.part_number}")
    else:
        print(f"\n✓ Test part already exists: {existing_part.part_number}")

    # Test WorkOrder model
    part = session.query(Part).filter(Part.part_number == 'TEST-2X4-BRICK').first()
    if part:
        existing_wo = session.query(WorkOrder).filter(
            WorkOrder.work_order_number.like('WO-TEST-%')
        ).first()

        if not existing_wo:
            work_order = WorkOrder(
                work_order_number='WO-TEST-0001',
                part_id=part.id,
                quantity_ordered=10,
                priority=3,
                status=WorkOrderStatus.PLANNED.value
            )
            session.add(work_order)
            session.flush()
            print(f"\n✓ Created test work order: {work_order.work_order_number}")
        else:
            print(f"\n✓ Test work order already exists: {existing_wo.work_order_number}")

print("\n" + "=" * 60)
print("All model tests passed!")
print("=" * 60)
