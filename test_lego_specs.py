"""Verify LEGO specifications are correct."""
from shared.lego_specs import LegoStandard, LEGO, MANUFACTURING_TOLERANCES

print("=" * 60)
print("LEGO Specification Verification")
print("=" * 60)

# Core dimensions
print("\n📐 Core LEGO Dimensions:")
print(f"  LDU (LEGO Design Unit): {LEGO.LDU}mm")
print(f"  Stud Pitch: {LEGO.STUD_PITCH}mm")
print(f"  Plate Height: {LEGO.PLATE_HEIGHT}mm")
print(f"  Brick Height: {LEGO.BRICK_HEIGHT}mm")

# Critical corrected values
print("\n✅ Corrected Values (previously incorrect):")
print(f"  Wall Thickness: {LEGO.WALL_THICKNESS}mm (was 1.5mm)")
print(f"  Clearance Per Side: {LEGO.CLEARANCE_PER_SIDE}mm (was 0mm)")
print(f"  Rib Bottom Recess: {LEGO.RIB_BOTTOM_RECESS}mm (was 2.0mm)")

# Stud dimensions
print("\n🔴 Stud Dimensions:")
print(f"  Diameter: {LEGO.STUD_DIAMETER}mm")
print(f"  Height: {LEGO.STUD_HEIGHT}mm")

# Tube dimensions
print("\n⭕ Anti-Stud Tube Dimensions:")
print(f"  Outer Diameter: {LEGO.TUBE_OUTER_DIAMETER}mm")
print(f"  Inner Diameter: {LEGO.TUBE_INNER_DIAMETER}mm")

# Technic
print("\n🔧 Technic Dimensions:")
print(f"  Pin Hole Diameter: {LEGO.TECHNIC_PIN_HOLE_DIAMETER}mm (was 4.8mm)")
print(f"  Axle Hole Size: {LEGO.TECHNIC_AXLE_HOLE_SIZE}mm")
print(f"  Bar Diameter: {LEGO.BAR_DIAMETER}mm")

# Duplo
print("\n🟢 Duplo (2:1 Scale):")
print(f"  Scale Factor: {LEGO.DUPLO_SCALE}x")
print(f"  Stud Pitch: {LEGO.DUPLO_STUD_PITCH}mm")
print(f"  Brick Height: {LEGO.DUPLO_BRICK_HEIGHT}mm")

# Tolerances
print("\n📏 Manufacturing Tolerances:")
print(f"  LEGO Mold Precision: ±{LEGO.LEGO_MOLD_TOLERANCE}mm (2 microns)")
print(f"  LEGO Part Tolerance: ±{LEGO.LEGO_PART_TOLERANCE}mm")
print(f"  Stud Tolerance: ±{LEGO.STUD_TOLERANCE}mm")

# FDM adjustments
print("\n🖨️ 3D Printing Tolerances:")
print(f"  FDM Standard: ±{LEGO.FDM_TOLERANCE}mm")
print(f"  FDM Fine: ±{LEGO.FDM_FINE_TOLERANCE}mm")
print(f"  SLA Resin: ±{LEGO.SLA_TOLERANCE}mm")
print(f"  CNC Milling: ±{LEGO.CNC_TOLERANCE}mm")

# Manufacturing tolerance profiles
print("\n🏭 Manufacturing Tolerance Profiles:")
for process, tolerances in MANUFACTURING_TOLERANCES.items():
    print(f"  {process}: general ±{tolerances['general']}mm, xy_comp {tolerances['xy_compensation']}mm")

# Validate calculated values
print("\n🔢 Calculated Values Check:")
brick_width_2x = 2 * LEGO.STUD_PITCH - 2 * LEGO.CLEARANCE_PER_SIDE
print(f"  2-stud brick width: {brick_width_2x}mm (should be 15.8mm)")

brick_width_4x = 4 * LEGO.STUD_PITCH - 2 * LEGO.CLEARANCE_PER_SIDE
print(f"  4-stud brick width: {brick_width_4x}mm (should be 31.8mm)")

# Verify the LEGO Unit math
print("\n🔢 LEGO Unit (LDU) Verification:")
print(f"  5 LDU = {5 * LEGO.LDU}mm (should equal stud pitch: {LEGO.STUD_PITCH}mm) ✓" if 5 * LEGO.LDU == LEGO.STUD_PITCH else f"  5 LDU = {5 * LEGO.LDU}mm != stud pitch ✗")
print(f"  6 LDU = {6 * LEGO.LDU}mm (should equal brick height: {LEGO.BRICK_HEIGHT}mm) ✓" if 6 * LEGO.LDU == LEGO.BRICK_HEIGHT else f"  6 LDU = {6 * LEGO.LDU}mm != brick height ✗")
print(f"  2 LDU = {2 * LEGO.LDU}mm (should equal plate height: {LEGO.PLATE_HEIGHT}mm) ✓" if 2 * LEGO.LDU == LEGO.PLATE_HEIGHT else f"  2 LDU = {2 * LEGO.LDU}mm != plate height ✗")
print(f"  1 LDU = {LEGO.LDU}mm (should equal wall thickness: {LEGO.WALL_THICKNESS}mm) ✓" if LEGO.LDU == LEGO.WALL_THICKNESS else f"  1 LDU = {LEGO.LDU}mm != wall thickness ✗")

print("\n" + "=" * 60)
print("Specification verification complete!")
print("=" * 60)
