"""
LEGO Brick Dimension Standards

Official LEGO dimensions based on measurements and patents.
All measurements in millimeters unless otherwise noted.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LegoStandard:
    """Official LEGO brick dimensions (in mm)"""

    # === Fundamental Units ===
    STUD_PITCH: float = 8.0  # Distance between stud centers

    # === Stud Dimensions ===
    STUD_DIAMETER: float = 4.8  # Cylinder on top
    STUD_HEIGHT: float = 1.7  # Height above brick surface
    STUD_INNER_DIAMETER: float = 3.0  # Hollow stud interior (some bricks)

    # === Brick Heights ===
    BRICK_HEIGHT: float = 9.6  # Standard brick (3 plates)
    PLATE_HEIGHT: float = 3.2  # Plate = 1/3 brick height
    TILE_HEIGHT: float = 3.2  # Same as plate, no studs

    # === Wall Structure ===
    WALL_THICKNESS: float = 1.5  # Outer walls
    TOP_THICKNESS: float = 1.0  # Top surface thickness
    BOTTOM_THICKNESS: float = 0.0  # Open bottom (hollow)

    # === Bottom Tubes (for clutch power) ===
    TUBE_OUTER_DIAMETER: float = 6.51
    TUBE_INNER_DIAMETER: float = 4.8  # Matches stud for grip

    # === Bottom Ribs (for 1xN bricks) ===
    RIB_THICKNESS: float = 1.0
    RIB_HEIGHT: float = 8.0  # From bottom to just below top

    # === Manufacturing Tolerances ===
    TOLERANCE: float = 0.1  # General tolerance
    STUD_TOLERANCE: float = 0.2  # Slightly looser for clutch
    FDM_TOLERANCE: float = 0.15  # Extra tolerance for 3D printing

    # === Slope Angles (degrees) ===
    SLOPE_18: float = 18.0
    SLOPE_33: float = 33.0
    SLOPE_45: float = 45.0
    SLOPE_65: float = 65.0
    SLOPE_75: float = 75.0

    # === Technic ===
    TECHNIC_HOLE_DIAMETER: float = 4.8
    TECHNIC_HOLE_SPACING: float = 8.0  # Same as stud pitch


# Singleton instance
LEGO = LegoStandard()


def brick_dimensions(
    studs_x: int, studs_y: int, height_units: float = 1.0
) -> Tuple[float, float, float]:
    """
    Calculate brick dimensions for given stud configuration.

    Args:
        studs_x: Number of studs in X direction
        studs_y: Number of studs in Y direction
        height_units: Height in brick units (1.0 = brick, 0.333 = plate)

    Returns:
        Tuple of (width, depth, height) in mm
    """
    width = studs_x * LEGO.STUD_PITCH
    depth = studs_y * LEGO.STUD_PITCH
    height = height_units * LEGO.BRICK_HEIGHT
    return (width, depth, height)


def stud_positions(studs_x: int, studs_y: int) -> list:
    """
    Calculate center positions for all studs.

    Returns:
        List of (x, y) tuples for stud centers
    """
    positions = []
    for i in range(studs_x):
        for j in range(studs_y):
            x = (i + 0.5) * LEGO.STUD_PITCH
            y = (j + 0.5) * LEGO.STUD_PITCH
            positions.append((x, y))
    return positions


def tube_positions(studs_x: int, studs_y: int) -> list:
    """
    Calculate center positions for bottom tubes.
    Tubes go between studs, so for NxM brick there are (N-1)x(M-1) tubes.

    Returns:
        List of (x, y) tuples for tube centers
    """
    if studs_x < 2 or studs_y < 2:
        return []  # No tubes for 1xN bricks

    positions = []
    for i in range(studs_x - 1):
        for j in range(studs_y - 1):
            x = (i + 1) * LEGO.STUD_PITCH
            y = (j + 1) * LEGO.STUD_PITCH
            positions.append((x, y))
    return positions


def rib_positions(studs_x: int, studs_y: int) -> list:
    """
    Calculate center positions for bottom ribs (1xN bricks only).

    Returns:
        List of (x, y, orientation) tuples
    """
    if studs_x == 1 and studs_y > 1:
        # Ribs perpendicular to length
        return [(LEGO.STUD_PITCH / 2, (i + 1) * LEGO.STUD_PITCH, "x") for i in range(studs_y - 1)]
    elif studs_y == 1 and studs_x > 1:
        # Ribs perpendicular to length
        return [((i + 1) * LEGO.STUD_PITCH, LEGO.STUD_PITCH / 2, "y") for i in range(studs_x - 1)]
    return []


# Brick type definitions
BRICK_TYPES = {
    "standard": {
        "description": "Standard brick with studs",
        "has_studs": True,
        "hollow": True,
        "height_units": 1.0,
    },
    "plate": {
        "description": "Thin plate (1/3 brick height)",
        "has_studs": True,
        "hollow": True,
        "height_units": 1 / 3,
    },
    "tile": {
        "description": "Flat tile without studs",
        "has_studs": False,
        "hollow": False,
        "height_units": 1 / 3,
    },
    "slope": {
        "description": "Sloped brick",
        "has_studs": True,
        "hollow": True,
        "height_units": 1.0,
        "slope_angle": 45.0,
    },
    "technic": {
        "description": "Technic brick with holes",
        "has_studs": True,
        "hollow": True,
        "height_units": 1.0,
        "has_holes": True,
    },
}


# Common brick configurations
COMMON_BRICKS = [
    {"name": "1x1", "studs_x": 1, "studs_y": 1},
    {"name": "1x2", "studs_x": 1, "studs_y": 2},
    {"name": "1x3", "studs_x": 1, "studs_y": 3},
    {"name": "1x4", "studs_x": 1, "studs_y": 4},
    {"name": "1x6", "studs_x": 1, "studs_y": 6},
    {"name": "1x8", "studs_x": 1, "studs_y": 8},
    {"name": "2x2", "studs_x": 2, "studs_y": 2},
    {"name": "2x3", "studs_x": 2, "studs_y": 3},
    {"name": "2x4", "studs_x": 2, "studs_y": 4},
    {"name": "2x6", "studs_x": 2, "studs_y": 6},
    {"name": "2x8", "studs_x": 2, "studs_y": 8},
    {"name": "2x10", "studs_x": 2, "studs_y": 10},
    {"name": "4x4", "studs_x": 4, "studs_y": 4},
    {"name": "4x6", "studs_x": 4, "studs_y": 6},
    {"name": "6x6", "studs_x": 6, "studs_y": 6},
]
