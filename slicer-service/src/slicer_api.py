"""
CuraEngine HTTP API Service

Provides a REST API for slicing STL files using CuraEngine CLI.
Optimized for LEGO brick printing with appropriate profiles.
Works on both x64 and ARM64 (Apple Silicon).
"""

import os
import subprocess
import json
import re
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


app = FastAPI(
    title="LEGO Slicer Service",
    description="CuraEngine-based G-code generation for LEGO bricks",
    version="1.0.0",
)

# Directories
PROFILES_DIR = Path("/app/profiles")
OUTPUT_DIR = Path("/output/gcode/3dprint")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# === Models ===


class SliceRequest(BaseModel):
    """Request to slice an STL file."""

    stl_path: str = Field(..., description="Path to input STL file")
    printer: str = Field("generic", description="Printer profile name")
    material: str = Field("pla", description="Material type")
    quality: str = Field("fine", description="Print quality preset")
    infill_percent: int = Field(20, ge=0, le=100, description="Infill percentage")
    output_filename: Optional[str] = Field(None, description="Custom output filename")


class SliceResponse(BaseModel):
    """Response from slicing operation."""

    success: bool
    path: str = ""
    estimated_time_min: float = 0
    filament_meters: float = 0
    filament_grams: float = 0
    layer_height_mm: float = 0
    layer_count: int = 0
    error: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    engine: str
    engine_available: bool


# === Quality Presets ===

QUALITY_PRESETS = {
    "draft": {"layer_height": 0.30, "wall_count": 2, "top_layers": 3, "bottom_layers": 2},
    "normal": {"layer_height": 0.20, "wall_count": 3, "top_layers": 4, "bottom_layers": 3},
    "fine": {"layer_height": 0.15, "wall_count": 3, "top_layers": 5, "bottom_layers": 4},
    "ultra": {"layer_height": 0.10, "wall_count": 4, "top_layers": 6, "bottom_layers": 5},
}

# Material-specific settings
MATERIAL_SETTINGS = {
    "pla": {"temp_extruder": 215, "temp_bed": 60, "fan_speed": 100},
    "petg": {"temp_extruder": 240, "temp_bed": 85, "fan_speed": 50},
    "abs": {"temp_extruder": 255, "temp_bed": 100, "fan_speed": 0},
    "asa": {"temp_extruder": 260, "temp_bed": 105, "fan_speed": 0},
}


# === Helper Functions ===


def is_curaengine_available() -> bool:
    """Check if CuraEngine CLI is available."""
    try:
        result = subprocess.run(
            ["CuraEngine", "--help"], capture_output=True, text=True, timeout=10
        )
        # CuraEngine --help returns 0, but --version may return different codes
        # Check if we got any output indicating it exists
        return result.returncode == 0 or "Cura" in result.stdout or "Cura" in result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def get_curaengine_version() -> str:
    """Get CuraEngine version."""
    try:
        result = subprocess.run(
            ["CuraEngine", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip() or result.stderr.strip()
    except Exception:
        pass
    return "unknown"


def create_cura_config(quality: dict, material: dict, infill: int) -> Path:
    """Create a CuraEngine JSON config for LEGO brick printing."""
    config = {
        # Layer settings
        "layer_height": quality["layer_height"],
        "layer_height_0": quality["layer_height"] * 1.2,  # Slightly thicker first layer

        # Wall settings
        "wall_line_count": quality["wall_count"],
        "wall_thickness": quality["wall_count"] * 0.4,  # 0.4mm nozzle

        # Top/bottom
        "top_layers": quality["top_layers"],
        "bottom_layers": quality["bottom_layers"],
        "top_thickness": quality["top_layers"] * quality["layer_height"],
        "bottom_thickness": quality["bottom_layers"] * quality["layer_height"],

        # Infill
        "infill_sparse_density": infill,
        "infill_pattern": "grid",

        # Temperature
        "material_print_temperature": material["temp_extruder"],
        "material_print_temperature_layer_0": material["temp_extruder"] + 5,
        "material_bed_temperature": material["temp_bed"],
        "material_bed_temperature_layer_0": material["temp_bed"] + 5,

        # Fan
        "cool_fan_speed": material["fan_speed"],
        "cool_fan_speed_0": 0,  # No fan on first layer

        # Speed settings (conservative for LEGO precision)
        "speed_print": 50,
        "speed_infill": 60,
        "speed_wall_0": 30,  # Slower outer wall for quality
        "speed_wall_x": 40,
        "speed_topbottom": 40,
        "speed_travel": 150,
        "speed_layer_0": 20,  # Slow first layer

        # Retraction
        "retraction_enable": True,
        "retraction_amount": 0.8,
        "retraction_speed": 45,

        # Quality settings for LEGO
        "outer_inset_first": True,  # External perimeters first
        "fill_outline_gaps": True,  # Fill thin walls
        "filter_out_tiny_gaps": False,  # Keep small details

        # Nozzle/machine
        "machine_nozzle_size": 0.4,
        "line_width": 0.4,

        # Adhesion
        "adhesion_type": "skirt",
        "skirt_line_count": 3,
    }

    config_path = PROFILES_DIR / "lego_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def parse_gcode_stats(gcode_path: Path) -> dict:
    """Parse G-code file for print statistics."""
    stats = {"time_min": 0, "filament_m": 0, "filament_g": 0, "layers": 0}

    try:
        with open(gcode_path, "r") as f:
            content = f.read()

            # CuraEngine time format: ;TIME:1234 (seconds)
            time_match = re.search(r";TIME:(\d+)", content)
            if time_match:
                stats["time_min"] = int(time_match.group(1)) / 60

            # Filament used: ;Filament used: 1.234m
            filament_match = re.search(r";Filament used:\s*([\d.]+)m", content)
            if filament_match:
                stats["filament_m"] = float(filament_match.group(1))
                # Estimate grams (PLA ~1.24 g/cm³, 1.75mm filament)
                stats["filament_g"] = stats["filament_m"] * 2.98  # ~2.98g per meter

            # Layer count: ;LAYER_COUNT:64
            layer_match = re.search(r";LAYER_COUNT:(\d+)", content)
            if layer_match:
                stats["layers"] = int(layer_match.group(1))

    except FileNotFoundError:
        logger.warning(f"G-code file not found: {gcode_path}")
    except Exception as e:
        logger.warning(f"Error parsing G-code stats from {gcode_path}: {e}")

    return stats


# === Endpoints ===


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and CuraEngine availability."""
    engine_available = is_curaengine_available()
    version = get_curaengine_version() if engine_available else "not installed"

    return HealthResponse(
        status="ok" if engine_available else "degraded",
        service="LEGO Slicer Service",
        version="1.0.0",
        engine="CuraEngine",
        engine_available=engine_available,
    )


@app.get("/printers")
async def list_printers():
    """List available printer profiles."""
    # CuraEngine supports any printer with proper config
    printers = ["generic", "prusa_mk3s", "prusa_mk4", "ender3", "voron", "bambu_x1"]
    return {"printers": printers}


@app.get("/materials")
async def list_materials():
    """List available material profiles."""
    return {"materials": list(MATERIAL_SETTINGS.keys()), "settings": MATERIAL_SETTINGS}


@app.get("/qualities")
async def list_qualities():
    """List available quality presets."""
    return {"qualities": list(QUALITY_PRESETS.keys()), "settings": QUALITY_PRESETS}


class LegoSliceRequest(BaseModel):
    """Request to slice a LEGO brick STL file with optimized settings."""

    stl_path: str = Field(..., description="Path to input STL file")
    printer: str = Field("generic", description="Printer profile name")
    brick_type: str = Field("standard", description="LEGO brick type")
    output_path: Optional[str] = Field(None, description="Custom output path")


@app.post("/slice/lego", response_model=SliceResponse)
async def slice_lego_stl(request: LegoSliceRequest):
    """
    Slice a LEGO brick STL with optimized settings.

    Uses fine quality with settings tuned for LEGO dimensions.
    """
    # Convert to standard slice request with LEGO-optimized settings
    standard_request = SliceRequest(
        stl_path=request.stl_path,
        printer=request.printer,
        material="pla",  # PLA is best for LEGO
        quality="fine",  # Fine quality for precision
        infill_percent=20,  # 20% infill for bricks
        output_filename=request.output_path.split("/")[-1] if request.output_path else None,
    )
    return await slice_stl(standard_request)


@app.post("/slice", response_model=SliceResponse)
async def slice_stl(request: SliceRequest):
    """
    Slice an STL file and generate G-code.

    Uses CuraEngine CLI with LEGO-optimized settings.
    """
    # Validate input file
    stl_path = Path(request.stl_path)
    if not stl_path.exists():
        raise HTTPException(404, f"STL file not found: {request.stl_path}")

    # Check if CuraEngine is available
    if not is_curaengine_available():
        return SliceResponse(
            success=False,
            error="CuraEngine not available. Please check installation.",
        )

    # Get quality settings
    quality = QUALITY_PRESETS.get(request.quality, QUALITY_PRESETS["fine"])
    material = MATERIAL_SETTINGS.get(request.material, MATERIAL_SETTINGS["pla"])

    # Create config file
    config_path = create_cura_config(quality, material, request.infill_percent)

    # Determine output path
    if request.output_filename:
        output_name = request.output_filename
    else:
        base_name = stl_path.stem
        output_name = f"{base_name}_{request.printer}_{request.quality}.gcode"

    output_path = OUTPUT_DIR / output_name

    # Build CuraEngine command
    # CuraEngine slice -j config.json -o output.gcode -l input.stl
    cmd = [
        "CuraEngine",
        "slice",
        "-j", str(config_path),
        "-o", str(output_path),
        "-l", str(stl_path),
    ]

    try:
        # Run CuraEngine
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return SliceResponse(success=False, error=f"CuraEngine failed: {error_msg}")

        if not output_path.exists():
            return SliceResponse(success=False, error="G-code file was not created")

        # Parse G-code for statistics
        stats = parse_gcode_stats(output_path)

        return SliceResponse(
            success=True,
            path=str(output_path),
            estimated_time_min=stats.get("time_min", 0),
            filament_meters=stats.get("filament_m", 0),
            filament_grams=stats.get("filament_g", 0),
            layer_height_mm=quality["layer_height"],
            layer_count=stats.get("layers", 0),
        )

    except subprocess.TimeoutExpired:
        return SliceResponse(success=False, error="Slicing timed out (>5 minutes)")
    except Exception as e:
        return SliceResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766)
