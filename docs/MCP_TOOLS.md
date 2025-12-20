# MCP Tools Reference

Complete reference for all 33 MCP tools available in LEGO MCP Studio.

---

## Overview

MCP (Model Context Protocol) tools allow Claude to interact with the LEGO MCP system. When connected via Claude Desktop, you can use natural language to invoke these tools.

---

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| Brick Creation | 8 | Create LEGO bricks |
| Export | 5 | Export to various formats |
| 3D Printing | 5 | Slicing and print prep |
| CNC Milling | 4 | CAM operations |
| History | 4 | Undo/redo operations |
| Batch | 4 | Bulk operations |
| Utility | 3 | System utilities |

---

## Brick Creation Tools

### create_brick

Create a standard LEGO brick.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| width | int | Yes | Width in studs (1-48) |
| depth | int | Yes | Depth in studs (1-48) |
| height | int | No | Height in plates (default: 3) |
| hollow | bool | No | Hollow interior (default: true) |

**Example:**
```
Create a 2x4 brick
```

**Response:**
```json
{
  "brick_id": "brick_2x4_h3",
  "dimensions": {"width": 2, "depth": 4, "height": 3},
  "size_mm": {"x": 16, "y": 32, "z": 9.6}
}
```

---

### create_plate

Create a LEGO plate (⅓ brick height).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| width | int | Yes | Width in studs |
| depth | int | Yes | Depth in studs |

**Example:**
```
Create a 4x4 plate
```

---

### create_tile

Create a LEGO tile (plate without studs).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| width | int | Yes | Width in studs |
| depth | int | Yes | Depth in studs |

**Example:**
```
Create a 2x2 tile
```

---

### create_slope

Create a slope brick.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| width | int | Yes | Width in studs |
| depth | int | Yes | Depth in studs |
| angle | int | Yes | Slope angle (18, 33, 45, 65, 75) |
| inverted | bool | No | Inverted slope (default: false) |

**Example:**
```
Create a 45-degree slope, 2 studs wide and 3 deep
```

---

### create_technic

Create a Technic brick or beam.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| length | int | Yes | Length in studs |
| type | string | No | "brick" or "beam" |
| holes | bool | No | Include holes (default: true) |

**Example:**
```
Create a Technic beam with 8 holes
```

---

### create_custom

Create a custom brick with advanced features.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| width | int | Yes | Width in studs |
| depth | int | Yes | Depth in studs |
| height | int | No | Height in plates |
| features | array | No | List of features |
| hollow | bool | No | Hollow interior |
| chamfer | bool | No | Chamfered edges |

**Features:**
- `ball_joint_top`: Ball joint on top
- `ball_joint_side`: Ball joint on side
- `socket_top`: Socket on top
- `socket_side`: Socket on side
- `clip_horizontal`: Horizontal clip
- `clip_vertical`: Vertical clip
- `hinge_male`: Hinge (male)
- `hinge_female`: Hinge (female)
- `pattern_grille`: Grille pattern
- `pattern_vent`: Vent pattern

**Example:**
```
Create a 2x2 brick with a ball joint on top and a clip on the side
```

---

### list_catalog

Browse the brick catalog.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| category | string | No | Filter by category |
| search | string | No | Search term |
| limit | int | No | Max results (default: 20) |

**Example:**
```
Show me all slope bricks
```

---

### get_brick_details

Get detailed information about a brick.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_id | string | Yes | Brick identifier |

**Example:**
```
Tell me about the 2x4 brick
```

---

## Export Tools

### export_stl

Export to STL format for 3D printing.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_id | string | Yes | Brick to export |
| quality | string | No | "low", "medium", "high" |
| output_path | string | No | Custom output path |

**Example:**
```
Export the 2x4 brick to STL at high quality
```

---

### export_step

Export to STEP format for CAD.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_id | string | Yes | Brick to export |

**Example:**
```
Export as STEP file
```

---

### export_3mf

Export to 3MF format.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_id | string | Yes | Brick to export |
| color | string | No | Color for the model |

---

### export_batch

Export multiple bricks at once.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_ids | array | Yes | List of brick IDs |
| format | string | No | Export format |
| output_dir | string | No | Output directory |

**Example:**
```
Export all my bricks to STL
```

---

### list_export_formats

List available export formats.

**Example:**
```
What export formats are available?
```

---

## 3D Printing Tools

### generate_print_config

Generate slicer configuration.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| stl_path | string | Yes | Path to STL file |
| printer | string | Yes | Printer profile |
| material | string | No | Material (default: PLA) |

**Supported Printers:**
- prusa_mk3s, prusa_mk4, prusa_xl
- bambu_x1c, bambu_p1s, bambu_a1
- voron_24, voron_02
- ender3_v2, ender3_s1
- ratrig_vcore3
- anycubic_kobra2
- qidi_xmax3
- flashforge_adv5m

**Example:**
```
Generate print settings for my Prusa MK3S with PLA
```

---

### slice_for_print

Slice STL and generate G-code.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| stl_path | string | Yes | Path to STL file |
| printer | string | Yes | Printer profile |
| material | string | No | Material |
| supports | bool | No | Generate supports |
| infill | int | No | Infill percentage |

**Example:**
```
Slice the brick for my Bambu X1C with 20% infill
```

---

### estimate_print_time

Estimate printing time.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| stl_path | string | Yes | Path to STL |
| printer | string | Yes | Printer profile |

---

### list_printers

List available printer profiles.

**Example:**
```
What printers are supported?
```

---

### list_materials

List available materials.

**Example:**
```
What materials can I use?
```

---

## CNC Milling Tools

### generate_milling_operations

Generate CNC toolpaths.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_id | string | Yes | Brick to mill |
| machine | string | Yes | CNC machine |
| material | string | No | Stock material |

**Supported Machines:**
- haas_vf2
- tormach_1100mx
- datron_neo
- shapeoko_4
- nomad_3
- bantam_desktop
- roland_srm20
- grbl_generic

**Example:**
```
Generate CNC operations for the 2x4 brick on my Shapeoko
```

---

### calculate_cutting_params

Calculate speeds and feeds.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| tool | string | Yes | Cutting tool |
| material | string | Yes | Workpiece material |
| operation | string | Yes | Operation type |

---

### list_machines

List available CNC machines.

---

### list_cutting_tools

List available cutting tools.

**Available Tools:**
- flat_endmill_3mm, flat_endmill_6mm
- ball_endmill_3mm, ball_endmill_6mm
- drill_2mm, drill_3mm, drill_4.8mm
- chamfer_45deg, chamfer_90deg
- face_mill_25mm
- thread_mill_m3
- engraver_30deg
- roughing_endmill_6mm

---

## History Tools

### undo

Undo the last operation.

**Example:**
```
Undo
```

---

### redo

Redo the last undone operation.

**Example:**
```
Redo
```

---

### get_history

Get operation history.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| limit | int | No | Number of entries |
| session_id | string | No | Filter by session |

**Example:**
```
Show my recent operations
```

---

### get_session_statistics

Get statistics for current session.

**Example:**
```
How many bricks have I created?
```

---

## Batch Tools

### batch_create_bricks

Create multiple bricks at once.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| bricks | array | Yes | List of brick specs |

**Example:**
```
Create a 1x1, 1x2, 2x2, and 2x4 brick
```

---

### batch_export_stl

Export multiple bricks to STL.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_ids | array | Yes | Bricks to export |
| quality | string | No | Export quality |

**Example:**
```
Export all my bricks as STL files
```

---

### generate_brick_set

Generate a predefined set of bricks.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| set_name | string | Yes | Set name |

**Available Sets:**
- basic: 1x1 through 2x8 bricks
- plates: Common plates
- slopes: All slope angles
- technic: Basic Technic pieces
- starter: Good starting collection

**Example:**
```
Generate the basic brick set
```

---

### clear_session

Clear all bricks from current session.

---

## Utility Tools

### health_check

Check system health and connectivity.

**Example:**
```
Is everything working?
```

---

### get_server_info

Get server version and capabilities.

---

### generate_preview

Generate preview image of a brick.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| brick_id | string | Yes | Brick to preview |
| angle | string | No | Camera angle |
| size | int | No | Image size |

---

## Claude Desktop Setup

### Configuration

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lego-mcp": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/lego-mcp-fusion360/mcp-server",
      "env": {
        "FUSION_API_URL": "http://localhost:8080",
        "SLICER_URL": "http://localhost:8081"
      }
    }
  }
}
```

### Testing Connection

After restarting Claude Desktop:

```
Can you create a simple 2x2 brick?
```

If working, you'll get a response with brick details.

---

## Natural Language Examples

Claude understands various ways to request the same thing:

**Creating Bricks:**
- "Create a 2x4 brick"
- "Make a 2 by 4 LEGO brick"
- "I need a standard 2x4"

**Slopes:**
- "Create a 45-degree slope"
- "Make a roof piece, 2 wide, 3 deep"
- "I need a 45° 2x3 slope"

**Technic:**
- "Create a Technic beam with 8 holes"
- "Make a 1x8 Technic brick"
- "I need an 8-long beam"

**Export:**
- "Export to STL"
- "Save as an STL file"
- "Get the STL for printing"

**Printing:**
- "Slice for my Prusa"
- "Generate G-code for Prusa MK3S"
- "Prepare for 3D printing on my Prusa"

---

## Error Handling

Tools return errors in this format:

```json
{
  "error": true,
  "message": "Invalid brick dimensions: width must be 1-48",
  "code": "INVALID_PARAMS"
}
```

Common error codes:
- `INVALID_PARAMS`: Invalid parameters
- `BRICK_NOT_FOUND`: Brick doesn't exist
- `FUSION_UNAVAILABLE`: Fusion 360 not connected
- `SLICER_UNAVAILABLE`: Slicer service not running
- `EXPORT_FAILED`: Export operation failed
