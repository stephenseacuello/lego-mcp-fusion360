# Getting Started with LegoMCP v5.0

## World-Class Manufacturing Platform for LEGO-Compatible Brick Production

Welcome to LegoMCP, a PhD-level cyber-physical production system (CPPS) that combines Fusion 360 CAD, 3D printing, and Industry 4.0 manufacturing capabilities.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Architecture Overview](#architecture-overview)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Start All Services (Docker Compose)

```bash
# Clone and navigate to project
cd lego_mcp_fusion360

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 2. Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: http://localhost:5000
- **API Docs**: http://localhost:5000/api/health

### 3. Create Your First Brick

```bash
# Using curl
curl -X POST http://localhost:5000/api/brick/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_first_brick",
    "width_studs": 2,
    "depth_studs": 4,
    "height_plates": 3,
    "brick_type": "standard"
  }'
```

Or use Python:

```python
import requests

response = requests.post('http://localhost:5000/api/brick/create', json={
    'name': 'my_first_brick',
    'width_studs': 2,
    'depth_studs': 4,
    'height_plates': 3,
    'brick_type': 'standard'
})

print(response.json())
```

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, macOS 10.15+, Linux |
| **RAM** | 8 GB |
| **Storage** | 10 GB free space |
| **Docker** | Docker Desktop 4.0+ |
| **Python** | 3.9+ (for development) |

### For Fusion 360 Integration

| Component | Requirement |
|-----------|-------------|
| **Fusion 360** | Latest version with API access |
| **License** | Personal, Startup, or Commercial |

### For AI/Vision Features

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GPU with CUDA 11.8+ (optional) |
| **VRAM** | 4 GB+ for YOLO11 inference |

---

## Installation

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/lego_mcp_fusion360.git
cd lego_mcp_fusion360

# 2. Create environment file
cp .env.example .env

# 3. Start services
docker-compose up -d

# 4. Check status
docker-compose ps
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Redis (required)
docker run -d -p 6379:6379 redis:alpine

# 4. Start the dashboard
cd dashboard
python app.py
```

### Option 3: Fusion 360 Add-in Only

1. Open Fusion 360
2. Go to **Tools > ADD-INS > Scripts and Add-Ins**
3. Click **Add-Ins** tab, then **+** (green plus)
4. Navigate to `fusion360-addin/LegoMCP`
5. Select the folder and click **Open**
6. Check **Run on Startup** and click **Run**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LegoMCP v5.0 Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Fusion 360 │  │   Dashboard  │  │    Slicer    │          │
│  │   Add-in     │  │   (Flask)    │  │   Service    │          │
│  │   :8765      │  │   :5000      │  │   :8081      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────┬────┴────────────────┘                   │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │  MCP Server   │                                  │
│              │  (Protocol)   │                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│  ┌───────────────────┼───────────────────┐                     │
│  │          Services Layer               │                     │
│  ├───────────────────────────────────────┤                     │
│  │  Manufacturing │ Quality │ Scheduling │                     │
│  │  Digital Twin  │ ERP     │ Supply Chain│                    │
│  │  Sustainability│ Compliance│ AI/ML    │                     │
│  └───────────────────────────────────────┘                     │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │    Redis      │                                  │
│              │   (Events)    │                                  │
│              └───────────────┘                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | 5000 | Web UI and REST API |
| Fusion 360 Add-in | 8765 | CAD operations |
| Slicer Service | 8081 | 3D print slicing |
| Redis | 6379 | Event streaming |
| PostgreSQL | 5432 | Database (optional) |

---

## Basic Usage

### 1. Creating Bricks

#### Standard Brick (2x4)

```python
import requests

# Create a classic 2x4 LEGO brick
response = requests.post('http://localhost:5000/api/brick/create', json={
    'width_studs': 2,
    'depth_studs': 4,
    'height_plates': 3,  # Standard brick height
    'brick_type': 'standard',
    'hollow': True,
    'studs': True,
    'tubes': True
})

brick = response.json()
print(f"Created: {brick['component_name']}")
print(f"Volume: {brick['volume_mm3']} mm³")
```

#### Plate (1/3 height)

```python
response = requests.post('http://localhost:5000/api/brick/create', json={
    'width_studs': 4,
    'depth_studs': 4,
    'height_plates': 1,  # Plate = 1 plate height
    'brick_type': 'plate'
})
```

#### Slope Brick

```python
response = requests.post('http://localhost:5000/api/brick/create', json={
    'width_studs': 2,
    'depth_studs': 3,
    'brick_type': 'slope',
    'slope_angle': 45,
    'slope_direction': 'front'
})
```

#### Technic Brick with Holes

```python
response = requests.post('http://localhost:5000/api/brick/create', json={
    'width_studs': 1,
    'depth_studs': 8,
    'brick_type': 'technic',
    'technic_holes': True
})
```

### 2. Exporting Bricks

```python
# Export to STL for 3D printing
response = requests.post('http://localhost:5000/api/export/stl', json={
    'component': 'Brick_2x4',
    'output_path': '/output/brick_2x4.stl',
    'refinement': 'high'
})

result = response.json()
print(f"Exported to: {result['path']}")
print(f"File size: {result['size_kb']} KB")
```

### 3. Using the Dashboard

1. **Home**: Overview of system status
2. **Builder**: Visual brick creation interface
3. **Catalog**: Browse 323+ brick types
4. **Collection**: Manage your brick inventory
5. **Workspace**: Vision-based brick detection
6. **MES**: Manufacturing execution
7. **Quality**: SPC charts, inspections
8. **Scheduling**: Production optimization

---

## Advanced Features

### Manufacturing Execution System (MES)

```python
# Create a work order
response = requests.post('http://localhost:5000/api/mes/work-orders', json={
    'part_id': 'BRICK-2X4-RED',
    'quantity': 100,
    'priority': 'high',
    'due_date': '2024-12-31T00:00:00Z'
})

work_order = response.json()
print(f"Work Order: {work_order['id']}")

# Release to production
requests.post(f"http://localhost:5000/api/mes/work-orders/{work_order['id']}/release")

# Get OEE metrics
oee = requests.get('http://localhost:5000/api/mes/oee/WC-PRINT-01?period=7d').json()
print(f"OEE: {oee['oee']}%")
```

### Quality Management

```python
# Add SPC data point
response = requests.post('http://localhost:5000/api/quality/spc/data-point', json={
    'chart_id': 'BRICK-2X4-HEIGHT',
    'value': 9.62,
    'sample_size': 5
})

result = response.json()
if not result['in_control']:
    print(f"⚠️ Out of control: {result['violation']}")

# Run AI defect detection
with open('part_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/quality/vision/detect',
        files={'image': f}
    )

defects = response.json()
print(f"Defects found: {defects['defect_count']}")
```

### Advanced Scheduling

```python
# Optimize schedule with CP-SAT
response = requests.post('http://localhost:5000/api/scheduling/optimize/cp-sat', json={
    'work_orders': ['WO-001', 'WO-002', 'WO-003'],
    'objective': 'makespan',
    'time_limit_seconds': 60
})

schedule = response.json()
print(f"Makespan: {schedule['makespan']} hours")
print(f"Optimal: {schedule['optimal']}")

# Multi-objective optimization with NSGA-II
response = requests.post('http://localhost:5000/api/scheduling/optimize/nsga2', json={
    'work_orders': ['WO-001', 'WO-002', 'WO-003'],
    'objectives': ['makespan', 'tardiness', 'energy'],
    'population_size': 100,
    'generations': 50
})

pareto = response.json()
print(f"Pareto solutions: {len(pareto['solutions'])}")
```

### AI Copilot

```python
# Ask manufacturing questions
response = requests.post('http://localhost:5000/api/ai/ask', json={
    'question': 'Why is the defect rate high on printer WC-PRINT-02 today?',
    'include_context': True,
    'explain': True
})

answer = response.json()
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence']}%")

# Get XAI explanation
if 'xai' in answer:
    for feature in answer['xai']['feature_importance']:
        print(f"  {feature['feature']}: {feature['importance']:.2f}")
```

### Sustainability Tracking

```python
# Calculate carbon footprint
response = requests.post('http://localhost:5000/api/sustainability/carbon/footprint', json={
    'work_order': 'WO-001',
    'include_scope3': True
})

footprint = response.json()
print(f"Total CO2e: {footprint['total_kg_co2e']} kg")
print(f"  Scope 1: {footprint['scope1']} kg")
print(f"  Scope 2: {footprint['scope2']} kg")
print(f"  Scope 3: {footprint['scope3']} kg")

# Get LCA impact
lca = requests.get('http://localhost:5000/api/sustainability/lca/BRICK-2X4/impact').json()
print(f"GWP: {lca['impact_categories']['gwp']['value']} kg CO2-eq")
```

### Real-Time Events (WebSocket)

```javascript
// JavaScript client
const socket = io('http://localhost:5000/manufacturing');

socket.on('connect', () => {
    console.log('Connected to manufacturing events');
    socket.emit('subscribe', {
        topics: ['mes:WC-PRINT-01', 'quality:defects', 'copilot:insights']
    });
});

socket.on('mes:oee_update', (data) => {
    console.log(`OEE Update: ${data.work_center} - ${data.oee}%`);
});

socket.on('quality:defect_detected', (data) => {
    console.log(`Defect detected: ${data.type} (${data.confidence}%)`);
});
```

---

## API Reference

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/brick/create` | POST | Create a brick |
| `/api/catalog` | GET | List brick catalog |
| `/api/mes/work-orders` | GET/POST | Manage work orders |
| `/api/quality/spc/data-point` | POST | Add SPC measurement |
| `/api/scheduling/optimize/cp-sat` | POST | Optimize schedule |
| `/api/ai/ask` | POST | Ask AI copilot |
| `/api/sustainability/carbon/footprint` | POST | Calculate carbon |

### Full Documentation

- **OpenAPI Spec**: [docs/openapi.yaml](docs/openapi.yaml)
- **API Reference**: [docs/API.md](docs/API.md)
- **Developer Guide**: [docs/DEVELOPER.md](docs/DEVELOPER.md)

---

## Troubleshooting

### Common Issues

#### 1. Docker services not starting

```bash
# Check logs
docker-compose logs dashboard
docker-compose logs slicer

# Restart services
docker-compose restart
```

#### 2. Fusion 360 not connecting

1. Ensure Fusion 360 is running
2. Check the add-in is loaded (Tools > ADD-INS)
3. Verify port 8765 is not blocked

```bash
# Test Fusion 360 connection
curl http://127.0.0.1:8765/health
```

#### 3. Redis connection failed

```bash
# Start Redis if not running
docker run -d -p 6379:6379 --name redis redis:alpine

# Or check if already running
docker ps | grep redis
```

#### 4. AI/Vision features not working

```bash
# Check if YOLO model is downloaded
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# For GPU support
nvidia-smi  # Check GPU availability
```

### Getting Help

- **Documentation**: [docs/](docs/)
- **Issues**: Open a GitHub issue
- **IEEE Paper**: [docs/ieee_paper/main.tex](docs/ieee_paper/main.tex)

---

## Next Steps

1. **Explore the Dashboard**: http://localhost:5000
2. **Create Custom Bricks**: Use the Builder interface
3. **Set Up Production**: Configure work centers and routings
4. **Enable Quality Control**: Set up SPC charts and inspection plans
5. **Optimize Scheduling**: Use CP-SAT or NSGA-II for production planning
6. **Track Sustainability**: Monitor carbon footprint and LCA metrics

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/benchmarks/ -v              # Performance benchmarks

# Run with coverage
pytest tests/ --cov=dashboard --cov-report=html
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**LegoMCP v5.0** - World-Class Manufacturing for LEGO-Compatible Bricks
