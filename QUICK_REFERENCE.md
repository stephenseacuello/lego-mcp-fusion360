# LEGO MCP Quick Reference Card

## Quick Start

```bash
# 1. Setup (one time)
cd lego-mcp-fusion360
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start Dashboard
cd dashboard && python app.py

# 3. Open Browser
open http://localhost:5000
```

---

## Dashboard URLs

### Core Pages
| Page | URL |
|------|-----|
| Home | http://localhost:5000 |
| Catalog | http://localhost:5000/catalog |
| Collection | http://localhost:5000/collection |

### Manufacturing
| Page | URL |
|------|-----|
| Shop Floor | http://localhost:5000/api/manufacturing/shop-floor/page |
| Work Orders | http://localhost:5000/api/mes/work-orders/page |
| WIP Tracking | http://localhost:5000/api/mes/work-orders/wip/page |
| OEE | http://localhost:5000/api/manufacturing/oee/page |

### Quality
| Page | URL |
|------|-----|
| Quality Dashboard | http://localhost:5000/api/quality/dashboard/page |
| SPC | http://localhost:5000/api/quality/spc/page |
| FMEA | http://localhost:5000/api/quality/fmea/page |
| House of Quality | http://localhost:5000/api/quality/qfd/hoq/page |

### ERP & Finance
| Page | URL |
|------|-----|
| Vendors | http://localhost:5000/api/erp/vendors/page |
| Accounts Receivable | http://localhost:5000/api/erp/financials/ar/page |
| Accounts Payable | http://localhost:5000/api/erp/financials/ap/page |
| General Ledger | http://localhost:5000/api/erp/financials/gl/page |
| Customer Orders | http://localhost:5000/api/erp/orders/page |
| BOM | http://localhost:5000/api/erp/bom/page |
| Costing | http://localhost:5000/api/erp/costing/page |

### Planning & AI
| Page | URL |
|------|-----|
| Materials/MRP | http://localhost:5000/api/mrp/materials/page |
| Scheduling | http://localhost:5000/api/scheduling/dashboard/page |
| AI Copilot | http://localhost:5000/api/ai/copilot/page |
| Digital Twin | http://localhost:5000/api/twin/dashboard/page |
| Sustainability | http://localhost:5000/api/sustainability/dashboard/page |

---

## Essential API Commands

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Brick Catalog
```bash
# List categories
curl http://localhost:5000/api/catalog/categories

# Get bricks
curl http://localhost:5000/api/catalog/bricks?category=basic
```

### Work Orders
```bash
# Create
curl -X POST http://localhost:5000/api/manufacturing/work-orders \
  -H "Content-Type: application/json" \
  -d '{"part_id": "brick-2x4", "quantity": 100, "priority": "high"}'

# List
curl http://localhost:5000/api/manufacturing/work-orders

# Release
curl -X POST http://localhost:5000/api/manufacturing/work-orders/WO-001/release

# Start
curl -X POST http://localhost:5000/api/manufacturing/work-orders/WO-001/start

# Complete
curl -X POST http://localhost:5000/api/manufacturing/work-orders/WO-001/complete \
  -d '{"quantity_completed": 98}'
```

### Quality Inspection
```bash
# Inspect image
curl -X POST http://localhost:5000/api/quality/inspect \
  -d '{"image_path": "/path/to/image.jpg"}'

# SPC data
curl http://localhost:5000/api/quality/spc/brick-2x4?parameter=stud_diameter

# Record measurement
curl -X POST http://localhost:5000/api/quality/spc/measurements \
  -d '{"part_id": "brick-2x4", "parameter": "stud_diameter", "value": 4.81}'
```

### Vendors
```bash
# Create vendor
curl -X POST http://localhost:5000/api/erp/vendors \
  -d '{"code": "SUP001", "name": "Acme", "vendor_type": "raw_material"}'

# List vendors
curl http://localhost:5000/api/erp/vendors

# Get scorecard
curl http://localhost:5000/api/erp/vendors/SUP001/scorecard
```

### Financials
```bash
# Create invoice
curl -X POST http://localhost:5000/api/erp/financials/ar/invoices \
  -d '{"customer_id": "C001", "line_items": [{"description": "Bricks", "quantity": 100, "unit_price": 0.25}]}'

# Create bill
curl -X POST http://localhost:5000/api/erp/financials/ap/bills \
  -d '{"vendor_id": "SUP001", "line_items": [{"description": "Material", "quantity": 10, "unit_price": 25}]}'

# Trial balance
curl http://localhost:5000/api/erp/financials/gl/trial-balance
```

### AI Copilot
```bash
# Query
curl -X POST http://localhost:5000/api/ai/copilot/query \
  -d '{"query": "Optimize print parameters for 2x4 brick"}'

# Root cause analysis
curl -X POST http://localhost:5000/api/ai/causal/root-cause \
  -d '{"defect_type": "warping", "part_id": "brick-2x4"}'
```

---

## Environment Variables

```bash
# Core
export FLASK_ENV=development
export FLASK_PORT=5000
export SECRET_KEY=your-secret-key

# Database
export DATABASE_URL=sqlite:///lego_mcp.db

# Vision
export DETECTION_BACKEND=mock  # mock, yolo, roboflow

# Printer (Moonraker/Klipper)
export PRINTER_PROTOCOL=moonraker
export PRINTER_HOST=192.168.1.100
export PRINTER_PORT=7125

# Printer (Bambu Lab)
export PRINTER_PROTOCOL=bambu
export BAMBU_DEVICE_ID=your-device-id
export BAMBU_ACCESS_TOKEN=your-token
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `S` | Scan page |
| `C` | Collection |
| `B` | Builds |
| `W` | Workspace |
| `/` | Search |
| `?` | Help |
| `Esc` | Close modal |

---

## Docker

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f

# Shell
docker-compose exec dashboard bash
```

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Unit only
python -m pytest tests/unit/ -v

# With coverage
python -m pytest tests/ --cov=dashboard
```

---

## Troubleshooting

```bash
# Reset database
rm dashboard/lego_mcp.db
cd dashboard && flask db upgrade

# Kill port 5000
lsof -ti:5000 | xargs kill -9

# Check logs
tail -f dashboard/logs/app.log

# Use mock vision
export DETECTION_BACKEND=mock
```

---

## File Structure

```
lego-mcp-fusion360/
├── dashboard/           # Flask web application
│   ├── app.py          # Main entry point
│   ├── routes/         # API endpoints
│   ├── services/       # Business logic
│   ├── models/         # Database models
│   └── templates/      # HTML templates
├── fusion360-addin/    # Fusion 360 integration
├── mcp-server/         # MCP protocol server
├── slicer-service/     # G-code generation
├── tests/              # Test suite
└── docs/               # Documentation
```

---

*LEGO MCP v6.0 - Quick Reference*
