# Developer Guide

Contributing to and extending LEGO MCP Studio.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Desktop                          │
│                           │                                  │
│                      MCP Protocol                            │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      MCP Server                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Brick Tools │  │ Export Tools │  │ Printing Tools  │    │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘    │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
┌─────────▼────────┐  ┌────▼────┐  ┌──────────▼──────────┐
│   Fusion 360     │  │  Files  │  │   Slicer Service    │
│   Add-in         │  │  System │  │   (PrusaSlicer)     │
└──────────────────┘  └─────────┘  └─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Flask Dashboard                          │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Workspace│  │ Collection │  │  Builds  │  │ Insights │  │
│  └────┬─────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │             │         │
│  ┌────▼──────────────▼──────────────▼─────────────▼─────┐  │
│  │                    Services Layer                     │  │
│  │  ┌─────────┐  ┌───────────┐  ┌────────┐  ┌────────┐  │  │
│  │  │ Vision  │  │ Inventory │  │ Builds │  │ Bridge │  │  │
│  │  └─────────┘  └───────────┘  └────────┘  └────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
lego-mcp-fusion360/
├── dashboard/                    # Flask Web Application
│   ├── app.py                   # Application factory
│   ├── config.py                # Configuration
│   ├── routes/                  # Route blueprints
│   │   ├── __init__.py
│   │   ├── main.py             # Home page
│   │   ├── workspace.py        # Digital twin
│   │   ├── scan.py             # Bulk scanning
│   │   ├── collection.py       # Inventory
│   │   ├── builds_routes.py    # Build planner
│   │   ├── insights.py         # Analytics
│   │   ├── catalog.py          # Brick catalog
│   │   ├── builder.py          # Custom builder
│   │   ├── files.py            # File browser
│   │   ├── history.py          # Operation history
│   │   ├── tools.py            # MCP tools
│   │   ├── status.py           # Service status
│   │   ├── settings.py         # Configuration
│   │   └── api.py              # REST API
│   ├── services/                # Business logic
│   │   ├── vision/             # Detection system
│   │   │   ├── __init__.py
│   │   │   ├── detector.py     # YOLO/Roboflow
│   │   │   └── camera_manager.py
│   │   ├── inventory/          # Inventory management
│   │   │   ├── __init__.py
│   │   │   ├── inventory_manager.py
│   │   │   └── workspace_state.py
│   │   ├── builds/             # Build planning
│   │   │   ├── __init__.py
│   │   │   └── build_planner.py
│   │   ├── catalog_service.py
│   │   ├── mcp_bridge.py
│   │   ├── builder_service.py
│   │   ├── file_service.py
│   │   └── status_service.py
│   ├── templates/               # Jinja2 templates
│   │   ├── base.html
│   │   ├── pages/
│   │   └── errors/
│   ├── static/                  # Static assets
│   │   ├── css/main.css
│   │   ├── js/app.js
│   │   └── vendor/
│   └── websocket/               # WebSocket handlers
│       ├── __init__.py
│       └── events.py
│
├── mcp-server/                   # MCP Server
│   ├── src/
│   │   ├── server.py           # Main entry
│   │   ├── tools/              # Tool definitions
│   │   ├── fusion_client.py
│   │   ├── slicer_client.py
│   │   ├── history_manager.py
│   │   ├── batch_operations.py
│   │   └── error_recovery.py
│   └── requirements.txt
│
├── fusion360-addin/              # Fusion 360 Add-in
│   └── LegoMCP/
│       ├── LegoMCP.py
│       ├── LegoMCP.manifest
│       ├── core/
│       ├── api/
│       └── ui/
│
├── slicer-service/               # Slicing Service
│   ├── src/slicer_api.py
│   └── profiles/
│
├── shared/                       # Shared modules
│   ├── lego_specs.py
│   ├── brick_catalog.py
│   ├── brick_catalog_extended.py
│   ├── custom_brick_builder.py
│   ├── advanced_features.py
│   └── validation.py
│
├── tests/                        # Test suite
├── docs/                         # Documentation
└── output/                       # Generated files
```

---

## Setting Up Development Environment

### Prerequisites

- Python 3.9+
- Git
- Node.js (optional, for JS tooling)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r dashboard/requirements.txt
pip install -r tests/requirements-test.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running in Development Mode

```bash
cd dashboard
FLASK_ENV=development python app.py
```

This enables:
- Debug mode
- Auto-reload on file changes
- Detailed error pages

---

## Code Style

### Python

Follow PEP 8 with these specifics:

```python
# Imports: stdlib, third-party, local
import os
import sys

from flask import Flask, render_template
import requests

from services.vision import get_detector

# Classes: PascalCase
class BrickDetector:
    pass

# Functions/variables: snake_case
def detect_bricks(frame):
    detection_result = []
    return detection_result

# Constants: UPPER_SNAKE_CASE
MAX_DETECTION_COUNT = 100

# Type hints encouraged
def add_brick(brick_id: str, quantity: int = 1) -> bool:
    return True
```

### JavaScript

```javascript
// Use const/let, not var
const detector = new BrickDetector();
let currentFrame = null;

// camelCase for variables and functions
function updateWorkspace() {
    const brickCount = getBrickCount();
}

// PascalCase for classes
class WorkspaceManager {
    constructor() {
        this.bricks = [];
    }
}
```

### HTML/CSS

```html
<!-- Use semantic HTML -->
<main class="workspace-container">
    <section class="camera-panel">
        <!-- Content -->
    </section>
</main>
```

```css
/* BEM-like naming */
.brick-card { }
.brick-card__header { }
.brick-card--selected { }

/* CSS custom properties for theming */
:root {
    --color-primary: #e3000b;
}
```

---

## Adding a New Feature

### 1. Create Route

```python
# dashboard/routes/my_feature.py
from flask import Blueprint, render_template, jsonify

bp = Blueprint('my_feature', __name__, url_prefix='/my-feature')

@bp.route('/')
def index():
    return render_template('pages/my_feature.html')

@bp.route('/api/data')
def get_data():
    return jsonify({'data': 'value'})
```

### 2. Register Blueprint

```python
# dashboard/app.py
from routes.my_feature import bp as my_feature_bp
app.register_blueprint(my_feature_bp)
```

### 3. Create Template

```html
<!-- dashboard/templates/pages/my_feature.html -->
{% extends "base.html" %}

{% block title %}My Feature{% endblock %}

{% block content %}
<div class="page-header">
    <h1>My Feature</h1>
</div>
<!-- Content here -->
{% endblock %}
```

### 4. Add Navigation

```html
<!-- dashboard/templates/base.html -->
<li class="nav-item">
    <a href="{{ url_for('my_feature.index') }}">
        <span class="nav-icon">🆕</span>
        <span class="nav-text">My Feature</span>
    </a>
</li>
```

### 5. Write Tests

```python
# tests/test_my_feature.py
def test_my_feature_page(client):
    response = client.get('/my-feature/')
    assert response.status_code == 200
```

---

## Adding an MCP Tool

### 1. Define Tool

```python
# mcp-server/src/tools/my_tools.py
from typing import Dict, Any

def register_tools(server):
    @server.tool()
    async def my_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
        """
        Description of what this tool does.
        
        Args:
            param1: Description of param1
            param2: Description of param2 (default: 10)
        
        Returns:
            Dictionary with result
        """
        result = do_something(param1, param2)
        return {"success": True, "data": result}
```

### 2. Register in Server

```python
# mcp-server/src/server.py
from tools.my_tools import register_tools as register_my_tools

register_my_tools(server)
```

---

## Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_phase2_digital_twin.py -v

# With coverage
python -m pytest tests/ --cov=dashboard --cov-report=html
```

### Writing Tests

```python
# tests/test_example.py
import pytest

class TestMyFeature:
    @pytest.fixture
    def client(self):
        from dashboard.app import create_app
        app = create_app('testing')
        with app.test_client() as client:
            yield client
    
    def test_page_loads(self, client):
        response = client.get('/my-feature/')
        assert response.status_code == 200
    
    def test_api_returns_data(self, client):
        response = client.get('/my-feature/api/data')
        data = response.get_json()
        assert data['success'] == True
```

---

## Database (Future)

Currently using JSON file storage. Migration to SQLite planned:

### Planned Schema

```sql
-- Users
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE,
    created_at TIMESTAMP
);

-- Inventory
CREATE TABLE inventory (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    brick_id TEXT,
    color TEXT,
    quantity INTEGER,
    added_at TIMESTAMP
);

-- Builds
CREATE TABLE builds (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name TEXT,
    description TEXT,
    created_at TIMESTAMP
);

-- Build Parts
CREATE TABLE build_parts (
    build_id INTEGER REFERENCES builds(id),
    brick_id TEXT,
    color TEXT,
    quantity INTEGER
);
```

---

## API Design Principles

1. **RESTful URLs**
   - `GET /collection/` - List
   - `GET /collection/123` - Get one
   - `POST /collection/` - Create
   - `PUT /collection/123` - Update
   - `DELETE /collection/123` - Delete

2. **Consistent Response Format**
   ```json
   {
     "success": true,
     "data": { },
     "error": null
   }
   ```

3. **Pagination**
   ```
   GET /collection/?page=1&per_page=50
   ```

4. **Filtering**
   ```
   GET /collection/?category=brick&color=red
   ```

---

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests: `python -m pytest tests/`
5. Commit: `git commit -m 'Add my feature'`
6. Push: `git push origin feature/my-feature`
7. Open Pull Request

### Commit Messages

```
feat: Add new brick detection algorithm
fix: Correct color classification for trans-clear
docs: Update API documentation
test: Add tests for inventory manager
refactor: Simplify workspace state tracking
```

### Code Review Checklist

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No console.log or print statements
- [ ] Error handling in place

---

## Deployment

### Docker

```bash
docker-compose up -d
```

### Manual

```bash
# Dashboard
cd dashboard
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# MCP Server
cd mcp-server
python -m src.server

# Slicer
cd slicer-service
python src/slicer_api.py
```

### Environment Variables

See `.env.example` for all options.

---

## Debugging

### Flask Debug Mode

```bash
FLASK_ENV=development python app.py
```

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Browser DevTools

- Network tab for API calls
- Console for JavaScript errors
- Elements for HTML inspection

---

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [LEGO Dimensions](https://www.ldraw.org/)

---

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Read the documentation
