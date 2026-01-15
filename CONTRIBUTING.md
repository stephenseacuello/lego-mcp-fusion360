# Contributing to LEGO MCP v8.0

Thank you for your interest in contributing to LEGO MCP v8.0, a DoD/ONR-class manufacturing system. This document provides comprehensive guidelines for contributing.

**Version:** 8.0.0 | **Classification:** UNCLASSIFIED

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Security Considerations](#security-considerations)
- [Quick Start](#-quick-start-for-contributors)
- [Development Setup](#development-setup)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@lego-mcp.io.

## Security Considerations

**IMPORTANT:** Before contributing, review [SECURITY.md](./SECURITY.md).

- Never commit secrets, API keys, or credentials
- Run `pip-audit` before submitting PRs
- Security-sensitive changes require additional review
- Report vulnerabilities to security@lego-mcp.io

## Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/lego-mcp-fusion360.git
cd lego-mcp-fusion360

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r dashboard/requirements.txt
pip install -r tests/requirements-test.txt

# Run tests
python -m pytest tests/ -v

# Start development server
cd dashboard
FLASK_ENV=development python app.py
```

## 📁 Project Structure

```
lego-mcp-fusion360/
├── dashboard/          # Flask web app
│   ├── routes/        # URL handlers
│   ├── services/      # Business logic
│   ├── templates/     # HTML templates
│   └── static/        # CSS, JS
├── mcp-server/        # Claude MCP server
├── fusion360-addin/   # Fusion 360 plugin
├── shared/            # Shared Python modules
├── tests/             # Test suite
└── docs/              # Documentation
```

## 🔧 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the code style (see below)
- Add tests for new features
- Update documentation

### 3. Test

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_phase2_digital_twin.py -v

# Check dashboard routes
cd dashboard
python -c "from app import create_app; app = create_app('testing'); print(len(list(app.url_map.iter_rules())), 'routes')"
```

### 4. Commit

```bash
git add .
git commit -m "feat: add amazing feature"
```

Use conventional commit messages:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `style:` Formatting
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## 📝 Code Style

### Python

```python
# Imports: stdlib, third-party, local
import os
from typing import List, Dict

from flask import Flask
import requests

from services.vision import get_detector

# Type hints encouraged
def add_brick(brick_id: str, quantity: int = 1) -> bool:
    """Add a brick to inventory.
    
    Args:
        brick_id: The brick identifier
        quantity: Number to add (default 1)
        
    Returns:
        True if successful
    """
    return True

# Classes: PascalCase
class BrickManager:
    pass

# Functions/variables: snake_case
def get_brick_count():
    total_count = 0
    return total_count

# Constants: UPPER_SNAKE_CASE
MAX_BRICKS = 1000
```

### JavaScript

```javascript
// Use const/let, not var
const detector = new BrickDetector();
let currentFrame = null;

// camelCase for functions
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

### HTML/Jinja2

```html
<!-- Semantic HTML -->
<main class="workspace-container">
    <section class="camera-panel">
        <h2>Camera Feed</h2>
    </section>
</main>

<!-- Jinja2 templates -->
{% for brick in bricks %}
    <div class="brick-card">
        {{ brick.name }}
    </div>
{% endfor %}
```

### CSS

```css
/* Use CSS custom properties */
:root {
    --color-primary: #e3000b;
    --spacing-md: 16px;
}

/* BEM-like naming */
.brick-card { }
.brick-card__header { }
.brick-card--selected { }
```

## 🧪 Testing Guidelines

### Write Tests For

- New features
- Bug fixes
- API endpoints
- Service methods

### Test Structure

```python
# tests/test_my_feature.py
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
    
    def test_api_works(self, client):
        response = client.post('/my-feature/api',
            json={'key': 'value'},
            content_type='application/json'
        )
        assert response.get_json()['success']
```

## 📚 Documentation

Update documentation when you:

- Add new features
- Change API endpoints
- Modify configuration options
- Add new MCP tools

Documentation files:
- `README.md` - Overview
- `docs/USER_GUIDE.md` - User documentation
- `docs/API.md` - API reference
- `docs/DEVELOPER.md` - Developer guide

## 🐛 Bug Reports

Include:
1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Error messages/logs
5. Environment (OS, Python version)

## 💡 Feature Requests

Include:
1. Use case / problem to solve
2. Proposed solution
3. Alternatives considered

## ✅ Pull Request Checklist

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Changelog updated (for significant changes)
- [ ] No console.log or print statements
- [ ] Error handling in place

## 🏷️ Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `feature` | New feature request |
| `docs` | Documentation |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |

## 📞 Getting Help

- Check existing issues
- Read the documentation
- Ask in discussions

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🧱
