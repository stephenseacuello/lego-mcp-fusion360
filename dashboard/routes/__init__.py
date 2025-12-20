"""
Dashboard Routes

All Flask route blueprints.
"""

from .main import main_bp
from .catalog import catalog_bp
from .builder import builder_bp
from .files import files_bp
from .history import history_bp
from .status import status_bp
from .tools import tools_bp
from .settings import settings_bp
from .api import api_bp

__all__ = [
    "main_bp",
    "catalog_bp",
    "builder_bp",
    "files_bp",
    "history_bp",
    "status_bp",
    "tools_bp",
    "settings_bp",
    "api_bp",
]
