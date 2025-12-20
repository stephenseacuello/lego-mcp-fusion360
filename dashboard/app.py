"""
LEGO MCP Dashboard - Flask Application

A web dashboard for visualizing, testing, and monitoring the LEGO MCP system.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template

# Optional SocketIO support
try:
    from flask_socketio import SocketIO

    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None

# Add paths for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "shared"))
sys.path.insert(0, str(BASE_DIR / "mcp-server" / "src"))

# Initialize SocketIO (optional)
socketio = SocketIO() if SOCKETIO_AVAILABLE else None


def create_app(config_name=None):
    """Application factory."""

    app = Flask(__name__)

    # Load configuration
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    from config import config

    app.config.from_object(config.get(config_name, config["default"]))

    # Initialize extensions
    if SOCKETIO_AVAILABLE and socketio:
        socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Register blueprints
    from routes.main import main_bp
    from routes.catalog import catalog_bp
    from routes.builder import builder_bp
    from routes.files import files_bp
    from routes.history import history_bp
    from routes.status import status_bp
    from routes.tools import tools_bp
    from routes.settings import settings_bp
    from routes.api import api_bp

    # Phase 2: Digital Twin blueprints
    from routes.workspace import workspace_bp
    from routes.scan import scan_bp
    from routes.collection import collection_bp
    from routes.builds_routes import builds_bp
    from routes.insights import insights_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(catalog_bp, url_prefix="/catalog")
    app.register_blueprint(builder_bp, url_prefix="/builder")
    app.register_blueprint(files_bp, url_prefix="/files")
    app.register_blueprint(history_bp, url_prefix="/history")
    app.register_blueprint(status_bp, url_prefix="/status")
    app.register_blueprint(tools_bp, url_prefix="/tools")
    app.register_blueprint(settings_bp, url_prefix="/settings")
    app.register_blueprint(api_bp, url_prefix="/api")

    # Phase 2: Digital Twin routes
    app.register_blueprint(workspace_bp, url_prefix="/workspace")
    app.register_blueprint(scan_bp, url_prefix="/scan")
    app.register_blueprint(collection_bp, url_prefix="/collection")
    app.register_blueprint(builds_bp, url_prefix="/builds")
    app.register_blueprint(insights_bp, url_prefix="/insights")

    # Register WebSocket events (if available)
    if SOCKETIO_AVAILABLE and socketio:
        from websocket.events import register_events

        register_events(socketio)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("errors/500.html"), 500

    # Context processors
    @app.context_processor
    def inject_globals():
        """Inject global variables into templates."""
        from services.status_service import get_quick_status

        return {"services_status": get_quick_status()}

    return app


# Create app instance for development
app = create_app()


if __name__ == "__main__":
    if SOCKETIO_AVAILABLE and socketio:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True)
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)
