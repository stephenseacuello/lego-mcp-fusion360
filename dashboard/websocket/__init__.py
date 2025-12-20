"""
WebSocket Events

Real-time event handling for the dashboard.
"""

import time
import threading

# Optional SocketIO support
try:
    from flask_socketio import emit, join_room, leave_room
    from flask import request

    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    emit = join_room = leave_room = None
    request = None


# Track connected clients
connected_clients = set()

# Background threads
_status_thread = None
_status_thread_running = False


def register_events(socketio):
    """Register all WebSocket event handlers."""

    if not SOCKETIO_AVAILABLE or socketio is None:
        return

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        client_id = request.sid
        connected_clients.add(client_id)

        # Send welcome message
        emit(
            "connected",
            {
                "client_id": client_id,
                "message": "Connected to LEGO MCP Dashboard",
                "timestamp": time.time(),
            },
        )

        # Start status monitoring if not running
        start_status_monitoring(socketio)

    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        client_id = request.sid
        connected_clients.discard(client_id)

        # Stop monitoring if no clients
        if not connected_clients:
            stop_status_monitoring()

    @socketio.on("subscribe")
    def handle_subscribe(data):
        """Subscribe to a room/channel."""
        room = data.get("room")
        if room:
            join_room(room)
            emit("subscribed", {"room": room})

    @socketio.on("unsubscribe")
    def handle_unsubscribe(data):
        """Unsubscribe from a room/channel."""
        room = data.get("room")
        if room:
            leave_room(room)
            emit("unsubscribed", {"room": room})

    @socketio.on("ping")
    def handle_ping():
        """Handle ping (keep-alive)."""
        emit("pong", {"timestamp": time.time()})

    @socketio.on("check_status")
    def handle_check_status():
        """Check and emit system status."""
        from services.status_service import StatusService

        status = StatusService.get_all_status(use_cache=False)
        emit("status_update", status)

    @socketio.on("execute_tool")
    def handle_execute_tool(data):
        """Execute an MCP tool and emit progress/result."""
        from services.mcp_bridge import MCPBridge

        tool_name = data.get("tool")
        params = data.get("params", {})
        request_id = data.get("request_id", str(time.time()))

        # Emit start
        emit(
            "tool_started", {"request_id": request_id, "tool": tool_name, "timestamp": time.time()}
        )

        # Execute
        result = MCPBridge.execute_tool(tool_name, params)

        # Emit result
        emit(
            "tool_complete",
            {
                "request_id": request_id,
                "tool": tool_name,
                "result": result,
                "timestamp": time.time(),
            },
        )

    @socketio.on("watch_files")
    def handle_watch_files(data):
        """Watch a directory for file changes."""
        directory = data.get("directory", "")
        join_room(f"files:{directory}")
        emit("watching", {"directory": directory})


def start_status_monitoring(socketio):
    """Start background status monitoring."""
    global _status_thread, _status_thread_running

    if _status_thread_running:
        return

    _status_thread_running = True

    def monitor():
        from services.status_service import StatusService

        last_status = {}

        while _status_thread_running:
            try:
                status = StatusService.get_all_status(use_cache=False)

                # Check for changes
                if status != last_status:
                    socketio.emit("status_update", status)

                    # Check for specific changes
                    for service_id, service_status in status.get("services", {}).items():
                        old_status = last_status.get("services", {}).get(service_id, {})

                        if service_status.get("status") != old_status.get("status"):
                            socketio.emit(
                                "service_status_changed",
                                {
                                    "service": service_id,
                                    "old_status": old_status.get("status"),
                                    "new_status": service_status.get("status"),
                                    "timestamp": time.time(),
                                },
                            )

                    last_status = status

            except Exception as e:
                socketio.emit("error", {"type": "status_monitor", "message": str(e)})

            # Wait before next check
            time.sleep(10)

    _status_thread = threading.Thread(target=monitor, daemon=True)
    _status_thread.start()


def stop_status_monitoring():
    """Stop background status monitoring."""
    global _status_thread_running
    _status_thread_running = False


def emit_operation_progress(operation_id, step, total_steps, message, percent):
    """Emit operation progress update."""
    from app import socketio

    socketio.emit(
        "operation_progress",
        {
            "operation_id": operation_id,
            "step": step,
            "total_steps": total_steps,
            "message": message,
            "percent": percent,
            "timestamp": time.time(),
        },
    )


def emit_operation_complete(operation_id, result, duration_ms):
    """Emit operation complete."""
    from app import socketio

    socketio.emit(
        "operation_complete",
        {
            "operation_id": operation_id,
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        },
    )


def emit_file_created(file_path, size):
    """Emit file created event."""
    from app import socketio

    socketio.emit("file_created", {"path": file_path, "size": size, "timestamp": time.time()})


def emit_error(error_type, message, code=None):
    """Emit error event."""
    from app import socketio

    socketio.emit(
        "error", {"type": error_type, "message": message, "code": code, "timestamp": time.time()}
    )


# ============================================================
# Phase 2: Digital Twin WebSocket Events
# ============================================================


def emit_workspace_update(data):
    """Emit workspace update to all subscribers."""
    try:
        from app import socketio

        socketio.emit("workspace:update", {**data, "timestamp": time.time()}, room="workspace")
    except Exception:
        pass


def emit_detection_result(detections, detection_time_ms):
    """Emit detection results."""
    try:
        from app import socketio

        socketio.emit(
            "detection:result",
            {
                "detections": [d.to_dict() if hasattr(d, "to_dict") else d for d in detections],
                "detection_time_ms": detection_time_ms,
                "count": len(detections),
                "timestamp": time.time(),
            },
            room="workspace",
        )
    except Exception:
        pass


def emit_collection_update(item, action="add"):
    """Emit collection update."""
    try:
        from app import socketio

        socketio.emit(
            "collection:update",
            {
                "action": action,
                "item": item.to_dict() if hasattr(item, "to_dict") else item,
                "timestamp": time.time(),
            },
            room="collection",
        )
    except Exception:
        pass


def emit_scan_progress(batch_id, scanned, added, pending):
    """Emit scan progress update."""
    try:
        from app import socketio

        socketio.emit(
            "scan:progress",
            {
                "batch_id": batch_id,
                "scanned": scanned,
                "added": added,
                "pending": pending,
                "timestamp": time.time(),
            },
            room="scan",
        )
    except Exception:
        pass


def emit_build_check(build_id, check_result):
    """Emit build check result."""
    try:
        from app import socketio

        socketio.emit(
            "build:checked",
            {
                "build_id": build_id,
                "result": (
                    check_result.to_dict() if hasattr(check_result, "to_dict") else check_result
                ),
                "timestamp": time.time(),
            },
            room="builds",
        )
    except Exception:
        pass


def register_phase2_events(socketio):
    """Register Phase 2 WebSocket event handlers."""

    if not SOCKETIO_AVAILABLE or socketio is None:
        return

    @socketio.on("workspace:subscribe")
    def handle_workspace_subscribe():
        """Subscribe to workspace updates."""
        join_room("workspace")
        emit("workspace:subscribed", {"status": "ok"})

    @socketio.on("workspace:unsubscribe")
    def handle_workspace_unsubscribe():
        """Unsubscribe from workspace updates."""
        leave_room("workspace")

    @socketio.on("workspace:detect")
    def handle_workspace_detect():
        """Trigger detection on workspace."""
        from services.vision import get_camera_manager, get_detector
        from services.inventory import get_workspace_manager

        camera_manager = get_camera_manager()
        detector = get_detector()
        workspace_manager = get_workspace_manager()

        frame = camera_manager.get_frame()
        if frame is None:
            emit("workspace:error", {"error": "No frame available"})
            return

        start = time.time()
        detections = detector.detect(frame)
        detection_time = (time.time() - start) * 1000

        result = workspace_manager.update_from_detections([d.to_dict() for d in detections])
        result["detection_time_ms"] = round(detection_time, 1)

        emit("workspace:update", result)
        socketio.emit("workspace:update", result, room="workspace", include_self=False)

    @socketio.on("workspace:state")
    def handle_workspace_state():
        """Get current workspace state."""
        from services.inventory import get_workspace_manager

        ws = get_workspace_manager()
        emit(
            "workspace:state",
            {"bricks": [b.to_dict() for b in ws.get_current_bricks()], "summary": ws.get_summary()},
        )

    @socketio.on("collection:subscribe")
    def handle_collection_subscribe():
        """Subscribe to collection updates."""
        join_room("collection")
        emit("collection:subscribed", {"status": "ok"})

    @socketio.on("scan:subscribe")
    def handle_scan_subscribe():
        """Subscribe to scan updates."""
        join_room("scan")
        emit("scan:subscribed", {"status": "ok"})

    @socketio.on("scan:detect")
    def handle_scan_detect():
        """Run detection for scanning."""
        from services.vision import get_camera_manager, get_detector

        camera_manager = get_camera_manager()
        detector = get_detector()

        frame = camera_manager.get_frame()
        if frame is None:
            emit("scan:error", {"error": "No frame available"})
            return

        start = time.time()
        detections = detector.detect(frame)
        detection_time = (time.time() - start) * 1000

        results = []
        for det in detections:
            d = det.to_dict()
            d["status"] = "confirmed" if det.confidence >= 0.8 else "review"
            results.append(d)

        emit(
            "scan:detected",
            {
                "detections": results,
                "detection_time_ms": round(detection_time, 1),
                "total": len(results),
            },
        )

    @socketio.on("builds:subscribe")
    def handle_builds_subscribe():
        """Subscribe to builds updates."""
        join_room("builds")
        emit("builds:subscribed", {"status": "ok"})

    @socketio.on("builds:check")
    def handle_builds_check(data):
        """Check a build against inventory."""
        from services.builds import get_build_planner

        build_id = data.get("build_id")
        if not build_id:
            emit("builds:error", {"error": "No build_id"})
            return

        planner = get_build_planner()
        check = planner.check_build(build_id)

        if check:
            emit("builds:checked", check.to_dict())
        else:
            emit("builds:error", {"error": "Build not found"})

    @socketio.on("camera:subscribe")
    def handle_camera_subscribe():
        """Subscribe to camera frames."""
        join_room("camera")
        emit("camera:subscribed", {"status": "ok"})

    @socketio.on("camera:frame")
    def handle_camera_frame():
        """Get a single camera frame as base64."""
        from services.vision import get_camera_manager

        camera_manager = get_camera_manager()
        frame_b64 = camera_manager.get_frame_base64(quality=70)

        if frame_b64:
            emit("camera:frame", {"image": frame_b64})
        else:
            emit("camera:error", {"error": "No frame"})
