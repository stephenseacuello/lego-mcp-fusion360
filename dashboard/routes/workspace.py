"""
Workspace Routes

Live digital twin of the physical LEGO workspace.
Real-time camera feed with brick detection and sync.
"""

from flask import Blueprint, render_template, request, jsonify, Response
import time
import json

workspace_bp = Blueprint("workspace", __name__)


@workspace_bp.route("/")
def workspace():
    """Main workspace page - the digital twin."""
    from services.vision import get_camera_manager, get_detector
    from services.inventory import get_workspace_manager

    camera_manager = get_camera_manager()
    workspace_manager = get_workspace_manager()

    cameras = camera_manager.list_cameras()
    current_bricks = workspace_manager.get_current_bricks()
    summary = workspace_manager.get_summary()
    config = workspace_manager.get_config()

    # Get detector info
    try:
        detector = get_detector()
        detector_info = detector.get_info()
    except Exception:
        detector_info = {"backend": "not_initialized"}

    return render_template(
        "pages/workspace.html",
        cameras=cameras,
        current_bricks=[b.to_dict() for b in current_bricks],
        summary=summary,
        config=config.__dict__,
        detector_info=detector_info,
    )


@workspace_bp.route("/frame")
def get_frame():
    """Get a single camera frame as JPEG."""
    from services.vision import get_camera_manager

    camera_manager = get_camera_manager()

    # Ensure camera is open
    if camera_manager._active_camera is None:
        cameras = camera_manager.list_cameras()
        if cameras:
            camera_manager.open(cameras[0].id)

    jpeg_bytes = camera_manager.get_frame_jpeg(quality=75)

    if jpeg_bytes is None:
        return jsonify({"error": "No frame available"}), 500

    return Response(jpeg_bytes, mimetype="image/jpeg")


@workspace_bp.route("/stream")
def stream():
    """Stream camera frames as MJPEG."""
    from services.vision import get_camera_manager

    camera_manager = get_camera_manager()

    # Ensure camera is open
    if camera_manager._active_camera is None:
        cameras = camera_manager.list_cameras()
        if cameras:
            camera_manager.open(cameras[0].id)

    def generate():
        while True:
            jpeg_bytes = camera_manager.get_frame_jpeg(quality=70)

            if jpeg_bytes:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")

            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@workspace_bp.route("/detect", methods=["POST"])
def detect():
    """Run detection on current frame and update workspace."""
    from services.vision import get_camera_manager, get_detector
    from services.inventory import get_workspace_manager

    camera_manager = get_camera_manager()
    detector = get_detector()
    workspace_manager = get_workspace_manager()

    # Get frame
    frame = camera_manager.get_frame()
    if frame is None:
        return jsonify({"error": "No frame available"}), 500

    # Run detection
    start_time = time.time()
    detections = detector.detect(frame)
    detection_time = (time.time() - start_time) * 1000

    # Update workspace state
    result = workspace_manager.update_from_detections([d.to_dict() for d in detections])
    result["detection_time_ms"] = round(detection_time, 1)
    result["detections"] = [d.to_dict() for d in detections]

    return jsonify(result)


@workspace_bp.route("/state")
def get_state():
    """Get current workspace state."""
    from services.inventory import get_workspace_manager

    workspace_manager = get_workspace_manager()

    bricks = workspace_manager.get_current_bricks()
    summary = workspace_manager.get_summary()

    return jsonify({"bricks": [b.to_dict() for b in bricks], "summary": summary})


@workspace_bp.route("/clear", methods=["POST"])
def clear_workspace():
    """Clear all bricks from workspace."""
    from services.inventory import get_workspace_manager

    workspace_manager = get_workspace_manager()
    workspace_manager.clear()

    return jsonify({"success": True, "message": "Workspace cleared"})


@workspace_bp.route("/add-all-to-collection", methods=["POST"])
def add_all_to_collection():
    """Add all workspace bricks to permanent collection."""
    from services.inventory import get_workspace_manager

    workspace_manager = get_workspace_manager()
    added = workspace_manager.add_all_to_inventory()

    return jsonify({"success": True, "added": len(added), "items": added})


@workspace_bp.route("/camera/<int:camera_id>/open", methods=["POST"])
def open_camera(camera_id):
    """Open a specific camera."""
    from services.vision import get_camera_manager

    camera_manager = get_camera_manager()
    success = camera_manager.open(camera_id)

    if success:
        return jsonify({"success": True, "camera_id": camera_id})
    else:
        return jsonify({"success": False, "error": "Failed to open camera"}), 500


@workspace_bp.route("/camera/close", methods=["POST"])
def close_camera():
    """Close the active camera."""
    from services.vision import get_camera_manager

    camera_manager = get_camera_manager()
    camera_manager.close()

    return jsonify({"success": True})


@workspace_bp.route("/cameras")
def list_cameras():
    """List available cameras."""
    from services.vision import get_camera_manager

    camera_manager = get_camera_manager()
    cameras = camera_manager.list_cameras()

    return jsonify(
        {"cameras": [c.__dict__ for c in cameras], "status": camera_manager.get_status()}
    )


@workspace_bp.route("/config", methods=["GET", "POST"])
def workspace_config():
    """Get or update workspace configuration."""
    from services.inventory import get_workspace_manager

    workspace_manager = get_workspace_manager()

    if request.method == "POST":
        data = request.get_json()
        config = workspace_manager.update_config(**data)
        return jsonify({"success": True, "config": config.__dict__})
    else:
        config = workspace_manager.get_config()
        return jsonify(config.__dict__)


@workspace_bp.route("/calibrate", methods=["POST"])
def calibrate():
    """Calibrate workspace from corner positions."""
    from services.inventory import get_workspace_manager

    data = request.get_json()

    workspace_manager = get_workspace_manager()
    workspace_manager.calibrate_from_corners(
        top_left=tuple(data["top_left"]),
        top_right=tuple(data["top_right"]),
        bottom_left=tuple(data["bottom_left"]),
        bottom_right=tuple(data["bottom_right"]),
    )

    return jsonify({"success": True, "message": "Calibration saved"})


@workspace_bp.route("/capture", methods=["POST"])
def capture_image():
    """Capture and save current frame."""
    from services.vision import get_camera_manager
    import os
    from datetime import datetime

    camera_manager = get_camera_manager()

    # Create captures directory
    captures_dir = "/home/claude/lego-mcp-fusion360/dashboard/data/captures"
    os.makedirs(captures_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(captures_dir, f"capture_{timestamp}.jpg")

    success = camera_manager.capture_image(filepath)

    if success:
        return jsonify({"success": True, "filepath": filepath})
    else:
        return jsonify({"success": False, "error": "Capture failed"}), 500


@workspace_bp.route("/detector/info")
def detector_info():
    """Get detector information."""
    from services.vision import get_detector

    try:
        detector = get_detector()
        return jsonify(detector.get_info())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
