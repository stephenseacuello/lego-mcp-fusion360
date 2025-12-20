"""
Vision Services

YOLO11 + Roboflow LEGO brick detection and camera management.
"""

from .detector import (
    LegoDetector,
    BrickDetection,
    DetectionBackend,
    LegoColorClassifier,
    BrickMatcher,
    get_detector,
    reset_detector,
)

from .camera_manager import CameraManager, CameraInfo, CameraType, get_camera_manager

__all__ = [
    # Detector
    "LegoDetector",
    "BrickDetection",
    "DetectionBackend",
    "LegoColorClassifier",
    "BrickMatcher",
    "get_detector",
    "reset_detector",
    # Camera
    "CameraManager",
    "CameraInfo",
    "CameraType",
    "get_camera_manager",
]
