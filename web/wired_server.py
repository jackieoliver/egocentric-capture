#!/usr/bin/env python3
"""
Wired GoPro Dev GUI

Simple Flask app for wired GoPro webcams via USB.
Supports up to 3 cameras (HEAD, LWRIST, RWRIST).
Uses direct OpenCV capture - no ffmpeg/UDP complexity.
"""

import cv2
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# Recording output directory
RECORDINGS_DIR = Path(__file__).parent.parent / "recordings" / "wired"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CameraState:
    """State for a single camera."""
    index: int
    role: str  # HEAD, LWRIST, RWRIST
    name: str
    capture: Optional[cv2.VideoCapture] = None
    recording: bool = False
    writer: Optional[cv2.VideoWriter] = None
    recording_path: Optional[Path] = None
    frame_count: int = 0
    last_frame: Optional[Any] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class WiredCameraManager:
    """Manages multiple wired GoPro webcams."""

    # Role assignment by camera index (can be customized)
    ROLE_ORDER = ["LWRIST", "HEAD", "RWRIST"]

    def __init__(self):
        self.cameras: dict[int, CameraState] = {}
        self.lock = threading.Lock()
        self._scan_cameras()

    def _scan_cameras(self) -> None:
        """Scan for available cameras."""
        print("Scanning for cameras...")
        gopro_indices = []

        # Check first 5 camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name (not always available)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Heuristic: GoPro webcams are typically 1080p30
                # FaceTime is usually 1080p15 or 720p
                is_likely_gopro = (w == 1920 and h == 1080 and fps >= 25)

                print(f"  Camera {i}: {w}x{h} @ {fps}fps {'(likely GoPro)' if is_likely_gopro else ''}")

                if is_likely_gopro:
                    gopro_indices.append(i)

                cap.release()

        # Assign roles to detected GoPros
        with self.lock:
            for idx, cam_idx in enumerate(gopro_indices):
                role = self.ROLE_ORDER[idx] if idx < len(self.ROLE_ORDER) else f"CAM{idx}"
                self.cameras[cam_idx] = CameraState(
                    index=cam_idx,
                    role=role,
                    name=f"GoPro {role}",
                )
                print(f"  Assigned camera {cam_idx} as {role}")

    def rescan(self) -> dict[str, Any]:
        """Rescan for cameras."""
        # Close existing captures
        with self.lock:
            for cam in self.cameras.values():
                if cam.capture and cam.capture.isOpened():
                    cam.capture.release()
                if cam.writer:
                    cam.writer.release()
            self.cameras.clear()

        self._scan_cameras()
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        """Get status of all cameras."""
        with self.lock:
            cameras = []
            for cam in self.cameras.values():
                cameras.append({
                    "index": cam.index,
                    "role": cam.role,
                    "name": cam.name,
                    "recording": cam.recording,
                    "frame_count": cam.frame_count,
                    "recording_path": str(cam.recording_path) if cam.recording_path else None,
                })
            return {
                "cameras": cameras,
                "recording_dir": str(RECORDINGS_DIR),
            }

    def _ensure_capture(self, cam: CameraState) -> bool:
        """Ensure camera capture is open."""
        if cam.capture is None or not cam.capture.isOpened():
            cam.capture = cv2.VideoCapture(cam.index)
            if not cam.capture.isOpened():
                return False
            # Set buffer size to minimize latency
            cam.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def get_frame(self, camera_index: int) -> Optional[bytes]:
        """Get JPEG frame from camera."""
        with self.lock:
            cam = self.cameras.get(camera_index)
            if not cam:
                return None

        with cam.lock:
            if not self._ensure_capture(cam):
                return None

            ret, frame = cam.capture.read()
            if not ret:
                return None

            cam.last_frame = frame

            # Write to recording if active
            if cam.recording and cam.writer:
                cam.writer.write(frame)
                cam.frame_count += 1

            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpeg.tobytes()

    def start_recording(self, camera_index: int) -> dict[str, Any]:
        """Start recording for a camera."""
        with self.lock:
            cam = self.cameras.get(camera_index)
            if not cam:
                return {"ok": False, "error": "Camera not found"}

        with cam.lock:
            if cam.recording:
                return {"ok": False, "error": "Already recording"}

            if not self._ensure_capture(cam):
                return {"ok": False, "error": "Cannot open camera"}

            # Create recording path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cam.role}_{timestamp}.mp4"
            cam.recording_path = RECORDINGS_DIR / filename

            # Get camera properties
            w = int(cam.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cam.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cam.capture.get(cv2.CAP_PROP_FPS) or 30.0

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            cam.writer = cv2.VideoWriter(str(cam.recording_path), fourcc, fps, (w, h))

            if not cam.writer.isOpened():
                cam.writer = None
                cam.recording_path = None
                return {"ok": False, "error": "Cannot create video writer"}

            cam.recording = True
            cam.frame_count = 0

            return {
                "ok": True,
                "path": str(cam.recording_path),
                "role": cam.role,
            }

    def stop_recording(self, camera_index: int) -> dict[str, Any]:
        """Stop recording for a camera."""
        with self.lock:
            cam = self.cameras.get(camera_index)
            if not cam:
                return {"ok": False, "error": "Camera not found"}

        with cam.lock:
            if not cam.recording:
                return {"ok": False, "error": "Not recording"}

            cam.recording = False
            if cam.writer:
                cam.writer.release()
                cam.writer = None

            result = {
                "ok": True,
                "path": str(cam.recording_path),
                "role": cam.role,
                "frames": cam.frame_count,
            }
            cam.recording_path = None
            cam.frame_count = 0

            return result

    def start_all_recording(self) -> dict[str, Any]:
        """Start recording on all cameras."""
        results = []
        with self.lock:
            indices = list(self.cameras.keys())

        for idx in indices:
            results.append(self.start_recording(idx))

        return {"ok": all(r["ok"] for r in results), "results": results}

    def stop_all_recording(self) -> dict[str, Any]:
        """Stop recording on all cameras."""
        results = []
        with self.lock:
            indices = list(self.cameras.keys())

        for idx in indices:
            results.append(self.stop_recording(idx))

        return {"ok": True, "results": results}

    def assign_role(self, camera_index: int, role: str) -> dict[str, Any]:
        """Manually assign a role to a camera."""
        with self.lock:
            cam = self.cameras.get(camera_index)
            if not cam:
                return {"ok": False, "error": "Camera not found"}

            cam.role = role.upper()
            cam.name = f"GoPro {cam.role}"
            return {"ok": True, "role": cam.role}


# Global camera manager
manager = WiredCameraManager()


def generate_mjpeg(camera_index: int):
    """Generate MJPEG stream for a camera."""
    while True:
        frame = manager.get_frame(camera_index)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # No frame, wait a bit
            time.sleep(0.1)


@app.route('/')
def index():
    """Main page."""
    return render_template('wired.html')


@app.route('/api/status')
def api_status():
    """Get camera status."""
    return jsonify(manager.get_status())


@app.route('/api/rescan', methods=['POST'])
def api_rescan():
    """Rescan for cameras."""
    return jsonify(manager.rescan())


@app.route('/api/stream/<int:camera_index>')
def api_stream(camera_index: int):
    """MJPEG stream for a camera."""
    return Response(
        generate_mjpeg(camera_index),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/record/start/<int:camera_index>', methods=['POST'])
def api_record_start(camera_index: int):
    """Start recording for a camera."""
    return jsonify(manager.start_recording(camera_index))


@app.route('/api/record/stop/<int:camera_index>', methods=['POST'])
def api_record_stop(camera_index: int):
    """Stop recording for a camera."""
    return jsonify(manager.stop_recording(camera_index))


@app.route('/api/record/start-all', methods=['POST'])
def api_record_start_all():
    """Start recording on all cameras."""
    return jsonify(manager.start_all_recording())


@app.route('/api/record/stop-all', methods=['POST'])
def api_record_stop_all():
    """Stop recording on all cameras."""
    return jsonify(manager.stop_all_recording())


@app.route('/api/assign-role/<int:camera_index>', methods=['POST'])
def api_assign_role(camera_index: int):
    """Assign role to a camera."""
    data = request.get_json() or {}
    role = data.get("role", "")
    if not role:
        return jsonify({"ok": False, "error": "Role required"})
    return jsonify(manager.assign_role(camera_index, role))


# ============ Storage Management ============

def get_recordings_list() -> list[dict[str, Any]]:
    """Get list of all recordings with metadata."""
    recordings = []
    for f in sorted(RECORDINGS_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        recordings.append({
            "filename": f.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "created": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return recordings


def get_storage_stats() -> dict[str, Any]:
    """Get storage statistics."""
    recordings = get_recordings_list()
    total_bytes = sum(r["size_bytes"] for r in recordings)
    return {
        "total_files": len(recordings),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 1),
        "total_gb": round(total_bytes / (1024 * 1024 * 1024), 2),
        "recording_dir": str(RECORDINGS_DIR),
    }


@app.route('/api/recordings')
def api_recordings():
    """List all recordings."""
    return jsonify({
        "recordings": get_recordings_list(),
        "storage": get_storage_stats(),
    })


@app.route('/api/recordings/delete/<filename>', methods=['POST'])
def api_delete_recording(filename: str):
    """Delete a specific recording."""
    # Security: only allow deleting .mp4 files in recordings dir
    if not filename.endswith('.mp4') or '/' in filename or '\\' in filename:
        return jsonify({"ok": False, "error": "Invalid filename"})

    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"ok": False, "error": "File not found"})

    try:
        filepath.unlink()
        return jsonify({"ok": True, "deleted": filename, "storage": get_storage_stats()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route('/api/recordings/clear-all', methods=['POST'])
def api_clear_recordings():
    """Delete all recordings."""
    deleted = []
    errors = []
    for f in RECORDINGS_DIR.glob("*.mp4"):
        try:
            f.unlink()
            deleted.append(f.name)
        except Exception as e:
            errors.append({"file": f.name, "error": str(e)})

    return jsonify({
        "ok": len(errors) == 0,
        "deleted": deleted,
        "errors": errors,
        "storage": get_storage_stats(),
    })


def main():
    """Run the server."""
    print("\n" + "=" * 50)
    print("WIRED GOPRO DEV GUI")
    print("=" * 50)
    print(f"Recordings will be saved to: {RECORDINGS_DIR}")
    print(f"Open http://127.0.0.1:5001 in your browser")
    print("=" * 50 + "\n")

    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    main()
