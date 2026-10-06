#!/usr/bin/env python3
"""
Minimal Flask web GUI for transcriptions CLI.
Brutalist design, core features only.

Streaming approach adapted from apps/recorder-ui/server/main.py which
used ffmpeg subprocess to decode HEVC and output MJPEG directly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Generator, Optional

from flask import Flask, Response, jsonify, render_template, request

from recording.models import DeviceState, SessionState
from recording.recorder import Recorder
from recording.store import get_session_store

app = Flask(__name__, template_folder=Path(__file__).parent / "templates")

# Find ffmpeg binary - check vendor/ first, then system PATH
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_FFMPEG = _REPO_ROOT / "vendor" / "ffmpeg"
FFMPEG_BIN = str(_VENDOR_FFMPEG) if _VENDOR_FFMPEG.exists() else "ffmpeg"


# ---------------------------------------------------------------------------
# MJPEG Streaming (ffmpeg subprocess approach from old UI)
# ---------------------------------------------------------------------------

# Target resolution for web streaming
STREAM_HEIGHT = 360

# Placeholder JPEG for "Connecting..." state (generated once at startup)
_PLACEHOLDER_JPEG: Optional[bytes] = None


def _get_placeholder_jpeg() -> bytes:
    """Get a simple gray placeholder JPEG with 'Connecting...' text."""
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is not None:
        return _PLACEHOLDER_JPEG

    # Create a simple gray JPEG using ffmpeg (no opencv dependency)
    # Generate a 640x360 gray frame with text overlay
    try:
        cmd = [
            FFMPEG_BIN,
            "-f", "lavfi",
            "-i", f"color=c=0x333333:s=640x360:d=1",
            "-vf", "drawtext=text='Connecting...':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=(h-text_h)/2",
            "-frames:v", "1",
            "-f", "mjpeg",
            "-q:v", "5",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            _PLACEHOLDER_JPEG = result.stdout
            return _PLACEHOLDER_JPEG
    except Exception:
        pass

    # Fallback: minimal valid JPEG (1x1 gray pixel)
    _PLACEHOLDER_JPEG = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xA2, 0x80, 0x0A,
        0x28, 0xA0, 0x02, 0x8A, 0x28, 0x00, 0xFF, 0xD9
    ])
    return _PLACEHOLDER_JPEG


def _has_ffmpeg() -> bool:
    """Check if ffmpeg binary is available."""
    if Path(FFMPEG_BIN).exists():
        return True
    return shutil.which(FFMPEG_BIN) is not None


class StreamManager:
    """Manages UDP video stream capture and MJPEG serving.

    Uses ffmpeg subprocess to decode HEVC and output MJPEG directly.
    This approach is more robust for handling UDP packet loss than OpenCV.

    Key technique from old apps/recorder-ui/server/main.py:
    - ffmpeg decodes HEVC UDP stream
    - Outputs MJPEG via image2pipe format
    - We parse JPEG SOI/EOI markers from the pipe
    - Store latest frame per port for serving
    """

    def __init__(self) -> None:
        self._streams: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _ffmpeg_capture_loop(self, port: int) -> None:
        """Background thread: run ffmpeg to decode HEVC and output MJPEG.

        The key insight is that ffmpeg's MJPEG encoder produces valid JPEG
        frames even when HEVC decode errors occur (due to UDP packet loss).
        This is much more robust than OpenCV's VideoCapture.
        """
        # LATENCY FIX: increased buffer from 50000 to 200000 (revert: fifo_size=50000)
        url = f"udp://@0.0.0.0:{port}?overrun_nonfatal=1&fifo_size=200000&reuse=1"

        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "error",
            # Low-latency input options
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32768",
            # NOTE: -hwaccel videotoolbox crashes with UDP input, disabled for now
            "-i", url,
            # LATENCY FIX: 30fps instead of 15fps (revert: fps=15)
            "-vf", f"fps=30,scale=-2:{STREAM_HEIGHT}",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "5",
            "pipe:1",
        ]

        while True:
            with self._lock:
                stream = self._streams.get(port)
                if not stream or stream.get("stop"):
                    break

            print(f"[StreamManager] Starting ffmpeg for port {port}...", flush=True)

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except FileNotFoundError:
                print(f"[StreamManager] ffmpeg not found at {FFMPEG_BIN}", flush=True)
                time.sleep(5)
                continue
            except Exception as e:
                print(f"[StreamManager] Failed to start ffmpeg: {e}", flush=True)
                time.sleep(2)
                continue

            # Store process reference for cleanup
            with self._lock:
                stream = self._streams.get(port)
                if stream:
                    stream["proc"] = proc

            # Parse JPEG frames from ffmpeg stdout
            buffer = bytearray()
            try:
                while proc.poll() is None:
                    with self._lock:
                        stream = self._streams.get(port)
                        if not stream or stream.get("stop"):
                            break

                    chunk = proc.stdout.read(8192)
                    if not chunk:
                        break

                    buffer.extend(chunk)

                    # Extract complete JPEG frames (SOI to EOI)
                    while True:
                        start = buffer.find(b"\xff\xd8")  # JPEG SOI marker
                        if start == -1:
                            # No SOI found, clear buffer up to last 2 bytes
                            if len(buffer) > 2:
                                del buffer[:-2]
                            break

                        end = buffer.find(b"\xff\xd9", start + 2)  # JPEG EOI marker
                        if end == -1:
                            # No EOI yet, wait for more data
                            break

                        # Extract complete frame
                        frame = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]

                        # Store as latest frame
                        with self._lock:
                            stream = self._streams.get(port)
                            if stream:
                                stream["frame"] = frame
                                stream["updated"] = time.time()

            except Exception as e:
                print(f"[StreamManager] Error reading from ffmpeg: {e}", flush=True)

            # Cleanup process
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            with self._lock:
                stream = self._streams.get(port)
                if stream and stream.get("stop"):
                    break

            # Brief delay before retry
            time.sleep(0.5)

        print(f"[StreamManager] Stopped capture for port {port}", flush=True)

    def start_stream(self, port: int) -> None:
        """Start capturing from a port if not already running."""
        with self._lock:
            if port in self._streams:
                return

            stream: dict[str, Any] = {
                "frame": None,
                "updated": 0,
                "stop": False,
                "thread": None,
                "proc": None,
            }
            self._streams[port] = stream

        thread = threading.Thread(target=self._ffmpeg_capture_loop, args=(port,), daemon=True)
        with self._lock:
            self._streams[port]["thread"] = thread
        thread.start()

    def stop_stream(self, port: int) -> None:
        """Stop capturing from a port."""
        with self._lock:
            stream = self._streams.get(port)
            if stream:
                stream["stop"] = True
                proc = stream.get("proc")
                if proc:
                    try:
                        proc.terminate()
                    except Exception:
                        pass

    def get_frame(self, port: int) -> Optional[bytes]:
        """Get the latest JPEG frame for a port."""
        with self._lock:
            stream = self._streams.get(port)
            if stream:
                return stream.get("frame")
        return None

    def generate_mjpeg(self, port: int) -> Generator[bytes, None, None]:
        """Generate MJPEG stream for a port."""
        self.start_stream(port)

        placeholder = _get_placeholder_jpeg()
        last_frame: Optional[bytes] = None

        while True:
            frame = self.get_frame(port)

            if frame is not None:
                last_frame = frame
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )
            elif last_frame is not None:
                # Show last good frame if we have one
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n'
                )
            else:
                # Show placeholder while connecting
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n'
                )

            time.sleep(0.025)  # LATENCY FIX: ~40fps polling (revert: 0.066 for ~15fps)


# Global stream manager
stream_manager = StreamManager()


def _device_to_dict(dev: Any, stream_port: Optional[int] = None) -> dict[str, Any]:
    """Convert DeviceStateEntry to JSON-serializable dict."""
    return {
        "device_id": dev.device_id,
        "role": dev.role,
        "kind": dev.kind,
        "state": dev.state.value if hasattr(dev.state, "value") else str(dev.state),
        "battery_pct": dev.battery_pct,
        "storage_gb": round(dev.storage_gb, 1) if dev.storage_gb else None,
        "connection": dev.connection,
        "stream_port": stream_port,
    }


def _take_to_dict(take: Any) -> dict[str, Any]:
    """Convert TakeEntry to JSON-serializable dict."""
    return {
        "take_number": take.take_number,
        "label": take.label,
        "started_at": take.started_at,
        "stopped_at": take.stopped_at,
        "duration_sec": take.duration_sec,
        "is_recording": take.is_recording,
        "recordings_count": len(take.recordings),
    }


def _get_status(device: str = "gopro") -> dict[str, Any]:
    """Get combined daemon + session status."""
    store = get_session_store()
    session = store.get_active()

    # Get daemon status
    daemon_status: dict[str, Any] = {"ok": False, "cameras": []}
    try:
        recorder = Recorder(device=device, store=store)
        daemon_status = recorder.status()
        daemon_status["ok"] = True
    except Exception as e:
        daemon_status["error"] = str(e)

    # Build response
    result: dict[str, Any] = {
        "daemon": {
            "ok": daemon_status.get("ok", False),
            "error": daemon_status.get("error"),
            "camera_count": len(daemon_status.get("cameras", [])),
        },
        "session": None,
        "cameras": [],
        "takes": [],
        "is_recording": False,
        "current_take": None,
    }

    # Build map of serial -> stream port from daemon
    stream_ports: dict[str, int] = {}
    for cam in daemon_status.get("cameras", []):
        serial = cam.get("serial")
        stream_info = cam.get("stream") or {}
        port = stream_info.get("port")
        if serial and port and stream_info.get("active"):
            stream_ports[serial] = port

    if session:
        # Refresh session with latest daemon state
        try:
            recorder = Recorder(device=device, store=store)
            recorder.refresh_session(session)
        except Exception:
            pass

        result["session"] = {
            "session_id": session.session_id,
            "name": session.name,
            "participant": session.participant,
            "created_at": session.created_at,
            "state": session.state.value,
            "is_closed": session.state == SessionState.CLOSED,
            "take_count": len(session.takes),
        }

        result["cameras"] = [
            _device_to_dict(dev, stream_port=stream_ports.get(dev.device_id))
            for dev in session.devices
        ]
        result["takes"] = [_take_to_dict(take) for take in reversed(session.takes)]

        current = session.current_take
        if current:
            result["is_recording"] = True
            result["current_take"] = _take_to_dict(current)

    return result


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/status")
def api_status() -> Any:
    device = request.args.get("device", "gopro")
    return jsonify(_get_status(device))


@app.route("/api/session/init", methods=["POST"])
def api_session_init() -> Any:
    data = request.get_json() or {}
    device = data.get("device", "gopro")
    name = data.get("name")
    participant = data.get("participant")
    rig = data.get("rig")

    store = get_session_store()
    recorder = Recorder(device=device, store=store)

    try:
        session = recorder.create_session(
            name=name,
            participant=participant,
            rig=rig,
        )
        return jsonify({"ok": True, "session_id": session.session_id})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/session/close", methods=["POST"])
def api_session_close() -> Any:
    data = request.get_json() or {}
    force = data.get("force", False)

    store = get_session_store()
    session = store.get_active()

    if not session:
        return jsonify({"ok": False, "error": "No active session"}), 400

    # Check if recording
    recording = [d for d in session.devices if d.state == DeviceState.RECORDING]
    if recording and not force:
        return jsonify({"ok": False, "error": "Session is recording. Use force=true to close anyway."}), 400

    store.close(session)
    return jsonify({"ok": True, "session_id": session.session_id})


@app.route("/api/take/start", methods=["POST"])
def api_take_start() -> Any:
    data = request.get_json() or {}
    device = data.get("device", "gopro")
    label = data.get("label")
    force = data.get("force", False)

    store = get_session_store()
    session = store.get_active()

    if not session:
        return jsonify({"ok": False, "error": "No active session"}), 400

    if session.current_take:
        return jsonify({"ok": False, "error": "Take already recording"}), 400

    recorder = Recorder(device=device, store=store)

    try:
        take = recorder.start_take(session, label=label, force=force)
        if not take:
            return jsonify({"ok": False, "error": "Failed to start take on any device"}), 500
        return jsonify({"ok": True, "take_number": take.take_number})
    except ValueError as e:
        error_msg = str(e)
        # If devices not ready, return which ones so UI can offer to force
        if error_msg.startswith("devices_not_ready:"):
            devices = error_msg.split(":", 1)[-1]
            return jsonify({
                "ok": False,
                "error": "devices_not_ready",
                "devices": devices,
                "message": f"Devices not ready: {devices}. Continue anyway?"
            }), 400
        return jsonify({"ok": False, "error": error_msg}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/take/stop", methods=["POST"])
def api_take_stop() -> Any:
    data = request.get_json() or {}
    device = data.get("device", "gopro")

    store = get_session_store()
    session = store.get_active()

    if not session:
        return jsonify({"ok": False, "error": "No active session"}), 400

    if not session.current_take:
        return jsonify({"ok": False, "error": "No take currently recording"}), 400

    recorder = Recorder(device=device, store=store)

    try:
        take = recorder.stop_take(session)
        return jsonify({"ok": True, "take_number": take.take_number if take else None})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions")
def api_sessions() -> Any:
    store = get_session_store()
    sessions = store.list_sessions(limit=20)
    active_id = store.get_active_id()

    result = []
    for sid in sessions:
        session = store.load(sid)
        if session:
            result.append({
                "session_id": session.session_id,
                "name": session.name,
                "participant": session.participant,
                "created_at": session.created_at,
                "state": session.state.value,
                "take_count": len(session.takes),
                "is_active": sid == active_id,
            })

    return jsonify(result)


@app.route("/api/session/use", methods=["POST"])
def api_session_use() -> Any:
    data = request.get_json() or {}
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"ok": False, "error": "session_id required"}), 400

    store = get_session_store()
    session = store.load(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    if session.state == SessionState.CLOSED:
        return jsonify({"ok": False, "error": "Session is closed"}), 400

    store.set_active(session_id)
    return jsonify({"ok": True, "session_id": session_id})


@app.route("/api/stream/<int:port>")
def api_stream(port: int) -> Response:
    """MJPEG stream for a camera by UDP port."""
    return Response(
        stream_manager.generate_mjpeg(port),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcriptions Web GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Starting web GUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
