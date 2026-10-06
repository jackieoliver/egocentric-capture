#!/usr/bin/env python3
"""
Hand Tracking Test Server

Simple Flask server for testing Apple Vision hand tracking via browser webcam.
Receives JPEG frames from browser, runs VisionDetector, returns landmarks.
"""

import cv2
import json
import subprocess
import threading
import numpy as np
from pathlib import Path
from queue import Queue, Empty
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# Path to VisionDetector
VISION_DETECTOR_PATH = Path(__file__).parent.parent / "vision-detector" / ".build" / "release" / "VisionDetector"


class HandDetector:
    """Manages VisionDetector subprocess for hand tracking."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.process = None
        self.running = False
        self.lock = threading.Lock()

        # Landmark output from stderr
        self._stderr_queue = Queue(maxsize=10)
        self._stderr_thread = None

        # Frame output from stdout (we discard the overlay)
        self._stdout_thread = None

    def start(self) -> bool:
        """Start VisionDetector subprocess."""
        if not VISION_DETECTOR_PATH.exists():
            print(f"VisionDetector not found at {VISION_DETECTOR_PATH}")
            return False

        with self.lock:
            if self.running:
                return True

            cmd = [
                str(VISION_DETECTOR_PATH),
                "transform",
                "--width", str(self.width),
                "--height", str(self.height),
                "--max-hands", "2",
                "--no-body",       # Hands only for speed
                "--no-overlay",    # Skip rendering, we do it in browser
            ]

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                self.running = True

                # Read stderr for landmarks
                self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
                self._stderr_thread.start()

                # Drain stdout (we use --no-overlay but still need to read)
                self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
                self._stdout_thread.start()

                print(f"VisionDetector started: {' '.join(cmd)}")
                return True

            except Exception as e:
                print(f"Failed to start VisionDetector: {e}")
                self.running = False
                return False

    def stop(self):
        """Stop VisionDetector."""
        with self.lock:
            self.running = False
            if self.process:
                try:
                    self.process.stdin.close()
                    self.process.terminate()
                    self.process.wait(timeout=2)
                except:
                    self.process.kill()
                self.process = None

    def _read_stderr(self):
        """Read landmark JSONL from stderr."""
        while self.running and self.process:
            try:
                line = self.process.stderr.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode('utf-8').strip())
                    # Put in queue, drop old if full
                    try:
                        self._stderr_queue.put_nowait(data)
                    except:
                        try:
                            self._stderr_queue.get_nowait()
                            self._stderr_queue.put_nowait(data)
                        except:
                            pass
                except json.JSONDecodeError:
                    # Print non-JSON (startup messages)
                    print(f"[VisionDetector] {line.decode('utf-8', errors='replace').strip()}")
            except:
                break

    def _drain_stdout(self):
        """Drain stdout (frames with no overlay are still output)."""
        frame_size = self.width * self.height * 4  # BGRA
        while self.running and self.process:
            try:
                # Read and discard
                data = self.process.stdout.read(frame_size)
                if not data:
                    break
            except:
                break

    def detect(self, jpeg_bytes: bytes) -> dict:
        """
        Run detection on a JPEG frame.

        Returns dict with 'hands' array.
        """
        if not self.running:
            if not self.start():
                return {"error": "VisionDetector not available", "hands": []}

        try:
            # Decode JPEG to numpy
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"error": "Invalid image", "hands": []}

            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # Convert BGR to BGRA
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            # Send to VisionDetector
            with self.lock:
                if not self.process or not self.running:
                    return {"error": "Detector not running", "hands": []}

                try:
                    self.process.stdin.write(bgra.tobytes())
                    self.process.stdin.flush()
                except Exception as e:
                    return {"error": f"Write failed: {e}", "hands": []}

            # Wait for result (with timeout)
            try:
                result = self._stderr_queue.get(timeout=0.5)
                return result
            except Empty:
                return {"hands": []}

        except Exception as e:
            return {"error": str(e), "hands": []}


# Global detector
detector = HandDetector()


@app.route('/')
def index():
    """Hand tracking test page."""
    return render_template('hand_test.html')


@app.route('/record')
def recorder():
    """Video recorder page."""
    return render_template('recorder.html')


@app.route('/api/status')
def api_status():
    """Check if VisionDetector is available."""
    return jsonify({
        "available": VISION_DETECTOR_PATH.exists(),
        "path": str(VISION_DETECTOR_PATH),
        "running": detector.running,
    })


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    Receive JPEG frame, return hand landmarks.

    Expects multipart form with 'frame' as JPEG file.
    """
    if 'frame' not in request.files:
        return jsonify({"error": "No frame provided", "hands": []})

    frame_file = request.files['frame']
    jpeg_bytes = frame_file.read()

    result = detector.detect(jpeg_bytes)

    # Debug: log when hands detected
    hands = result.get('hands', [])
    if hands:
        h = hands[0]
        all_joints = ['wrist', 'thumbCMC', 'thumbMP', 'thumbIP', 'thumbTip',
                      'indexMCP', 'indexPIP', 'indexDIP', 'indexTip',
                      'middleMCP', 'middlePIP', 'middleDIP', 'middleTip',
                      'ringMCP', 'ringPIP', 'ringDIP', 'ringTip',
                      'littleMCP', 'littlePIP', 'littleDIP', 'littleTip']
        present = [j for j in all_joints if h.get(j)]
        missing = [j for j in all_joints if not h.get(j)]
        # Show actual wrist values
        wrist = h.get('wrist')
        wrist_info = f"wrist={wrist}" if wrist else "wrist=None"
        print(f"[DEBUG] Hand: {h.get('chirality')}, present={len(present)}/21, {wrist_info}")

    return jsonify(result)


def main():
    print("\n" + "=" * 50)
    print("HAND TRACKING TEST SERVER")
    print("=" * 50)
    print(f"VisionDetector: {'Found' if VISION_DETECTOR_PATH.exists() else 'NOT FOUND'}")
    if not VISION_DETECTOR_PATH.exists():
        print(f"  Build with: cd vision-detector && swift build -c release")
    print(f"\nOpen http://127.0.0.1:5002 in your browser")
    print("=" * 50 + "\n")

    app.run(host='127.0.0.1', port=5002, debug=False, threaded=True)


if __name__ == '__main__':
    main()
