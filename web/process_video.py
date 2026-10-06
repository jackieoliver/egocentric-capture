#!/usr/bin/env python3
"""
Video Hand Tracking Processor

Takes a raw video file and outputs a version with hand tracking skeleton overlay.
Uses VisionDetector (Swift) for hand pose detection.

Usage:
    python process_video.py input.webm output.mp4
    python process_video.py test_videos/raw/footage.webm test_videos/processed/footage_tracked.mp4
"""

import argparse
import cv2
import json
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue, Empty

# Path to VisionDetector
VISION_DETECTOR_PATH = Path(__file__).parent.parent / "vision-detector" / ".build" / "release" / "VisionDetector"

# Hand skeleton connections
FINGER_CONNECTIONS = {
    'thumb': ['wrist', 'thumbCMC', 'thumbMP', 'thumbIP', 'thumbTip'],
    'index': ['wrist', 'indexMCP', 'indexPIP', 'indexDIP', 'indexTip'],
    'middle': ['wrist', 'middleMCP', 'middlePIP', 'middleDIP', 'middleTip'],
    'ring': ['wrist', 'ringMCP', 'ringPIP', 'ringDIP', 'ringTip'],
    'little': ['wrist', 'littleMCP', 'littlePIP', 'littleDIP', 'littleTip'],
}

# Colors (BGR for OpenCV)
COLORS = {
    'left': (255, 0, 255),   # Magenta
    'right': (255, 255, 0),  # Cyan
}


class VisionDetector:
    """Manages VisionDetector subprocess for hand tracking."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.process = None
        self.running = False
        self._stderr_queue = Queue(maxsize=100)
        self._stderr_thread = None
        self._stdout_thread = None

    def start(self) -> bool:
        if not VISION_DETECTOR_PATH.exists():
            print(f"ERROR: VisionDetector not found at {VISION_DETECTOR_PATH}")
            print("Build with: cd vision-detector && swift build -c release")
            return False

        cmd = [
            str(VISION_DETECTOR_PATH),
            "transform",
            "--width", str(self.width),
            "--height", str(self.height),
            "--max-hands", "2",
            "--no-body",
            "--no-overlay",
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

            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_thread.start()

            self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
            self._stdout_thread.start()

            return True
        except Exception as e:
            print(f"Failed to start VisionDetector: {e}")
            return False

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()

    def _read_stderr(self):
        while self.running and self.process:
            try:
                line = self.process.stderr.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode('utf-8').strip())
                    try:
                        self._stderr_queue.put_nowait(data)
                    except:
                        try:
                            self._stderr_queue.get_nowait()
                            self._stderr_queue.put_nowait(data)
                        except:
                            pass
                except json.JSONDecodeError:
                    pass  # Ignore non-JSON output
            except:
                break

    def _drain_stdout(self):
        frame_size = self.width * self.height * 4
        while self.running and self.process:
            try:
                data = self.process.stdout.read(frame_size)
                if not data:
                    break
            except:
                break

    def detect(self, frame) -> dict:
        if not self.running:
            return {"hands": []}

        try:
            # Convert BGR to BGRA
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            # Send to VisionDetector
            self.process.stdin.write(bgra.tobytes())
            self.process.stdin.flush()

            # Wait for result
            try:
                return self._stderr_queue.get(timeout=0.5)
            except Empty:
                return {"hands": []}
        except Exception as e:
            return {"hands": []}


def draw_rounded_rect(frame, x, y, w, h, color, alpha=0.85):
    """Draw a semi-transparent rounded rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ui_overlay(frame, data):
    """Draw UI panels with stats and coordinates."""
    hands = data.get('hands', [])
    height, width = frame.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_mono = cv2.FONT_HERSHEY_SIMPLEX

    # Colors
    white = (255, 255, 255)
    gray = (150, 150, 150)
    green = (0, 255, 0)
    magenta = (255, 0, 255)
    cyan = (255, 255, 0)
    dark_bg = (20, 20, 20)

    # === TOP-LEFT: Stats Panel ===
    draw_rounded_rect(frame, 10, 10, 160, 70, dark_bg)

    # Overall confidence
    if hands:
        avg_conf = sum(h.get('avgConfidence', 0) for h in hands) / len(hands)
        conf_text = f"{int(avg_conf * 100)}%"
    else:
        conf_text = "--%"

    cv2.putText(frame, "Confidence", (20, 35), font, 0.45, gray, 1, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (110, 35), font, 0.5, green, 1, cv2.LINE_AA)

    cv2.putText(frame, "Joints", (20, 60), font, 0.45, gray, 1, cv2.LINE_AA)
    cv2.putText(frame, "21", (110, 60), font, 0.5, green, 1, cv2.LINE_AA)

    # === LEFT HAND PANEL ===
    left_hand = next((h for h in hands if h.get('chirality') == 'left'), None)
    if left_hand:
        panel_x, panel_y = 10, 90
        draw_rounded_rect(frame, panel_x, panel_y, 150, 140, dark_bg)

        # Left border accent
        cv2.line(frame, (panel_x, panel_y), (panel_x, panel_y + 140), magenta, 3)

        # Title
        cv2.putText(frame, "LEFT", (panel_x + 15, panel_y + 22), font, 0.5, magenta, 1, cv2.LINE_AA)

        # Confidence
        conf = int(left_hand.get('avgConfidence', 0) * 100)
        cv2.putText(frame, f"{conf}%", (panel_x + 15, panel_y + 52), font, 0.9, magenta, 2, cv2.LINE_AA)

        # Wrist
        wrist = left_hand.get('wrist')
        if wrist:
            cv2.putText(frame, "Wrist", (panel_x + 15, panel_y + 75), font, 0.35, gray, 1, cv2.LINE_AA)
            cv2.putText(frame, f"({wrist['x']:.2f}, {wrist['y']:.2f})", (panel_x + 15, panel_y + 92), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # Fingertips
        cv2.putText(frame, "Tips", (panel_x + 15, panel_y + 112), font, 0.35, gray, 1, cv2.LINE_AA)
        tips = ['thumbTip', 'indexTip', 'middleTip', 'ringTip', 'littleTip']
        tip_str = " ".join([f"{left_hand.get(t, {}).get('x', 0):.1f}" if left_hand.get(t) else "--" for t in tips])
        cv2.putText(frame, tip_str, (panel_x + 15, panel_y + 130), font, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    # === RIGHT HAND PANEL ===
    right_hand = next((h for h in hands if h.get('chirality') == 'right'), None)
    if right_hand:
        panel_x, panel_y = width - 160, 10
        draw_rounded_rect(frame, panel_x, panel_y, 150, 140, dark_bg)

        # Left border accent
        cv2.line(frame, (panel_x, panel_y), (panel_x, panel_y + 140), cyan, 3)

        # Title
        cv2.putText(frame, "RIGHT", (panel_x + 15, panel_y + 22), font, 0.5, cyan, 1, cv2.LINE_AA)

        # Confidence
        conf = int(right_hand.get('avgConfidence', 0) * 100)
        cv2.putText(frame, f"{conf}%", (panel_x + 15, panel_y + 52), font, 0.9, cyan, 2, cv2.LINE_AA)

        # Wrist
        wrist = right_hand.get('wrist')
        if wrist:
            cv2.putText(frame, "Wrist", (panel_x + 15, panel_y + 75), font, 0.35, gray, 1, cv2.LINE_AA)
            cv2.putText(frame, f"({wrist['x']:.2f}, {wrist['y']:.2f})", (panel_x + 15, panel_y + 92), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # Fingertips
        cv2.putText(frame, "Tips", (panel_x + 15, panel_y + 112), font, 0.35, gray, 1, cv2.LINE_AA)
        tips = ['thumbTip', 'indexTip', 'middleTip', 'ringTip', 'littleTip']
        tip_str = " ".join([f"{right_hand.get(t, {}).get('x', 0):.1f}" if right_hand.get(t) else "--" for t in tips])
        cv2.putText(frame, tip_str, (panel_x + 15, panel_y + 130), font, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    # === BOTTOM CENTER: Tracking Status ===
    if hands:
        status_text = "Tracking..."
        status_color = green
        border_color = green
    else:
        status_text = "Place hand in frame"
        status_color = gray
        border_color = (80, 80, 80)

    text_size = cv2.getTextSize(status_text, font, 0.5, 1)[0]
    status_w = text_size[0] + 40
    status_x = (width - status_w) // 2
    status_y = height - 50

    draw_rounded_rect(frame, status_x, status_y, status_w, 35, dark_bg)
    cv2.rectangle(frame, (status_x, status_y), (status_x + status_w, status_y + 35), border_color, 1)
    cv2.putText(frame, status_text, (status_x + 20, status_y + 24), font, 0.5, status_color, 1, cv2.LINE_AA)

    return frame


def draw_skeleton(frame, data):
    """Draw hand skeleton overlay on frame."""
    hands = data.get('hands', [])
    height, width = frame.shape[:2]

    for hand in hands:
        chirality = hand.get('chirality', 'right')
        color = COLORS.get(chirality, COLORS['right'])

        # Draw joints
        joints = [
            'wrist',
            'thumbCMC', 'thumbMP', 'thumbIP', 'thumbTip',
            'indexMCP', 'indexPIP', 'indexDIP', 'indexTip',
            'middleMCP', 'middlePIP', 'middleDIP', 'middleTip',
            'ringMCP', 'ringPIP', 'ringDIP', 'ringTip',
            'littleMCP', 'littlePIP', 'littleDIP', 'littleTip'
        ]

        for joint_name in joints:
            joint = hand.get(joint_name)
            if not joint:
                continue
            if joint.get('confidence', 0) < 0.1:
                continue

            x = int(joint['x'] * width)
            y = int((1 - joint['y']) * height)  # Flip Y
            radius = 12 if 'Tip' in joint_name else 8

            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)

        # Draw connections
        for finger, joint_names in FINGER_CONNECTIONS.items():
            points = []
            for joint_name in joint_names:
                joint = hand.get(joint_name)
                if not joint or joint.get('confidence', 0) < 0.1:
                    continue
                x = int(joint['x'] * width)
                y = int((1 - joint['y']) * height)
                points.append((x, y))

            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 4, cv2.LINE_AA)

    # Draw UI overlay on top
    frame = draw_ui_overlay(frame, data)

    return frame


def process_video(input_path: str, output_path: str):
    """Process video file and add hand tracking overlay."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return False

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.1f}s")
    print()

    # Initialize VisionDetector
    detector = VisionDetector(width, height)
    if not detector.start():
        cap.release()
        return False

    # Create output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"ERROR: Could not create output video: {output_path}")
        detector.stop()
        cap.release()
        return False

    # Create JSONL file for landmarks data
    jsonl_path = output_path.with_suffix('.jsonl')
    jsonl_file = open(jsonl_path, 'w')

    print(f"Processing...")

    frame_count = 0
    hands_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_ms = (frame_count - 1) * (1000 / fps)

        # Detect hands
        data = detector.detect(frame)

        if data.get('hands'):
            hands_detected += 1

        # Write frame data to JSONL
        frame_data = {
            'frame': frame_count,
            'timestamp_ms': round(timestamp_ms, 2),
            'hands_detected': len(data.get('hands', [])),
            'hands': data.get('hands', [])
        }
        jsonl_file.write(json.dumps(frame_data) + '\n')

        # Draw skeleton overlay
        frame = draw_skeleton(frame, data)

        # Write frame
        out.write(frame)

        # Progress
        if frame_count % 30 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"  {frame_count}/{total_frames} ({pct:.0f}%) - Hands detected: {hands_detected}/{frame_count}")

    # Cleanup
    cap.release()
    out.release()
    jsonl_file.close()
    detector.stop()

    print()
    print(f"Done!")
    print(f"Output video: {output_path}")
    print(f"Output data:  {jsonl_path}")
    print(f"Frames with hands: {hands_detected}/{frame_count} ({100*hands_detected/frame_count:.0f}%)")

    return True


def main():
    parser = argparse.ArgumentParser(description='Process video with hand tracking overlay')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output video file (will be .mp4)')
    args = parser.parse_args()

    success = process_video(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
