#!/usr/bin/env python3
"""
Compare smoothed vs unsmoothed MANO skeleton rendering.

Loads per-frame MANO params from Dyn-HaMR track_preds output,
renders skeleton overlay with and without RTS smoothing,
and creates a side-by-side comparison video.

Usage:
    python 07_compare_smoothing.py <track_preds_dir> <video> -o <output.mp4>
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


# =============================================================================
# RTS Smoother (from 03_smooth_params.py)
# =============================================================================

class RTSSmoother:
    """Rauch-Tung-Striebel two-pass smoother."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self.q = process_noise
        self.r = measurement_noise

    def smooth(self, measurements: np.ndarray) -> np.ndarray:
        """Smooth a sequence of measurements. Input: (T, D), Output: (T, D)"""
        if len(measurements) < 3:
            return measurements.copy()

        T, D = measurements.shape

        # State: [position, velocity] for each dimension
        # Forward pass (Kalman filter)
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])  # State transition
        H = np.array([[1, 0]])  # Measurement matrix
        Q = self.q * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])  # Process noise
        R = np.array([[self.r]])  # Measurement noise

        smoothed = np.zeros_like(measurements)

        for d in range(D):
            z = measurements[:, d]

            # Forward pass
            x_fwd = np.zeros((T, 2))
            P_fwd = np.zeros((T, 2, 2))

            # Initialize
            x_fwd[0] = [z[0], 0]
            P_fwd[0] = np.eye(2) * 1.0

            for t in range(1, T):
                # Predict
                x_pred = F @ x_fwd[t-1]
                P_pred = F @ P_fwd[t-1] @ F.T + Q

                # Update
                y = z[t] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x_fwd[t] = x_pred + (K @ y).flatten()
                P_fwd[t] = (np.eye(2) - K @ H) @ P_pred

            # Backward pass (RTS smoothing)
            x_smooth = np.zeros((T, 2))
            x_smooth[-1] = x_fwd[-1]

            for t in range(T - 2, -1, -1):
                x_pred = F @ x_fwd[t]
                P_pred = F @ P_fwd[t] @ F.T + Q
                G = P_fwd[t] @ F.T @ np.linalg.inv(P_pred)
                x_smooth[t] = x_fwd[t] + G @ (x_smooth[t+1] - x_pred)

            smoothed[:, d] = x_smooth[:, 0]

        return smoothed


# =============================================================================
# MANO Joint Computation
# =============================================================================

# MANO joint connections (21 joints)
MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

# Haptica website teal: #00d4aa = RGB(0, 212, 170) = BGR(170, 212, 0)
HAPTICA_TEAL = (170, 212, 0)
FINGER_COLORS = {
    'thumb': HAPTICA_TEAL,
    'index': HAPTICA_TEAL,
    'middle': HAPTICA_TEAL,
    'ring': HAPTICA_TEAL,
    'pinky': HAPTICA_TEAL,
}


def load_track_data(track_dir: Path) -> Dict:
    """Load all MANO params from a track directory."""
    mano_files = sorted(track_dir.glob("*_mano.json"))

    frames = []
    for mano_file in mano_files:
        frame_num = int(mano_file.stem.split('_')[0])
        with open(mano_file) as f:
            data = json.load(f)

        # Also load keypoints for 2D projection
        kp_file = track_dir / f"{frame_num:06d}_keypoints.json"
        if kp_file.exists():
            with open(kp_file) as f:
                kp_data = json.load(f)
            # Keypoints are in people[0]["pose_keypoints_2d"] format: [[x,y,conf], ...]
            people = kp_data.get('people', [])
            if people:
                kp_list = people[0].get('pose_keypoints_2d', [])
                # Extract just x,y (ignore confidence)
                data['keypoints_2d'] = [[kp[0], kp[1]] for kp in kp_list]
            else:
                data['keypoints_2d'] = []

        data['frame_num'] = frame_num
        frames.append(data)

    return frames


def extract_params_array(frames: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Extract translation and keypoints as arrays for smoothing."""
    frame_nums = [f['frame_num'] for f in frames]

    # Extract 2D keypoints (21 joints x 2 coords = 42 values)
    keypoints = []
    for f in frames:
        kp = f.get('keypoints_2d', [])
        if len(kp) >= 21:
            kp_arr = np.array(kp[:21])  # (21, 2)
            kp_flat = kp_arr.flatten()  # (42,)
        else:
            kp_flat = np.zeros(42)
        keypoints.append(kp_flat)

    keypoints = np.array(keypoints)  # (T, 42)

    # Extract translation
    trans = []
    for f in frames:
        t = f.get('trans', [0, 0, 0])
        trans.append(t)
    trans = np.array(trans)  # (T, 3)

    return keypoints, trans, frame_nums


def get_joint_color(joint_idx: int) -> tuple:
    """Get color for a joint based on finger."""
    if joint_idx <= 4:
        return FINGER_COLORS['thumb']
    elif joint_idx <= 8:
        return FINGER_COLORS['index']
    elif joint_idx <= 12:
        return FINGER_COLORS['middle']
    elif joint_idx <= 16:
        return FINGER_COLORS['ring']
    else:
        return FINGER_COLORS['pinky']


def draw_skeleton(frame: np.ndarray, keypoints_2d: np.ndarray,
                  line_thickness: int = 2, point_radius: int = 4,
                  alpha: float = 0.8) -> np.ndarray:
    """Draw skeleton on frame from 2D keypoints."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Reshape to (21, 2)
    if keypoints_2d.shape[0] == 42:
        joints = keypoints_2d.reshape(21, 2)
    else:
        joints = keypoints_2d[:21, :2] if len(keypoints_2d) >= 21 else keypoints_2d

    if len(joints) < 21:
        return frame

    # Draw connections
    for (ja, jb) in MANO_CONNECTIONS:
        pt_a = joints[ja].astype(int)
        pt_b = joints[jb].astype(int)

        if not (0 <= pt_a[0] < w and 0 <= pt_a[1] < h):
            continue
        if not (0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
            continue

        color = get_joint_color(max(ja, jb))
        cv2.line(overlay, tuple(pt_a), tuple(pt_b), color, line_thickness, cv2.LINE_AA)

    # Draw joints
    for i, joint in enumerate(joints):
        pt = joint.astype(int)
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            continue

        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        cv2.circle(overlay, tuple(pt), point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(pt), point_radius, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def render_comparison(video_path: str, track_dirs: List[Path], output_path: str,
                      process_noise: float = 0.005, measurement_noise: float = 2.0):
    """Render side-by-side comparison of smoothed vs unsmoothed skeleton."""

    print("Loading track data...")
    all_tracks = []
    for track_dir in track_dirs:
        frames = load_track_data(track_dir)
        if frames:
            keypoints, trans, frame_nums = extract_params_array(frames)
            all_tracks.append({
                'frames': frames,
                'keypoints': keypoints,
                'trans': trans,
                'frame_nums': frame_nums,
                'frame_to_idx': {fn: i for i, fn in enumerate(frame_nums)}
            })

    if not all_tracks:
        print("ERROR: No track data found")
        return False

    print(f"Loaded {len(all_tracks)} tracks")

    # Apply smoothing
    print("Applying RTS smoothing...")
    smoother = RTSSmoother(process_noise=process_noise, measurement_noise=measurement_noise)

    for track in all_tracks:
        track['keypoints_smooth'] = smoother.smooth(track['keypoints'])

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output is side by side
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_idx in tqdm(range(total_frames), desc="Rendering comparison"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_unsmoothed = frame.copy()
        frame_smoothed = frame.copy()

        # Draw skeleton for each track
        for track in all_tracks:
            if frame_idx in track['frame_to_idx']:
                idx = track['frame_to_idx'][frame_idx]

                # Unsmoothed
                kp_raw = track['keypoints'][idx]
                frame_unsmoothed = draw_skeleton(frame_unsmoothed, kp_raw)

                # Smoothed
                kp_smooth = track['keypoints_smooth'][idx]
                frame_smoothed = draw_skeleton(frame_smoothed, kp_smooth)

        # Add labels
        cv2.putText(frame_unsmoothed, f"Raw (Unsmoothed) - Frame {frame_idx}", (10, 30),
                    font, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_smoothed, f"RTS Smoothed - Frame {frame_idx}", (10, 30),
                    font, 0.7, (0, 255, 0), 2)

        # Combine side by side
        combined = np.hstack([frame_unsmoothed, frame_smoothed])
        out.write(combined)

    cap.release()
    out.release()

    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare smoothed vs unsmoothed skeleton")
    parser.add_argument("track_preds_dir", help="Path to track_preds/<video_name>/ directory")
    parser.add_argument("video", help="Path to original video")
    parser.add_argument("-o", "--output", default="smoothing_comparison.mp4",
                        help="Output video path")
    parser.add_argument("--process-noise", type=float, default=0.005,
                        help="RTS process noise (lower = smoother)")
    parser.add_argument("--measurement-noise", type=float, default=2.0,
                        help="RTS measurement noise (higher = smoother)")

    args = parser.parse_args()

    track_preds_dir = Path(args.track_preds_dir)

    # Find track subdirectories (000, 001, etc.)
    track_dirs = sorted([d for d in track_preds_dir.iterdir() if d.is_dir()])

    if not track_dirs:
        print(f"ERROR: No track directories found in {track_preds_dir}")
        return

    print(f"Found tracks: {[d.name for d in track_dirs]}")

    render_comparison(
        args.video,
        track_dirs,
        args.output,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise
    )


if __name__ == "__main__":
    main()
