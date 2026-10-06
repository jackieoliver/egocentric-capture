#!/usr/bin/env python3
"""
Render side-by-side GoPro video and 3D reconstruction using matplotlib.

This version uses matplotlib's 3D plotting which is more reliable for
headless rendering than Open3D.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _quat_wxyz_to_rotmat(q):
    """Convert quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def _nearest_index(timestamps, t):
    """Find nearest index in timestamps for time t."""
    if timestamps is None or len(timestamps) == 0:
        return None
    idx = int(np.searchsorted(timestamps, t, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - t) < abs(timestamps[idx - 1] - t):
        return idx
    return idx - 1


def _build_frustum(scale=0.3):
    """Build camera frustum vertices in camera coordinates."""
    # Simple pyramid frustum
    pts = np.array([
        [0, 0, 0],           # Camera origin
        [-0.5, -0.3, 1],     # Near plane corners
        [0.5, -0.3, 1],
        [0.5, 0.3, 1],
        [-0.5, 0.3, 1],
    ], dtype=np.float64) * scale

    # Lines connecting the frustum
    lines = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Rays from origin
        (1, 2), (2, 3), (3, 4), (4, 1),  # Near plane
    ]
    return pts, lines


def _render_frame(fig, ax, trail_positions, cam_pos, cam_rot, frustum_pts, frustum_lines,
                  width, height, trail_color='dodgerblue', frustum_color='crimson'):
    """Render a single 3D frame."""
    ax.clear()

    # Plot trajectory trail
    if len(trail_positions) >= 2:
        ax.plot3D(trail_positions[:, 0], trail_positions[:, 1], trail_positions[:, 2],
                  color=trail_color, linewidth=2.5, alpha=0.8)
        # Mark current position
        ax.scatter3D([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                     color=frustum_color, s=50, marker='o')

    # Transform and plot frustum
    frustum_world = (cam_rot @ frustum_pts.T).T + cam_pos
    for i, j in frustum_lines:
        ax.plot3D([frustum_world[i, 0], frustum_world[j, 0]],
                  [frustum_world[i, 1], frustum_world[j, 1]],
                  [frustum_world[i, 2], frustum_world[j, 2]],
                  color=frustum_color, linewidth=1.5)

    # Set view to follow camera (look from behind and above)
    # Compute viewing position relative to camera
    look_back = cam_rot @ np.array([0, 0, -2.0])  # Behind camera
    look_up = cam_rot @ np.array([0, -1.0, 0])    # Above camera
    view_pos = cam_pos + look_back + look_up * 0.5

    # Set axis limits centered on recent trajectory
    if len(trail_positions) >= 2:
        center = np.mean(trail_positions[-100:], axis=0)  # Recent center
        spread = max(1.0, np.max(np.std(trail_positions[-100:], axis=0)) * 4)
    else:
        center = cam_pos
        spread = 2.0

    ax.set_xlim(center[0] - spread, center[0] + spread)
    ax.set_ylim(center[1] - spread, center[1] + spread)
    ax.set_zlim(center[2] - spread, center[2] + spread)

    # Styling
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_facecolor('#f8f8f8')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    # View angle that follows camera direction
    # Compute azimuth and elevation from camera orientation
    forward = cam_rot @ np.array([0, 0, 1])
    azim = np.degrees(np.arctan2(forward[0], forward[2]))
    elev = np.degrees(np.arcsin(-forward[1] / (np.linalg.norm(forward) + 1e-8)))
    ax.view_init(elev=max(-60, min(60, elev + 30)), azim=azim + 180)

    # Render to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Resize to match video dimensions
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def main():
    p = argparse.ArgumentParser(description="Render side-by-side video + 3D trajectory (matplotlib version).")
    p.add_argument("--video", required=True, help="Path to input MP4 video")
    p.add_argument("--trajectory", required=True, help="Path to trajectory NPZ")
    p.add_argument("--output", required=True, help="Path to output MP4")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds to render")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--trail-seconds", type=float, default=3.0, help="Camera trail length")
    p.add_argument("--fps", type=float, default=0, help="Output FPS (0 = match input)")
    args = p.parse_args()

    video_path = Path(args.video)
    traj_path = Path(args.trajectory)
    out_path = Path(args.output)

    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not traj_path.exists():
        print(f"Trajectory not found: {traj_path}", file=sys.stderr)
        sys.exit(1)

    # Load trajectory
    traj = np.load(traj_path, allow_pickle=True)
    timestamps = traj["timestamps"]
    positions = traj["positions"]
    orientations = traj["orientations"]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.fps > 0:
        fps = args.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width * 2, height))

    # Setup matplotlib figure
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout(pad=0.5)
    canvas = FigureCanvasAgg(fig)

    # Build frustum
    frustum_pts, frustum_lines = _build_frustum(scale=0.5)

    frame_count = int(round(args.duration * fps))
    print(f"Rendering {frame_count} frames at {fps:.1f} fps...")

    for i in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            print(f"Video ended at frame {i}")
            break

        t = args.start + (i / fps)
        ti = _nearest_index(timestamps, t)
        if ti is None:
            break

        # Get trail
        t0 = t - args.trail_seconds
        start_idx = max(0, int(np.searchsorted(timestamps, t0, side="left")))
        trail_positions = positions[start_idx:ti + 1]

        # Current pose
        cam_pos = positions[ti]
        cam_rot = _quat_wxyz_to_rotmat(orientations[ti])

        # Render 3D view
        render = _render_frame(fig, ax, trail_positions, cam_pos, cam_rot,
                               frustum_pts, frustum_lines, width, height)

        # Combine side by side
        combo = np.hstack([frame, render])

        # Add overlay text
        label = f"Frame {i+1}/{frame_count} | t={t:.2f}s | pos=[{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}]m"
        cv2.putText(combo, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(combo, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(combo)

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{frame_count} frames...")

    writer.release()
    cap.release()
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
