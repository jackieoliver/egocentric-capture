#!/usr/bin/env python3
"""
Render side-by-side GoPro video and 3D reconstruction (trajectory + hand + frustum).

Requirements:
  - OpenCV
  - numpy
  - Open3D (preferred) or PyVista (fallback)

Example:
  python scripts/render_3d_comparison.py \
    --video /path/to/video.mp4 \
    --trajectory /path/to/trajectory.npz \
    --hand-tracking /path/to/hand.npz \
    --output /path/to/output.mp4 \
    --duration 10 \
    --start 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# Prefer Open3D; fall back to PyVista if not available.
try:
    import open3d as o3d
    from open3d.visualization import rendering as o3dr
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

try:
    import pyvista as pv
    _HAS_PV = True
except Exception:
    _HAS_PV = False


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
    # choose closer of idx-1 and idx
    if abs(timestamps[idx] - t) < abs(timestamps[idx - 1] - t):
        return idx
    return idx - 1


def _load_hand_data(path):
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    # Try to find joint positions in the NPZ
    joints = None
    for key in ("joints", "joints_3d", "joints_cam", "joints_camera"):
        if key in data:
            joints = data[key]
            break

    # Normalize to shape (T, J, 3)
    if joints is not None:
        if joints.ndim == 4:  # (B, T, J, 3)
            joints = joints[0]
        if joints.ndim == 3:  # (T, J, 3)
            pass
        else:
            joints = None

    # Root translation fallback
    trans = data["trans"] if "trans" in data else None
    if trans is not None and trans.ndim == 3:  # (B, T, 3)
        trans = trans[0]

    intrins = data["intrins"] if "intrins" in data else None

    # Time base if present
    timestamps = data["timestamps"] if "timestamps" in data else None

    return {
        "joints": joints,
        "trans": trans,
        "intrins": intrins,
        "timestamps": timestamps,
    }


def _default_hand_edges(num_joints):
    """
    Default hand skeleton edges for 15 joints (approx MANO-like).
    This is a best-effort fallback; adjust if you have a known topology.
    """
    if num_joints < 5:
        return [(i, i + 1) for i in range(num_joints - 1)]
    edges = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14),
    ]
    # Filter edges if joints count is smaller
    return [(a, b) for a, b in edges if a < num_joints and b < num_joints]


def _build_frustum_lines(intrins, width, height, near=0.05, far=0.25):
    """
    Build a simple camera frustum in camera coordinates.

    Assumes camera frame: +X right, +Y down, +Z forward.
    """
    if intrins is not None and len(intrins) == 4:
        fx, fy, cx, cy = [float(x) for x in intrins]
    else:
        # Approximate intrinsics if unknown
        fx = fy = max(width, height)
        cx, cy = width * 0.5, height * 0.5

    def corner(z, x, y):
        return np.array([
            (x - cx) / fx * z,
            (y - cy) / fy * z,
            z,
        ], dtype=np.float64)

    # Image corners at near/far planes
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    near_pts = [corner(near, x, y) for x, y in corners]
    far_pts = [corner(far, x, y) for x, y in corners]

    pts = np.array(near_pts + far_pts, dtype=np.float64)
    # Lines: near loop, far loop, and 4 connecting edges
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return pts, lines


def _o3d_lineset(points, lines, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    if np.asarray(color).ndim == 1:
        colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
    else:
        colors = np.asarray(color, dtype=np.float64)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _render_o3d_frame(renderer, scene, width, height, geom_items, view):
    scene.clear_geometry()
    for name, geom, mat in geom_items:
        scene.add_geometry(name, geom, mat)
    scene.camera.look_at(view["center"], view["eye"], view["up"])
    img = np.asarray(renderer.render_to_image())
    # Open3D returns RGBA
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def _render_pyvista_frame(plotter):
    img = plotter.screenshot(transparent_background=False)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def main():
    p = argparse.ArgumentParser(description="Render side-by-side GoPro video and 3D reconstruction.")
    p.add_argument("--video", required=True, help="Path to input MP4 video")
    p.add_argument("--trajectory", required=True, help="Path to trajectory NPZ")
    p.add_argument("--hand-tracking", default=None, help="Path to Dyn-HaMR NPZ (optional)")
    p.add_argument("--output", required=True, help="Path to output MP4")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds to render")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--trail-seconds", type=float, default=2.0, help="Camera trail length in seconds")
    args = p.parse_args()

    video_path = Path(args.video)
    traj_path = Path(args.trajectory)
    hand_path = Path(args.hand_tracking) if args.hand_tracking else None
    out_path = Path(args.output)

    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not traj_path.exists():
        print(f"Trajectory not found: {traj_path}", file=sys.stderr)
        sys.exit(1)
    if hand_path is not None and not hand_path.exists():
        print(f"Hand tracking not found: {hand_path}", file=sys.stderr)
        sys.exit(1)

    # Load trajectory
    traj = np.load(traj_path, allow_pickle=True)
    timestamps = traj["timestamps"] if "timestamps" in traj else None
    positions = traj["positions"] if "positions" in traj else None
    orientations = traj["orientations"] if "orientations" in traj else None
    if timestamps is None or positions is None or orientations is None:
        print("Trajectory NPZ missing required keys: timestamps, positions, orientations.", file=sys.stderr)
        sys.exit(1)

    # Load hand data (optional)
    hand = _load_hand_data(hand_path) if hand_path else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek to start time
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = width * 2
    out_h = height
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    # Prepare 3D renderer
    if _HAS_O3D:
        renderer = o3dr.OffscreenRenderer(width, height)
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])

        mat = o3dr.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.line_width = 2.0

        # Precompute global view to keep stable framing
        traj_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(positions.astype(np.float64))
        )
        center = traj_bbox.get_center()
        extent = traj_bbox.get_extent()
        radius = max(1.0, np.linalg.norm(extent))
        view = {
            "center": center,
            "eye": center + np.array([radius, radius, radius], dtype=np.float64),
            "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        }

    elif _HAS_PV:
        pv.global_theme.background = "white"
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.camera_position = "iso"
    else:
        print("Neither Open3D nor PyVista is available.", file=sys.stderr)
        sys.exit(1)

    # Build frustum in camera coords (used each frame)
    frustum_pts_cam, frustum_lines = _build_frustum_lines(
        hand["intrins"] if hand else None, width, height
    )

    frame_count = int(round(args.duration * fps))
    for i in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        t = args.start + (i / fps)

        # Trajectory index and window
        ti = _nearest_index(timestamps, t)
        if ti is None:
            break

        # Trail window indices
        t0 = t - args.trail_seconds
        if t0 <= timestamps[0]:
            start_idx = 0
        else:
            start_idx = int(np.searchsorted(timestamps, t0, side="left"))
        trail_positions = positions[start_idx:ti + 1]

        # Current pose
        cam_pos = positions[ti]
        cam_rot = _quat_wxyz_to_rotmat(orientations[ti])

        # Transform frustum to world
        frustum_world = (cam_rot @ frustum_pts_cam.T).T + cam_pos

        # Hand skeleton (if available)
        hand_points = None
        hand_lines = None
        if hand and hand["joints"] is not None:
            hj = hand["joints"]
            if hand["timestamps"] is not None:
                hi = _nearest_index(hand["timestamps"], t)
            else:
                # assume same time base as trajectory
                hi = min(ti, len(hj) - 1)
            if hi is not None and hi < len(hj):
                hand_points = hj[hi]
                hand_lines = _default_hand_edges(hand_points.shape[0])
        elif hand and hand["trans"] is not None:
            # Fallback: use translation as a single point if joints are unavailable
            hi = min(ti, len(hand["trans"]) - 1)
            hand_points = hand["trans"][hi][None, :]
            hand_lines = []

        if _HAS_O3D:
            geom_items = []

            # Trajectory trail
            if len(trail_positions) >= 2:
                lines = [(j, j + 1) for j in range(len(trail_positions) - 1)]
                trail_ls = _o3d_lineset(trail_positions, lines, color=[0.1, 0.6, 0.9])
                geom_items.append(("traj", trail_ls, mat))

            # Current camera frustum
            frustum_ls = _o3d_lineset(frustum_world, frustum_lines, color=[1.0, 0.2, 0.2])
            geom_items.append(("frustum", frustum_ls, mat))

            # Hand skeleton
            if hand_points is not None:
                if len(hand_lines) > 0:
                    hand_ls = _o3d_lineset(hand_points, hand_lines, color=[0.2, 0.8, 0.2])
                    geom_items.append(("hand", hand_ls, mat))
                else:
                    # Single point fallback
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(hand_points.astype(np.float64))
                    pcd.paint_uniform_color([0.2, 0.8, 0.2])
                    geom_items.append(("hand_point", pcd, mat))

            render = _render_o3d_frame(renderer, scene, width, height, geom_items, view)

        else:
            plotter.clear()
            if len(trail_positions) >= 2:
                plotter.add_lines(trail_positions, color="dodgerblue", width=2.0)
            plotter.add_lines(frustum_world[frustum_lines].reshape(-1, 2, 3), color="red", width=2.0)
            if hand_points is not None and len(hand_points) >= 2:
                for a, b in hand_lines:
                    plotter.add_lines(hand_points[[a, b]], color="limegreen", width=2.0)
            elif hand_points is not None:
                plotter.add_points(hand_points, color="limegreen", point_size=10.0)
            render = _render_pyvista_frame(plotter)

        # Compose side-by-side
        if render.shape[:2] != frame.shape[:2]:
            render = cv2.resize(render, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        combo = np.hstack([frame, render])

        # Overlay frame counter + time
        label = f"Frame {i+1}/{frame_count} | t={t:.2f}s"
        cv2.putText(combo, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

        writer.write(combo)

    writer.release()
    cap.release()
    # Note: renderer cleanup not needed in newer Open3D versions

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
