#!/usr/bin/env python3
"""
Render side-by-side GoPro video + 3D world visualization of Dyn-HaMR hands.

LEFT: original video
RIGHT: 3D scene with camera trajectory + hand skeletons in world coordinates

This script intentionally favors a working prototype over perfect hand accuracy.
If MANO/torch is not available, it uses a lightweight forward-kinematics (FK)
approximation to generate 3D joints from pose_body/root_orient/trans.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# Prefer Open3D if available; fallback to matplotlib (headless).
try:
    import open3d as o3d
    from open3d.visualization import rendering as o3dr
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


HAND_EDGES_21 = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _quat_wxyz_to_rotmat(q):
    """Convert quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _axis_angle_to_rotmat(aa):
    """Rodrigues formula for axis-angle (3,) -> (3,3)."""
    theta = np.linalg.norm(aa)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = aa / theta
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


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


def _load_dynhamr_npz(path):
    data = np.load(path, allow_pickle=True)
    pose = None
    for key in ("pose_body", "latent_pose"):
        if key in data:
            pose = data[key]
            break
    if pose is None:
        raise ValueError("Dyn-HaMR NPZ missing pose_body/latent_pose.")

    root_orient = data["root_orient"] if "root_orient" in data else None
    trans = data["trans"] if "trans" in data else None
    betas = data["betas"] if "betas" in data else None
    is_right = data["is_right"] if "is_right" in data else None
    intrins = data["intrins"] if "intrins" in data else None
    timestamps = data["timestamps"] if "timestamps" in data else None

    # Normalize shapes to (B, T, ...)
    if pose.ndim == 3:  # (T, 15, 3)
        pose = pose[None, ...]
    if root_orient is not None and root_orient.ndim == 2:
        root_orient = root_orient[None, ...]
    if trans is not None and trans.ndim == 2:
        trans = trans[None, ...]
    if is_right is not None and is_right.ndim == 1:
        is_right = is_right[None, ...]

    return {
        "pose": pose,
        "root_orient": root_orient,
        "trans": trans,
        "betas": betas,
        "is_right": is_right,
        "intrins": intrins,
        "timestamps": timestamps,
    }


def _build_frustum_lines(intrins, width, height, near=0.05, far=0.25):
    """Camera frustum in camera coordinates (+X right, +Y down, +Z forward)."""
    if intrins is not None and len(intrins) == 4:
        fx, fy, cx, cy = [float(x) for x in intrins]
    else:
        fx = fy = max(width, height)
        cx, cy = width * 0.5, height * 0.5

    def corner(z, x, y):
        return np.array([(x - cx) / fx * z, (y - cy) / fy * z, z], dtype=np.float64)

    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    near_pts = [corner(near, x, y) for x, y in corners]
    far_pts = [corner(far, x, y) for x, y in corners]

    pts = np.array(near_pts + far_pts, dtype=np.float64)
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return pts, lines


def _build_grid(size=2.0, step=0.25, z=0.0):
    """Build XY grid lines at a fixed Z."""
    lines = []
    pts = []
    num = int(np.ceil(size / step))
    coords = np.linspace(-size, size, 2 * num + 1)
    for c in coords:
        pts.append([-size, c, z])
        pts.append([size, c, z])
        lines.append([len(pts) - 2, len(pts) - 1])
        pts.append([c, -size, z])
        pts.append([c, size, z])
        lines.append([len(pts) - 2, len(pts) - 1])
    return np.array(pts, dtype=np.float64), lines


def _o3d_lineset(points, lines, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    color = np.asarray(color, dtype=np.float64)
    if color.ndim == 1:
        colors = np.tile(color, (len(lines), 1))
    else:
        colors = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _view_from_camera(cam_pos, cam_rot, follow_dist=1.5, follow_up=0.6, look_ahead=0.3):
    forward = cam_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = cam_rot @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    eye = cam_pos - forward * follow_dist + up * follow_up
    center = cam_pos + forward * look_ahead
    return {"eye": eye, "center": center, "up": up}


def _hand_fk_joints(pose_body, root_orient, trans, is_right):
    """
    Lightweight FK hand joints (21 joints) from axis-angle pose.
    This is an approximation; it ignores betas and MANO shape.
    """
    parents = [
        -1,  # 0 wrist
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19,
    ]

    offsets = np.zeros((21, 3), dtype=np.float64)
    offsets[1] = np.array([-0.03, 0.015, 0.02])
    offsets[2] = np.array([0.0, 0.0, 0.035])
    offsets[3] = np.array([0.0, 0.0, 0.025])
    offsets[4] = np.array([0.0, 0.0, 0.02])

    offsets[5] = np.array([-0.015, 0.0, 0.03])
    offsets[6] = np.array([0.0, 0.0, 0.04])
    offsets[7] = np.array([0.0, 0.0, 0.025])
    offsets[8] = np.array([0.0, 0.0, 0.02])

    offsets[9] = np.array([0.0, 0.0, 0.035])
    offsets[10] = np.array([0.0, 0.0, 0.045])
    offsets[11] = np.array([0.0, 0.0, 0.03])
    offsets[12] = np.array([0.0, 0.0, 0.022])

    offsets[13] = np.array([0.015, 0.0, 0.03])
    offsets[14] = np.array([0.0, 0.0, 0.043])
    offsets[15] = np.array([0.0, 0.0, 0.028])
    offsets[16] = np.array([0.0, 0.0, 0.021])

    offsets[17] = np.array([0.03, 0.005, 0.025])
    offsets[18] = np.array([0.0, 0.0, 0.038])
    offsets[19] = np.array([0.0, 0.0, 0.022])
    offsets[20] = np.array([0.0, 0.0, 0.018])

    if is_right < 0.5:
        offsets[:, 0] *= -1.0

    pose_to_joint = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    local_rot = [np.eye(3, dtype=np.float64) for _ in range(21)]
    for i, j_idx in enumerate(pose_to_joint):
        local_rot[j_idx] = _axis_angle_to_rotmat(pose_body[i])

    global_rot = [np.eye(3, dtype=np.float64) for _ in range(21)]
    joints = np.zeros((21, 3), dtype=np.float64)

    global_rot[0] = _axis_angle_to_rotmat(root_orient)
    joints[0] = trans

    for j in range(1, 21):
        p = parents[j]
        global_rot[j] = global_rot[p] @ local_rot[j]
        joints[j] = joints[p] + global_rot[p] @ offsets[j]

    return joints


def _render_o3d_frame(renderer, scene, width, height, geom_items, view):
    scene.clear_geometry()
    for name, geom, mat in geom_items:
        scene.add_geometry(name, geom, mat)
    scene.camera.look_at(view["center"], view["eye"], view["up"])
    img = np.asarray(renderer.render_to_image())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def _render_mpl_frame(fig, ax, width, height, trail_positions, frustum_world, frustum_lines,
                      hands_world, grid_pts, grid_lines, view):
    ax.clear()

    if grid_pts is not None and len(grid_pts) > 0:
        segs = np.array([[grid_pts[a], grid_pts[b]] for a, b in grid_lines])
        lc = Line3DCollection(segs, colors="#dddddd", linewidths=0.6, alpha=0.7)
        ax.add_collection3d(lc)

    if len(trail_positions) >= 2:
        ax.plot3D(trail_positions[:, 0], trail_positions[:, 1], trail_positions[:, 2],
                  color="#1f77b4", linewidth=2.0, alpha=0.9)

    frustum_segs = np.array([[frustum_world[a], frustum_world[b]] for a, b in frustum_lines])
    lc_f = Line3DCollection(frustum_segs, colors="#d62728", linewidths=1.5, alpha=0.9)
    ax.add_collection3d(lc_f)

    for hand in hands_world:
        color = hand["color"]
        joints = hand["joints"]
        segs = np.array([[joints[a], joints[b]] for a, b in HAND_EDGES_21])
        lc_h = Line3DCollection(segs, colors=[color], linewidths=2.0, alpha=0.95)
        ax.add_collection3d(lc_h)
        ax.scatter3D(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=6)

    ax.set_xlim(view["center"][0] - 2.0, view["center"][0] + 2.0)
    ax.set_ylim(view["center"][1] - 2.0, view["center"][1] + 2.0)
    ax.set_zlim(view["center"][2] - 2.0, view["center"][2] + 2.0)
    ax.view_init(elev=25, azim=135)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def main():
    p = argparse.ArgumentParser(description="Render side-by-side GoPro video + 3D hand world view.")
    p.add_argument("--video", required=True, help="Path to GoPro video")
    p.add_argument("--trajectory", required=True, help="Trajectory NPZ (from estimate_trajectory.py)")
    p.add_argument("--hands", required=True, help="Dyn-HaMR NPZ (world_results or raw_params)")
    p.add_argument("--output", required=True, help="Output MP4")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    p.add_argument("--trail-seconds", type=float, default=2.0, help="Trail length in seconds")
    p.add_argument("--traj-time-offset", type=float, default=0.0, help="Seconds added to video time for trajectory")
    p.add_argument("--hand-time-offset", type=float, default=0.0, help="Seconds added to video time for hand data")
    p.add_argument("--hand-fps", type=float, default=0.0, help="Hand FPS if no timestamps (0 = use video fps)")
    p.add_argument("--hand-frame-offset", type=int, default=0, help="Frame offset applied to hand index")
    p.add_argument("--grid-size", type=float, default=2.0, help="Half-size of ground grid (meters)")
    p.add_argument("--grid-step", type=float, default=0.25, help="Grid spacing (meters)")
    p.add_argument("--grid-z", type=float, default=0.0, help="Grid plane Z")
    p.add_argument("--renderer", choices=["auto", "open3d", "mpl"], default="auto")
    args = p.parse_args()

    video_path = Path(args.video)
    traj_path = Path(args.trajectory)
    hands_path = Path(args.hands)
    out_path = Path(args.output)

    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not traj_path.exists():
        print(f"Trajectory not found: {traj_path}", file=sys.stderr)
        sys.exit(1)
    if not hands_path.exists():
        print(f"Hands not found: {hands_path}", file=sys.stderr)
        sys.exit(1)

    traj = np.load(traj_path, allow_pickle=True)
    if "timestamps" not in traj or "positions" not in traj or "orientations" not in traj:
        print("Trajectory NPZ missing required keys: timestamps, positions, orientations.", file=sys.stderr)
        sys.exit(1)
    traj_t = traj["timestamps"]
    traj_pos = traj["positions"]
    traj_q = traj["orientations"]

    hand = _load_dynhamr_npz(hands_path)
    pose = hand["pose"]
    root_orient = hand["root_orient"]
    trans = hand["trans"]
    is_right = hand["is_right"]
    hand_t = hand["timestamps"]
    intrins = hand["intrins"]

    if root_orient is None or trans is None or is_right is None:
        print("Hands NPZ missing required keys: root_orient, trans, is_right.", file=sys.stderr)
        sys.exit(1)

    num_hands, num_frames = pose.shape[0], pose.shape[1]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 1e-3:
        video_fps = 30.0
    hand_fps = args.hand_fps if args.hand_fps > 0 else video_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (width * 2, height))

    use_o3d = _HAS_O3D and (args.renderer in ("auto", "open3d"))
    use_mpl = _HAS_MPL and (args.renderer in ("auto", "mpl")) and not use_o3d

    if not use_o3d and not use_mpl:
        print("No available renderer: install Open3D or matplotlib.", file=sys.stderr)
        sys.exit(1)

    if use_o3d:
        renderer = o3dr.OffscreenRenderer(width, height)
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = o3dr.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.line_width = 2.0
    else:
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        fig.tight_layout(pad=0.2)
        FigureCanvasAgg(fig)

    frustum_pts_cam, frustum_lines = _build_frustum_lines(intrins, width, height)
    grid_pts, grid_lines = _build_grid(size=args.grid_size, step=args.grid_step, z=args.grid_z)

    frame_count = int(round(args.duration * video_fps))
    print(f"Rendering {frame_count} frames at {video_fps:.1f} fps...")

    for i in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        t_video = args.start + (i / video_fps)

        t_traj = t_video + args.traj_time_offset
        ti = _nearest_index(traj_t, t_traj)
        if ti is None:
            break

        t0 = t_traj - args.trail_seconds
        start_idx = max(0, int(np.searchsorted(traj_t, t0, side="left")))
        trail_positions = traj_pos[start_idx:ti + 1]

        cam_pos = traj_pos[ti]
        cam_rot = _quat_wxyz_to_rotmat(traj_q[ti])

        frustum_world = (cam_rot @ frustum_pts_cam.T).T + cam_pos

        t_hand = t_video + args.hand_time_offset
        hands_world = []
        for h in range(num_hands):
            if hand_t is not None:
                hi = _nearest_index(hand_t, t_hand)
            else:
                hi = int(round(t_hand * hand_fps)) + args.hand_frame_offset
                hi = max(0, min(num_frames - 1, hi))
            if hi is None:
                continue
            joints_cam = _hand_fk_joints(
                pose[h, hi],
                root_orient[h, hi],
                trans[h, hi],
                is_right[h, hi],
            )
            joints_world = (cam_rot @ joints_cam.T).T + cam_pos
            color = [0.1, 0.7, 0.2] if is_right[h, hi] > 0.5 else [0.2, 0.4, 0.9]
            hands_world.append({"joints": joints_world, "color": color})

        view = _view_from_camera(cam_pos, cam_rot)

        if use_o3d:
            geom_items = []
            if grid_pts is not None and len(grid_pts) > 0:
                grid_ls = _o3d_lineset(grid_pts, grid_lines, color=[0.85, 0.85, 0.85])
                geom_items.append(("grid", grid_ls, mat))

            if len(trail_positions) >= 2:
                lines = [(j, j + 1) for j in range(len(trail_positions) - 1)]
                trail_ls = _o3d_lineset(trail_positions, lines, color=[0.1, 0.6, 0.9])
                geom_items.append(("traj", trail_ls, mat))

            frustum_ls = _o3d_lineset(frustum_world, frustum_lines, color=[1.0, 0.2, 0.2])
            geom_items.append(("frustum", frustum_ls, mat))

            for idx, hand_item in enumerate(hands_world):
                joints = hand_item["joints"]
                color = hand_item["color"]
                hand_ls = _o3d_lineset(joints, HAND_EDGES_21, color=color)
                geom_items.append((f"hand_{idx}", hand_ls, mat))

            render = _render_o3d_frame(renderer, scene, width, height, geom_items, view)
        else:
            render = _render_mpl_frame(
                fig, ax, width, height,
                trail_positions, frustum_world, frustum_lines,
                hands_world, grid_pts, grid_lines, view,
            )

        if render.shape[:2] != frame.shape[:2]:
            render = cv2.resize(render, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        combo = np.hstack([frame, render])

        label = f"Frame {i+1}/{frame_count} | t={t_video:.2f}s"
        cv2.putText(combo, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        writer.write(combo)

    writer.release()
    cap.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
