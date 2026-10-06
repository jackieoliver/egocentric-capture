#!/usr/bin/env python3
"""
Render Dyn-HaMR hands in camera space - simple visualization without IMU trajectory.

This shows the hand skeletons as Dyn-HaMR sees them, directly overlaid or side-by-side.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 21-joint MANO hand skeleton
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

def axis_angle_to_rotmat(aa):
    """Rodrigues formula."""
    theta = np.linalg.norm(aa)
    if theta < 1e-10:
        return np.eye(3)
    axis = aa / theta
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])

def hand_fk_joints(pose_body, root_orient, trans, is_right):
    """Simple FK to get 21 joint positions."""
    parents = [-1, 0,1,2,3, 0,5,6,7, 0,9,10,11, 0,13,14,15, 0,17,18,19]

    # Approximate bone lengths (meters)
    offsets = np.zeros((21, 3))
    # Thumb
    offsets[1] = [-0.025, 0.01, 0.015]
    offsets[2] = [0, 0, 0.03]
    offsets[3] = [0, 0, 0.02]
    offsets[4] = [0, 0, 0.018]
    # Index
    offsets[5] = [-0.01, 0, 0.06]
    offsets[6] = [0, 0, 0.035]
    offsets[7] = [0, 0, 0.022]
    offsets[8] = [0, 0, 0.018]
    # Middle
    offsets[9] = [0, 0, 0.065]
    offsets[10] = [0, 0, 0.04]
    offsets[11] = [0, 0, 0.025]
    offsets[12] = [0, 0, 0.02]
    # Ring
    offsets[13] = [0.01, 0, 0.06]
    offsets[14] = [0, 0, 0.035]
    offsets[15] = [0, 0, 0.022]
    offsets[16] = [0, 0, 0.018]
    # Pinky
    offsets[17] = [0.025, 0.005, 0.05]
    offsets[18] = [0, 0, 0.028]
    offsets[19] = [0, 0, 0.018]
    offsets[20] = [0, 0, 0.015]

    if is_right < 0.5:  # Left hand - flip X
        offsets[:, 0] *= -1

    # Map pose_body indices to joints
    pose_to_joint = [1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19]
    local_rot = [np.eye(3) for _ in range(21)]
    for i, j in enumerate(pose_to_joint):
        local_rot[j] = axis_angle_to_rotmat(pose_body[i])

    global_rot = [np.eye(3) for _ in range(21)]
    joints = np.zeros((21, 3))

    global_rot[0] = axis_angle_to_rotmat(root_orient)
    joints[0] = trans

    for j in range(1, 21):
        p = parents[j]
        global_rot[j] = global_rot[p] @ local_rot[j]
        joints[j] = joints[p] + global_rot[p] @ offsets[j]

    return joints

def render_3d_view(joints_list, colors, width, height):
    """Render hands in 3D matplotlib view."""
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Collect all points for axis limits
    all_pts = []
    for joints in joints_list:
        all_pts.append(joints)
    all_pts = np.vstack(all_pts) if all_pts else np.zeros((1, 3))

    center = np.mean(all_pts, axis=0)
    spread = max(0.15, np.max(np.abs(all_pts - center)))

    for joints, color in zip(joints_list, colors):
        # Draw skeleton
        for i, j in HAND_EDGES:
            ax.plot3D([joints[i,0], joints[j,0]],
                     [joints[i,1], joints[j,1]],
                     [joints[i,2], joints[j,2]],
                     color=color, linewidth=2.5)
        # Draw joints
        ax.scatter3D(joints[:,0], joints[:,1], joints[:,2],
                    color=color, s=25)

    ax.set_xlim(center[0] - spread, center[0] + spread)
    ax.set_ylim(center[1] - spread, center[1] + spread)
    ax.set_zlim(center[2] - spread, center[2] + spread)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=15, azim=-60)
    ax.set_facecolor('#f5f5f5')
    fig.tight_layout(pad=0.3)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    return cv2.resize(img, (width, height))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--hands", required=True, help="Dyn-HaMR NPZ")
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--hand-fps", type=float, default=0.0)
    args = p.parse_args()

    # Load hand data
    data = np.load(args.hands, allow_pickle=True)
    pose = data.get('pose_body', data.get('latent_pose'))
    root_orient = data['root_orient']
    trans = data['trans']
    is_right = data['is_right']

    if pose.ndim == 3:
        pose = pose[None, ...]
        root_orient = root_orient[None, ...]
        trans = trans[None, ...]
        is_right = is_right[None, ...]

    num_hands, num_frames = pose.shape[0], pose.shape[1]
    print(f"Loaded {num_hands} hands, {num_frames} frames")

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hand_fps = args.hand_fps if args.hand_fps > 0 else fps

    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width * 2, height))

    frame_count = int(args.duration * fps)
    print(f"Rendering {frame_count} frames...")

    for i in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        t = args.start + i / fps
        hi = min(int(t * hand_fps), num_frames - 1)

        joints_list = []
        colors = []
        for h in range(num_hands):
            joints = hand_fk_joints(pose[h, hi], root_orient[h, hi],
                                   trans[h, hi], is_right[h, hi])
            joints_list.append(joints)
            colors.append('#2ecc71' if is_right[h, hi] > 0.5 else '#3498db')

        render = render_3d_view(joints_list, colors, width, height)
        combo = np.hstack([frame, render])

        label = f"Frame {i+1}/{frame_count} | t={t:.2f}s"
        cv2.putText(combo, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255,255,255), 2)
        cv2.putText(combo, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,0,0), 1)

        writer.write(combo)

        if (i+1) % 30 == 0:
            print(f"  {i+1}/{frame_count}")

    writer.release()
    cap.release()
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
