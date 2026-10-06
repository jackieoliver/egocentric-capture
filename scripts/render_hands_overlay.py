#!/usr/bin/env python3
"""
Overlay Dyn-HaMR hand skeletons directly on video frames.
Projects 3D joints to 2D using camera intrinsics.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# 21-joint MANO hand skeleton
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

# Colors for each finger (BGR)
FINGER_COLORS = [
    (147, 20, 255),   # Thumb - pink
    (0, 255, 255),    # Index - yellow
    (0, 255, 0),      # Middle - green
    (255, 255, 0),    # Ring - cyan
    (255, 0, 0),      # Pinky - blue
]

def axis_angle_to_rotmat(aa):
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
    """Simple FK to get 21 joint positions in camera space."""
    parents = [-1, 0,1,2,3, 0,5,6,7, 0,9,10,11, 0,13,14,15, 0,17,18,19]

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

    if is_right < 0.5:
        offsets[:, 0] *= -1

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

def project_to_2d(joints_3d, intrins, width, height, bbox=None, trans=None):
    """Project 3D joints to 2D pixel coordinates.

    Uses HaMeR's coordinate conventions:
    - If bbox is provided, uses bbox center for wrist and scales offsets
    - Otherwise falls back to pinhole projection with focal=5000

    Args:
        joints_3d: (21, 3) array of joint positions in 3D
        intrins: Optional [fx, fy, cx, cy] intrinsics
        width, height: Image dimensions
        bbox: Optional [cx, cy, w, h] bounding box from HaMeR
        trans: Optional cam_trans for computing projection offset
    """
    if intrins is not None and len(intrins) >= 4:
        fx, fy, cx, cy = intrins[:4]
    else:
        # HaMeR uses base focal length = 5000 for cam_crop_to_full
        fx = fy = 5000.0
        cx, cy = width / 2, height / 2

    # If bbox is provided, use it for accurate positioning
    # The bbox center should correspond to the wrist position
    if bbox is not None and trans is not None and len(bbox) >= 2:
        bbox_cx, bbox_cy = bbox[0], bbox[1]

        # Compute projection offset: difference between naive projection and bbox center
        wrist_proj_x = fx * trans[0] / trans[2] + cx
        wrist_proj_y = fy * trans[1] / trans[2] + cy
        offset_x = wrist_proj_x - bbox_cx
        offset_y = wrist_proj_y - bbox_cy
    else:
        offset_x, offset_y = 0, 0

    pts_2d = []
    for pt in joints_3d:
        if pt[2] > 0.01:  # Only project if in front of camera
            x = int(fx * pt[0] / pt[2] + cx - offset_x)
            y = int(fy * pt[1] / pt[2] + cy - offset_y)
            pts_2d.append((x, y))
        else:
            pts_2d.append(None)
    return pts_2d

def draw_skeleton(frame, pts_2d, is_right, line_thickness=3, joint_radius=5):
    """Draw hand skeleton on frame with colored fingers."""
    if is_right > 0.5:
        base_color = (0, 200, 100)  # Green for right
    else:
        base_color = (200, 100, 0)  # Blue for left

    # Draw bones by finger
    finger_edges = [
        [(0, 1), (1, 2), (2, 3), (3, 4)],      # Thumb
        [(0, 5), (5, 6), (6, 7), (7, 8)],      # Index
        [(0, 9), (9, 10), (10, 11), (11, 12)], # Middle
        [(0, 13), (13, 14), (14, 15), (15, 16)], # Ring
        [(0, 17), (17, 18), (18, 19), (19, 20)], # Pinky
    ]

    for finger_idx, edges in enumerate(finger_edges):
        color = FINGER_COLORS[finger_idx]
        for i, j in edges:
            if pts_2d[i] is not None and pts_2d[j] is not None:
                cv2.line(frame, pts_2d[i], pts_2d[j], color, line_thickness, cv2.LINE_AA)

    # Draw joints
    for idx, pt in enumerate(pts_2d):
        if pt is not None:
            # Wrist is white, fingertips are brighter
            if idx == 0:
                cv2.circle(frame, pt, joint_radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            elif idx in [4, 8, 12, 16, 20]:  # Fingertips
                finger_idx = [4, 8, 12, 16, 20].index(idx)
                cv2.circle(frame, pt, joint_radius + 1, FINGER_COLORS[finger_idx], -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, joint_radius, base_color, -1, cv2.LINE_AA)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--hands", required=True, help="Dyn-HaMR NPZ")
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--hand-fps", type=float, default=0.0)
    p.add_argument("--line-thickness", type=int, default=3)
    p.add_argument("--joint-radius", type=int, default=5)
    args = p.parse_args()

    # Load hand data
    data = np.load(args.hands, allow_pickle=True)
    pose = data.get('pose_body', data.get('latent_pose'))
    root_orient = data['root_orient']
    trans = data['trans']
    is_right = data['is_right']
    intrins = data.get('intrins', None)
    bbox = data.get('bbox', None)

    if pose.ndim == 3:
        pose = pose[None, ...]
        root_orient = root_orient[None, ...]
        trans = trans[None, ...]
        is_right = is_right[None, ...]
        if bbox is not None:
            bbox = bbox[None, ...]

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
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = int(args.duration * fps)
    print(f"Rendering {frame_count} frames at {fps:.1f} fps...")

    for i in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        t = args.start + i / fps
        hi = min(int(t * hand_fps), num_frames - 1)

        for h in range(num_hands):
            joints_3d = hand_fk_joints(pose[h, hi], root_orient[h, hi],
                                       trans[h, hi], is_right[h, hi])
            hand_bbox = bbox[h, hi] if bbox is not None else None
            hand_trans = trans[h, hi]
            pts_2d = project_to_2d(joints_3d, intrins, width, height,
                                   bbox=hand_bbox, trans=hand_trans)
            draw_skeleton(frame, pts_2d, is_right[h, hi],
                         args.line_thickness, args.joint_radius)

        # Frame counter
        label = f"t={t:.2f}s"
        cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)

        if (i+1) % 30 == 0:
            print(f"  {i+1}/{frame_count}")

    writer.release()
    cap.release()
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
