#!/usr/bin/env python3
"""
Render skeleton overlay from Dyn-HaMR MANO params.

Takes the params.npz output from Dyn-HaMR and renders a clean skeleton
overlay on the original video - better for demos than full mesh.

Usage:
    python 04_render_skeleton.py params.npz video.mp4 -o output.mp4
"""

import argparse
import numpy as np
import cv2
from pathlib import Path

# MANO joint connections (21 joints)
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

SKELETON_CONNECTIONS = [
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
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]

# Colors for each finger (BGR)
# Haptica website teal: #00d4aa = RGB(0, 212, 170) = BGR(170, 212, 0)
HAPTICA_TEAL = (170, 212, 0)
FINGER_COLORS = {
    'wrist': HAPTICA_TEAL,
    'thumb': HAPTICA_TEAL,
    'index': HAPTICA_TEAL,
    'middle': HAPTICA_TEAL,
    'ring': HAPTICA_TEAL,
    'pinky': HAPTICA_TEAL,
    'palm': HAPTICA_TEAL,
}

def get_connection_color(joint_a: int, joint_b: int) -> tuple:
    """Get color for a skeleton connection based on finger."""
    if joint_a == 0 or joint_b == 0:
        # Connection to wrist
        if max(joint_a, joint_b) <= 4:
            return FINGER_COLORS['thumb']
        elif max(joint_a, joint_b) <= 8:
            return FINGER_COLORS['index']
        elif max(joint_a, joint_b) <= 12:
            return FINGER_COLORS['middle']
        elif max(joint_a, joint_b) <= 16:
            return FINGER_COLORS['ring']
        else:
            return FINGER_COLORS['pinky']
    elif 1 <= joint_a <= 4 or 1 <= joint_b <= 4:
        return FINGER_COLORS['thumb']
    elif 5 <= joint_a <= 8 or 5 <= joint_b <= 8:
        return FINGER_COLORS['index']
    elif 9 <= joint_a <= 12 or 9 <= joint_b <= 12:
        return FINGER_COLORS['middle']
    elif 13 <= joint_a <= 16 or 13 <= joint_b <= 16:
        return FINGER_COLORS['ring']
    elif 17 <= joint_a <= 20 or 17 <= joint_b <= 20:
        return FINGER_COLORS['pinky']
    else:
        return FINGER_COLORS['palm']


def load_mano_joints(params_path: str) -> np.ndarray:
    """
    Load joint positions from Dyn-HaMR output.

    Dyn-HaMR saves various formats - this handles the common ones.
    Returns: (num_frames, num_hands, 21, 3) or (num_frames, 21, 3)
    """
    data = np.load(params_path, allow_pickle=True)

    # Check what keys are available
    keys = list(data.keys())
    print(f"Available keys in params: {keys}")

    # Dyn-HaMR typically saves joints or we need to compute from MANO params
    if 'joints_3d' in data:
        return data['joints_3d']
    elif 'joints' in data:
        return data['joints']
    elif 'pred_joints' in data:
        return data['pred_joints']
    elif 'j3d' in data:
        return data['j3d']
    else:
        # Need to compute from MANO params - requires MANO model
        print("WARNING: No precomputed joints found. Keys available:", keys)
        print("You may need to run MANO forward pass to get joints.")
        print("Looking for camera-space vertices or keypoints...")

        # Try to find any 3D keypoint data
        for key in keys:
            arr = data[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                print(f"  {key}: shape {arr.shape}")

        return None


def project_to_2d(joints_3d: np.ndarray,
                  img_width: int,
                  img_height: int,
                  focal_length: float = None,
                  camera_matrix: np.ndarray = None) -> np.ndarray:
    """
    Project 3D joints to 2D image coordinates.

    If no camera matrix provided, uses simple perspective projection.
    """
    if camera_matrix is not None:
        # Use provided camera matrix
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    else:
        # Assume simple perspective with focal length ~ image width
        fx = fy = focal_length or img_width
        cx, cy = img_width / 2, img_height / 2

    # Project: x_2d = fx * X/Z + cx, y_2d = fy * Y/Z + cy
    z = joints_3d[..., 2:3]
    z = np.where(z == 0, 1e-6, z)  # Avoid division by zero

    x_2d = fx * joints_3d[..., 0:1] / z + cx
    y_2d = fy * joints_3d[..., 1:2] / z + cy

    return np.concatenate([x_2d, y_2d], axis=-1)


def draw_skeleton(frame: np.ndarray,
                  joints_2d: np.ndarray,
                  line_thickness: int = 3,
                  point_radius: int = 5,
                  alpha: float = 0.8) -> np.ndarray:
    """
    Draw skeleton on frame.

    Args:
        frame: BGR image
        joints_2d: (21, 2) or (num_hands, 21, 2) array of 2D joint positions
        line_thickness: Thickness of skeleton lines
        point_radius: Radius of joint circles
        alpha: Transparency (1.0 = opaque)
    """
    overlay = frame.copy()

    # Handle single or multiple hands
    if joints_2d.ndim == 2:
        joints_2d = joints_2d[np.newaxis, ...]  # Add hand dimension

    for hand_joints in joints_2d:
        # Draw connections
        for (joint_a, joint_b) in SKELETON_CONNECTIONS:
            pt_a = tuple(hand_joints[joint_a].astype(int))
            pt_b = tuple(hand_joints[joint_b].astype(int))

            # Skip if points are outside frame
            h, w = frame.shape[:2]
            if not (0 <= pt_a[0] < w and 0 <= pt_a[1] < h):
                continue
            if not (0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
                continue

            color = get_connection_color(joint_a, joint_b)
            cv2.line(overlay, pt_a, pt_b, color, line_thickness, cv2.LINE_AA)

        # Draw joints
        for i, joint in enumerate(hand_joints):
            pt = tuple(joint.astype(int))
            h, w = frame.shape[:2]
            if not (0 <= pt[0] < w and 0 <= pt[1] < h):
                continue

            # Wrist is white, fingertips are brighter
            if i == 0:
                color = FINGER_COLORS['wrist']
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                color = (255, 255, 255)
            else:
                color = (200, 200, 200)

            cv2.circle(overlay, pt, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, point_radius, (0, 0, 0), 1, cv2.LINE_AA)  # Border

    # Blend
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def render_skeleton_video(params_path: str,
                          video_path: str,
                          output_path: str,
                          line_thickness: int = 3,
                          point_radius: int = 6):
    """
    Render skeleton overlay video from Dyn-HaMR params.
    """
    # Load joints
    joints_3d = load_mano_joints(params_path)
    if joints_3d is None:
        print("ERROR: Could not load joints from params file")
        return False

    print(f"Loaded joints: shape {joints_3d.shape}")

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
    print(f"Joints: {len(joints_3d)} frames")

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(joints_3d):
            # Project 3D joints to 2D
            joints_2d = project_to_2d(joints_3d[frame_idx], width, height)

            # Draw skeleton
            frame = draw_skeleton(frame, joints_2d,
                                  line_thickness=line_thickness,
                                  point_radius=point_radius)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render skeleton from Dyn-HaMR params")
    parser.add_argument("params", help="Path to Dyn-HaMR params.npz")
    parser.add_argument("video", help="Path to original video")
    parser.add_argument("-o", "--output", default="skeleton_overlay.mp4",
                        help="Output video path")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Line thickness (default: 3)")
    parser.add_argument("--radius", type=int, default=6,
                        help="Joint point radius (default: 6)")

    args = parser.parse_args()

    render_skeleton_video(
        args.params,
        args.video,
        args.output,
        line_thickness=args.thickness,
        point_radius=args.radius
    )


if __name__ == "__main__":
    main()
