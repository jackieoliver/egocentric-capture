#!/usr/bin/env python3
"""
Render skeleton overlay from Dyn-HaMR MANO params by computing MANO forward pass.

Takes the world_results.npz output from Dyn-HaMR and renders skeleton overlay.

IMPORTANT: This script uses Dyn-HaMR's internal MANO module (not smplx directly)
to ensure correct coordinate transformations.

Usage:
    python 05_render_mano_skeleton.py params.npz video.mp4 -o output.mp4 --dynhamr_dir path/to/Dyn-HaMR
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path

# MANO joint connections (21 joints)
MANO_CONNECTIONS_21 = [
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

# Colors for each finger (BGR)
# Medium teal: #00E6BB = RGB(0, 230, 187) = BGR(187, 230, 0)
BRIGHT_TEAL = (187, 230, 0)  # #00E6BB in BGR - medium brightness
FINGER_COLORS = {
    'wrist': BRIGHT_TEAL,
    'thumb': BRIGHT_TEAL,
    'index': BRIGHT_TEAL,
    'middle': BRIGHT_TEAL,
    'ring': BRIGHT_TEAL,
    'pinky': BRIGHT_TEAL,
    'palm': BRIGHT_TEAL,
}


def compute_mano_joints(params_path: str, dynhamr_dir: str):
    """
    Compute 3D joint positions from MANO parameters using Dyn-HaMR's run_mano.

    This uses Dyn-HaMR's internal MANO implementation which correctly handles:
    - X-flip for left hands
    - Proper pose2rot conversion
    - Correct mean pose handling

    Returns: (joints, intrins) where joints is (num_tracks, num_frames, num_joints, 3)
    """
    import sys
    import torch

    # Add Dyn-HaMR to path
    dyn_hamr_path = Path(dynhamr_dir) / "dyn-hamr"
    sys.path.insert(0, str(dyn_hamr_path))

    try:
        from body_model import MANO
        from body_model.utils import run_mano
    except ImportError as e:
        print(f"ERROR: Could not import from Dyn-HaMR: {e}")
        print(f"Make sure --dynhamr_dir points to the Dyn-HaMR repository root")
        return None, None

    # Load parameters
    data = np.load(params_path, allow_pickle=True)
    print(f"Available keys: {list(data.keys())}")

    # Extract MANO params
    pose_body = torch.tensor(data['pose_body'], dtype=torch.float32)  # (B, T, 15, 3)
    root_orient = torch.tensor(data['root_orient'], dtype=torch.float32)  # (B, T, 3)
    trans = torch.tensor(data['trans'], dtype=torch.float32)  # (B, T, 3)
    betas = torch.tensor(data['betas'], dtype=torch.float32)  # (B, 10)
    is_right = torch.tensor(data['is_right'], dtype=torch.float32)  # (B, T)
    intrins = data['intrins']  # (4,) - fx, fy, cx, cy

    num_tracks, num_frames = pose_body.shape[:2]
    print(f"Processing {num_tracks} tracks, {num_frames} frames")
    print(f"Intrinsics: {intrins}")

    # Reshape pose for run_mano: (B, T, 45)
    pose_body_flat = pose_body.reshape(num_tracks, num_frames, -1)

    # Initialize MANO model using Dyn-HaMR's config
    mano_dir = Path(dynhamr_dir) / "_DATA" / "data" / "mano"
    if not mano_dir.exists():
        print(f"ERROR: MANO model not found at {mano_dir}")
        return None, None

    mano_cfg = {
        'model_path': str(mano_dir),
        'gender': 'neutral',
        'num_hand_joints': 15,
        'create_body_pose': False,
    }

    print(f"Initializing MANO model from {mano_dir}")
    hand_model = MANO(batch_size=num_tracks * num_frames, pose2rot=True, **mano_cfg)

    # Use Dyn-HaMR's run_mano which handles all the coordinate transforms
    print("Running MANO forward pass...")
    result = run_mano(hand_model, trans, root_orient, pose_body_flat, is_right, betas)

    joints = result['joints'].detach().numpy()  # (B, T, 21, 3)
    print(f"Output joints shape: {joints.shape}")

    return joints, intrins


def project_to_2d(joints_3d: np.ndarray, intrins: np.ndarray) -> np.ndarray:
    """
    Project 3D joints to 2D using camera intrinsics.

    joints_3d: (..., 3)
    intrins: (4,) - fx, fy, cx, cy

    Note: No additional flips needed - run_mano already applied them!
    """
    fx, fy, cx, cy = intrins

    z = joints_3d[..., 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    x_2d = fx * joints_3d[..., 0:1] / z + cx
    y_2d = fy * joints_3d[..., 1:2] / z + cy

    return np.concatenate([x_2d, y_2d], axis=-1)


def get_connection_color(joint_a: int, joint_b: int) -> tuple:
    """Get color for a skeleton connection based on finger."""
    max_joint = max(joint_a, joint_b)
    min_joint = min(joint_a, joint_b)

    if min_joint == 0:
        if max_joint <= 4:
            return FINGER_COLORS['thumb']
        elif max_joint <= 8:
            return FINGER_COLORS['index']
        elif max_joint <= 12:
            return FINGER_COLORS['middle']
        elif max_joint <= 16:
            return FINGER_COLORS['ring']
        else:
            return FINGER_COLORS['pinky']
    elif 1 <= max_joint <= 4:
        return FINGER_COLORS['thumb']
    elif 5 <= max_joint <= 8:
        return FINGER_COLORS['index']
    elif 9 <= max_joint <= 12:
        return FINGER_COLORS['middle']
    elif 13 <= max_joint <= 16:
        return FINGER_COLORS['ring']
    else:
        return FINGER_COLORS['pinky']


def draw_skeleton(frame: np.ndarray, joints_2d: np.ndarray,
                  line_thickness: int = 3, point_radius: int = 5,
                  alpha: float = 0.9, visibility_threshold: float = 0.2) -> np.ndarray:
    """Draw skeleton on frame.

    Args:
        visibility_threshold: Minimum fraction of joints that must be on screen
                              to draw the hand (0.0-1.0). Default 0.5 = 50%.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Handle (num_tracks, num_joints, 2) shape
    if joints_2d.ndim == 2:
        joints_2d = joints_2d[np.newaxis, ...]

    num_joints = joints_2d.shape[1]
    connections = MANO_CONNECTIONS_21

    for hand_idx, hand_joints in enumerate(joints_2d):
        # Check how many joints are visible (on screen)
        visible_count = 0
        for joint in hand_joints:
            pt = joint.astype(int)
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                visible_count += 1

        # Skip this hand if not enough joints are visible
        visibility_ratio = visible_count / num_joints
        if visibility_ratio < visibility_threshold:
            continue

        # Also skip if wrist (joint 0) is off screen - it's the anchor point
        wrist = hand_joints[0].astype(int)
        if not (0 <= wrist[0] < w and 0 <= wrist[1] < h):
            continue

        # Draw connections
        for (ja, jb) in connections:
            if ja >= num_joints or jb >= num_joints:
                continue

            pt_a = hand_joints[ja].astype(int)
            pt_b = hand_joints[jb].astype(int)

            if not (0 <= pt_a[0] < w and 0 <= pt_a[1] < h):
                continue
            if not (0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
                continue

            color = BRIGHT_TEAL  # Same color for all fingers and both hands

            cv2.line(overlay, tuple(pt_a), tuple(pt_b), color, line_thickness, cv2.LINE_AA)

        # Draw joints
        for i, joint in enumerate(hand_joints):
            pt = joint.astype(int)
            if not (0 <= pt[0] < w and 0 <= pt[1] < h):
                continue

            cv2.circle(overlay, tuple(pt), point_radius, BRIGHT_TEAL, -1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(pt), point_radius, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def load_visibility_masks_and_offset(output_dir: str, num_tracks: int, num_frames: int):
    """Load visibility masks and frame offset from track_info.json.

    Returns:
        vis_masks: (num_tracks, num_frames) boolean array
        start_frame: The video frame number where MANO data starts (offset)
    """
    track_info_path = Path(output_dir) / "track_info.json"

    if not track_info_path.exists():
        print(f"WARNING: No track_info.json found at {track_info_path}")
        # Return all-visible masks and no offset
        return np.ones((num_tracks, num_frames), dtype=bool), 0

    with open(track_info_path) as f:
        track_info = json.load(f)

    # Get the frame offset from seq_interval
    meta = track_info.get('meta', {})
    seq_interval = meta.get('seq_interval', [0, 0])
    start_frame = seq_interval[0] if seq_interval else 0
    print(f"Frame offset from track_info: start_frame={start_frame}")

    vis_masks = []
    for i in range(num_tracks):
        track_key = str(i)
        if track_key in track_info.get('tracks', {}):
            mask = track_info['tracks'][track_key].get('vis_mask', [True] * num_frames)
            # Pad or truncate to match num_frames
            if len(mask) < num_frames:
                mask = mask + [False] * (num_frames - len(mask))
            elif len(mask) > num_frames:
                mask = mask[:num_frames]
            vis_masks.append(mask)
        else:
            # Missing tracks default to INVISIBLE (safer - avoids ghost skeletons)
            vis_masks.append([False] * num_frames)

    return np.array(vis_masks, dtype=bool), start_frame


def render_skeleton_video(params_path: str, video_path: str, output_path: str,
                          dynhamr_dir: str, line_thickness: int = 3,
                          point_radius: int = 6):
    """Render skeleton overlay video from Dyn-HaMR MANO params."""

    print(f"Computing MANO joints from {params_path}...")
    joints_3d, intrins = compute_mano_joints(params_path, dynhamr_dir)

    if joints_3d is None:
        print("ERROR: Failed to compute joints")
        return False

    num_tracks, num_frames, num_joints, _ = joints_3d.shape
    print(f"Computed joints: {num_tracks} tracks, {num_frames} frames, {num_joints} joints")

    # Load visibility masks and frame offset from track_info.json
    output_dir = Path(params_path).parent.parent  # Go up from smooth_fit/ to output dir
    vis_masks, start_frame = load_visibility_masks_and_offset(output_dir, num_tracks, num_frames)
    print(f"Loaded visibility masks: {vis_masks.sum(axis=1)} visible frames per track")
    print(f"Video-to-MANO offset: video frame N uses MANO index (N - {start_frame})")

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

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Map video frame to MANO data index (accounting for offset)
        mano_idx = video_frame_idx - start_frame

        if 0 <= mano_idx < num_frames:
            # Get joints for visible tracks only at this frame
            visible_joints = []
            for track_idx in range(num_tracks):
                if vis_masks[track_idx, mano_idx]:
                    track_joints = joints_3d[track_idx, mano_idx]  # (num_joints, 3)

                    # VALIDATION: Skip tracks with garbage data (ghost skeleton fix)
                    # 1. Check for NaN or Inf values
                    if not np.isfinite(track_joints).all():
                        continue
                    # 2. Check for behind-camera joints (Z <= 0)
                    z_values = track_joints[:, 2]
                    if (z_values <= 0.01).any():  # Behind camera or too close
                        continue
                    # 3. Sanity check: reject if joints spread too far (garbage extrapolation)
                    joint_spread = np.max(track_joints[:, :2]) - np.min(track_joints[:, :2])
                    if joint_spread > 2.0:  # More than 2 meters spread is unrealistic
                        continue

                    visible_joints.append(track_joints)

            if visible_joints:
                frame_joints_3d = np.array(visible_joints)  # (num_visible, num_joints, 3)

                # Project to 2D
                frame_joints_2d = project_to_2d(frame_joints_3d, intrins)

                # Draw skeleton
                frame = draw_skeleton(frame, frame_joints_2d,
                                      line_thickness=line_thickness,
                                      point_radius=point_radius)

        out.write(frame)
        video_frame_idx += 1

        if video_frame_idx % 100 == 0:
            print(f"  Rendered {video_frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render skeleton from Dyn-HaMR MANO params")
    parser.add_argument("params", help="Path to Dyn-HaMR world_results.npz")
    parser.add_argument("video", help="Path to original video")
    parser.add_argument("-o", "--output", default="skeleton_overlay.mp4",
                        help="Output video path")
    parser.add_argument("--dynhamr_dir", required=True,
                        help="Path to Dyn-HaMR repository root")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Line thickness")
    parser.add_argument("--radius", type=int, default=6,
                        help="Joint point radius")

    args = parser.parse_args()

    render_skeleton_video(
        args.params,
        args.video,
        args.output,
        args.dynhamr_dir,
        line_thickness=args.thickness,
        point_radius=args.radius
    )


if __name__ == "__main__":
    main()
