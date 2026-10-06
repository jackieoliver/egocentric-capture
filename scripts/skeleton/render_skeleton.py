#!/usr/bin/env python3
"""
Render skeleton overlay from Dyn-HaMR smooth_fit_params.pth files.
Uses MANO forward pass to compute joint positions.

Includes ghost skeleton fix with validation checks.
"""

import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# Add Dyn-HaMR paths
DYNHAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR")
sys.path.insert(0, str(DYNHAMR_ROOT / "dyn-hamr"))
sys.path.insert(0, str(DYNHAMR_ROOT / "third-party" / "hamer"))

from body_model import MANO

# MANO skeleton connections (21 joints)
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
]

# Haptica website teal: #00d4aa = RGB(0, 212, 170) = BGR(170, 212, 0)
HAPTICA_TEAL = (170, 212, 0)


def load_mano_model(device='cuda'):
    mano_cfg = {
        'model_path': str(DYNHAMR_ROOT / '_DATA' / 'data' / 'mano'),
        'use_pca': False,
        'flat_hand_mean': False,
    }
    return MANO(batch_size=1, pose2rot=True, **mano_cfg).to(device)


def get_mano_joints(mano_model, pose, root_orient, trans, betas, is_right, device='cuda'):
    """Get 3D joint positions from MANO params.

    IMPORTANT: Applies handedness-based X-flip (from Dyn-HaMR's run_mano).
    Do NOT apply the 180-degree X-rotation used in HaMeR's renderer.py,
    since that is specific to pyrender/OpenGL and will mirror projections
    if used with standard pinhole projection.
    """
    pose = pose.unsqueeze(0).to(device)
    root_orient = root_orient.unsqueeze(0).to(device)
    trans = trans.unsqueeze(0).to(device)
    betas = betas.unsqueeze(0).to(device) if betas.dim() == 1 else betas.to(device)

    pose_flat = pose.reshape(1, -1)

    with torch.no_grad():
        output = mano_model(
            betas=betas,
            global_orient=root_orient,
            hand_pose=pose_flat,
            transl=trans,
            return_verts=True,
        )

    # joints shape: (1, 21, 3)
    joints = output.joints[0].cpu().numpy()

    # Apply handedness-based X-flip (from Dyn-HaMR's run_mano)
    # is_right: 1 for right hand, 0 for left hand
    # Formula: x = (2*is_right - 1) * x
    # Right hand: 2*1-1 = 1, X unchanged
    # Left hand: 2*0-1 = -1, X negated
    is_right_val = is_right.item() if hasattr(is_right, 'item') else is_right
    x_flip = 2 * is_right_val - 1
    joints[:, 0] = x_flip * joints[:, 0]

    # NOTE: Do NOT apply Y-flip here. Testing showed that HaMeR's cam_trans is already
    # computed in a coordinate system where raw MANO Y values project correctly.
    # The Y-flip causes ~200px vertical error vs HaMeR's ground-truth 2D keypoints.

    return joints


def validate_joints(joints_3d):
    """
    GHOST SKELETON FIX: Validate joints to filter garbage data.

    Returns True if joints are valid, False otherwise.

    NOTE: Z values should be positive in camera/world space for valid points.
    """
    # 1. Check for NaN or Inf values
    if not np.isfinite(joints_3d).all():
        return False

    # 2. Check for behind-camera joints or too close
    z_values = joints_3d[:, 2]
    if (z_values <= 0.01).any():  # Behind camera or too close
        return False

    # 3. Sanity check: reject if joints spread too far (garbage extrapolation)
    joint_spread = np.max(joints_3d[:, :2]) - np.min(joints_3d[:, :2])
    if joint_spread > 2.0:  # More than 2 meters spread is unrealistic
        return False

    return True


def project_to_2d(joints_3d, intrinsics, cam_R=None, cam_t=None):
    """Project 3D joints to 2D using camera intrinsics.

    If cam_R/cam_t are provided, first transform joints to camera space:
    joints_cam = cam_R @ joints + cam_t.
    """
    fx, fy, cx, cy = intrinsics
    joints = joints_3d
    if cam_R is not None and cam_t is not None:
        joints = (cam_R @ joints.T).T + cam_t

    z = joints[:, 2]
    valid = z > 0.01

    joints_2d = np.zeros((len(joints_3d), 2))
    joints_2d[valid, 0] = (joints[valid, 0] / z[valid]) * fx + cx
    joints_2d[valid, 1] = (joints[valid, 1] / z[valid]) * fy + cy

    return joints_2d, valid


def draw_skeleton(frame, joints_2d, valid, color=HAPTICA_TEAL, thickness=3, radius=5,
                  visibility_threshold=0.2):
    """Draw skeleton on frame with visibility threshold."""
    H, W = frame.shape[:2]

    # Check how many joints are visible (on screen and valid)
    visible_count = 0
    for i, (x, y) in enumerate(joints_2d):
        if valid[i] and 0 <= x < W and 0 <= y < H:
            visible_count += 1

    # Skip if not enough joints visible
    visibility_ratio = visible_count / len(joints_2d)
    if visibility_ratio < visibility_threshold:
        return frame

    # Skip if wrist is off screen
    wrist = joints_2d[0]
    if not valid[0] or not (0 <= wrist[0] < W and 0 <= wrist[1] < H):
        return frame

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if valid[i] and valid[j]:
            pt1 = tuple(joints_2d[i].astype(int))
            pt2 = tuple(joints_2d[j].astype(int))
            if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                0 <= pt2[0] < W and 0 <= pt2[1] < H):
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw joints
    for i, (x, y) in enumerate(joints_2d):
        if valid[i] and 0 <= x < W and 0 <= y < H:
            pt = (int(x), int(y))
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, (255, 255, 255), 2, cv2.LINE_AA)  # White outline

    return frame


def load_track_info(output_dir, num_tracks, num_frames):
    """Load visibility masks and frame offset from track_info.json.

    IMPORTANT: track_info.json uses arbitrary string keys (e.g., "1" for wrist cameras)
    with an "index" field that maps to the actual tensor index. We must use the
    "index" field, not the track key itself.
    """
    track_info_path = Path(output_dir) / "track_info.json"

    if not track_info_path.exists():
        print(f"WARNING: No track_info.json found, defaulting to all invisible (safe)")
        # Default to INVISIBLE to avoid ghost skeletons
        return np.zeros((num_tracks, num_frames), dtype=bool), 0

    with open(track_info_path) as f:
        track_info = json.load(f)

    # Get frame offset
    meta = track_info.get('meta', {})
    seq_interval = meta.get('seq_interval', [0, 0])
    start_frame = seq_interval[0] if seq_interval else 0
    print(f"Frame offset: start_frame={start_frame}")

    # Initialize all tracks as invisible (safe default)
    vis_masks = [[False] * num_frames for _ in range(num_tracks)]

    # Fill in visibility from track_info using the "index" field
    tracks = track_info.get('tracks', {})
    for track_key, track_data in tracks.items():
        # The "index" field maps to the tensor index, not the track key
        tensor_idx = track_data.get('index', None)
        if tensor_idx is None:
            # Fallback: try to parse key as integer
            try:
                tensor_idx = int(track_key)
            except ValueError:
                print(f"WARNING: Track {track_key} has no 'index' field, skipping")
                continue

        if 0 <= tensor_idx < num_tracks:
            mask = track_data.get('vis_mask', [])
            if len(mask) < num_frames:
                mask = mask + [False] * (num_frames - len(mask))
            elif len(mask) > num_frames:
                mask = mask[:num_frames]
            vis_masks[tensor_idx] = mask
            print(f"Track key={track_key} -> tensor_idx={tensor_idx}: {sum(mask)} visible frames")
        else:
            print(f"WARNING: Track {track_key} has index {tensor_idx} but num_tracks={num_tracks}")

    return np.array(vis_masks, dtype=bool), start_frame


def render_skeleton_video(params_path, cameras_path, video_path, output_path, device='cuda'):
    """Render skeleton overlay video with ghost skeleton fix."""
    # Load params
    params = torch.load(params_path, map_location='cpu')
    pose = params['latent_pose']
    root_orient = params['root_orient']
    trans = params['trans']
    betas = params['betas']
    is_right = params['is_right']
    cam_R = params.get('cam_R', None)
    cam_t = params.get('cam_t', None)

    num_hands, T = pose.shape[:2]
    print(f"Loaded {num_hands} hands, {T} frames")

    # Load camera intrinsics
    with open(cameras_path) as f:
        cam_data = json.load(f)
    intrinsics = cam_data['intrinsics'][0] if isinstance(cam_data['intrinsics'][0], list) else cam_data['intrinsics']
    print(f"Intrinsics: {intrinsics}")

    # Load visibility masks and frame offset
    output_dir = Path(params_path).parent
    vis_masks, start_frame = load_track_info(output_dir, num_hands, T)
    print(f"Visibility: {vis_masks.sum(axis=1)} visible frames per track")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Load MANO
    print("Loading MANO model...")
    mano_model = load_mano_model(device)

    print(f"Rendering to {output_path}")
    print(f"Using Haptica teal #00d4aa")

    ghost_filtered = 0
    rendered_hands = 0

    for video_frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # Map video frame to MANO index (accounting for offset)
        mano_idx = video_frame_idx - start_frame

        if 0 <= mano_idx < T:
            # Render each hand
            for hand_idx in range(num_hands):
                # Check visibility mask first
                if not vis_masks[hand_idx, mano_idx]:
                    continue

                hand_is_right = is_right[hand_idx, mano_idx].item()
                if not (0.0 <= hand_is_right <= 1.0):
                    continue

                try:
                    joints_3d = get_mano_joints(
                        mano_model,
                        pose[hand_idx, mano_idx],
                        root_orient[hand_idx, mano_idx],
                        trans[hand_idx, mano_idx],
                        betas[hand_idx],
                        is_right[hand_idx, mano_idx],
                        device=device,
                    )

                    # GHOST SKELETON FIX: Validate joints
                    if not validate_joints(joints_3d):
                        ghost_filtered += 1
                        continue

                    cam_R_frame = None
                    cam_t_frame = None
                    if cam_R is not None and cam_t is not None:
                        cam_R_frame = cam_R[hand_idx, mano_idx].cpu().numpy()
                        cam_t_frame = cam_t[hand_idx, mano_idx].cpu().numpy()

                    joints_2d, valid = project_to_2d(
                        joints_3d,
                        intrinsics,
                        cam_R=cam_R_frame,
                        cam_t=cam_t_frame,
                    )
                    frame = draw_skeleton(frame, joints_2d, valid)
                    rendered_hands += 1

                except Exception as e:
                    pass

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done! Rendered {rendered_hands} hand instances, filtered {ghost_filtered} ghost skeletons")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Render Haptica teal skeleton overlay on video using Dyn-HaMR params',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dynamic mode - specify all paths:
  python render_skeleton.py --video input.mp4 --params smooth_fit_params.pth --output skeleton.mp4

  # With explicit cameras file:
  python render_skeleton.py --video input.mp4 --params params.pth --cameras cameras.json --output out.mp4

  # Legacy mode - use preset paths for old footage:
  python render_skeleton.py --camera head
        '''
    )

    # Dynamic mode arguments
    parser.add_argument('--video', type=str, help='Input video file path')
    parser.add_argument('--params', type=str, help='Path to smooth_fit_params.pth from Dyn-HaMR')
    parser.add_argument('--cameras', type=str, help='Path to cameras.json (optional, derived from params if not given)')
    parser.add_argument('--output', type=str, help='Output video path')

    # Legacy mode argument (for old hardcoded paths)
    parser.add_argument('--camera', choices=['head', 'lwrist', 'rwrist'],
                        help='Legacy mode: use preset paths for January footage')

    parser.add_argument('--device', default='cuda', help='Device for MANO model (cuda/cpu)')
    args = parser.parse_args()

    # Validate arguments
    if args.camera:
        # Legacy mode - use hardcoded paths for old footage
        base = Path("/home/example/dyn_hamr_workspace")
        cam_dirs = {
            'head': 'head_60fps-all-shot-0-0--1',
            'lwrist': 'lwrist_clip_60fps-all-shot-0-0--1',
            'rwrist': 'rwrist_60fps-all-shot-0-0--1',
        }
        video_files = {
            'head': 'head_60fps.mp4',
            'lwrist': 'lwrist_clip_60fps.mp4',
            'rwrist': 'rwrist_60fps.mp4',
        }
        cam_dir = base / "outputs" / "logs" / "video-custom" / "2026-01-23" / cam_dirs[args.camera]
        params_path = cam_dir / "smooth_fit_params.pth"
        cameras_path = cam_dir / "cameras.json"
        video_path = base / "Dyn-HaMR" / "test" / "videos" / video_files[args.camera]
        output_path = base / "outputs" / f"{args.camera}_skeleton.mp4"
    elif args.video and args.params:
        # Dynamic mode - use provided paths
        video_path = Path(args.video)
        params_path = Path(args.params)

        if args.cameras:
            cameras_path = Path(args.cameras)
        else:
            # Derive cameras.json from params directory
            cameras_path = params_path.parent / "cameras.json"

        if args.output:
            output_path = Path(args.output)
        else:
            # Default output next to input video
            output_path = video_path.parent / f"{video_path.stem}_skeleton.mp4"

        # Validate paths exist
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not params_path.exists():
            raise FileNotFoundError(f"Params not found: {params_path}")
        if not cameras_path.exists():
            print(f"Warning: cameras.json not found at {cameras_path}, will use default intrinsics")
    else:
        parser.error("Either --camera (legacy) or --video + --params (dynamic) required")

    render_skeleton_video(params_path, cameras_path, video_path, output_path, args.device)
