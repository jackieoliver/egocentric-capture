#!/usr/bin/env python3
"""
Convert HaMeR 3-pass output (per-frame results.pkl) to smooth_fit_params.pth format.

This is a simplified conversion without Dyn-HaMeR's temporal smoothing.
Use this when Dyn-HaMeR's camera estimation (VIPE/DROID-SLAM) is unavailable.

Usage:
    python hamer_to_skeleton.py --hamer_pkl results.pkl --output params.pth --video input.mp4
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


def load_hamer_results(pkl_path):
    """Load HaMeR 3-pass output pickle."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_video_info(video_path):
    """Get video dimensions and fps for camera intrinsics."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total_frames


def create_default_intrinsics(width, height, focal_length=500.0):
    """Create camera intrinsics [fx, fy, cx, cy].

    IMPORTANT: HaMeR computes cam_trans using a scaled focal length:
        scaled_focal = 500.0 / 224 * max(width, height)

    We must use the same scaling for correct projection.
    """
    # HaMeR's focal length scaling formula (from hamer/models/components/pose_transformer.py)
    img_size_max = max(width, height)
    scaled_focal = focal_length / 224.0 * img_size_max

    cx = width / 2.0
    cy = height / 2.0
    return [scaled_focal, scaled_focal, cx, cy]


def rotmat_to_axis_angle(rotmat):
    """Convert rotation matrix to axis-angle representation."""
    # rotmat: (..., 3, 3) -> (..., 3) axis-angle
    if isinstance(rotmat, np.ndarray):
        rotmat = torch.from_numpy(rotmat).float()

    # Handle batch dimensions
    original_shape = rotmat.shape[:-2]
    rotmat = rotmat.reshape(-1, 3, 3)

    # Compute axis-angle using Rodrigues formula inverse
    batch_size = rotmat.shape[0]
    axis_angle = torch.zeros(batch_size, 3)

    for i in range(batch_size):
        R = rotmat[i]
        theta = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1, 1))

        if theta < 1e-6:
            # Near identity
            axis_angle[i] = torch.zeros(3)
        else:
            # Extract axis from skew-symmetric part
            axis = torch.stack([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * torch.sin(theta))
            axis_angle[i] = axis * theta

    return axis_angle.reshape(*original_shape, 3)


def convert_hamer_to_skeleton_params(hamer_data, num_frames, width, height):
    """
    Convert HaMeR per-frame results to smooth_fit_params.pth format.

    HaMeR 3-pass output structure (per-frame):
        - 'mano': list of dicts with:
            - 'global_orient': (1, 3, 3) rotation matrix
            - 'hand_pose': (15, 3, 3) rotation matrices
            - 'betas': (10,) shape params
            - 'is_right': int (0 or 1)
        - 'cam_trans': list of (3,) translations
        - 'tracked_ids': list of track IDs

    Target format:
        - latent_pose: (num_tracks, T, 45) - MANO pose (axis-angle)
        - root_orient: (num_tracks, T, 3) - root rotation (axis-angle)
        - trans: (num_tracks, T, 3) - translation
        - betas: (num_tracks, 10) - shape
        - is_right: (num_tracks, T) - handedness
    """
    # Parse frame data
    frame_keys = sorted([k for k in hamer_data.keys() if k.endswith('.jpg')])

    # Organize hands by track ID (left=0, right=1)
    left_data = []  # List of (frame_idx, mano_dict, cam_trans)
    right_data = []

    for frame_idx, frame_key in enumerate(frame_keys):
        frame_data = hamer_data[frame_key]

        # Handle our HaMeR 3-pass format
        if 'mano' in frame_data and 'cam_trans' in frame_data:
            mano_list = frame_data['mano']
            cam_trans_list = frame_data['cam_trans']

            for h_idx, mano_dict in enumerate(mano_list):
                cam_trans = cam_trans_list[h_idx] if h_idx < len(cam_trans_list) else np.zeros(3)
                is_right_val = mano_dict.get('is_right', 0)

                if is_right_val > 0.5:
                    right_data.append((frame_idx, mano_dict, cam_trans))
                else:
                    left_data.append((frame_idx, mano_dict, cam_trans))

        # Handle other formats for compatibility
        elif isinstance(frame_data, dict) and 'left' in frame_data:
            if frame_data.get('left'):
                left_data.append((frame_idx, frame_data['left'], np.zeros(3)))
            if frame_data.get('right'):
                right_data.append((frame_idx, frame_data['right'], np.zeros(3)))

    # Determine number of tracks (at least 2 for left/right)
    num_tracks = 2
    T = num_frames

    # Initialize tensors
    latent_pose = torch.zeros(num_tracks, T, 45)
    root_orient = torch.zeros(num_tracks, T, 3)
    trans = torch.zeros(num_tracks, T, 3)
    betas = torch.zeros(num_tracks, 10)
    is_right = torch.zeros(num_tracks, T)

    # Visibility mask (which frames have detections)
    vis_mask = [[False] * T for _ in range(num_tracks)]

    # Fill in data for left hand (track 0)
    betas_counts = [0, 0]
    for frame_idx, mano_dict, cam_trans in left_data:
        if frame_idx < T:
            extract_hand_params_v2(mano_dict, cam_trans,
                                   latent_pose[0, frame_idx], root_orient[0, frame_idx],
                                   trans[0, frame_idx], betas[0], betas_counts, 0)
            is_right[0, frame_idx] = 0.0
            vis_mask[0][frame_idx] = True

    # Fill in data for right hand (track 1)
    for frame_idx, mano_dict, cam_trans in right_data:
        if frame_idx < T:
            extract_hand_params_v2(mano_dict, cam_trans,
                                   latent_pose[1, frame_idx], root_orient[1, frame_idx],
                                   trans[1, frame_idx], betas[1], betas_counts, 1)
            is_right[1, frame_idx] = 1.0
            vis_mask[1][frame_idx] = True

    # Average betas over frames
    for i in range(2):
        if betas_counts[i] > 0:
            betas[i] /= betas_counts[i]

    print(f"Converted {len(left_data)} left hand, {len(right_data)} right hand detections")
    print(f"Left visible frames: {sum(vis_mask[0])}, Right visible frames: {sum(vis_mask[1])}")

    return {
        'latent_pose': latent_pose,
        'root_orient': root_orient,
        'trans': trans,
        'betas': betas,
        'is_right': is_right,
    }, vis_mask


def extract_hand_params_v2(mano_dict, cam_trans, pose_out, orient_out, trans_out, betas_out, betas_counts, track_idx):
    """Extract MANO parameters from HaMeR 3-pass mano dict."""
    # Global orient: (1, 3, 3) rotation matrix -> (3,) axis-angle
    if 'global_orient' in mano_dict:
        global_orient = mano_dict['global_orient']
        if isinstance(global_orient, np.ndarray):
            global_orient = torch.from_numpy(global_orient).float()
        # Convert rotation matrix to axis-angle
        aa = rotmat_to_axis_angle(global_orient.squeeze(0))  # (3,)
        orient_out[:] = aa

    # Hand pose: (15, 3, 3) rotation matrices -> (45,) axis-angle
    if 'hand_pose' in mano_dict:
        hand_pose = mano_dict['hand_pose']
        if isinstance(hand_pose, np.ndarray):
            hand_pose = torch.from_numpy(hand_pose).float()
        # Convert each joint's rotation matrix to axis-angle
        aa = rotmat_to_axis_angle(hand_pose)  # (15, 3)
        pose_out[:] = aa.flatten()

    # Translation from cam_trans
    if isinstance(cam_trans, np.ndarray):
        cam_trans = torch.from_numpy(cam_trans).float()
    trans_out[:] = cam_trans

    # Betas - accumulate for averaging
    if 'betas' in mano_dict:
        betas_val = mano_dict['betas']
        if isinstance(betas_val, np.ndarray):
            betas_val = torch.from_numpy(betas_val).float()
        betas_out[:] += betas_val.flatten()[:10]
        betas_counts[track_idx] += 1


def extract_hand_params(hand, pose_out, orient_out, trans_out, betas_out):
    """Extract MANO parameters from a single hand detection."""
    # Handle different key names
    pose_keys = ['pred_hand_pose', 'hand_pose', 'pose']
    orient_keys = ['pred_mano_params', 'global_orient', 'root_orient']
    trans_keys = ['pred_cam_t_full', 'transl', 'trans', 'cam_t']
    betas_keys = ['pred_betas', 'betas', 'shape']

    for key in pose_keys:
        if key in hand:
            val = hand[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            if val.numel() >= 45:
                pose_out[:] = val.flatten()[:45]
            break

    for key in orient_keys:
        if key in hand:
            val = hand[key]
            if isinstance(val, dict) and 'global_orient' in val:
                val = val['global_orient']
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            if hasattr(val, 'numel') and val.numel() >= 3:
                orient_out[:] = val.flatten()[:3]
            break

    for key in trans_keys:
        if key in hand:
            val = hand[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            if hasattr(val, 'numel') and val.numel() >= 3:
                trans_out[:] = val.flatten()[:3]
            break

    for key in betas_keys:
        if key in hand:
            val = hand[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            if hasattr(val, 'numel') and val.numel() >= 10:
                betas_out[:] = val.flatten()[:10]
            break


def save_cameras_json(output_dir, intrinsics):
    """Save camera intrinsics in Dyn-HaMeR format."""
    cameras_data = {
        'intrinsics': intrinsics,
    }
    cameras_path = output_dir / 'cameras.json'
    with open(cameras_path, 'w') as f:
        json.dump(cameras_data, f, indent=2)
    return cameras_path


def save_track_info(output_dir, vis_mask, start_frame=0):
    """Save track visibility info for skeleton renderer."""
    num_tracks = len(vis_mask)

    track_info = {
        'meta': {
            'seq_interval': [start_frame, start_frame + len(vis_mask[0])],
        },
        'tracks': {}
    }

    for idx in range(num_tracks):
        track_info['tracks'][str(idx)] = {
            'index': idx,
            'vis_mask': vis_mask[idx],
            'is_right': idx == 1,  # Track 0 = left, Track 1 = right
        }

    track_info_path = output_dir / 'track_info.json'
    with open(track_info_path, 'w') as f:
        json.dump(track_info, f, indent=2)
    return track_info_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert HaMeR 3-pass output to skeleton-compatible format'
    )
    parser.add_argument('--hamer_pkl', type=str, required=True,
                       help='Path to HaMeR results.pkl')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video (for resolution/fps)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for params and cameras')
    parser.add_argument('--focal_length', type=float, default=500.0,
                       help='Assumed focal length (default: 500)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video info
    width, height, fps, total_frames = extract_video_info(args.video)
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Load HaMeR results
    print(f"Loading HaMeR results from {args.hamer_pkl}")
    hamer_data = load_hamer_results(args.hamer_pkl)
    print(f"Found {len(hamer_data)} frame entries")

    # Convert to skeleton format
    params, vis_mask = convert_hamer_to_skeleton_params(hamer_data, total_frames, width, height)

    # Save params
    params_path = output_dir / 'smooth_fit_params.pth'
    torch.save(params, params_path)
    print(f"Saved params to {params_path}")

    # Save cameras
    intrinsics = create_default_intrinsics(width, height, args.focal_length)
    cameras_path = save_cameras_json(output_dir, intrinsics)
    print(f"Saved cameras to {cameras_path}")

    # Save track info
    track_info_path = save_track_info(output_dir, vis_mask)
    print(f"Saved track info to {track_info_path}")

    print("\nConversion complete! Run skeleton overlay with:")
    print(f"  python render_skeleton.py --video {args.video} --params {params_path} --cameras {cameras_path}")


if __name__ == '__main__':
    main()
