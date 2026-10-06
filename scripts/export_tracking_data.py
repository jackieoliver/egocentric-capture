#!/usr/bin/env python3
"""
Export comprehensive hand tracking data from Dyn-HaMeR outputs.

This script consolidates data from multiple Dyn-HaMeR output files into a single
NPZ file with all useful metrics including detection confidence, visibility masks,
and 3D joint positions.

Usage:
    conda activate dynhamr
    python export_tracking_data.py \
        --dyn-hamr-output ~/dyn_hamr_workspace/outputs/logs/video-custom/2026-01-23/head_60fps-all-shot-0-0--1 \
        --hamer-pickle ~/dyn_hamr_workspace/Dyn-HaMR/test/dynhamr/hamer_out/head_60fps/head_60fps.pkl \
        --output tracking_data.npz
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Add Dyn-HaMR paths
DYN_HAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR/dyn-hamr")
if DYN_HAMR_ROOT.exists():
    sys.path.insert(0, str(DYN_HAMR_ROOT))


def load_hamer_confidence(pkl_path: Path) -> dict:
    """
    Load detection confidence and 2D keypoints from HaMeR pickle.

    Returns dict with:
        - bbox_conf: (num_frames, max_hands) detection confidence per hand
        - keypoints_2d: (num_frames, max_hands, 21, 3) 2D keypoints [x, y, conf]
        - tracked_ids: list of track ID mappings per frame
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    frame_keys = sorted(data.keys())
    num_frames = len(frame_keys)

    # Collect data
    all_bbox_conf = []  # list of lists
    all_keypoints_2d = []  # list of lists of arrays
    all_tracked_ids = []

    max_hands = 0
    for frame_key in frame_keys:
        frame_data = data[frame_key]
        bbox_conf = frame_data.get('bbox_conf', [])
        extra_data = frame_data.get('extra_data', [])
        tracked_ids = frame_data.get('tracked_ids', [])

        max_hands = max(max_hands, len(bbox_conf))
        all_bbox_conf.append(bbox_conf)
        all_keypoints_2d.append([np.array(kp) if kp else np.zeros((21, 3)) for kp in extra_data])
        all_tracked_ids.append(tracked_ids)

    # Pad to uniform size
    bbox_conf_arr = np.zeros((num_frames, max_hands), dtype=np.float32)
    keypoints_2d_arr = np.zeros((num_frames, max_hands, 21, 3), dtype=np.float32)

    for i, (conf, kpts) in enumerate(zip(all_bbox_conf, all_keypoints_2d)):
        for j, c in enumerate(conf):
            bbox_conf_arr[i, j] = c
        for j, kp in enumerate(kpts):
            if j < max_hands:
                keypoints_2d_arr[i, j] = kp

    return {
        'bbox_conf': bbox_conf_arr,
        'keypoints_2d': keypoints_2d_arr,
        'tracked_ids': all_tracked_ids,
        'num_frames': num_frames,
        'max_hands': max_hands,
    }


def load_track_info(json_path: Path) -> dict:
    """Load visibility masks from track_info.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    vis_masks = {}
    for track_id, track_data in data['tracks'].items():
        vis_masks[int(track_id)] = np.array(track_data['vis_mask'], dtype=bool)

    return {
        'vis_masks': vis_masks,
        'meta': data.get('meta', {}),
    }


def load_smooth_fit_params(pth_path: Path) -> dict:
    """Load optimized MANO parameters from smooth_fit_params.pth."""
    params = torch.load(pth_path, map_location='cpu', weights_only=False)

    result = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.numpy()
        else:
            result[key] = val

    return result


def load_world_results(npz_path: Path) -> dict:
    """Load world results NPZ with camera parameters."""
    data = np.load(npz_path, allow_pickle=True)
    return dict(data)


def compute_3d_joints(pose_data: dict, device: str = 'cpu') -> np.ndarray:
    """
    Compute 3D joint positions from MANO parameters.

    Args:
        pose_data: dict with pose_body, root_orient, trans, betas, is_right
        device: torch device

    Returns:
        joints_3d: (num_hands, num_frames, 16, 3) world-space joint positions
    """
    try:
        from smplx import MANO
    except ImportError:
        print("Warning: smplx not available, skipping 3D joint computation")
        return None

    mano_path = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR/_DATA/data/mano")
    if not mano_path.exists():
        print(f"Warning: MANO path not found: {mano_path}")
        return None

    pose_body = pose_data.get('pose_body', pose_data.get('latent_pose'))
    root_orient = pose_data['root_orient']
    trans = pose_data['trans']
    is_right = pose_data['is_right']
    betas = pose_data.get('betas')

    num_hands, num_frames = pose_body.shape[:2]

    # Convert to tensors
    pose_t = torch.from_numpy(pose_body).float().to(device)
    root_t = torch.from_numpy(root_orient).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)
    is_right_t = torch.from_numpy(is_right).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device) if betas is not None else None

    # Load MANO model
    body_model = MANO(
        str(mano_path),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=num_frames,
    ).to(device)
    body_model.eval()

    all_joints = []

    with torch.no_grad():
        for h in range(num_hands):
            # Reshape pose for MANO
            hand_pose = pose_t[h].reshape(num_frames, -1)
            global_orient = root_t[h]
            transl = trans_t[h]

            if betas_t is not None:
                b = betas_t[h:h+1].expand(num_frames, -1)
            else:
                b = torch.zeros(num_frames, 10, device=device)

            output = body_model(
                hand_pose=hand_pose,
                betas=b,
                global_orient=global_orient,
                transl=transl,
            )

            joints = output.joints  # (T, 16, 3)

            # Flip X for left hand
            if is_right_t[h, 0] < 0.5:
                joints[:, :, 0] = -joints[:, :, 0]

            all_joints.append(joints.cpu().numpy())

    return np.stack(all_joints, axis=0)  # (H, T, 16, 3)


def main():
    parser = argparse.ArgumentParser(description='Export comprehensive tracking data from Dyn-HaMeR')
    parser.add_argument('--dyn-hamr-output', required=True, type=Path,
                        help='Dyn-HaMeR output directory (containing smooth_fit_params.pth)')
    parser.add_argument('--hamer-pickle', type=Path, default=None,
                        help='HaMeR pickle file (for detection confidence)')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output NPZ file')
    parser.add_argument('--compute-joints', action='store_true',
                        help='Compute 3D joint positions (requires MANO)')
    parser.add_argument('--device', default='cpu',
                        help='Device for MANO computation')
    args = parser.parse_args()

    output_dir = args.dyn_hamr_output

    # Find required files
    smooth_fit_path = output_dir / 'smooth_fit_params.pth'
    track_info_path = output_dir / 'track_info.json'

    # Find world results NPZ (latest checkpoint)
    smooth_fit_dir = output_dir / 'smooth_fit'
    world_results_files = sorted(smooth_fit_dir.glob('*_world_results.npz'))
    world_results_path = world_results_files[-1] if world_results_files else None

    print(f"Loading smooth_fit_params from: {smooth_fit_path}")
    pose_data = load_smooth_fit_params(smooth_fit_path)

    num_hands = pose_data['trans'].shape[0]
    num_frames = pose_data['trans'].shape[1]
    print(f"  Loaded {num_hands} hands, {num_frames} frames")

    # Load track info for visibility
    if track_info_path.exists():
        print(f"Loading track_info from: {track_info_path}")
        track_data = load_track_info(track_info_path)

        # Build visibility mask array
        vis_mask = np.zeros((num_hands, num_frames), dtype=bool)
        for track_id, mask in track_data['vis_masks'].items():
            if track_id < num_hands:
                vis_mask[track_id, :len(mask)] = mask[:num_frames]

        print(f"  Visibility: {vis_mask.sum()} / {vis_mask.size} frames visible")
    else:
        print(f"Warning: track_info.json not found, assuming all visible")
        vis_mask = np.ones((num_hands, num_frames), dtype=bool)

    # Load world results for camera params
    if world_results_path and world_results_path.exists():
        print(f"Loading world_results from: {world_results_path}")
        world_data = load_world_results(world_results_path)
        cam_R = world_data.get('cam_R')
        cam_t = world_data.get('cam_t')
        intrins = world_data.get('intrins')
        print(f"  Camera params loaded: R={cam_R.shape if cam_R is not None else None}, "
              f"t={cam_t.shape if cam_t is not None else None}")
    else:
        print("Warning: world_results.npz not found, camera params unavailable")
        cam_R = None
        cam_t = None
        intrins = None

    # Load HaMeR confidence if available
    if args.hamer_pickle and args.hamer_pickle.exists():
        print(f"Loading HaMeR confidence from: {args.hamer_pickle}")
        hamer_data = load_hamer_confidence(args.hamer_pickle)
        detection_conf = hamer_data['bbox_conf']
        keypoints_2d = hamer_data['keypoints_2d']

        # Transpose to (H, T) format matching pose data
        # Note: HaMeR may have different hand ordering
        detection_conf = detection_conf.T[:num_hands, :num_frames]
        keypoints_2d = keypoints_2d.transpose(1, 0, 2, 3)[:num_hands, :num_frames]

        print(f"  Detection confidence: mean={detection_conf.mean():.3f}, "
              f"min={detection_conf.min():.3f}, max={detection_conf.max():.3f}")
    else:
        print("Warning: HaMeR pickle not provided, detection confidence unavailable")
        detection_conf = np.ones((num_hands, num_frames), dtype=np.float32)
        keypoints_2d = None

    # Compute 3D joints if requested
    if args.compute_joints:
        print("Computing 3D joint positions...")
        joints_3d = compute_3d_joints(pose_data, args.device)
        if joints_3d is not None:
            print(f"  Joints shape: {joints_3d.shape}")
    else:
        joints_3d = None

    # Build output dictionary
    output_data = {
        # Pose parameters
        'pose_body': pose_data.get('pose_body', pose_data.get('latent_pose')),
        'root_orient': pose_data['root_orient'],
        'trans': pose_data['trans'],
        'betas': pose_data.get('betas'),
        'is_right': pose_data['is_right'],

        # Quality metrics
        'detection_conf': detection_conf,
        'vis_mask': vis_mask,

        # Metadata
        'num_hands': np.array(num_hands),
        'num_frames': np.array(num_frames),
    }

    # Optional data
    if cam_R is not None:
        output_data['cam_R'] = cam_R
    if cam_t is not None:
        output_data['cam_t'] = cam_t
    if intrins is not None:
        output_data['intrins'] = intrins
    if keypoints_2d is not None:
        output_data['keypoints_2d'] = keypoints_2d
    if joints_3d is not None:
        output_data['joints_3d'] = joints_3d

    # Save
    print(f"\nSaving to: {args.output}")
    np.savez_compressed(args.output, **output_data)

    # Print summary
    print("\n=== Export Summary ===")
    for key, val in output_data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")

    print(f"\nOutput saved: {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
