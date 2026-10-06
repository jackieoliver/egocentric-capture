#!/usr/bin/env python3
"""
Export Dyn-HaMR hand tracking data to JSON for browser visualization.
Converts NPZ MANO parameters to pre-computed 3D joint positions.

Must be run in the dynhamr conda environment for MANO model access.

Usage:
    conda activate dynhamr
    python export_hands_json.py --npz PATH --output hands.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add Dyn-HaMR paths for MANO model
DYN_HAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR/dyn-hamr")
sys.path.insert(0, str(DYN_HAMR_ROOT))

# MANO skeleton connectivity (16 joints)
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3],      # Index finger
    [0, 4], [4, 5], [5, 6],      # Middle finger
    [0, 7], [7, 8], [8, 9],      # Pinky finger
    [0, 10], [10, 11], [11, 12], # Ring finger
    [0, 13], [13, 14], [14, 15], # Thumb
]

# Finger names for visualization
FINGER_NAMES = ["index", "middle", "pinky", "ring", "thumb"]
FINGER_COLORS = ["#FFFF00", "#00FF00", "#FF0000", "#00FFFF", "#FF14FF"]


def run_mano_joints(body_model, trans, root_orient, body_pose, is_right, betas=None):
    """
    Run MANO model and return 3D joints.
    Adapted from Dyn-HaMR's run_mano function.
    """
    B, T, _ = trans.shape
    bm_batch_size = body_model.batch_size
    seq_len = bm_batch_size // B
    bm_num_betas = body_model.num_betas

    # Pad if needed
    if T != seq_len:
        pad_size = seq_len - T
        padding_trans = torch.zeros((B, pad_size, 3), device=trans.device)
        padding_root = torch.zeros((B, pad_size, 3), device=root_orient.device)
        padding_pose = torch.zeros((B, pad_size, body_pose.shape[-1]), device=body_pose.device)
        trans = torch.cat([trans, padding_trans], dim=1)
        root_orient = torch.cat([root_orient, padding_root], dim=1)
        body_pose = torch.cat([body_pose, padding_pose], dim=1)

    if betas is None:
        betas = torch.zeros(B, bm_num_betas, device=trans.device)
    betas = betas.reshape((B, 1, bm_num_betas)).expand((B, seq_len, bm_num_betas))

    mano_output = body_model(
        hand_pose=body_pose.reshape((B * seq_len, -1)),
        betas=betas.reshape((B * seq_len, -1)),
        global_orient=root_orient.reshape((B * seq_len, -1)),
        transl=trans.reshape((B * seq_len, -1)),
    )

    joints = mano_output.joints
    joints = joints.reshape(B, seq_len, -1, 3)[:, :T]

    # Flip X coordinate for left hand (is_right=0)
    is_right_exp = is_right.unsqueeze(-1)  # (B, T, 1)
    joints[:, :, :, 0] = (2 * is_right_exp - 1) * joints[:, :, :, 0]

    return joints


def reproject(points3d, cam_R, cam_t, cam_f, cam_center):
    """
    Reproject 3D points to 2D using camera parameters.
    Matches Dyn-HaMR's reproject function.
    """
    B, T, N, _ = points3d.shape
    # Apply camera rotation
    points3d = torch.einsum("btij,btnj->btni", cam_R, points3d)
    # Add translation
    points3d = points3d + cam_t[..., None, :]
    # Perspective divide
    points2d = points3d[..., :2] / points3d[..., 2:3]
    # Apply intrinsics
    points2d = cam_f[None, :, None] * points2d + cam_center[None, :, None]
    return points2d


def main():
    parser = argparse.ArgumentParser(description="Export Dyn-HaMR NPZ to JSON for browser")
    parser.add_argument("--npz", required=True, help="Input NPZ file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--fps", type=float, default=30.0, help="Hand data FPS (default: 30)")
    parser.add_argument("--precision", type=int, default=4, help="Decimal precision for coordinates")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Import MANO model
    try:
        from smplx import MANO
    except ImportError:
        print("Error: smplx not found. Run in dynhamr conda environment.", file=sys.stderr)
        sys.exit(1)

    # Load hand data
    print(f"Loading: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)

    pose = data.get('pose_body', data.get('latent_pose'))
    root_orient = data['root_orient']
    trans = data['trans']
    is_right = data['is_right']
    betas = data.get('betas', None)
    intrins = data['intrins']
    cam_R = data.get('cam_R', None)
    cam_t = data.get('cam_t', None)

    # Ensure 4D shape: (num_hands, num_frames, ...)
    if pose.ndim == 3:
        pose = pose[None, ...]
        root_orient = root_orient[None, ...]
        trans = trans[None, ...]
        is_right = is_right[None, ...]
        if betas is not None:
            betas = betas[None, ...]
        if cam_R is not None:
            cam_R = cam_R[None, ...]
        if cam_t is not None:
            cam_t = cam_t[None, ...]

    num_hands, num_frames = pose.shape[0], pose.shape[1]
    print(f"Found {num_hands} hands, {num_frames} frames")

    # Convert to tensors
    device = args.device
    pose_t = torch.from_numpy(pose).float().to(device)
    root_orient_t = torch.from_numpy(root_orient).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)
    is_right_t = torch.from_numpy(is_right).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device) if betas is not None else None

    # Setup camera parameters
    if cam_R is None:
        cam_R = np.tile(np.eye(3), (num_hands, num_frames, 1, 1))
    if cam_t is None:
        cam_t = np.zeros((num_hands, num_frames, 3))

    cam_R_t = torch.from_numpy(cam_R).float().to(device)
    cam_t_t = torch.from_numpy(cam_t).float().to(device)

    # Intrinsics
    fx, fy, cx, cy = intrins[:4]
    cam_f = torch.tensor([[fx, fy]] * num_frames, device=device)
    cam_center = torch.tensor([[cx, cy]] * num_frames, device=device)

    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Load MANO model
    mano_path = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR/_DATA/data/mano")
    print(f"Loading MANO model from {mano_path}")

    body_model = MANO(
        str(mano_path),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=num_frames,
    ).to(device)
    body_model.eval()

    # Compute joints for all hands
    print("Computing 3D joints...")
    all_hands_data = []

    with torch.no_grad():
        for h in range(num_hands):
            hand_is_right = float(is_right[h, 0])

            # Get 3D joints from MANO
            joints3d = run_mano_joints(
                body_model,
                trans_t[h:h+1],
                root_orient_t[h:h+1],
                pose_t[h:h+1].reshape(1, num_frames, -1),
                is_right_t[h:h+1],
                betas_t[h:h+1] if betas_t is not None else None
            )  # (1, T, J, 3)

            # Reproject to 2D
            joints2d = reproject(
                joints3d,
                cam_R_t[h:h+1],
                cam_t_t[h:h+1],
                cam_f,
                cam_center
            )  # (1, T, J, 2)

            joints3d_np = joints3d[0].cpu().numpy()
            joints2d_np = joints2d[0].cpu().numpy()

            # Build frame data
            frames = []
            for t in range(num_frames):
                frame_data = {
                    "joints3d": joints3d_np[t, :16].round(args.precision).tolist(),
                    "joints2d": joints2d_np[t, :16].round(args.precision).tolist(),
                }
                frames.append(frame_data)

            hand_data = {
                "is_right": hand_is_right > 0.5,
                "chirality": "right" if hand_is_right > 0.5 else "left",
                "num_frames": num_frames,
                "frames": frames,
            }
            all_hands_data.append(hand_data)
            print(f"  Hand {h}: {'right' if hand_is_right > 0.5 else 'left'}")

    # Build output JSON
    output = {
        "version": "1.0",
        "source": str(args.npz),
        "fps": args.fps,
        "num_frames": num_frames,
        "num_hands": num_hands,
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        },
        "skeleton": {
            "edges": HAND_EDGES,
            "fingers": FINGER_NAMES,
            "colors": FINGER_COLORS,
            "num_joints": 16,
        },
        "hands": all_hands_data,
    }

    # Write JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {output_path} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
