#!/usr/bin/env python3
"""
Quick test script to diagnose skeleton overlay misalignment.
Renders just the wrist position to verify projection is correct.

The issue: HaMeR renderer applies two transformations that we need to match:
1. camera_translation[0] *= -1  (X-flip on camera translation)
2. 180-degree rotation around X-axis on the mesh (Y -> -Y, Z -> -Z)
"""

import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path

# Add Dyn-HaMR paths
DYNHAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR")
sys.path.insert(0, str(DYNHAMR_ROOT / "dyn-hamr"))
sys.path.insert(0, str(DYNHAMR_ROOT / "third-party" / "hamer"))

from body_model import MANO


def load_mano_model(device='cuda'):
    mano_cfg = {
        'model_path': str(DYNHAMR_ROOT / '_DATA' / 'data' / 'mano'),
        'use_pca': False,
        'flat_hand_mean': False,
    }
    return MANO(batch_size=1, pose2rot=True, **mano_cfg).to(device)


def get_mano_joints_raw(mano_model, pose, root_orient, trans, betas, device='cuda'):
    """Get raw 3D joint positions from MANO params (no flips yet)."""
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

    joints = output.joints[0].cpu().numpy()
    return joints


def apply_hamer_transforms_v1(joints_3d, is_right):
    """
    Version 1: Just handedness X-flip (what skeleton script currently does)
    """
    joints = joints_3d.copy()
    x_flip = 2 * is_right - 1
    joints[:, 0] = x_flip * joints[:, 0]
    return joints


def apply_hamer_transforms_v2(joints_3d, is_right):
    """
    Version 2: Handedness X-flip + 180-degree rotation around X-axis
    The 180-degree rotation is what HaMeR applies to the mesh.
    But we also need to consider that pyrender camera looks down -Z.
    """
    joints = joints_3d.copy()

    # Step 1: Handedness X-flip (from run_mano)
    x_flip = 2 * is_right - 1
    joints[:, 0] = x_flip * joints[:, 0]

    # Step 2: 180-degree rotation around X-axis (from HaMeR renderer)
    # Rotation matrix for 180 deg around X: [[1,0,0],[0,-1,0],[0,0,-1]]
    joints[:, 1] = -joints[:, 1]
    joints[:, 2] = -joints[:, 2]

    return joints


def apply_hamer_transforms_v3(joints_3d, is_right, cam_trans):
    """
    Version 3: Full HaMeR-style rendering pipeline

    In HaMeR:
    1. Mesh vertices are rotated 180 deg around X
    2. Camera translation has X flipped
    3. Camera is placed at cam_trans (with flipped X)
    4. Camera looks down -Z

    For simple perspective projection (no pyrender), we need to:
    - Apply handedness X-flip to joints
    - Then apply the 180 deg X rotation to match the mesh transform
    - This puts the joints in the same space as HaMeR's rendered mesh
    - But since camera is at cam_trans looking down -Z, we need to
      transform joints to camera space: joints_cam = joints_world - cam_trans
    """
    joints = joints_3d.copy()

    # Step 1: Handedness X-flip (from run_mano)
    x_flip = 2 * is_right - 1
    joints[:, 0] = x_flip * joints[:, 0]

    # Step 2: 180-degree rotation around X-axis (from HaMeR renderer)
    joints[:, 1] = -joints[:, 1]
    joints[:, 2] = -joints[:, 2]

    # Step 3: Transform to camera space
    # In pyrender, camera pose sets where camera is located
    # Camera looks down -Z, so objects with negative Z are in front
    # We need to flip camera translation X like HaMeR does
    cam_t = cam_trans.copy()
    cam_t[0] *= -1.0  # HaMeR's X-flip on camera translation

    # Transform to camera space: joints relative to camera
    joints = joints - cam_t

    return joints


def apply_hamer_transforms_v4(joints_3d, is_right, cam_trans):
    """
    Version 4: Alternative interpretation

    The key insight: HaMeR uses pyrender which has OpenGL conventions.
    In OpenGL, camera looks down -Z, Y is up.

    The 180-degree X rotation flips Y and Z of the mesh.
    The camera at cam_trans (with X-flipped) looks at the rotated mesh.

    For our skeleton projection:
    - We want to project joints to 2D without pyrender
    - We need the joints in camera space where Z is depth (positive = in front)

    After 180 deg rotation: Y_new = -Y, Z_new = -Z
    So points that had positive Z (in front) now have negative Z.
    But pyrender camera looks down -Z, so negative Z is in front.

    For standard perspective projection (Z positive = in front):
    - We need to negate Z after the 180 deg rotation
    """
    joints = joints_3d.copy()

    # Step 1: Handedness X-flip (from run_mano)
    x_flip = 2 * is_right - 1
    joints[:, 0] = x_flip * joints[:, 0]

    # Step 2: Apply 180 deg X rotation, then fix Z for standard projection
    # 180 deg X: Y->-Y, Z->-Z
    # Then negate Z again so positive Z is in front for our projection
    joints[:, 1] = -joints[:, 1]
    # Z stays the same (two negations cancel out)

    return joints


def project_to_2d(joints_3d, fx, fy, cx, cy):
    """Project 3D joints to 2D using camera intrinsics."""
    z = joints_3d[:, 2]
    valid = np.abs(z) > 0.01  # Need to handle negative Z after rotation

    joints_2d = np.zeros((len(joints_3d), 2))
    joints_2d[valid, 0] = (joints_3d[valid, 0] / z[valid]) * fx + cx
    joints_2d[valid, 1] = (joints_3d[valid, 1] / z[valid]) * fy + cy

    return joints_2d, valid


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default='head', choices=['head', 'lwrist', 'rwrist'])
    parser.add_argument('--frame', type=int, default=50)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

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

    # Load params
    params = torch.load(params_path, map_location='cpu')
    pose = params['latent_pose']
    root_orient = params['root_orient']
    trans = params['trans']
    betas = params['betas']
    is_right = params['is_right']

    num_hands, T = pose.shape[:2]
    print(f"Loaded {num_hands} hands, {T} frames")

    # Load camera intrinsics
    with open(cameras_path) as f:
        cam_data = json.load(f)
    intrinsics = cam_data['intrinsics'][0] if isinstance(cam_data['intrinsics'][0], list) else cam_data['intrinsics']
    fx, fy, cx, cy = intrinsics
    print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # Load MANO model
    print("Loading MANO model...")
    mano_model = load_mano_model(args.device)

    # Read video frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to read frame {args.frame}")
        return

    H, W = frame.shape[:2]
    print(f"Frame size: {W}x{H}")

    # Make copies for different versions
    frame_v1 = frame.copy()
    frame_v2 = frame.copy()
    frame_v4 = frame.copy()

    # Process each hand
    for hand_idx in range(num_hands):
        is_right_val = is_right[hand_idx, args.frame].item()
        cam_trans = trans[hand_idx, args.frame].numpy()
        print(f"\n=== Hand {hand_idx} (is_right={is_right_val}) ===")
        print(f"cam_trans: {cam_trans}")

        # Get raw joints
        joints_raw = get_mano_joints_raw(
            mano_model,
            pose[hand_idx, args.frame],
            root_orient[hand_idx, args.frame],
            trans[hand_idx, args.frame],
            betas[hand_idx],
            device=args.device,
        )

        wrist_raw = joints_raw[0]
        print(f"Raw wrist 3D: X={wrist_raw[0]:.4f}, Y={wrist_raw[1]:.4f}, Z={wrist_raw[2]:.4f}")

        # Test V1: Just X-flip (current skeleton script)
        joints_v1 = apply_hamer_transforms_v1(joints_raw, is_right_val)
        wrist_v1 = joints_v1[0]
        print(f"V1 (current): X={wrist_v1[0]:.4f}, Y={wrist_v1[1]:.4f}, Z={wrist_v1[2]:.4f}")
        joints_2d_v1, valid_v1 = project_to_2d(joints_v1, fx, fy, cx, cy)
        print(f"V1 2D: X={joints_2d_v1[0][0]:.1f}, Y={joints_2d_v1[0][1]:.1f}")

        # Test V4: X-flip + Y-flip only
        joints_v4 = apply_hamer_transforms_v4(joints_raw, is_right_val, cam_trans)
        wrist_v4 = joints_v4[0]
        print(f"V4 (Y-flip): X={wrist_v4[0]:.4f}, Y={wrist_v4[1]:.4f}, Z={wrist_v4[2]:.4f}")
        joints_2d_v4, valid_v4 = project_to_2d(joints_v4, fx, fy, cx, cy)
        print(f"V4 2D: X={joints_2d_v4[0][0]:.1f}, Y={joints_2d_v4[0][1]:.1f}")

        # Draw on frames
        color = (0, 255, 0) if is_right_val > 0.5 else (255, 0, 0)  # Green=right, Blue=left

        if valid_v1[0]:
            x, y = int(joints_2d_v1[0][0]), int(joints_2d_v1[0][1])
            cv2.circle(frame_v1, (x, y), 15, color, -1)
            cv2.circle(frame_v1, (x, y), 15, (255, 255, 255), 2)
            cv2.putText(frame_v1, f"V1-H{hand_idx}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if valid_v4[0]:
            x, y = int(joints_2d_v4[0][0]), int(joints_2d_v4[0][1])
            cv2.circle(frame_v4, (x, y), 15, color, -1)
            cv2.circle(frame_v4, (x, y), 15, (255, 255, 255), 2)
            cv2.putText(frame_v4, f"V4-H{hand_idx}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Save results
    output_v1 = f"/tmp/wrist_test_v1_frame_{args.frame}.png"
    output_v4 = f"/tmp/wrist_test_v4_frame_{args.frame}.png"
    cv2.imwrite(output_v1, frame_v1)
    cv2.imwrite(output_v4, frame_v4)
    print(f"\nSaved:")
    print(f"  V1 (current): {output_v1}")
    print(f"  V4 (Y-flip): {output_v4}")


if __name__ == '__main__':
    main()
