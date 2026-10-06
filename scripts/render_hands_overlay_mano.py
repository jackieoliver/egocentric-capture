#!/usr/bin/env python3
"""
Overlay Dyn-HaMR hand skeletons using the actual MANO model.
Must be run in the dynhamr conda environment.

Usage:
    conda activate dynhamr
    python render_hands_overlay_mano.py --video VIDEO --hands NPZ --output OUTPUT
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

# Add Dyn-HaMR paths
DYN_HAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR/dyn-hamr")
sys.path.insert(0, str(DYN_HAMR_ROOT))

# MANO hand skeleton connectivity (16 joints)
# MANO joints: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
HAND_EDGES = [
    # Index finger: wrist -> index1 -> index2 -> index3
    (0, 1), (1, 2), (2, 3),
    # Middle finger: wrist -> middle1 -> middle2 -> middle3
    (0, 4), (4, 5), (5, 6),
    # Pinky finger: wrist -> pinky1 -> pinky2 -> pinky3
    (0, 7), (7, 8), (8, 9),
    # Ring finger: wrist -> ring1 -> ring2 -> ring3
    (0, 10), (10, 11), (11, 12),
    # Thumb: wrist -> thumb1 -> thumb2 -> thumb3
    (0, 13), (13, 14), (14, 15),
]

# Map edges to finger indices for coloring
EDGE_TO_FINGER = {
    (0, 1): 0, (1, 2): 0, (2, 3): 0,      # Index
    (0, 4): 1, (4, 5): 1, (5, 6): 1,      # Middle
    (0, 7): 2, (7, 8): 2, (8, 9): 2,      # Pinky
    (0, 10): 3, (10, 11): 3, (11, 12): 3, # Ring
    (0, 13): 4, (13, 14): 4, (14, 15): 4, # Thumb
}

# Colors for each finger (BGR)
FINGER_COLORS = [
    (0, 255, 255),    # Index - yellow
    (0, 255, 0),      # Middle - green
    (255, 0, 0),      # Pinky - blue
    (255, 255, 0),    # Ring - cyan
    (147, 20, 255),   # Thumb - pink
]

# Fingertip joints for highlighting (last joint of each finger)
FINGERTIP_JOINTS = [3, 6, 9, 12, 15]  # index3, middle3, pinky3, ring3, thumb3


def reproject(points3d, cam_R, cam_t, cam_f, cam_center):
    """
    Reproject 3D points to 2D using camera parameters.
    Matches Dyn-HaMR's reproject function exactly.

    Args:
        points3d: (B, T, N, 3) - 3D joint positions
        cam_R: (B, T, 3, 3) - camera rotation matrices
        cam_t: (B, T, 3) - camera translations
        cam_f: (T, 2) - focal lengths [fx, fy]
        cam_center: (T, 2) - principal points [cx, cy]

    Returns:
        points2d: (B, T, N, 2) - 2D projected points
    """
    B, T, N, _ = points3d.shape
    # Apply camera rotation
    points3d = torch.einsum("btij,btnj->btni", cam_R, points3d)
    # Add translation
    points3d = points3d + cam_t[..., None, :]  # (B, T, N, 3)
    # Perspective divide
    points2d = points3d[..., :2] / points3d[..., 2:3]
    # Apply intrinsics
    points2d = cam_f[None, :, None] * points2d + cam_center[None, :, None]
    return points2d


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


def draw_skeleton(frame, pts_2d, is_right, line_thickness=3, joint_radius=5):
    """Draw hand skeleton on frame with colored fingers."""
    if is_right > 0.5:
        base_color = (0, 200, 100)  # Green for right
    else:
        base_color = (200, 100, 0)  # Blue for left

    # Draw bones
    for edge in HAND_EDGES:
        i, j = edge
        if i < len(pts_2d) and j < len(pts_2d):
            if pts_2d[i] is not None and pts_2d[j] is not None:
                p1 = (int(pts_2d[i][0]), int(pts_2d[i][1]))
                p2 = (int(pts_2d[j][0]), int(pts_2d[j][1]))
                finger_idx = EDGE_TO_FINGER.get(edge, 0)
                color = FINGER_COLORS[finger_idx]
                cv2.line(frame, p1, p2, color, line_thickness, cv2.LINE_AA)

    # Draw joints
    for idx, pt in enumerate(pts_2d):
        if pt is not None:
            px = (int(pt[0]), int(pt[1]))
            # Wrist is white, fingertips are brighter
            if idx == 0:
                cv2.circle(frame, px, joint_radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            elif idx in FINGERTIP_JOINTS:
                finger_idx = FINGERTIP_JOINTS.index(idx)
                cv2.circle(frame, px, joint_radius + 1, FINGER_COLORS[finger_idx], -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, px, joint_radius, base_color, -1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--hands", required=True, help="Dyn-HaMR NPZ file")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    p.add_argument("--hand-fps", type=float, default=0.0, help="Hand data FPS (0=match video)")
    p.add_argument("--line-thickness", type=int, default=3)
    p.add_argument("--joint-radius", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Import MANO model
    try:
        from smplx import MANO
    except ImportError:
        print("Error: smplx not found. Run in dynhamr conda environment.", file=sys.stderr)
        sys.exit(1)

    # Load hand data
    print(f"Loading hand data from {args.hands}")
    data = np.load(args.hands, allow_pickle=True)

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
    print(f"Loaded {num_hands} hands, {num_frames} frames")

    # Convert to tensors
    device = args.device
    pose_t = torch.from_numpy(pose).float().to(device)
    root_orient_t = torch.from_numpy(root_orient).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)
    is_right_t = torch.from_numpy(is_right).float().to(device)
    if betas is not None:
        betas_t = torch.from_numpy(betas).float().to(device)
    else:
        betas_t = None

    # Setup camera parameters
    if cam_R is None:
        cam_R = np.tile(np.eye(3), (num_hands, num_frames, 1, 1))
    if cam_t is None:
        cam_t = np.zeros((num_hands, num_frames, 3))

    cam_R_t = torch.from_numpy(cam_R).float().to(device)
    cam_t_t = torch.from_numpy(cam_t).float().to(device)

    # Intrinsics: [fx, fy, cx, cy]
    fx, fy, cx, cy = intrins[:4]
    cam_f = torch.tensor([[fx, fy]] * num_frames, device=device)  # (T, 2)
    cam_center = torch.tensor([[cx, cy]] * num_frames, device=device)  # (T, 2)

    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Load MANO model
    mano_path = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR/_DATA/data/mano")
    print(f"Loading MANO model from {mano_path}")

    body_model = MANO(
        str(mano_path),
        is_rhand=True,  # Will flip for left hand
        use_pca=False,
        flat_hand_mean=False,
        batch_size=num_frames,  # Process one hand at a time
    ).to(device)
    body_model.eval()

    # Compute 3D joints for all hands
    print("Computing 3D joints with MANO model...")
    all_joints_2d = []

    with torch.no_grad():
        for h in range(num_hands):
            # Get joints3d from MANO
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

            all_joints_2d.append(joints2d[0].cpu().numpy())  # (T, J, 2)
            print(f"  Hand {h}: is_right={is_right[h, 0]:.1f}, "
                  f"sample 2D pos frame 0 joint 0: {joints2d[0, 0, 0].cpu().numpy()}")

    # Open video
    print(f"Opening video: {args.video}")
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
            pts_2d = all_joints_2d[h][hi]  # (J, 2)
            # Convert to list of tuples, checking bounds
            pts_list = []
            for j in range(pts_2d.shape[0]):
                x, y = pts_2d[j]
                if 0 <= x < width and 0 <= y < height:
                    pts_list.append((x, y))
                else:
                    pts_list.append(None)

            draw_skeleton(frame, pts_list, is_right[h, hi],
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
