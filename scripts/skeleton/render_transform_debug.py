#!/usr/bin/env python3
"""
Render a single video frame with different axis-flip transform combinations.

Outputs debug images to /tmp/transform_debug by default.
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch

# Add Dyn-HaMR paths
DYNHAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR")
sys.path.insert(0, str(DYNHAMR_ROOT / "dyn-hamr"))
sys.path.insert(0, str(DYNHAMR_ROOT / "third-party" / "hamer"))

from body_model import MANO

# MANO skeleton connections (21 joints)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Haptica website teal: #00d4aa = RGB(0, 212, 170) = BGR(170, 212, 0)
HAPTICA_TEAL = (170, 212, 0)


def load_mano_model(device="cuda"):
    mano_cfg = {
        "model_path": str(DYNHAMR_ROOT / "_DATA" / "data" / "mano"),
        "use_pca": False,
        "flat_hand_mean": False,
    }
    return MANO(batch_size=1, pose2rot=True, **mano_cfg).to(device)


def get_mano_joints_raw(mano_model, pose, root_orient, trans, betas, device="cuda"):
    """Get raw 3D joint positions from MANO params (no flips applied)."""
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

    return output.joints[0].cpu().numpy()


def apply_axis_flips(joints_3d, flip_x=False, flip_y=False, flip_z=False):
    joints = joints_3d.copy()
    if flip_x:
        joints[:, 0] = -joints[:, 0]
    if flip_y:
        joints[:, 1] = -joints[:, 1]
    if flip_z:
        joints[:, 2] = -joints[:, 2]
    return joints


def project_to_2d(joints_3d, intrinsics, cam_R=None, cam_t=None):
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


def draw_skeleton(frame, joints_2d, valid, color=HAPTICA_TEAL, thickness=3, radius=5):
    H, W = frame.shape[:2]

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
            cv2.circle(frame, pt, radius, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def load_track_info(track_info_path, num_tracks, num_frames):
    if not track_info_path.exists():
        return np.zeros((num_tracks, num_frames), dtype=bool), 0

    with open(track_info_path) as f:
        track_info = json.load(f)

    meta = track_info.get("meta", {})
    seq_interval = meta.get("seq_interval", [0, 0])
    start_frame = seq_interval[0] if seq_interval else 0

    vis_masks = [[False] * num_frames for _ in range(num_tracks)]
    tracks = track_info.get("tracks", {})
    for track_key, track_data in tracks.items():
        tensor_idx = track_data.get("index", None)
        if tensor_idx is None:
            try:
                tensor_idx = int(track_key)
            except ValueError:
                continue

        if 0 <= tensor_idx < num_tracks:
            mask = track_data.get("vis_mask", [])
            if len(mask) < num_frames:
                mask = mask + [False] * (num_frames - len(mask))
            elif len(mask) > num_frames:
                mask = mask[:num_frames]
            vis_masks[tensor_idx] = mask

    return np.array(vis_masks, dtype=bool), start_frame


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render transform debug frame.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--params", required=True, help="smooth_fit_params.pth path")
    parser.add_argument("--cameras", required=True, help="cameras.json path")
    parser.add_argument("--frame", type=int, default=50, help="Video frame index")
    parser.add_argument("--output-dir", default="/tmp/transform_debug", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device for MANO (cuda/cpu)")
    parser.add_argument("--ignore-visibility", action="store_true", help="Render even if vis_mask is false")
    parser.add_argument("--no-handedness", action="store_true",
                        help="Disable is_right-based X flip before combos")
    parser.add_argument("--hand-idx", type=int, default=None, help="Render only a specific hand index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = torch.load(args.params, map_location="cpu")
    pose = params["latent_pose"]
    root_orient = params["root_orient"]
    trans = params["trans"]
    betas = params["betas"]
    cam_R = params.get("cam_R", None)
    cam_t = params.get("cam_t", None)

    num_hands, num_frames = pose.shape[:2]

    track_info_path = Path(args.params).parent / "track_info.json"
    vis_masks, start_frame = load_track_info(track_info_path, num_hands, num_frames)
    mano_idx = args.frame - start_frame
    if not (0 <= mano_idx < num_frames):
        raise ValueError(
            f"Frame {args.frame} -> mano_idx {mano_idx} out of range (0..{num_frames - 1})."
        )

    with open(args.cameras) as f:
        cam_data = json.load(f)
    intrinsics = cam_data["intrinsics"][0] if isinstance(cam_data["intrinsics"][0], list) else cam_data["intrinsics"]

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, base_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {args.frame} from {args.video}")

    print(f"Loaded frame {args.frame} (mano_idx={mano_idx}), intrinsics={intrinsics}")

    mano_model = load_mano_model(args.device)

    combos = [
        ("no_flip", dict(fx=False, fy=False, fz=False, neg_cam=False)),
        ("flip_x", dict(fx=True, fy=False, fz=False, neg_cam=False)),
        ("flip_y", dict(fx=False, fy=True, fz=False, neg_cam=False)),
        ("flip_xy", dict(fx=True, fy=True, fz=False, neg_cam=False)),
        ("flip_xz", dict(fx=True, fy=False, fz=True, neg_cam=False)),
        ("flip_xyz", dict(fx=True, fy=True, fz=True, neg_cam=False)),
        ("flip_xyz_neg_cam", dict(fx=True, fy=True, fz=True, neg_cam=True)),
    ]

    for name, combo in combos:
        frame = base_frame.copy()

        hand_indices = [args.hand_idx] if args.hand_idx is not None else list(range(num_hands))
        for hand_idx in hand_indices:
            if not (0 <= hand_idx < num_hands):
                continue

            if (not args.ignore_visibility) and (not vis_masks[hand_idx, mano_idx]):
                continue

            joints_raw = get_mano_joints_raw(
                mano_model,
                pose[hand_idx, mano_idx],
                root_orient[hand_idx, mano_idx],
                trans[hand_idx, mano_idx],
                betas[hand_idx],
                device=args.device,
            )

            joints = joints_raw
            if not args.no_handedness:
                is_right_val = params["is_right"][hand_idx, mano_idx].item()
                joints = joints.copy()
                joints[:, 0] = (2 * is_right_val - 1) * joints[:, 0]

            joints = apply_axis_flips(joints, combo["fx"], combo["fy"], combo["fz"])

            cam_R_frame = None
            cam_t_frame = None
            if cam_R is not None and cam_t is not None:
                cam_R_frame = cam_R[hand_idx, mano_idx].cpu().numpy()
                cam_t_frame = cam_t[hand_idx, mano_idx].cpu().numpy()
                if combo["neg_cam"]:
                    cam_t_frame = -cam_t_frame

            joints_2d, valid = project_to_2d(
                joints,
                intrinsics,
                cam_R=cam_R_frame,
                cam_t=cam_t_frame,
            )
            frame = draw_skeleton(frame, joints_2d, valid)

        label = f"{name} (frame {args.frame}, mano_idx {mano_idx})"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = output_dir / f"{name}_frame{args.frame:04d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
