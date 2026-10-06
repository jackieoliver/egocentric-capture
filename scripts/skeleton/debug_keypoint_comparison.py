#!/usr/bin/env python3
"""
Compare HaMeR 2D keypoints vs projected skeleton keypoints.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

# Add Dyn-HaMR paths
DYNHAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR")
import sys
sys.path.insert(0, str(DYNHAMR_ROOT / "dyn-hamr"))
sys.path.insert(0, str(DYNHAMR_ROOT / "third-party" / "hamer"))

from body_model import MANO

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_COLOR = (255, 255, 255)

# MANO skeleton connections (21 joints)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_mano_model(device="cuda"):
    mano_cfg = {
        "model_path": str(DYNHAMR_ROOT / "_DATA" / "data" / "mano"),
        "use_pca": False,
        "flat_hand_mean": False,
    }
    return MANO(batch_size=1, pose2rot=True, **mano_cfg).to(device)


def get_mano_joints(mano_model, pose, root_orient, trans, betas, is_right, device="cuda"):
    """Get 3D joint positions from MANO params with Dyn-HaMR X-flip."""
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

    # Dyn-HaMR handedness X-flip
    is_right_val = is_right.item() if hasattr(is_right, "item") else is_right
    x_flip = 2 * is_right_val - 1
    joints[:, 0] = x_flip * joints[:, 0]

    return joints


def project_to_2d(joints_3d, intrinsics, cam_R=None, cam_t=None):
    """Project 3D joints to 2D using camera intrinsics."""
    fx, fy, cx, cy = intrinsics
    joints = joints_3d
    if cam_R is not None and cam_t is not None:
        joints = (cam_R @ joints.T).T + cam_t

    z = joints[:, 2]
    valid = z > 0.01

    joints_2d = np.zeros((len(joints_3d), 2), dtype=np.float32)
    joints_2d[valid, 0] = (joints[valid, 0] / z[valid]) * fx + cx
    joints_2d[valid, 1] = (joints[valid, 1] / z[valid]) * fy + cy

    return joints_2d, valid


def load_hamer_results(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def extract_hamer_frame(frame_data):
    bbox_conf = frame_data.get("bbox_conf", [])
    extra_data = frame_data.get("extra_data", [])
    mano_list = frame_data.get("mano", [])

    keypoints_2d = [np.array(kp, dtype=np.float32) if kp is not None else np.zeros((21, 3), dtype=np.float32)
                    for kp in extra_data]
    handedness = []
    for i in range(len(keypoints_2d)):
        is_right_val = None
        if i < len(mano_list):
            is_right_val = mano_list[i].get("is_right", None)
        handedness.append(is_right_val)

    return keypoints_2d, bbox_conf, handedness


def select_hamer_index(hand_is_right, hand_idx, num_hands, keypoints_2d, bbox_conf, handedness):
    if not keypoints_2d:
        return None

    # Prefer handedness match if available
    candidates = []
    for i, h in enumerate(handedness):
        if h is None:
            continue
        if (h >= 0.5) == (hand_is_right >= 0.5):
            candidates.append(i)

    if candidates:
        if bbox_conf:
            return max(candidates, key=lambda i: bbox_conf[i] if i < len(bbox_conf) else -1)
        return candidates[0]

    # Fall back to index match if counts align
    if num_hands == len(keypoints_2d) and 0 <= hand_idx < len(keypoints_2d):
        return hand_idx

    # Fall back to highest confidence
    if len(keypoints_2d) >= 1:
        if bbox_conf:
            return int(np.argmax(bbox_conf))
        return 0

    return None


def draw_keypoints(frame, keypoints, color, conf_threshold=0.2, radius=4):
    for x, y, c in keypoints:
        if c < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def draw_skeleton_points(frame, joints_2d, valid, color, radius=4):
    H, W = frame.shape[:2]
    for i, (x, y) in enumerate(joints_2d):
        if not valid[i]:
            continue
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def draw_offset_lines(frame, joints_2d, valid, keypoints, conf_threshold=0.2):
    H, W = frame.shape[:2]
    for i, (x, y, c) in enumerate(keypoints):
        if c < conf_threshold or not valid[i]:
            continue
        sx, sy = joints_2d[i]
        if not (0 <= sx < W and 0 <= sy < H):
            continue
        cv2.line(frame, (int(sx), int(sy)), (int(x), int(y)), LINE_COLOR, 1, cv2.LINE_AA)


def main():
    video_path = Path("/home/example/dyn_hamr_workspace/pipeline_output/test_5sec/undistorted.mp4")
    hamer_path = Path("/home/example/dyn_hamr_workspace/pipeline_output/test_5sec/hamer_out/results_3pass.pkl")
    params_path = Path("/home/example/dyn_hamr_workspace/pipeline_output/test_5sec/skeleton_params/smooth_fit_params.pth")
    cameras_path = Path("/home/example/dyn_hamr_workspace/pipeline_output/test_5sec/skeleton_params/cameras.json")
    output_dir = Path("/tmp/skeleton_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_to_draw = [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 50, 75]

    hamer_data = load_hamer_results(hamer_path)
    frame_keys = sorted(hamer_data.keys())

    params = torch.load(params_path, map_location="cpu", weights_only=False)
    pose = params["latent_pose"]
    root_orient = params["root_orient"]
    trans = params["trans"]
    betas = params["betas"]
    is_right = params["is_right"]
    cam_R = params.get("cam_R", None)
    cam_t = params.get("cam_t", None)

    num_hands, T = pose.shape[:2]

    with open(cameras_path) as f:
        cam_data = json.load(f)
    intrinsics = cam_data.get("intrinsics", None)
    if isinstance(intrinsics, list) and intrinsics and isinstance(intrinsics[0], list):
        intrinsics = intrinsics[0]
    if isinstance(intrinsics, list) and len(intrinsics) >= 4:
        cx, cy = intrinsics[2], intrinsics[3]
    else:
        # Fallback to video center
        cap_tmp = cv2.VideoCapture(str(video_path))
        width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        cx, cy = width / 2.0, height / 2.0

    focal = 4285.71
    intrinsics = [focal, focal, cx, cy]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano_model = load_mano_model(device=device)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    offsets = []
    offsets_by_joint = [[] for _ in range(21)]
    offsets_by_frame = {}

    for frame_idx in frames_to_draw:
        if frame_idx >= total_frames or frame_idx >= len(frame_keys):
            print(f"Skipping frame {frame_idx}: out of range")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipping frame {frame_idx}: failed to read")
            continue

        frame_data = hamer_data[frame_keys[frame_idx]]
        keypoints_2d_list, bbox_conf, handedness = extract_hamer_frame(frame_data)

        frame_offsets = []

        for hand_idx in range(num_hands):
            if frame_idx >= T:
                continue

            hand_is_right = float(is_right[hand_idx, frame_idx])

            try:
                joints_3d = get_mano_joints(
                    mano_model,
                    pose[hand_idx, frame_idx],
                    root_orient[hand_idx, frame_idx],
                    trans[hand_idx, frame_idx],
                    betas[hand_idx],
                    is_right[hand_idx, frame_idx],
                    device=device,
                )
            except Exception as exc:
                print(f"Frame {frame_idx} hand {hand_idx}: MANO error {exc}")
                continue

            cam_R_frame = None
            cam_t_frame = None
            if cam_R is not None and cam_t is not None:
                cam_R_frame = cam_R[hand_idx, frame_idx].cpu().numpy()
                cam_t_frame = cam_t[hand_idx, frame_idx].cpu().numpy()

            joints_2d, valid = project_to_2d(joints_3d, intrinsics, cam_R=cam_R_frame, cam_t=cam_t_frame)

            h_idx = select_hamer_index(hand_is_right, hand_idx, num_hands, keypoints_2d_list, bbox_conf, handedness)
            if h_idx is None:
                continue

            hamer_kp = keypoints_2d_list[h_idx]

            draw_keypoints(frame, hamer_kp, GREEN)
            draw_skeleton_points(frame, joints_2d, valid, RED)
            draw_offset_lines(frame, joints_2d, valid, hamer_kp)

            # offsets
            for j, (x, y, c) in enumerate(hamer_kp):
                if c < 0.2 or not valid[j]:
                    continue
                sx, sy = joints_2d[j]
                dist = float(np.linalg.norm([x - sx, y - sy]))
                offsets.append(dist)
                offsets_by_joint[j].append(dist)
                frame_offsets.append(dist)

        offsets_by_frame[frame_idx] = frame_offsets
        cv2.putText(frame, f"frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"Saved {out_path}")

    cap.release()

    if offsets:
        offsets_np = np.array(offsets)
        print("\n=== Offset Summary ===")
        print(f"Total pairs: {len(offsets)}")
        print(f"Mean: {offsets_np.mean():.2f} px")
        print(f"Median: {np.median(offsets_np):.2f} px")
        print(f"P95: {np.percentile(offsets_np, 95):.2f} px")
        print(f"Max: {offsets_np.max():.2f} px")

        print("\nPer-joint mean offsets:")
        for j, vals in enumerate(offsets_by_joint):
            if vals:
                print(f"  Joint {j:02d}: {np.mean(vals):.2f} px (n={len(vals)})")
            else:
                print(f"  Joint {j:02d}: n=0")

        print("\nPer-frame mean offsets:")
        for frame_idx in frames_to_draw:
            vals = offsets_by_frame.get(frame_idx, [])
            if vals:
                print(f"  Frame {frame_idx:03d}: {np.mean(vals):.2f} px (n={len(vals)})")
            else:
                print(f"  Frame {frame_idx:03d}: n=0")
    else:
        print("\nNo offsets computed.")


if __name__ == "__main__":
    main()
