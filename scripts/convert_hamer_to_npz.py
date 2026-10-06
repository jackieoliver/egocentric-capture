#!/usr/bin/env python3
"""
Convert HaMeR pickle results to NPZ format for skeleton overlay script.
"""
import argparse
import pickle
import numpy as np
import cv2
from pathlib import Path

def rotmat_to_axis_angle(R):
    """Convert 3x3 rotation matrix to axis-angle (3D vector)."""
    rod, _ = cv2.Rodrigues(R.astype(np.float64))
    return rod.flatten()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="HaMeR pickle results file")
    p.add_argument("--output", required=True, help="Output NPZ file")
    args = p.parse_args()

    # Load pickle
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # Sort frames by image path (assumes numbered frames)
    frame_keys = sorted(data.keys())
    num_frames = len(frame_keys)
    print(f"Found {num_frames} frames")

    # Collect hands across all frames
    # Track left and right hands separately
    left_poses = []
    left_root = []
    left_trans = []
    left_bbox = []
    right_poses = []
    right_root = []
    right_trans = []
    right_bbox = []

    for frame_key in frame_keys:
        frame_data = data[frame_key]
        mano_list = frame_data.get('mano', [])
        cam_trans = frame_data.get('cam_trans', [])
        bboxes = frame_data.get('bboxes', [])

        left_found = False
        right_found = False

        for i, mano in enumerate(mano_list):
            is_right = mano.get('is_right', 1)
            global_orient = mano['global_orient']  # (1, 3, 3) or (3, 3)
            hand_pose = mano['hand_pose']  # (15, 3, 3)
            trans = cam_trans[i] if i < len(cam_trans) else np.zeros(3)
            bbox = bboxes[i] if i < len(bboxes) else np.zeros(4)

            # Convert rotation matrices to axis-angle
            if global_orient.ndim == 3:
                global_orient = global_orient[0]
            root_aa = rotmat_to_axis_angle(global_orient)

            pose_aa = np.zeros((15, 3))
            for j in range(15):
                pose_aa[j] = rotmat_to_axis_angle(hand_pose[j])

            if is_right:
                right_poses.append(pose_aa)
                right_root.append(root_aa)
                right_trans.append(trans)
                right_bbox.append(bbox)
                right_found = True
            else:
                left_poses.append(pose_aa)
                left_root.append(root_aa)
                left_trans.append(trans)
                left_bbox.append(bbox)
                left_found = True

        # Pad missing hands with zeros
        if not left_found:
            left_poses.append(np.zeros((15, 3)))
            left_root.append(np.zeros(3))
            left_trans.append(np.zeros(3))
            left_bbox.append(np.zeros(4))
        if not right_found:
            right_poses.append(np.zeros((15, 3)))
            right_root.append(np.zeros(3))
            right_trans.append(np.zeros(3))
            right_bbox.append(np.zeros(4))

    # Stack into arrays: (num_hands, num_frames, ...)
    hands_data = []
    is_right_arr = []

    if len(left_poses) == num_frames:
        hands_data.append({
            'pose': np.array(left_poses),
            'root': np.array(left_root),
            'trans': np.array(left_trans),
            'bbox': np.array(left_bbox),
        })
        is_right_arr.append(np.zeros(num_frames))

    if len(right_poses) == num_frames:
        hands_data.append({
            'pose': np.array(right_poses),
            'root': np.array(right_root),
            'trans': np.array(right_trans),
            'bbox': np.array(right_bbox),
        })
        is_right_arr.append(np.ones(num_frames))

    num_hands = len(hands_data)
    print(f"Exporting {num_hands} hands, {num_frames} frames each")

    # Build output arrays
    pose_body = np.stack([h['pose'] for h in hands_data], axis=0)  # (H, T, 15, 3)
    root_orient = np.stack([h['root'] for h in hands_data], axis=0)  # (H, T, 3)
    trans = np.stack([h['trans'] for h in hands_data], axis=0)  # (H, T, 3)
    bbox = np.stack([h['bbox'] for h in hands_data], axis=0)  # (H, T, 4)
    is_right = np.stack(is_right_arr, axis=0)  # (H, T)

    print(f"pose_body: {pose_body.shape}")
    print(f"root_orient: {root_orient.shape}")
    print(f"trans: {trans.shape}")
    print(f"bbox: {bbox.shape}")
    print(f"is_right: {is_right.shape}")

    np.savez(args.output,
             pose_body=pose_body,
             root_orient=root_orient,
             trans=trans,
             bbox=bbox,
             is_right=is_right)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
