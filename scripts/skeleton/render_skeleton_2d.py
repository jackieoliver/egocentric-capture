#!/usr/bin/env python3
"""
Render skeleton overlay using HaMeR's 2D keypoints directly.

This bypasses the MANO 3D→2D projection and uses the 2D keypoints
from HaMeR's 'extra_data' field, which come from the ViTPose detector.
This gives perfect alignment with the detected hand positions.
"""

import argparse
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

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


def draw_skeleton_2d(frame, keypoints_2d, color=HAPTICA_TEAL, thickness=3, radius=5,
                     conf_threshold=0.3, visibility_threshold=0.2):
    """Draw skeleton from 2D keypoints with confidence values."""
    H, W = frame.shape[:2]
    keypoints_2d = np.array(keypoints_2d)
    
    if len(keypoints_2d) != 21:
        return frame
    
    # Extract x, y, confidence
    xs = keypoints_2d[:, 0]
    ys = keypoints_2d[:, 1]
    confs = keypoints_2d[:, 2] if keypoints_2d.shape[1] > 2 else np.ones(21)
    
    # Check visibility
    valid = confs > conf_threshold
    on_screen = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    visible = valid & on_screen
    
    if visible.sum() / len(visible) < visibility_threshold:
        return frame
    
    # Skip if wrist not visible
    if not visible[0]:
        return frame
    
    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if visible[i] and visible[j]:
            pt1 = (int(xs[i]), int(ys[i]))
            pt2 = (int(xs[j]), int(ys[j]))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # Draw joints
    for idx in range(21):
        if visible[idx]:
            pt = (int(xs[idx]), int(ys[idx]))
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame


def render_skeleton_video(hamer_pkl_path, video_path, output_path):
    """Render skeleton overlay using HaMeR's 2D keypoints."""
    # Load HaMeR results
    print(f"Loading HaMeR results from {hamer_pkl_path}")
    with open(hamer_pkl_path, 'rb') as f:
        hamer_data = pickle.load(f)
    
    frame_keys = sorted([k for k in hamer_data.keys() if k.endswith('.jpg')])
    print(f"Found {len(frame_keys)} frames with detections")
    
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
    
    print(f"Rendering to {output_path}")
    print(f"Using Haptica teal #00d4aa")
    
    rendered_hands = 0
    
    for video_frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find matching HaMeR frame
        if video_frame_idx < len(frame_keys):
            frame_key = frame_keys[video_frame_idx]
            frame_data = hamer_data.get(frame_key, {})
            
            # Get 2D keypoints from extra_data
            extra_data = frame_data.get('extra_data', [])
            
            for hand_kps in extra_data:
                frame = draw_skeleton_2d(frame, hand_kps)
                rendered_hands += 1
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Done! Rendered {rendered_hands} hand instances")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render skeleton using HaMeR 2D keypoints')
    parser.add_argument('--hamer', type=str, required=True, help='Path to HaMeR results.pkl')
    parser.add_argument('--video', type=str, required=True, help='Input video')
    parser.add_argument('--output', type=str, required=True, help='Output video')
    args = parser.parse_args()
    
    render_skeleton_video(args.hamer, args.video, args.output)
