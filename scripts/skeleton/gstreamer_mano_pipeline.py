#!/usr/bin/env python3
"""Simple GStreamer + MANO test using OpenCV for video, GStreamer concepts"""
import sys
import json
import time
import cv2
import torch
import numpy as np

# Add Dyn-HaMR paths
sys.path.insert(0, "/home/example/dyn_hamr_workspace")
from render_skeleton import (
    load_mano_model, get_mano_joints, project_to_2d, 
    validate_joints, load_track_info
)

# Config
VIDEO = "/home/example/dyn_hamr_workspace/Dyn-HaMR/test/videos/head_60fps.mp4"
PARAMS_PATH = "/home/example/dyn_hamr_workspace/outputs/logs/video-custom/2026-01-23/head_60fps-all-shot-0-0--1/smooth_fit_params.pth"
CAMERAS_PATH = "/home/example/dyn_hamr_workspace/outputs/logs/video-custom/2026-01-23/head_60fps-all-shot-0-0--1/cameras.json"
MAX_FRAMES = 300  # 5 seconds at 60fps

print("🎬 GStreamer-style MANO Pipeline Test")
print("=" * 50)

# Load MANO model
print("\n1. Loading MANO model...")
mano = load_mano_model(device='cuda')
print("   ✓ MANO loaded on CUDA")

# Load smooth_fit params
print("\n2. Loading smooth_fit_params...")
params = torch.load(PARAMS_PATH, map_location='cpu')
pose = params['latent_pose']
root_orient = params['root_orient']
trans = params['trans']
betas = params['betas']
is_right = params['is_right']
num_hands, T = pose.shape[:2]
print(f"   ✓ {num_hands} hands, {T} frames")

# Load intrinsics
with open(CAMERAS_PATH) as f:
    cam_data = json.load(f)
intrinsics = cam_data['intrinsics'][0]
print(f"   ✓ Intrinsics: fx={intrinsics[0]}, fy={intrinsics[1]}")

# Load visibility
output_dir = "/home/example/dyn_hamr_workspace/outputs/logs/video-custom/2026-01-23/head_60fps-all-shot-0-0--1"
vis_masks, start_frame = load_track_info(output_dir, num_hands, T)
print(f"   ✓ Visibility loaded, start_frame={start_frame}")

# Open video (simulating GStreamer filesrc ! decodebin)
print("\n3. Opening video source...")
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"   ✓ {width}x{height} @ {fps}fps")

# Process frames (simulating appsink callback)
print(f"\n4. Processing {MAX_FRAMES} frames (5 seconds)...")
print("   [This simulates GStreamer appsink → Python → MANO inference]")
print()

results = []
hands_detected = 0
start_time = time.time()

for frame_idx in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    
    mano_idx = frame_idx - start_frame
    frame_result = {"frame": frame_idx, "hands": []}
    
    if 0 <= mano_idx < T:
        for hand_idx in range(num_hands):
            if not vis_masks[hand_idx, mano_idx]:
                continue
            
            hand_is_right = is_right[hand_idx, mano_idx].item()
            if not (0.0 <= hand_is_right <= 1.0):
                continue
            
            try:
                # MANO inference
                joints_3d = get_mano_joints(
                    mano,
                    pose[hand_idx, mano_idx],
                    root_orient[hand_idx, mano_idx],
                    trans[hand_idx, mano_idx],
                    betas[hand_idx],
                    is_right[hand_idx, mano_idx],
                    device='cuda'
                )
                
                if validate_joints(joints_3d):
                    joints_2d, valid = project_to_2d(joints_3d, intrinsics)
                    wrist = joints_2d[0]
                    
                    frame_result["hands"].append({
                        "hand": "right" if hand_is_right > 0.5 else "left",
                        "wrist_px": [int(wrist[0]), int(wrist[1])],
                        "depth_m": float(joints_3d[0, 2])
                    })
                    hands_detected += 1
            except:
                pass
    
    results.append(frame_result)
    
    # Progress
    if (frame_idx + 1) % 60 == 0:
        elapsed = time.time() - start_time
        print(f"   Frame {frame_idx+1:3d}: {(frame_idx+1)/elapsed:.1f} fps, {hands_detected} hands so far")

cap.release()
elapsed = time.time() - start_time

# Results
print()
print("=" * 50)
print("✅ Test Complete!")
print(f"   Frames processed: {len(results)}")
print(f"   Total time: {elapsed:.2f}s")
print(f"   Average FPS: {len(results)/elapsed:.1f}")
print(f"   Total hands detected: {hands_detected}")

# Sample JSONL output (what GStreamer pipeline would produce)
print()
print("📊 Sample JSONL output (what the pipeline produces):")
for r in results[50:55]:
    if r["hands"]:
        print(f'   {json.dumps(r)}')
