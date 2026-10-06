#!/usr/bin/env python3
"""
Render debug overlay showing all tracking data from HaMeR pkl.

Shows:
- Bounding boxes with confidence scores
- 2D keypoints with joint indices
- Track IDs
- Handedness labels
- Frame counter
"""

import argparse
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Colors
COLORS = {
    'left': (255, 100, 100),   # Blue-ish for left
    'right': (100, 255, 100),  # Green-ish for right
    'bbox': (0, 255, 255),     # Yellow for bbox
    'text': (255, 255, 255),   # White text
    'bg': (0, 0, 0),           # Black background
}

# MANO skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]


def draw_text_with_bg(frame, text, pos, font_scale=0.5, color=(255,255,255), thickness=1):
    """Draw text with dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x-2, y-h-2), (x+w+2, y+4), (0,0,0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def render_debug_overlay(pkl_path, video_path, output_path):
    """Render debug visualization of tracking data."""

    # Load pkl
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    frame_keys = sorted([k for k in data.keys() if isinstance(data[k], dict)])
    print(f"Found {len(frame_keys)} frames")

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

    print(f"Rendering debug overlay to {output_path}")

    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw frame counter
        draw_text_with_bg(frame, f"Frame: {frame_idx}", (10, 30), font_scale=0.7, color=(0, 255, 255))

        # Find matching pkl data
        if frame_idx < len(frame_keys):
            frame_key = frame_keys[frame_idx]
            frame_data = data.get(frame_key, {})

            # Get hand data
            mano_list = frame_data.get('mano', [])
            bboxes = frame_data.get('bboxes', [])
            bbox_conf = frame_data.get('bbox_conf', [])
            extra_data = frame_data.get('extra_data', [])  # 2D keypoints
            track_ids = frame_data.get('tracked_ids', [])

            num_hands = len(mano_list)
            draw_text_with_bg(frame, f"Hands: {num_hands}", (10, 60), font_scale=0.6, color=(0, 255, 0))

            for h_idx, mano_dict in enumerate(mano_list):
                is_right = mano_dict.get('is_right', 0)
                hand_label = 'R' if is_right else 'L'
                color = COLORS['right'] if is_right else COLORS['left']

                # Track ID
                tid = track_ids[h_idx] if h_idx < len(track_ids) else -1

                # Confidence
                conf = bbox_conf[h_idx] if h_idx < len(bbox_conf) else 0
                if hasattr(conf, 'item'):
                    conf = conf.item()

                # Draw bounding box
                if h_idx < len(bboxes):
                    bbox = bboxes[h_idx]
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Label with track ID, handedness, confidence
                        label = f"{hand_label} T{tid} {conf:.2f}"
                        draw_text_with_bg(frame, label, (x1, y1-5), font_scale=0.5, color=color)

                # Draw 2D keypoints
                if h_idx < len(extra_data):
                    kps = extra_data[h_idx]
                    if hasattr(kps, 'tolist'):
                        kps = kps.tolist()

                    if kps and len(kps) >= 21:
                        # Draw skeleton connections
                        for i, j in SKELETON_CONNECTIONS:
                            if i < len(kps) and j < len(kps):
                                x1, y1 = int(kps[i][0]), int(kps[i][1])
                                x2, y2 = int(kps[j][0]), int(kps[j][1])
                                c1 = kps[i][2] if len(kps[i]) > 2 else 1.0
                                c2 = kps[j][2] if len(kps[j]) > 2 else 1.0
                                if c1 > 0.3 and c2 > 0.3:
                                    cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                        # Draw keypoints with indices
                        for kp_idx, kp in enumerate(kps[:21]):
                            x, y = int(kp[0]), int(kp[1])
                            conf_kp = kp[2] if len(kp) > 2 else 1.0

                            if conf_kp > 0.3 and 0 <= x < width and 0 <= y < height:
                                # Draw point
                                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                                cv2.circle(frame, (x, y), 4, (255,255,255), 1, cv2.LINE_AA)

                                # Draw joint index (only for key joints to avoid clutter)
                                if kp_idx in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                                    cv2.putText(frame, str(kp_idx), (x+5, y-5),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        out.write(frame)

    cap.release()
    out.release()

    # Convert to H.264
    output_h264 = str(output_path).replace('.mp4', '_h264.mp4')
    print("Converting to H.264...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', str(output_path),
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        output_h264
    ], capture_output=True)

    print(f"Done! Output: {output_h264}")
    return output_h264


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render debug overlay from HaMeR pkl')
    parser.add_argument('--pkl', type=str, required=True, help='Path to results.pkl')
    parser.add_argument('--video', type=str, required=True, help='Input video')
    parser.add_argument('--output', type=str, required=True, help='Output video')
    args = parser.parse_args()

    render_debug_overlay(args.pkl, args.video, args.output)
