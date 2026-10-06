#!/usr/bin/env python3
"""
Re-render HaMeR mesh output with proper frame synchronization.

The HaMeR render_all output drops frames where no hands were detected,
causing the output video to be out of sync with the original.

This script:
1. Reads the original video frame by frame
2. For each frame N, checks if HaMeR rendered frame exists (NNNNNN.jpg)
3. Uses the HaMeR frame if it exists, otherwise uses original frame
4. Outputs a properly frame-synced video

Usage:
    python 06_sync_hamer_render.py <original_video> <hamer_render_dir> -o <output.mp4>
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def sync_hamer_render(original_video: str, hamer_render_dir: str, output_path: str):
    """Re-render HaMeR output with proper frame sync."""

    hamer_dir = Path(hamer_render_dir)

    # Get list of available HaMeR rendered frames
    hamer_frames = {}
    for jpg_file in hamer_dir.glob("*.jpg"):
        frame_num = int(jpg_file.stem)
        hamer_frames[frame_num] = jpg_file

    print(f"Found {len(hamer_frames)} HaMeR rendered frames")
    print(f"Frame range: {min(hamer_frames.keys())} - {max(hamer_frames.keys())}")

    # Open original video
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {original_video}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_with_mesh = 0
    frames_without_mesh = 0

    for frame_idx in tqdm(range(total_frames), desc="Syncing frames"):
        ret, original_frame = cap.read()
        if not ret:
            break

        # Check if HaMeR rendered this frame
        if frame_idx in hamer_frames:
            # Use HaMeR rendered frame
            hamer_frame = cv2.imread(str(hamer_frames[frame_idx]))
            if hamer_frame is not None:
                # Resize if needed (HaMeR might render at different resolution)
                if hamer_frame.shape[:2] != (height, width):
                    hamer_frame = cv2.resize(hamer_frame, (width, height))
                out.write(hamer_frame)
                frames_with_mesh += 1
            else:
                out.write(original_frame)
                frames_without_mesh += 1
        else:
            # No HaMeR detection for this frame, use original
            out.write(original_frame)
            frames_without_mesh += 1

    cap.release()
    out.release()

    print(f"\nSaved: {output_path}")
    print(f"Frames with mesh overlay: {frames_with_mesh}")
    print(f"Frames without mesh (original): {frames_without_mesh}")

    return True


def create_side_by_side(video_a: str, video_b: str, output_path: str,
                         label_a: str = "Original", label_b: str = "Synced"):
    """Create side-by-side comparison video."""

    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)

    if not cap_a.isOpened() or not cap_b.isOpened():
        print("ERROR: Could not open one of the videos")
        return False

    width_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_a.get(cv2.CAP_PROP_FPS)
    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video A: {total_a} frames")
    print(f"Video B: {total_b} frames")

    # Output is side by side
    out_width = width_a * 2
    out_height = height_a

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    frame_idx = 0
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        if not ret_a:
            break

        # If video B is shorter, use black frame
        if not ret_b:
            frame_b = np.zeros_like(frame_a)

        # Resize frame_b to match frame_a if needed
        if frame_b.shape[:2] != frame_a.shape[:2]:
            frame_b = cv2.resize(frame_b, (width_a, height_a))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_a, f"{label_a} - Frame {frame_idx}", (10, 30),
                    font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_b, f"{label_b} - Frame {frame_idx}", (10, 30),
                    font, 0.8, (0, 255, 0), 2)

        # Combine side by side
        combined = np.hstack([frame_a, frame_b])
        out.write(combined)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap_a.release()
    cap_b.release()
    out.release()

    print(f"Saved side-by-side: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync HaMeR render with original video")
    parser.add_argument("original_video", help="Path to original video")
    parser.add_argument("hamer_render_dir", help="Path to HaMeR render_all directory (with jpgs)")
    parser.add_argument("-o", "--output", default="hamer_synced.mp4", help="Output video path")
    parser.add_argument("--compare", help="Create side-by-side with this video")
    parser.add_argument("--compare-output", default="comparison.mp4", help="Comparison output path")

    args = parser.parse_args()

    # Sync the render
    success = sync_hamer_render(args.original_video, args.hamer_render_dir, args.output)

    if success and args.compare:
        create_side_by_side(args.compare, args.output, args.compare_output,
                           "Unsynced HaMeR", "Synced HaMeR")


if __name__ == "__main__":
    main()
