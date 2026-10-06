#!/usr/bin/env python3
"""Minimal test script to verify Rerun video logging works correctly."""
import sys
from pathlib import Path

import rerun as rr


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        sys.exit(1)

    print(f"Video: {video_path}")
    print(f"Size: {video_path.stat().st_size / 1e6:.1f} MB")

    # Initialize Rerun
    rr.init("video_test", spawn=True)

    # Log the video asset
    video_asset = rr.AssetVideo(path=video_path)
    print(f"AssetVideo created")

    # Read frame timestamps
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    print(f"Frame timestamps: {len(frame_timestamps_ns)} frames")
    if len(frame_timestamps_ns) > 0:
        print(f"  First: {frame_timestamps_ns[0] / 1e9:.3f}s")
        print(f"  Last: {frame_timestamps_ns[-1] / 1e9:.3f}s")

    # Log video as static
    rr.log("video", video_asset, static=True)
    print("Logged AssetVideo to 'video' entity")

    # Log frame references using send_columns
    # Key: use 'indexes' and 'columns' parameters with TimeColumn
    rr.send_columns(
        "video",
        indexes=[rr.TimeColumn("video_time", duration=1e-9 * frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
    )
    print(f"Logged {len(frame_timestamps_ns)} VideoFrameReferences")

    print("\nDone! Rerun viewer should show the video.")
    print("Use the timeline scrubber to navigate frames.")


if __name__ == "__main__":
    main()
