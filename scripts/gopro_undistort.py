#!/usr/bin/env python3
"""
GoPro lens distortion correction for Hero 13 Black (Wide mode).

Uses FFmpeg lenscorrection filter with tunable k1/k2 coefficients.
Default values are estimates for GoPro wide mode - calibrate for best results.

Usage:
    # Single file
    python gopro_undistort.py --input video.mp4 --output video_undistorted.mp4

    # Batch process folder
    python gopro_undistort.py --input-dir /path/to/videos --output-dir /path/to/output

    # Custom coefficients
    python gopro_undistort.py --input video.mp4 --output out.mp4 --k1 -0.25 --k2 0.08

Calibration:
    To find optimal k1/k2 for your GoPro Hero 13 Black:
    1. Record a checkerboard or straight lines (building edges, tiles)
    2. Run with --preview to test different values
    3. Adjust k1 (barrel distortion) and k2 (fine tuning) until lines are straight

    Typical GoPro wide mode values:
    - k1: -0.20 to -0.30 (negative = correct barrel distortion)
    - k2: 0.05 to 0.10 (positive = fine tune edges)
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Default coefficients for GoPro Hero 13 Black Wide mode (estimates)
# Negative k1 corrects barrel distortion, k2 fine-tunes edges
DEFAULT_K1 = -0.22
DEFAULT_K2 = 0.06

def undistort_video(input_path, output_path, k1, k2, preview=False):
    """Apply lens correction using FFmpeg."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg lenscorrection filter
    # cx/cy = center of distortion (0.5 = image center)
    # k1 = barrel/pincushion distortion
    # k2 = edge correction
    filter_str = f"lenscorrection=cx=0.5:cy=0.5:k1={k1}:k2={k2}"

    if preview:
        # Show preview with ffplay
        cmd = [
            "ffplay", "-i", str(input_path),
            "-vf", filter_str,
            "-window_title", f"Preview: k1={k1}, k2={k2}"
        ]
    else:
        # Full encode with high quality
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(output_path)
        ]

    print(f"Processing: {input_path.name}")
    print(f"  k1={k1}, k2={k2}")

    try:
        result = subprocess.run(cmd, check=True)
        if not preview:
            print(f"  Saved: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e}")
        return False

def batch_process(input_dir, output_dir, k1, k2, extensions=('.mp4', '.MP4', '.mov', '.MOV')):
    """Process all video files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = []
    for ext in extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"**/*{ext}"))  # Recursive

    video_files = sorted(set(video_files))

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to process")
    print(f"Using k1={k1}, k2={k2}")
    print()

    success = 0
    for video_path in video_files:
        # Preserve directory structure
        rel_path = video_path.relative_to(input_dir)
        output_path = output_dir / rel_path.parent / f"{video_path.stem}_undistorted{video_path.suffix}"

        if undistort_video(video_path, output_path, k1, k2):
            success += 1

    print()
    print(f"Processed {success}/{len(video_files)} videos")

def main():
    parser = argparse.ArgumentParser(
        description="GoPro Hero 13 Black lens distortion correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output
    parser.add_argument("--input", "-i", help="Input video file")
    parser.add_argument("--output", "-o", help="Output video file")
    parser.add_argument("--input-dir", help="Input directory for batch processing")
    parser.add_argument("--output-dir", help="Output directory for batch processing")

    # Distortion coefficients
    parser.add_argument("--k1", type=float, default=DEFAULT_K1,
                        help=f"Barrel distortion coefficient (default: {DEFAULT_K1})")
    parser.add_argument("--k2", type=float, default=DEFAULT_K2,
                        help=f"Edge correction coefficient (default: {DEFAULT_K2})")

    # Options
    parser.add_argument("--preview", action="store_true",
                        help="Preview with ffplay instead of encoding")

    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            print("Error: --output-dir required with --input-dir")
            sys.exit(1)
        batch_process(args.input_dir, args.output_dir, args.k1, args.k2)
    elif args.input:
        if not args.output and not args.preview:
            print("Error: --output required (or use --preview)")
            sys.exit(1)
        output = args.output or "preview"
        undistort_video(args.input, output, args.k1, args.k2, preview=args.preview)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python gopro_undistort.py -i video.mp4 -o video_fixed.mp4")
        print("  python gopro_undistort.py -i video.mp4 --preview  # Test settings")
        print("  python gopro_undistort.py --input-dir ./raw --output-dir ./undistorted")

if __name__ == "__main__":
    main()
