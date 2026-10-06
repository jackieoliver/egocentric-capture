#!/usr/bin/env python3
"""
Export HaMeR pickle data to JSON format for Haptica website.

Converts HaMeR output pickle to the JSON schema expected by SkeletonVideoHUD component.
Implements hybrid per-joint confidence using:
  1. Bbox detection confidence (40% weight) - from HaMeR detector
  2. Visibility confidence (30% weight) - based on joint position in frame
  3. Temporal jitter confidence (30% weight) - based on position stability

Output format:
{
  "fps": 30,
  "width": 1280,
  "height": 720,
  "total_frames": 1200,
  "duration_ms": 40000,
  "frames": [
    {
      "frame": 0,
      "timestamp_ms": 0,
      "joints": 21,
      "hands": 2,
      "confidence": 0.85,
      "joint_confidences": [0.9, 0.85, 0.7, ...],
      "left_wrist": {"x": 0.5, "y": 0.6, "confidence": 0.98},
      "right_wrist": {"x": 0.3, "y": 0.4, "confidence": 0.97},
      "fingertips": {...}
    },
    ...
  ]
}

Usage:
    python 09_export_stats_json.py <hamer_pickle> <video> -o <output.json>
"""

import argparse
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque


# Joint indices for specific keypoints
WRIST_IDX = 0
FINGERTIP_INDICES = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20,
}

JOINT_NAMES = [
    "wrist", "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# Hybrid confidence weights
BBOX_WEIGHT = 0.40
VISIBILITY_WEIGHT = 0.30
TEMPORAL_WEIGHT = 0.30

# Temporal jitter parameters
TEMPORAL_WINDOW_SIZE = 7  # frames to consider
JITTER_SCALE_FACTOR = 0.15  # scale factor for exponential decay (higher = more sensitive)

# Visibility edge margin (fraction of frame dimension)
EDGE_MARGIN = 0.10


def load_hamer_pickle(pickle_path: Path) -> Dict:
    """Load HaMeR output pickle."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def get_video_info(video_path: Path, hamer_frame_count: int = None) -> Dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle invalid frame count (common with webm files)
    if total_frames <= 0 and hamer_frame_count:
        total_frames = hamer_frame_count
        print(f"  (Using HaMeR frame count: {total_frames})")
    elif total_frames <= 0:
        # Count frames manually
        print("  (Counting frames manually...)")
        total_frames = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            total_frames += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Default FPS if invalid
    if fps <= 0:
        fps = 30.0
        print(f"  (Assuming FPS: {fps})")

    info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
    }
    info['duration_ms'] = int((info['total_frames'] / info['fps']) * 1000)
    cap.release()
    return info


def compute_visibility_confidence(x: float, y: float, width: int, height: int) -> float:
    """
    Compute visibility confidence based on joint position.

    Returns:
        1.0 if joint is comfortably within frame
        0.0-1.0 for joints near edges (linear decay)
        0.0 if joint is outside frame
    """
    # Normalize coordinates
    norm_x = x / width
    norm_y = y / height

    # Check if outside frame
    if norm_x < 0 or norm_x > 1 or norm_y < 0 or norm_y > 1:
        return 0.0

    # Calculate distance from edges
    dist_from_left = norm_x
    dist_from_right = 1.0 - norm_x
    dist_from_top = norm_y
    dist_from_bottom = 1.0 - norm_y

    min_edge_dist = min(dist_from_left, dist_from_right, dist_from_top, dist_from_bottom)

    # If within margin, apply linear penalty
    if min_edge_dist < EDGE_MARGIN:
        return min_edge_dist / EDGE_MARGIN

    return 1.0


def compute_temporal_jitter_confidence(
    joint_history: List[Tuple[float, float]],
    current_pos: Tuple[float, float],
    width: int,
    height: int
) -> float:
    """
    Compute temporal jitter confidence based on position stability.

    Uses variance of joint positions over recent frames.
    High variance = low confidence (unstable tracking)

    Args:
        joint_history: List of (x, y) positions from previous frames
        current_pos: Current frame (x, y) position
        width: Frame width for normalization
        height: Frame height for normalization

    Returns:
        Confidence value [0, 1] where 1 = stable, 0 = very jittery
    """
    if len(joint_history) < 2:
        # Not enough history, assume stable
        return 1.0

    # Collect all positions including current
    all_positions = list(joint_history) + [current_pos]

    # Normalize positions to [0, 1] range
    norm_positions = [(x / width, y / height) for x, y in all_positions]

    # Compute standard deviation
    xs = [p[0] for p in norm_positions]
    ys = [p[1] for p in norm_positions]

    std_x = np.std(xs)
    std_y = np.std(ys)

    # Combined jitter (Euclidean norm of stds)
    jitter = np.sqrt(std_x**2 + std_y**2)

    # Convert to confidence using exponential decay
    # jitter of 0 -> confidence 1.0
    # jitter of ~0.1 (10% of frame) -> confidence ~0.22
    confidence = np.exp(-jitter / JITTER_SCALE_FACTOR)

    return float(confidence)


class JointHistoryTracker:
    """Track joint positions over time for temporal confidence calculation."""

    def __init__(self, window_size: int = TEMPORAL_WINDOW_SIZE):
        self.window_size = window_size
        # Dict: hand_id -> Dict[joint_idx -> deque of (x, y) positions]
        self.history: Dict[int, Dict[int, deque]] = {}

    def update_and_get_confidence(
        self,
        hand_id: int,
        joint_idx: int,
        x: float,
        y: float,
        width: int,
        height: int
    ) -> float:
        """
        Update history and compute temporal confidence for a joint.

        Args:
            hand_id: Unique identifier for the hand (tracked_id)
            joint_idx: Index of the joint (0-20)
            x, y: Current joint position in pixels
            width, height: Frame dimensions

        Returns:
            Temporal jitter confidence [0, 1]
        """
        # Initialize hand history if needed
        if hand_id not in self.history:
            self.history[hand_id] = {}

        # Initialize joint history if needed
        if joint_idx not in self.history[hand_id]:
            self.history[hand_id][joint_idx] = deque(maxlen=self.window_size)

        joint_history = list(self.history[hand_id][joint_idx])

        # Compute confidence
        confidence = compute_temporal_jitter_confidence(
            joint_history, (x, y), width, height
        )

        # Update history
        self.history[hand_id][joint_idx].append((x, y))

        return confidence

    def clear_hand(self, hand_id: int):
        """Clear history for a hand that's no longer being tracked."""
        if hand_id in self.history:
            del self.history[hand_id]


def compute_hybrid_joint_confidence(
    bbox_conf: float,
    visibility_conf: float,
    temporal_conf: float
) -> float:
    """
    Compute hybrid confidence from three components.

    Args:
        bbox_conf: Detection confidence [0, 1]
        visibility_conf: Visibility confidence [0, 1]
        temporal_conf: Temporal jitter confidence [0, 1]

    Returns:
        Weighted hybrid confidence [0, 1]
    """
    hybrid = (
        BBOX_WEIGHT * bbox_conf +
        VISIBILITY_WEIGHT * visibility_conf +
        TEMPORAL_WEIGHT * temporal_conf
    )
    return float(np.clip(hybrid, 0.0, 1.0))


def extract_frame_stats(hamer_data: Dict, video_info: Dict) -> List[Dict]:
    """Extract per-frame statistics from HaMeR output with hybrid confidence."""
    frames = []
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']

    # Initialize temporal tracker
    tracker = JointHistoryTracker(window_size=TEMPORAL_WINDOW_SIZE)

    # Sort by frame number
    sorted_keys = sorted(hamer_data.keys(), key=lambda x: int(Path(x).stem))

    # Create a mapping of frame_num -> data
    frame_map = {}
    for img_path in sorted_keys:
        frame_num = int(Path(img_path).stem)
        frame_map[frame_num] = hamer_data[img_path]

    # Track which hands were seen in previous frame (for cleanup)
    prev_hand_ids = set()

    # Generate stats for all frames in video
    for frame_idx in range(video_info['total_frames']):
        timestamp_ms = int((frame_idx / fps) * 1000)

        frame_stats = {
            'frame': frame_idx,
            'timestamp_ms': timestamp_ms,
            'joints': 0,
            'hands': 0,
            'confidence': 0.0,
            'joint_confidences': [],
            'left_wrist': None,
            'right_wrist': None,
            'fingertips': None,
        }

        current_hand_ids = set()

        # Check if we have HaMeR data for this frame
        if frame_idx in frame_map:
            entry = frame_map[frame_idx]
            n_hands = len(entry.get('extra_data', []))
            frame_stats['hands'] = n_hands

            if n_hands > 0:
                frame_stats['joints'] = 21  # MANO has 21 joints

                all_joint_confidences = []
                fingertips = {}

                for hand_idx in range(n_hands):
                    # Get keypoints with confidence
                    if hand_idx < len(entry.get('extra_data', [])):
                        kp_data = entry['extra_data'][hand_idx]

                        # Get tracked hand ID
                        tracked_id = entry.get('tracked_ids', [0])[hand_idx] if hand_idx < len(entry.get('tracked_ids', [])) else hand_idx
                        current_hand_ids.add(tracked_id)

                        # Determine if right or left hand
                        is_right = tracked_id

                        # Get bbox confidence (detection confidence)
                        bbox_conf = 1.0
                        if 'bbox_conf' in entry and hand_idx < len(entry['bbox_conf']):
                            bbox_conf = float(entry['bbox_conf'][hand_idx])

                        hand_joint_confidences = []

                        # Process each joint
                        for j, kp in enumerate(kp_data):
                            x, y = float(kp[0]), float(kp[1])

                            # Compute visibility confidence
                            visibility_conf = compute_visibility_confidence(x, y, width, height)

                            # Compute temporal jitter confidence
                            temporal_conf = tracker.update_and_get_confidence(
                                tracked_id, j, x, y, width, height
                            )

                            # Compute hybrid confidence
                            hybrid_conf = compute_hybrid_joint_confidence(
                                bbox_conf, visibility_conf, temporal_conf
                            )

                            hand_joint_confidences.append(hybrid_conf)

                        all_joint_confidences.extend(hand_joint_confidences)

                        # Extract wrist with hybrid confidence
                        if len(kp_data) > WRIST_IDX:
                            wrist_kp = kp_data[WRIST_IDX]
                            wrist_conf = hand_joint_confidences[WRIST_IDX] if WRIST_IDX < len(hand_joint_confidences) else bbox_conf
                            wrist_data = {
                                'x': float(wrist_kp[0]) / width,  # Normalize to 0-1
                                'y': float(wrist_kp[1]) / height,
                                'confidence': wrist_conf
                            }
                            if is_right:
                                frame_stats['right_wrist'] = wrist_data
                            else:
                                frame_stats['left_wrist'] = wrist_data

                        # Extract fingertips with hybrid confidence
                        for finger_name, finger_idx in FINGERTIP_INDICES.items():
                            if len(kp_data) > finger_idx:
                                tip_kp = kp_data[finger_idx]
                                tip_conf = hand_joint_confidences[finger_idx] if finger_idx < len(hand_joint_confidences) else bbox_conf
                                key = f"{'right' if is_right else 'left'}_{finger_name}"
                                fingertips[key] = {
                                    'x': float(tip_kp[0]) / width,
                                    'y': float(tip_kp[1]) / height,
                                    'confidence': tip_conf
                                }

                # Average confidence across all joints
                if all_joint_confidences:
                    frame_stats['confidence'] = float(np.mean(all_joint_confidences))
                    frame_stats['joint_confidences'] = all_joint_confidences

                if fingertips:
                    frame_stats['fingertips'] = fingertips

        # Clean up history for hands that disappeared
        lost_hands = prev_hand_ids - current_hand_ids
        for hand_id in lost_hands:
            tracker.clear_hand(hand_id)

        prev_hand_ids = current_hand_ids
        frames.append(frame_stats)

    return frames


def compute_confidence_stats(frames: List[Dict]) -> Dict:
    """Compute summary statistics for confidence values."""
    all_confidences = []
    all_joint_confidences = []

    for frame in frames:
        if frame['confidence'] > 0:
            all_confidences.append(frame['confidence'])
        if frame['joint_confidences']:
            all_joint_confidences.extend(frame['joint_confidences'])

    stats = {
        'frame_confidence': {
            'count': len(all_confidences),
            'mean': float(np.mean(all_confidences)) if all_confidences else 0.0,
            'std': float(np.std(all_confidences)) if all_confidences else 0.0,
            'min': float(np.min(all_confidences)) if all_confidences else 0.0,
            'max': float(np.max(all_confidences)) if all_confidences else 0.0,
        },
        'joint_confidence': {
            'count': len(all_joint_confidences),
            'mean': float(np.mean(all_joint_confidences)) if all_joint_confidences else 0.0,
            'std': float(np.std(all_joint_confidences)) if all_joint_confidences else 0.0,
            'min': float(np.min(all_joint_confidences)) if all_joint_confidences else 0.0,
            'max': float(np.max(all_joint_confidences)) if all_joint_confidences else 0.0,
        }
    }

    # Compute percentiles
    if all_joint_confidences:
        percentiles = [10, 25, 50, 75, 90]
        stats['joint_confidence']['percentiles'] = {
            f'p{p}': float(np.percentile(all_joint_confidences, p))
            for p in percentiles
        }

    return stats


def export_to_json(hamer_pickle: Path, video_path: Path, output_path: Path):
    """Export HaMeR data to JSON format with hybrid confidence."""

    print("=" * 60)
    print("HaMeR to JSON Exporter (Haptica Website Format)")
    print("  with Hybrid Per-Joint Confidence")
    print("=" * 60)

    # Load data
    print(f"\nLoading HaMeR pickle: {hamer_pickle}")
    hamer_data = load_hamer_pickle(hamer_pickle)
    print(f"  Found {len(hamer_data)} frame entries")

    # Get video info (pass hamer frame count as fallback)
    hamer_frame_count = max(int(Path(k).stem) for k in hamer_data.keys()) + 1
    print(f"\nReading video info: {video_path}")
    video_info = get_video_info(video_path, hamer_frame_count)
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Total frames: {video_info['total_frames']}")
    print(f"  Duration: {video_info['duration_ms']}ms")

    # Show confidence configuration
    print(f"\nHybrid Confidence Configuration:")
    print(f"  Bbox detection weight: {BBOX_WEIGHT*100:.0f}%")
    print(f"  Visibility weight: {VISIBILITY_WEIGHT*100:.0f}%")
    print(f"  Temporal jitter weight: {TEMPORAL_WEIGHT*100:.0f}%")
    print(f"  Temporal window: {TEMPORAL_WINDOW_SIZE} frames")
    print(f"  Edge margin: {EDGE_MARGIN*100:.0f}% of frame")

    # Extract frame stats
    print("\nExtracting frame statistics with hybrid confidence...")
    frames = extract_frame_stats(hamer_data, video_info)

    # Compute confidence statistics
    conf_stats = compute_confidence_stats(frames)

    # Build output structure
    output = {
        'fps': video_info['fps'],
        'width': video_info['width'],
        'height': video_info['height'],
        'total_frames': video_info['total_frames'],
        'duration_ms': video_info['duration_ms'],
        'confidence_config': {
            'bbox_weight': BBOX_WEIGHT,
            'visibility_weight': VISIBILITY_WEIGHT,
            'temporal_weight': TEMPORAL_WEIGHT,
            'temporal_window_frames': TEMPORAL_WINDOW_SIZE,
            'edge_margin': EDGE_MARGIN,
            'jitter_scale_factor': JITTER_SCALE_FACTOR,
        },
        'confidence_stats': conf_stats,
        'frames': frames,
    }

    # Compute summary stats
    hands_detected = sum(1 for f in frames if f['hands'] > 0)

    print(f"\nSummary:")
    print(f"  Frames with hands: {hands_detected}/{len(frames)} ({100*hands_detected/len(frames):.1f}%)")

    print(f"\nConfidence Statistics:")
    fc = conf_stats['frame_confidence']
    jc = conf_stats['joint_confidence']
    print(f"  Frame-level confidence:")
    print(f"    Mean: {fc['mean']*100:.1f}%, Std: {fc['std']*100:.1f}%")
    print(f"    Min: {fc['min']*100:.1f}%, Max: {fc['max']*100:.1f}%")
    print(f"  Joint-level confidence:")
    print(f"    Mean: {jc['mean']*100:.1f}%, Std: {jc['std']*100:.1f}%")
    print(f"    Min: {jc['min']*100:.1f}%, Max: {jc['max']*100:.1f}%")
    if 'percentiles' in jc:
        print(f"    Percentiles: p10={jc['percentiles']['p10']*100:.1f}%, p50={jc['percentiles']['p50']*100:.1f}%, p90={jc['percentiles']['p90']*100:.1f}%")

    # Write JSON
    print(f"\nWriting JSON: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    file_size = output_path.stat().st_size / 1024
    print(f"  Size: {file_size:.1f} KB")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main():
    global BBOX_WEIGHT, VISIBILITY_WEIGHT, TEMPORAL_WEIGHT, TEMPORAL_WINDOW_SIZE, EDGE_MARGIN, JITTER_SCALE_FACTOR

    parser = argparse.ArgumentParser(description="Export HaMeR data to JSON for Haptica website")
    parser.add_argument("hamer_pickle", type=Path, help="Path to HaMeR output pickle")
    parser.add_argument("video", type=Path, help="Path to source video")
    parser.add_argument("-o", "--output", type=Path, default=Path("stats.json"),
                        help="Output JSON path (default: stats.json)")

    # Optional confidence tuning parameters
    parser.add_argument("--bbox-weight", type=float, default=BBOX_WEIGHT,
                        help=f"Bbox detection confidence weight (default: {BBOX_WEIGHT})")
    parser.add_argument("--visibility-weight", type=float, default=VISIBILITY_WEIGHT,
                        help=f"Visibility confidence weight (default: {VISIBILITY_WEIGHT})")
    parser.add_argument("--temporal-weight", type=float, default=TEMPORAL_WEIGHT,
                        help=f"Temporal jitter confidence weight (default: {TEMPORAL_WEIGHT})")
    parser.add_argument("--temporal-window", type=int, default=TEMPORAL_WINDOW_SIZE,
                        help=f"Temporal window size in frames (default: {TEMPORAL_WINDOW_SIZE})")
    parser.add_argument("--edge-margin", type=float, default=EDGE_MARGIN,
                        help=f"Edge margin as fraction of frame (default: {EDGE_MARGIN})")
    parser.add_argument("--jitter-scale", type=float, default=JITTER_SCALE_FACTOR,
                        help=f"Jitter scale factor for exponential decay (default: {JITTER_SCALE_FACTOR})")

    args = parser.parse_args()

    # Normalize weights to sum to 1
    total_weight = args.bbox_weight + args.visibility_weight + args.temporal_weight
    BBOX_WEIGHT = args.bbox_weight / total_weight
    VISIBILITY_WEIGHT = args.visibility_weight / total_weight
    TEMPORAL_WEIGHT = args.temporal_weight / total_weight

    TEMPORAL_WINDOW_SIZE = args.temporal_window
    EDGE_MARGIN = args.edge_margin
    JITTER_SCALE_FACTOR = args.jitter_scale

    export_to_json(args.hamer_pickle, args.video, args.output)


if __name__ == "__main__":
    main()
