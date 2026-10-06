#!/usr/bin/env python3
"""
Export HaMeR tracking data to JSON format.

Extracts all useful data from results.pkl into a well-organized JSON file:
- Per-frame 2D keypoints (21 joints per hand)
- 3D MANO parameters (pose, shape, orientation)
- Camera translation
- Bounding boxes with confidence
- Handedness labels
- Track IDs for temporal association

Usage:
    python export_tracking_data.py --input results.pkl --output tracking_data.json
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def numpy_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def extract_mano_params(mano_dict):
    """Extract MANO parameters from a single hand detection."""
    params = {}

    # Global orientation (rotation matrix 3x3)
    if 'global_orient' in mano_dict:
        params['global_orient'] = numpy_to_list(mano_dict['global_orient'])

    # Hand pose (15 joints x 3x3 rotation matrices)
    if 'hand_pose' in mano_dict:
        params['hand_pose'] = numpy_to_list(mano_dict['hand_pose'])

    # Shape parameters (10 betas)
    if 'betas' in mano_dict:
        params['betas'] = numpy_to_list(mano_dict['betas'])

    # Handedness (0=left, 1=right)
    if 'is_right' in mano_dict:
        val = mano_dict['is_right']
        params['is_right'] = bool(val) if isinstance(val, (bool, np.bool_)) else int(val) > 0

    return params


def export_tracking_data(input_pkl, output_json, output_summary=None):
    """Export HaMeR results to organized JSON format."""

    print(f"Loading {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    # Sort frames by frame index
    frame_keys = sorted([k for k in data.keys() if isinstance(data[k], dict)])

    print(f"Found {len(frame_keys)} frames")

    # Build organized output
    output = {
        'metadata': {
            'source': str(input_pkl),
            'total_frames': len(frame_keys),
            'format_version': '1.0',
        },
        'frames': []
    }

    # Statistics
    stats = {
        'left_hand_frames': 0,
        'right_hand_frames': 0,
        'both_hands_frames': 0,
        'no_hands_frames': 0,
        'total_detections': 0,
    }

    for frame_key in frame_keys:
        frame_data = data[frame_key]

        # Get frame index
        if 'frame_idx' in frame_data:
            frame_idx = int(frame_data['frame_idx'])
        else:
            # Extract from filename
            try:
                frame_idx = int(Path(frame_key).stem)
            except:
                frame_idx = frame_keys.index(frame_key)

        frame_output = {
            'frame_idx': frame_idx,
            'frame_file': Path(frame_key).name,
            'hands': []
        }

        # Extract hand data
        mano_list = frame_data.get('mano', [])
        cam_trans_list = frame_data.get('cam_trans', [])
        extra_data_list = frame_data.get('extra_data', [])
        bbox_list = frame_data.get('bboxes', [])
        bbox_conf_list = frame_data.get('bbox_conf', [])
        track_ids = frame_data.get('tracked_ids', [])

        has_left = False
        has_right = False

        for h_idx, mano_dict in enumerate(mano_list):
            hand = {}

            # MANO 3D parameters
            hand['mano_params'] = extract_mano_params(mano_dict)
            is_right = hand['mano_params'].get('is_right', False)

            if is_right:
                has_right = True
                hand['hand_label'] = 'right'
            else:
                has_left = True
                hand['hand_label'] = 'left'

            # Camera translation (for 3D positioning)
            if h_idx < len(cam_trans_list):
                hand['cam_translation'] = numpy_to_list(cam_trans_list[h_idx])

            # 2D keypoints (21 joints with x, y, confidence)
            if h_idx < len(extra_data_list):
                kps = extra_data_list[h_idx]
                if hasattr(kps, 'tolist'):
                    kps = kps.tolist()
                hand['keypoints_2d'] = kps
                hand['num_keypoints'] = len(kps) if kps else 0

            # Bounding box [x1, y1, x2, y2]
            if h_idx < len(bbox_list):
                hand['bbox'] = numpy_to_list(bbox_list[h_idx])

            # Detection confidence
            if h_idx < len(bbox_conf_list):
                conf = bbox_conf_list[h_idx]
                hand['confidence'] = float(conf) if isinstance(conf, (int, float, np.number)) else numpy_to_list(conf)

            # Track ID for temporal association
            if h_idx < len(track_ids):
                hand['track_id'] = int(track_ids[h_idx])

            frame_output['hands'].append(hand)
            stats['total_detections'] += 1

        # Update stats
        if has_left and has_right:
            stats['both_hands_frames'] += 1
        elif has_left:
            stats['left_hand_frames'] += 1
        elif has_right:
            stats['right_hand_frames'] += 1
        else:
            stats['no_hands_frames'] += 1

        output['frames'].append(frame_output)

    # Add statistics to metadata
    output['metadata']['statistics'] = stats

    # Save main JSON
    print(f"Saving {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)

    file_size = Path(output_json).stat().st_size / (1024 * 1024)
    print(f"Saved {file_size:.1f} MB")

    # Save summary (lighter weight)
    if output_summary:
        summary = {
            'metadata': output['metadata'],
            'frame_count': len(output['frames']),
            'sample_frame': output['frames'][len(output['frames'])//2] if output['frames'] else None
        }
        with open(output_summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {output_summary}")

    print(f"\nStatistics:")
    print(f"  Total frames: {stats['left_hand_frames'] + stats['right_hand_frames'] + stats['both_hands_frames'] + stats['no_hands_frames']}")
    print(f"  Both hands: {stats['both_hands_frames']}")
    print(f"  Left only: {stats['left_hand_frames']}")
    print(f"  Right only: {stats['right_hand_frames']}")
    print(f"  No hands: {stats['no_hands_frames']}")
    print(f"  Total detections: {stats['total_detections']}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Export HaMeR tracking data to JSON')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input results.pkl from HaMeR')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--summary', '-s', type=str,
                        help='Optional summary JSON (lighter weight)')
    args = parser.parse_args()

    export_tracking_data(args.input, args.output, args.summary)


if __name__ == '__main__':
    main()
