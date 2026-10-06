#!/usr/bin/env python3
"""
Create Interactive HTML Debug Viewer for Dyn-HaMR Output.

Generates an interactive HTML page showing:
- Frame scrubber
- Skeleton overlay with confidence-colored joints
- Bounding boxes with detection scores
- Per-joint confidence on hover
- Tracking ID and frame quality info

Usage:
    python 08_create_debug_viewer.py <hamer_pickle> <video> -o <output.html>
"""

import argparse
import json
import pickle
import base64
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import deque
import html


# ============== Hybrid Confidence Constants ==============
BBOX_WEIGHT = 0.40
VISIBILITY_WEIGHT = 0.30
TEMPORAL_WEIGHT = 0.30
TEMPORAL_WINDOW_SIZE = 7
JITTER_SCALE_FACTOR = 0.15
EDGE_MARGIN = 0.10


def compute_visibility_confidence(x: float, y: float, width: int, height: int) -> float:
    """Compute visibility confidence based on distance from frame edges."""
    if x < 0 or x > width or y < 0 or y > height:
        return 0.0

    margin_x = EDGE_MARGIN * width
    margin_y = EDGE_MARGIN * height

    dist_left = x
    dist_right = width - x
    dist_top = y
    dist_bottom = height - y

    min_dist_x = min(dist_left, dist_right)
    min_dist_y = min(dist_top, dist_bottom)

    conf_x = min(min_dist_x / margin_x, 1.0) if margin_x > 0 else 1.0
    conf_y = min(min_dist_y / margin_y, 1.0) if margin_y > 0 else 1.0

    return conf_x * conf_y


def compute_temporal_jitter_confidence(joint_history: deque, current_pos: tuple,
                                       width: int, height: int) -> float:
    """Compute temporal jitter confidence based on position stability."""
    if len(joint_history) < 2:
        return 1.0

    positions = list(joint_history)
    positions.append(current_pos)

    norm_positions = [(p[0] / width, p[1] / height) for p in positions]

    xs = [p[0] for p in norm_positions]
    ys = [p[1] for p in norm_positions]

    var_x = np.var(xs)
    var_y = np.var(ys)
    jitter = np.sqrt(var_x + var_y)

    return np.exp(-jitter / JITTER_SCALE_FACTOR)


def compute_hybrid_joint_confidence(bbox_conf: float, visibility_conf: float,
                                   temporal_conf: float) -> float:
    """Compute final hybrid confidence from component confidences."""
    return (BBOX_WEIGHT * bbox_conf +
            VISIBILITY_WEIGHT * visibility_conf +
            TEMPORAL_WEIGHT * temporal_conf)


class JointHistoryTracker:
    """Track joint positions over time for temporal confidence calculation."""

    def __init__(self, window_size: int = TEMPORAL_WINDOW_SIZE):
        self.window_size = window_size
        self.histories = {}  # (hand_id, joint_idx) -> deque of positions

    def update(self, hand_id: int, joint_idx: int, x: float, y: float) -> deque:
        key = (hand_id, joint_idx)
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.window_size)
        self.histories[key].append((x, y))
        return self.histories[key]

    def get_history(self, hand_id: int, joint_idx: int) -> deque:
        key = (hand_id, joint_idx)
        return self.histories.get(key, deque())


# MANO joint connections (21 joints)
MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

JOINT_NAMES = [
    "Wrist", "Thumb MCP", "Thumb PIP", "Thumb DIP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"
]

# Haptica website teal: #00d4aa
HAPTICA_TEAL = '#00d4aa'
FINGER_COLORS = {
    'thumb': HAPTICA_TEAL,
    'index': HAPTICA_TEAL,
    'middle': HAPTICA_TEAL,
    'ring': HAPTICA_TEAL,
    'pinky': HAPTICA_TEAL,
}


def get_finger_for_joint(joint_idx: int) -> str:
    """Get finger name for a joint index."""
    if joint_idx <= 4:
        return 'thumb'
    elif joint_idx <= 8:
        return 'index'
    elif joint_idx <= 12:
        return 'middle'
    elif joint_idx <= 16:
        return 'ring'
    else:
        return 'pinky'


def confidence_to_color(conf: float) -> str:
    """Convert confidence [0,1] to color (red=low, green=high)."""
    # Red to green gradient
    r = int(255 * (1 - conf))
    g = int(255 * conf)
    return f'rgb({r},{g},0)'


def load_hamer_pickle(pickle_path: Path) -> Dict:
    """Load HaMeR output pickle."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def extract_frame_data(hamer_data: Dict, width: int, height: int,
                       frame_dir: Optional[Path] = None) -> List[Dict]:
    """Extract per-frame debug data from HaMeR output with hybrid confidence."""
    frames = []
    joint_tracker = JointHistoryTracker()

    # Sort by frame number
    sorted_keys = sorted(hamer_data.keys(), key=lambda x: int(Path(x).stem))

    for img_path in sorted_keys:
        entry = hamer_data[img_path]
        frame_num = int(Path(img_path).stem)

        frame_info = {
            'frame': frame_num,
            'img_path': img_path,
            'hands': []
        }

        # Get number of hands detected
        n_hands = len(entry.get('extra_data', []))

        for hand_idx in range(n_hands):
            # Get bbox confidence (detection confidence from ViTDet)
            bbox_conf = 1.0
            if 'bbox_conf' in entry and hand_idx < len(entry['bbox_conf']):
                bbox_conf = float(entry['bbox_conf'][hand_idx])

            # Get hand ID for tracking
            hand_id = entry['tracked_ids'][hand_idx] if hand_idx < len(entry.get('tracked_ids', [])) else hand_idx

            hand_info = {
                'is_right': hand_id,
                'keypoints': [],
                'bbox': None,
                'bbox_conf': bbox_conf,
            }

            # Get keypoints with hybrid confidence
            if hand_idx < len(entry['extra_data']):
                kp_data = entry['extra_data'][hand_idx]
                for j, kp in enumerate(kp_data):
                    x, y = float(kp[0]), float(kp[1])

                    # Update joint history and get temporal confidence
                    history = joint_tracker.update(hand_id, j, x, y)
                    temporal_conf = compute_temporal_jitter_confidence(
                        history, (x, y), width, height
                    )

                    # Compute visibility confidence
                    visibility_conf = compute_visibility_confidence(x, y, width, height)

                    # Compute hybrid confidence
                    hybrid_conf = compute_hybrid_joint_confidence(
                        bbox_conf, visibility_conf, temporal_conf
                    )

                    hand_info['keypoints'].append({
                        'x': x,
                        'y': y,
                        'conf': hybrid_conf,
                        'name': JOINT_NAMES[j] if j < len(JOINT_NAMES) else f'Joint {j}',
                        'finger': get_finger_for_joint(j)
                    })

            # Get bbox if available
            if 'bboxes' in entry and hand_idx < len(entry['bboxes']):
                hand_info['bbox'] = entry['bboxes'][hand_idx]

            frame_info['hands'].append(hand_info)

        # Compute frame-level quality score (average hybrid confidence)
        if frame_info['hands']:
            all_confs = []
            for hand in frame_info['hands']:
                all_confs.extend([kp['conf'] for kp in hand['keypoints']])
            frame_info['quality_score'] = float(np.mean(all_confs)) if all_confs else 0.0
        else:
            frame_info['quality_score'] = 0.0

        frames.append(frame_info)

    return frames


def extract_video_frames_base64(video_path: Path, max_frames: int = None,
                                 sample_rate: int = 1) -> List[str]:
    """Extract video frames as base64-encoded JPEGs."""
    cap = cv2.VideoCapture(str(video_path))
    frames_b64 = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    frame_idx = 0
    pbar = tqdm(total=total, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            # Resize for web display
            scale = min(1.0, 1280 / frame.shape[1])
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(b64)

        frame_idx += 1
        pbar.update(1)

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    pbar.close()

    return frames_b64


def generate_html(frame_data: List[Dict], video_frames_b64: List[str],
                  output_path: Path, video_info: Dict):
    """Generate the interactive HTML viewer."""

    # Create minimal frame data for JS (just what we need for overlay)
    js_frame_data = json.dumps(frame_data)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dyn-HaMR Debug Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
        }}
        .video-container {{
            position: relative;
            display: inline-block;
            margin: 0 auto;
            display: block;
        }}
        #canvas {{
            border: 2px solid #333;
            border-radius: 8px;
            cursor: crosshair;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 15px 0;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
        }}
        .controls label {{
            color: #888;
            font-size: 14px;
        }}
        #scrubber {{
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: #333;
            border-radius: 4px;
            cursor: pointer;
        }}
        #scrubber::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        #frame-num {{
            min-width: 80px;
            text-align: right;
            font-family: monospace;
            font-size: 16px;
            color: #00d4ff;
        }}
        .info-panel {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .panel {{
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
        }}
        .panel h3 {{
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }}
        .stat-label {{
            color: #888;
        }}
        .stat-value {{
            font-family: monospace;
            color: #fff;
        }}
        .quality-bar {{
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .quality-fill {{
            height: 100%;
            transition: width 0.3s, background-color 0.3s;
        }}
        .joint-list {{
            max-height: 300px;
            overflow-y: auto;
            font-size: 12px;
        }}
        .joint-item {{
            display: flex;
            justify-content: space-between;
            padding: 4px 8px;
            border-radius: 4px;
            margin: 2px 0;
        }}
        .joint-item:hover {{
            background: #333;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        .toggle-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .toggle-btn {{
            padding: 6px 12px;
            border: 1px solid #333;
            background: transparent;
            color: #888;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .toggle-btn.active {{
            background: #00d4ff;
            color: #000;
            border-color: #00d4ff;
        }}
        /* Timeline styles */
        .timeline-panel {{
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }}
        .timeline-panel h3 {{
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .timeline-container {{
            position: relative;
            margin-bottom: 15px;
        }}
        .timeline-label {{
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .timeline-canvas {{
            width: 100%;
            height: 60px;
            background: #0d1321;
            border-radius: 4px;
            cursor: pointer;
        }}
        .timeline-playhead {{
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            background: #00d4ff;
            pointer-events: none;
            z-index: 10;
        }}
        .timeline-stats {{
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 11px;
            color: #666;
        }}
        .heatmap-canvas {{
            width: 100%;
            height: 80px;
            background: #0d1321;
            border-radius: 4px;
            cursor: pointer;
        }}
        .playback-controls {{
            display: flex;
            gap: 10px;
        }}
        .playback-btn {{
            padding: 8px 16px;
            border: none;
            background: #00d4ff;
            color: #000;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }}
        .playback-btn:hover {{
            background: #00b8e6;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 100;
            display: none;
            max-width: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dyn-HaMR Debug Viewer</h1>

        <div class="video-container">
            <canvas id="canvas"></canvas>
            <div id="tooltip"></div>
        </div>

        <div class="controls">
            <div class="playback-controls">
                <button class="playback-btn" id="prev-btn">◀</button>
                <button class="playback-btn" id="play-btn">▶</button>
                <button class="playback-btn" id="next-btn">▶▶</button>
            </div>
            <label>Frame:</label>
            <input type="range" id="scrubber" min="0" max="{len(frame_data)-1}" value="0">
            <span id="frame-num">0 / {len(frame_data)-1}</span>
        </div>

        <div class="controls">
            <label>Display:</label>
            <div class="toggle-group">
                <button class="toggle-btn active" data-toggle="skeleton">Skeleton</button>
                <button class="toggle-btn active" data-toggle="joints">Joints</button>
                <button class="toggle-btn active" data-toggle="bbox">Bounding Box</button>
                <button class="toggle-btn active" data-toggle="confidence">Confidence Colors</button>
                <button class="toggle-btn" data-toggle="labels">Joint Labels</button>
            </div>
        </div>

        <div class="timeline-panel">
            <h3>Timeline Analysis</h3>
            <div class="timeline-container">
                <div class="timeline-label">Quality Score Over Time</div>
                <canvas class="timeline-canvas" id="quality-timeline"></canvas>
                <div class="timeline-playhead" id="quality-playhead"></div>
                <div class="timeline-stats">
                    <span>Min: <span id="quality-min">-</span></span>
                    <span>Avg: <span id="quality-avg">-</span></span>
                    <span>Max: <span id="quality-max">-</span></span>
                </div>
            </div>
            <div class="timeline-container">
                <div class="timeline-label">Hands Detected Per Frame</div>
                <canvas class="timeline-canvas" id="hands-timeline"></canvas>
                <div class="timeline-playhead" id="hands-playhead"></div>
            </div>
            <div class="timeline-container">
                <div class="timeline-label">Joint Confidence Heatmap (21 joints × time)</div>
                <canvas class="heatmap-canvas" id="confidence-heatmap"></canvas>
                <div class="timeline-playhead" id="heatmap-playhead"></div>
            </div>
        </div>

        <div class="info-panel">
            <div class="panel">
                <h3>Frame Info</h3>
                <div class="stat">
                    <span class="stat-label">Frame Number</span>
                    <span class="stat-value" id="info-frame">-</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Hands Detected</span>
                    <span class="stat-value" id="info-hands">-</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Quality Score</span>
                    <span class="stat-value" id="info-quality">-</span>
                </div>
                <div class="quality-bar">
                    <div class="quality-fill" id="quality-fill"></div>
                </div>
            </div>

            <div class="panel">
                <h3>Hand Details</h3>
                <div id="hand-details">
                    <p style="color:#666;">Hover over joints for details</p>
                </div>
            </div>

            <div class="panel">
                <h3>Joint Confidence</h3>
                <div class="joint-list" id="joint-list">
                    <!-- Populated by JS -->
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FFFF00;"></div>
                        <span>Thumb</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#00FF00;"></div>
                        <span>Index</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FF00FF;"></div>
                        <span>Middle</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FFA500;"></div>
                        <span>Ring</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#0000FF;"></div>
                        <span>Pinky</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="panel" style="margin-top: 15px;">
            <h3>Video Info</h3>
            <div class="stat">
                <span class="stat-label">Resolution</span>
                <span class="stat-value">{video_info['width']}x{video_info['height']}</span>
            </div>
            <div class="stat">
                <span class="stat-label">FPS</span>
                <span class="stat-value">{video_info['fps']:.2f}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Total Frames</span>
                <span class="stat-value">{video_info['total_frames']}</span>
            </div>
        </div>
    </div>

    <script>
        // Frame data
        const frameData = {js_frame_data};

        // Video frames as base64
        const videoFrames = {json.dumps(video_frames_b64)};

        // Connections for skeleton drawing
        const connections = {json.dumps(MANO_CONNECTIONS)};

        // Finger colors
        const fingerColors = {json.dumps(FINGER_COLORS)};

        // Canvas setup
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');

        // Display toggles
        const displayOptions = {{
            skeleton: true,
            joints: true,
            bbox: true,
            confidence: true,
            labels: false
        }};

        // Current frame
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;

        // Image cache
        const imageCache = new Map();

        // Preload first few frames
        function preloadFrames(start, count) {{
            for (let i = start; i < Math.min(start + count, videoFrames.length); i++) {{
                if (!imageCache.has(i)) {{
                    const img = new Image();
                    img.src = 'data:image/jpeg;base64,' + videoFrames[i];
                    imageCache.set(i, img);
                }}
            }}
        }}
        preloadFrames(0, 10);

        // Get finger color for joint
        function getFingerColor(jointIdx) {{
            if (jointIdx <= 4) return fingerColors.thumb;
            if (jointIdx <= 8) return fingerColors.index;
            if (jointIdx <= 12) return fingerColors.middle;
            if (jointIdx <= 16) return fingerColors.ring;
            return fingerColors.pinky;
        }}

        // Confidence to color
        function confidenceToColor(conf) {{
            const r = Math.round(255 * (1 - conf));
            const g = Math.round(255 * conf);
            return `rgb(${{r}},${{g}},0)`;
        }}

        // Draw frame
        function drawFrame(frameIdx) {{
            const data = frameData[frameIdx];

            // Load and draw base image
            let img = imageCache.get(frameIdx);
            if (!img || !img.complete) {{
                img = new Image();
                img.src = 'data:image/jpeg;base64,' + videoFrames[frameIdx];
                imageCache.set(frameIdx, img);
            }}

            img.onload = () => render(img, data);
            if (img.complete) render(img, data);

            // Preload nearby frames
            preloadFrames(frameIdx, 5);
        }}

        function render(img, data) {{
            // Set canvas size
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw image
            ctx.drawImage(img, 0, 0);

            // Draw overlays for each hand
            for (const hand of data.hands) {{
                // Draw bounding box
                if (displayOptions.bbox && hand.bbox) {{
                    ctx.strokeStyle = hand.is_right ? '#00FF00' : '#FF0000';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    const [x1, y1, x2, y2] = hand.bbox;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    ctx.setLineDash([]);

                    // Label
                    ctx.fillStyle = hand.is_right ? '#00FF00' : '#FF0000';
                    ctx.font = '14px monospace';
                    ctx.fillText(
                        `${{hand.is_right ? 'R' : 'L'}} conf: ${{hand.bbox_conf.toFixed(2)}}`,
                        x1, y1 - 5
                    );
                }}

                // Draw skeleton
                if (displayOptions.skeleton && hand.keypoints.length >= 21) {{
                    ctx.lineWidth = 2;
                    for (const [a, b] of connections) {{
                        const kpA = hand.keypoints[a];
                        const kpB = hand.keypoints[b];

                        if (displayOptions.confidence) {{
                            const avgConf = (kpA.conf + kpB.conf) / 2;
                            ctx.strokeStyle = confidenceToColor(avgConf);
                        }} else {{
                            ctx.strokeStyle = getFingerColor(Math.max(a, b));
                        }}

                        ctx.beginPath();
                        ctx.moveTo(kpA.x, kpA.y);
                        ctx.lineTo(kpB.x, kpB.y);
                        ctx.stroke();
                    }}
                }}

                // Draw joints
                if (displayOptions.joints && hand.keypoints.length >= 21) {{
                    for (let i = 0; i < hand.keypoints.length; i++) {{
                        const kp = hand.keypoints[i];
                        const radius = i === 0 ? 6 : 4;

                        // Fill
                        if (displayOptions.confidence) {{
                            ctx.fillStyle = confidenceToColor(kp.conf);
                        }} else {{
                            ctx.fillStyle = getFingerColor(i);
                        }}

                        ctx.beginPath();
                        ctx.arc(kp.x, kp.y, radius, 0, Math.PI * 2);
                        ctx.fill();

                        // Outline
                        ctx.strokeStyle = '#000';
                        ctx.lineWidth = 1;
                        ctx.stroke();

                        // Label
                        if (displayOptions.labels) {{
                            ctx.fillStyle = '#fff';
                            ctx.font = '10px monospace';
                            ctx.fillText(kp.name, kp.x + 8, kp.y - 5);
                        }}
                    }}
                }}
            }}

            // Update info panel
            updateInfoPanel(data);
        }}

        function updateInfoPanel(data) {{
            document.getElementById('info-frame').textContent = data.frame;
            document.getElementById('info-hands').textContent = data.hands.length;
            document.getElementById('info-quality').textContent = (data.quality_score * 100).toFixed(1) + '%';

            // Quality bar
            const fill = document.getElementById('quality-fill');
            fill.style.width = (data.quality_score * 100) + '%';
            fill.style.backgroundColor = confidenceToColor(data.quality_score);

            // Joint list
            const jointList = document.getElementById('joint-list');
            jointList.innerHTML = '';

            for (let h = 0; h < data.hands.length; h++) {{
                const hand = data.hands[h];
                const header = document.createElement('div');
                header.style.fontWeight = 'bold';
                header.style.marginTop = '10px';
                header.style.color = hand.is_right ? '#00FF00' : '#FF0000';
                header.textContent = (hand.is_right ? 'Right' : 'Left') + ' Hand (conf: ' + hand.bbox_conf.toFixed(2) + ')';
                jointList.appendChild(header);

                for (const kp of hand.keypoints) {{
                    const item = document.createElement('div');
                    item.className = 'joint-item';
                    item.style.borderLeft = '3px solid ' + getFingerColor(hand.keypoints.indexOf(kp));
                    item.innerHTML = `
                        <span>${{kp.name}}</span>
                        <span style="color:${{confidenceToColor(kp.conf)}}">${{(kp.conf * 100).toFixed(0)}}%</span>
                    `;
                    jointList.appendChild(item);
                }}
            }}
        }}

        // ============================================
        // Timeline Drawing Functions
        // ============================================

        // Extract timeline data from frameData
        const qualityScores = frameData.map(f => f.quality_score);
        const handCounts = frameData.map(f => f.hands.length);

        // Compute stats
        const validScores = qualityScores.filter(s => s > 0);
        const qualityMin = validScores.length ? Math.min(...validScores) : 0;
        const qualityMax = validScores.length ? Math.max(...validScores) : 0;
        const qualityAvg = validScores.length ? validScores.reduce((a, b) => a + b, 0) / validScores.length : 0;

        // Update stats display
        document.getElementById('quality-min').textContent = (qualityMin * 100).toFixed(1) + '%';
        document.getElementById('quality-max').textContent = (qualityMax * 100).toFixed(1) + '%';
        document.getElementById('quality-avg').textContent = (qualityAvg * 100).toFixed(1) + '%';

        function drawTimeline(canvasId, data, options = {{}}) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();

            // Set actual canvas size (for crisp rendering)
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const width = rect.width;
            const height = rect.height;
            const padding = 5;
            const graphHeight = height - padding * 2;

            // Clear
            ctx.fillStyle = '#0d1321';
            ctx.fillRect(0, 0, width, height);

            // Find data range
            const minVal = options.minVal !== undefined ? options.minVal : Math.min(...data);
            const maxVal = options.maxVal !== undefined ? options.maxVal : Math.max(...data);
            const range = maxVal - minVal || 1;

            // Draw grid lines
            ctx.strokeStyle = '#1a2a3a';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {{
                const y = padding + (graphHeight / 4) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }}

            // Draw data
            const barWidth = width / data.length;

            if (options.drawMode === 'bars') {{
                // Bar chart mode (for hand counts)
                for (let i = 0; i < data.length; i++) {{
                    const val = data[i];
                    const normalizedVal = (val - minVal) / range;
                    const barHeight = normalizedVal * graphHeight;
                    const x = i * barWidth;
                    const y = height - padding - barHeight;

                    ctx.fillStyle = val === 0 ? '#333' :
                                    val === 1 ? '#4CAF50' :
                                    val === 2 ? '#2196F3' : '#9C27B0';
                    ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), barHeight);
                }}
            }} else {{
                // Line chart mode (for quality scores)
                ctx.beginPath();
                ctx.strokeStyle = '#00d4ff';
                ctx.lineWidth = 1.5;

                for (let i = 0; i < data.length; i++) {{
                    const val = data[i];
                    const normalizedVal = (val - minVal) / range;
                    const x = i * barWidth + barWidth / 2;
                    const y = height - padding - normalizedVal * graphHeight;

                    if (i === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }}
                ctx.stroke();

                // Fill area under curve
                ctx.lineTo((data.length - 1) * barWidth + barWidth / 2, height - padding);
                ctx.lineTo(barWidth / 2, height - padding);
                ctx.closePath();
                ctx.fillStyle = 'rgba(0, 212, 255, 0.1)';
                ctx.fill();

                // Draw colored segments based on quality
                for (let i = 0; i < data.length; i++) {{
                    const val = data[i];
                    const normalizedVal = (val - minVal) / range;
                    const x = i * barWidth;
                    const barHeight = normalizedVal * graphHeight;
                    const y = height - padding - barHeight;

                    // Color code: red=low, yellow=medium, green=high
                    const hue = val * 120; // 0=red, 60=yellow, 120=green
                    ctx.fillStyle = `hsla(${{hue}}, 70%, 50%, 0.3)`;
                    ctx.fillRect(x, y, Math.max(1, barWidth), barHeight);
                }}
            }}
        }}

        function drawConfidenceHeatmap() {{
            const canvas = document.getElementById('confidence-heatmap');
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();

            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const width = rect.width;
            const height = rect.height;

            // Clear
            ctx.fillStyle = '#0d1321';
            ctx.fillRect(0, 0, width, height);

            const numJoints = 21;
            const cellWidth = width / frameData.length;
            const cellHeight = height / numJoints;

            // Draw heatmap
            for (let frameIdx = 0; frameIdx < frameData.length; frameIdx++) {{
                const frame = frameData[frameIdx];

                // Get confidence for each joint (average across hands if multiple)
                for (let jointIdx = 0; jointIdx < numJoints; jointIdx++) {{
                    let conf = 0;
                    let count = 0;

                    for (const hand of frame.hands) {{
                        if (hand.keypoints[jointIdx]) {{
                            conf += hand.keypoints[jointIdx].conf;
                            count++;
                        }}
                    }}

                    if (count > 0) {{
                        conf /= count;
                        // Color: red (0) to green (1)
                        const r = Math.round(255 * (1 - conf));
                        const g = Math.round(255 * conf);
                        ctx.fillStyle = `rgb(${{r}},${{g}},0)`;
                    }} else {{
                        ctx.fillStyle = '#1a1a2e';
                    }}

                    const x = frameIdx * cellWidth;
                    const y = jointIdx * cellHeight;
                    ctx.fillRect(x, y, Math.max(1, cellWidth), cellHeight);
                }}
            }}

            // Draw joint labels on left
            ctx.fillStyle = '#666';
            ctx.font = '8px monospace';
            const jointLabels = ['W', 'T1', 'T2', 'T3', 'T4', 'I1', 'I2', 'I3', 'I4',
                                 'M1', 'M2', 'M3', 'M4', 'R1', 'R2', 'R3', 'R4',
                                 'P1', 'P2', 'P3', 'P4'];
        }}

        function updatePlayheads(frameIdx) {{
            const totalFrames = frameData.length;
            const percent = (frameIdx / (totalFrames - 1)) * 100;

            ['quality-playhead', 'hands-playhead', 'heatmap-playhead'].forEach(id => {{
                const playhead = document.getElementById(id);
                playhead.style.left = percent + '%';
            }});
        }}

        // Draw initial timelines
        function initTimelines() {{
            drawTimeline('quality-timeline', qualityScores, {{ minVal: 0, maxVal: 1 }});
            drawTimeline('hands-timeline', handCounts, {{ minVal: 0, maxVal: Math.max(2, ...handCounts), drawMode: 'bars' }});
            drawConfidenceHeatmap();
            updatePlayheads(0);
        }}

        // Click on timeline to seek
        function setupTimelineClick(canvasId) {{
            const canvas = document.getElementById(canvasId);
            canvas.addEventListener('click', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const frameIdx = Math.floor((x / rect.width) * frameData.length);
                currentFrame = Math.max(0, Math.min(frameData.length - 1, frameIdx));
                scrubber.value = currentFrame;
                frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
                drawFrame(currentFrame);
                updatePlayheads(currentFrame);
            }});
        }}

        // Controls
        const scrubber = document.getElementById('scrubber');
        const frameNum = document.getElementById('frame-num');

        scrubber.addEventListener('input', (e) => {{
            currentFrame = parseInt(e.target.value);
            frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
            drawFrame(currentFrame);
            updatePlayheads(currentFrame);
        }});

        document.getElementById('prev-btn').addEventListener('click', () => {{
            currentFrame = Math.max(0, currentFrame - 1);
            scrubber.value = currentFrame;
            frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
            drawFrame(currentFrame);
            updatePlayheads(currentFrame);
        }});

        document.getElementById('next-btn').addEventListener('click', () => {{
            currentFrame = Math.min(frameData.length - 1, currentFrame + 1);
            scrubber.value = currentFrame;
            frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
            drawFrame(currentFrame);
            updatePlayheads(currentFrame);
        }});

        document.getElementById('play-btn').addEventListener('click', (e) => {{
            isPlaying = !isPlaying;
            e.target.textContent = isPlaying ? '⏸' : '▶';

            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    currentFrame = (currentFrame + 1) % frameData.length;
                    scrubber.value = currentFrame;
                    frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
                    drawFrame(currentFrame);
                    updatePlayheads(currentFrame);
                }}, 1000 / {video_info['fps']});
            }} else {{
                clearInterval(playInterval);
            }}
        }});

        // Toggle buttons
        document.querySelectorAll('.toggle-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                const toggle = btn.dataset.toggle;
                displayOptions[toggle] = !displayOptions[toggle];
                btn.classList.toggle('active');
                drawFrame(currentFrame);
            }});
        }});

        // Tooltip on hover
        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const data = frameData[currentFrame];
            let found = false;

            for (const hand of data.hands) {{
                for (const kp of hand.keypoints) {{
                    const dist = Math.sqrt((kp.x - x) ** 2 + (kp.y - y) ** 2);
                    if (dist < 10) {{
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
                        tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
                        tooltip.innerHTML = `
                            <strong>${{kp.name}}</strong><br>
                            Position: (${{kp.x.toFixed(0)}}, ${{kp.y.toFixed(0)}})<br>
                            Confidence: <span style="color:${{confidenceToColor(kp.conf)}}">${{(kp.conf * 100).toFixed(1)}}%</span>
                        `;
                        found = true;
                        break;
                    }}
                }}
                if (found) break;
            }}

            if (!found) {{
                tooltip.style.display = 'none';
            }}
        }});

        canvas.addEventListener('mouseleave', () => {{
            tooltip.style.display = 'none';
        }});

        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') {{
                currentFrame = Math.max(0, currentFrame - 1);
                scrubber.value = currentFrame;
                frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
                drawFrame(currentFrame);
                updatePlayheads(currentFrame);
            }} else if (e.key === 'ArrowRight') {{
                currentFrame = Math.min(frameData.length - 1, currentFrame + 1);
                scrubber.value = currentFrame;
                frameNum.textContent = currentFrame + ' / ' + (frameData.length - 1);
                drawFrame(currentFrame);
                updatePlayheads(currentFrame);
            }} else if (e.key === ' ') {{
                e.preventDefault();
                document.getElementById('play-btn').click();
            }}
        }});

        // Initial draw
        drawFrame(0);

        // Initialize timelines after a short delay to ensure DOM is ready
        setTimeout(() => {{
            initTimelines();
            setupTimelineClick('quality-timeline');
            setupTimelineClick('hands-timeline');
            setupTimelineClick('confidence-heatmap');
        }}, 100);

        // Redraw timelines on window resize
        window.addEventListener('resize', () => {{
            initTimelines();
        }});
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML viewer saved to: {output_path}")
    print(f"  - {len(frame_data)} frames")
    print(f"  - {sum(len(f['hands']) for f in frame_data)} total hand detections")


def main():
    parser = argparse.ArgumentParser(description="Create interactive debug viewer for HaMeR output")
    parser.add_argument("hamer_pickle", help="Path to HaMeR output pickle file")
    parser.add_argument("video", help="Path to original video file")
    parser.add_argument("-o", "--output", default="debug_viewer.html", help="Output HTML file")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (for testing)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sample every Nth frame")

    args = parser.parse_args()

    print("=" * 60)
    print("Dyn-HaMR Debug Viewer Generator")
    print("=" * 60)

    # Load HaMeR data
    print(f"\nLoading HaMeR pickle: {args.hamer_pickle}")
    hamer_data = load_hamer_pickle(Path(args.hamer_pickle))
    print(f"  Found {len(hamer_data)} frame entries")

    # Get video info first (needed for hybrid confidence calculation)
    cap = cv2.VideoCapture(args.video)
    video_info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()

    print(f"\nVideo: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps")

    # Extract frame data with hybrid confidence
    print("\nExtracting frame debug data with hybrid confidence...")
    frame_data = extract_frame_data(hamer_data, video_info['width'], video_info['height'])

    # Limit frames if requested
    if args.max_frames:
        frame_data = frame_data[:args.max_frames]

    # Extract video frames
    print("\nExtracting video frames...")
    video_frames_b64 = extract_video_frames_base64(
        Path(args.video),
        max_frames=args.max_frames,
        sample_rate=args.sample_rate
    )

    # Sync frame counts
    min_frames = min(len(frame_data), len(video_frames_b64))
    frame_data = frame_data[:min_frames]
    video_frames_b64 = video_frames_b64[:min_frames]

    # Generate HTML
    print(f"\nGenerating HTML viewer...")
    generate_html(frame_data, video_frames_b64, Path(args.output), video_info)

    print("\n" + "=" * 60)
    print("Done! Open the HTML file in a browser to view.")
    print("=" * 60)


if __name__ == "__main__":
    main()
