#!/usr/bin/env python3
"""
Haptica Hand Tracking Pipeline CLI

Unified tool for hand tracking operations:
  - skeleton: Render skeleton overlay on video
  - mesh: Render MANO mesh overlay on video
  - export: Export joint positions to JSON
  - all: Run full pipeline (export + skeleton + mesh)

Usage:
  hand-pipeline skeleton --input video.mp4 --params smooth_fit.pth --output skeleton.mp4
  hand-pipeline export --params smooth_fit.pth --output joints.jsonl
  hand-pipeline all --input video.mp4 --params smooth_fit.pth --output-dir ./output/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add Dyn-HaMR paths
DYNHAMR_ROOT = Path("/home/example/dyn_hamr_workspace/Dyn-HaMR")
sys.path.insert(0, str(DYNHAMR_ROOT / "dyn-hamr"))
sys.path.insert(0, str(DYNHAMR_ROOT / "third-party" / "hamer"))

from body_model import MANO

# ============================================================================
# Constants
# ============================================================================

HAPTICA_TEAL = (170, 212, 0)  # BGR for OpenCV

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# ============================================================================
# Core Pipeline Class
# ============================================================================

class HandPipeline:
    """Unified hand tracking pipeline."""

    def __init__(self, params_path: str, cameras_path: str = None, device: str = 'cuda'):
        self.device = device
        self.params_path = Path(params_path)
        self.output_dir = self.params_path.parent

        # Load MANO model
        print("Loading MANO model...")
        self.mano = self._load_mano()

        # Load parameters
        print(f"Loading parameters from {params_path}...")
        self.params = torch.load(params_path, map_location='cpu')
        self.pose = self.params['latent_pose']
        self.root_orient = self.params['root_orient']
        self.trans = self.params['trans']
        self.betas = self.params['betas']
        self.is_right = self.params['is_right']
        self.num_hands, self.T = self.pose.shape[:2]
        print(f"  {self.num_hands} hands, {self.T} frames")

        # Load camera intrinsics
        if cameras_path:
            self.cameras_path = Path(cameras_path)
        else:
            self.cameras_path = self.output_dir / "cameras.json"

        if self.cameras_path.exists():
            with open(self.cameras_path) as f:
                cam_data = json.load(f)
            self.intrinsics = cam_data['intrinsics'][0] if isinstance(cam_data['intrinsics'][0], list) else cam_data['intrinsics']
        else:
            print("WARNING: No cameras.json found, using default intrinsics")
            self.intrinsics = [1500.0, 1500.0, 960.0, 540.0]
        print(f"  Intrinsics: {self.intrinsics}")

        # Load visibility masks
        self.vis_masks, self.start_frame = self._load_track_info()

    def _load_mano(self):
        mano_cfg = {
            'model_path': str(DYNHAMR_ROOT / '_DATA' / 'data' / 'mano'),
            'use_pca': False,
            'flat_hand_mean': False,
        }
        return MANO(batch_size=1, pose2rot=True, **mano_cfg).to(self.device)

    def _load_track_info(self):
        """Load visibility masks from track_info.json."""
        track_info_path = self.output_dir / "track_info.json"

        if not track_info_path.exists():
            print("WARNING: No track_info.json, defaulting to all visible")
            return np.ones((self.num_hands, self.T), dtype=bool), 0

        with open(track_info_path) as f:
            track_info = json.load(f)

        meta = track_info.get('meta', {})
        seq_interval = meta.get('seq_interval', [0, 0])
        start_frame = seq_interval[0] if seq_interval else 0

        vis_masks = [[True] * self.T for _ in range(self.num_hands)]

        tracks = track_info.get('tracks', {})
        for track_key, track_data in tracks.items():
            tensor_idx = track_data.get('index', None)
            if tensor_idx is None:
                try:
                    tensor_idx = int(track_key)
                except ValueError:
                    continue

            if 0 <= tensor_idx < self.num_hands:
                mask = track_data.get('vis_mask', [])
                if len(mask) < self.T:
                    mask = mask + [True] * (self.T - len(mask))
                vis_masks[tensor_idx] = mask[:self.T]

        return np.array(vis_masks, dtype=bool), start_frame

    def get_joints(self, hand_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """Get 3D joint positions for a hand at a frame."""
        mano_idx = frame_idx - self.start_frame

        if not (0 <= mano_idx < self.T):
            return None

        if not self.vis_masks[hand_idx, mano_idx]:
            return None

        hand_is_right = self.is_right[hand_idx, mano_idx].item()
        if not (0.0 <= hand_is_right <= 1.0):
            return None

        # MANO forward pass
        pose = self.pose[hand_idx, mano_idx].unsqueeze(0).to(self.device)
        root_orient = self.root_orient[hand_idx, mano_idx].unsqueeze(0).to(self.device)
        trans = self.trans[hand_idx, mano_idx].unsqueeze(0).to(self.device)
        betas = self.betas[hand_idx].unsqueeze(0).to(self.device)

        pose_flat = pose.reshape(1, -1)

        with torch.no_grad():
            output = self.mano(
                betas=betas,
                global_orient=root_orient,
                hand_pose=pose_flat,
                transl=trans,
                return_verts=True,
            )

        joints = output.joints[0].cpu().numpy()

        # Apply handedness X-flip
        x_flip = 2 * hand_is_right - 1
        joints[:, 0] = x_flip * joints[:, 0]

        # Validate
        if not np.isfinite(joints).all():
            return None
        if (joints[:, 2] <= 0.01).any():
            return None

        return joints

    def get_mesh(self, hand_idx: int, frame_idx: int) -> Optional[tuple]:
        """Get mesh vertices and faces for a hand at a frame."""
        mano_idx = frame_idx - self.start_frame

        if not (0 <= mano_idx < self.T):
            return None

        if not self.vis_masks[hand_idx, mano_idx]:
            return None

        hand_is_right = self.is_right[hand_idx, mano_idx].item()
        if not (0.0 <= hand_is_right <= 1.0):
            return None

        # MANO forward pass
        pose = self.pose[hand_idx, mano_idx].unsqueeze(0).to(self.device)
        root_orient = self.root_orient[hand_idx, mano_idx].unsqueeze(0).to(self.device)
        trans = self.trans[hand_idx, mano_idx].unsqueeze(0).to(self.device)
        betas = self.betas[hand_idx].unsqueeze(0).to(self.device)

        pose_flat = pose.reshape(1, -1)

        with torch.no_grad():
            output = self.mano(
                betas=betas,
                global_orient=root_orient,
                hand_pose=pose_flat,
                transl=trans,
                return_verts=True,
            )

        vertices = output.vertices[0].cpu().numpy()

        # Apply handedness X-flip
        x_flip = 2 * hand_is_right - 1
        vertices[:, 0] = x_flip * vertices[:, 0]

        # Get faces (flip winding for left hand)
        if hand_is_right > 0.5:
            faces = self.mano.faces.copy()
        else:
            faces = self.mano.faces[:, ::-1].copy()

        return vertices, faces, hand_is_right

    def project_to_2d(self, points_3d: np.ndarray) -> tuple:
        """Project 3D points to 2D using camera intrinsics."""
        fx, fy, cx, cy = self.intrinsics
        z = points_3d[:, 2]
        valid = z > 0.01

        points_2d = np.zeros((len(points_3d), 2))
        points_2d[valid, 0] = (points_3d[valid, 0] / z[valid]) * fx + cx
        points_2d[valid, 1] = (points_3d[valid, 1] / z[valid]) * fy + cy

        return points_2d, valid

    # ========================================================================
    # Export Operations
    # ========================================================================

    def export_joints(self, output_path: str, include_names: bool = True):
        """Export all joint positions to JSONL file."""
        print(f"\nExporting joints to {output_path}...")

        with open(output_path, 'w') as f:
            for frame_idx in tqdm(range(self.start_frame, self.start_frame + self.T), desc="Exporting"):
                frame_data = {
                    "frame": frame_idx,
                    "hands": []
                }

                for hand_idx in range(self.num_hands):
                    joints = self.get_joints(hand_idx, frame_idx)
                    if joints is not None:
                        mano_idx = frame_idx - self.start_frame
                        hand_is_right = self.is_right[hand_idx, mano_idx].item()

                        hand_data = {
                            "handedness": "right" if hand_is_right > 0.5 else "left",
                            "joints_3d": joints.tolist(),
                        }

                        if include_names:
                            hand_data["joint_names"] = JOINT_NAMES

                        # Add 2D projections
                        joints_2d, valid = self.project_to_2d(joints)
                        hand_data["joints_2d"] = joints_2d.tolist()

                        frame_data["hands"].append(hand_data)

                f.write(json.dumps(frame_data) + "\n")

        print(f"  Exported {self.T} frames")

    # ========================================================================
    # Rendering Operations
    # ========================================================================

    def render_skeleton(self, video_path: str, output_path: str,
                        color: tuple = HAPTICA_TEAL, thickness: int = 3):
        """Render skeleton overlay on video."""
        print(f"\nRendering skeleton to {output_path}...")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        rendered = 0
        for frame_idx in tqdm(range(total_frames), desc="Rendering skeleton"):
            ret, frame = cap.read()
            if not ret:
                break

            for hand_idx in range(self.num_hands):
                joints = self.get_joints(hand_idx, frame_idx)
                if joints is not None:
                    joints_2d, valid = self.project_to_2d(joints)
                    self._draw_skeleton(frame, joints_2d, valid, color, thickness)
                    rendered += 1

            out.write(frame)

        cap.release()
        out.release()
        print(f"  Rendered {rendered} hand instances")

    def render_mesh(self, video_path: str, output_path: str,
                    color: tuple = (0, 178, 170), alpha: float = 0.5):
        """Render mesh overlay on video."""
        print(f"\nRendering mesh to {output_path}...")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        rendered = 0
        for frame_idx in tqdm(range(total_frames), desc="Rendering mesh"):
            ret, frame = cap.read()
            if not ret:
                break

            for hand_idx in range(self.num_hands):
                mesh_data = self.get_mesh(hand_idx, frame_idx)
                if mesh_data is not None:
                    vertices, faces, _ = mesh_data
                    self._draw_mesh(frame, vertices, faces, color, alpha)
                    rendered += 1

            out.write(frame)

        cap.release()
        out.release()
        print(f"  Rendered {rendered} hand instances")

    def _draw_skeleton(self, frame, joints_2d, valid, color, thickness):
        """Draw skeleton on frame."""
        H, W = frame.shape[:2]

        # Check visibility
        visible = sum(1 for i, (x, y) in enumerate(joints_2d)
                      if valid[i] and 0 <= x < W and 0 <= y < H)
        if visible / len(joints_2d) < 0.2:
            return

        # Draw connections
        for i, j in SKELETON_CONNECTIONS:
            if valid[i] and valid[j]:
                pt1 = tuple(joints_2d[i].astype(int))
                pt2 = tuple(joints_2d[j].astype(int))
                if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                    0 <= pt2[0] < W and 0 <= pt2[1] < H):
                    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw joints
        for i, (x, y) in enumerate(joints_2d):
            if valid[i] and 0 <= x < W and 0 <= y < H:
                pt = (int(x), int(y))
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_mesh(self, frame, vertices, faces, color, alpha):
        """Draw mesh overlay on frame."""
        H, W = frame.shape[:2]

        # Project vertices
        verts_2d, valid = self.project_to_2d(vertices)

        # Create overlay
        overlay = frame.copy()

        for face in faces:
            if not all(valid[face]):
                continue
            pts = verts_2d[face].astype(np.int32)
            if not all((0 <= pts[:, 0]) & (pts[:, 0] < W) &
                       (0 <= pts[:, 1]) & (pts[:, 1] < H)):
                continue
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ============================================================================
# Multi-camera Sync Utilities
# ============================================================================

def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    """Smooth signal with windowed mean, handling NaN values."""
    if window <= 1:
        return signal.astype(np.float32, copy=True)
    valid = np.isfinite(signal)
    sig = np.where(valid, signal, 0.0)
    kernel = np.ones(window, dtype=np.float32)
    num = np.convolve(sig, kernel, mode='same')
    den = np.convolve(valid.astype(np.float32), kernel, mode='same')
    out = np.full_like(signal, np.nan, dtype=np.float32)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _build_motion_signal(pipeline: HandPipeline, total_frames: int,
                         trans_weight: float = 0.7, pose_weight: float = 0.3) -> np.ndarray:
    """Build a per-frame motion signal from MANO params for sync.

    Uses translation velocity and pose energy without running MANO forward pass.
    """
    signal = np.full(total_frames, np.nan, dtype=np.float32)
    start = pipeline.start_frame
    T = pipeline.T

    for hand_idx in range(pipeline.num_hands):
        trans = pipeline.trans[hand_idx].cpu().numpy()
        pose = pipeline.pose[hand_idx].cpu().numpy()

        # Speed: norm of translation delta
        if len(trans) >= 2:
            speed = np.linalg.norm(np.diff(trans, axis=0), axis=1)
            speed = np.concatenate([speed[:1], speed]).astype(np.float32)
        else:
            speed = np.zeros((T,), dtype=np.float32)

        # Pose energy: norm of flattened pose
        pose_flat = pose.reshape(T, -1)
        pose_energy = np.linalg.norm(pose_flat, axis=1).astype(np.float32)

        hand_signal = trans_weight * speed + pose_weight * pose_energy

        # Mask by visibility
        vis_mask = pipeline.vis_masks[hand_idx] if pipeline.vis_masks is not None else np.ones(T, dtype=bool)
        is_right = pipeline.is_right[hand_idx].cpu().numpy()
        valid = vis_mask & np.isfinite(is_right) & (is_right >= 0.0) & (is_right <= 1.0)
        hand_signal = hand_signal.astype(np.float32)
        hand_signal[~valid] = np.nan

        # Map to frame indices
        frame_idx = np.arange(T) + start
        valid_idx = frame_idx < total_frames
        frame_idx = frame_idx[valid_idx]
        vals = hand_signal[:len(frame_idx)]
        mask = np.isfinite(vals)
        frame_idx = frame_idx[mask]
        vals = vals[mask]

        if len(frame_idx) == 0:
            continue

        # Accumulate
        existing = signal[frame_idx]
        nan_mask = np.isnan(existing)
        existing[nan_mask] = 0.0
        signal[frame_idx] = existing + vals

    return signal


def _masked_corr(a: np.ndarray, b: np.ndarray, min_overlap: int) -> Optional[float]:
    """Compute correlation between two arrays, ignoring NaN values."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_overlap:
        return None
    aa = a[mask] - a[mask].mean()
    bb = b[mask] - b[mask].mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom < 1e-6:
        return None
    return float(np.dot(aa, bb) / denom)


def _estimate_lag(ref: np.ndarray, target: np.ndarray, max_lag: int,
                  min_overlap: int = 15) -> tuple:
    """Estimate lag where target best aligns to ref.

    Positive lag means target occurs later (delayed) relative to ref.
    """
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ref_slice = ref[-lag:]
            tgt_slice = target[:len(ref_slice)]
        else:
            ref_slice = ref[:len(target) - lag]
            tgt_slice = target[lag:lag + len(ref_slice)]

        corr = _masked_corr(ref_slice, tgt_slice, min_overlap)
        if corr is None:
            continue
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if not np.isfinite(best_corr):
        return 0, None
    return best_lag, float(best_corr)


def _video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": width, "height": height, "frames": frames}


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """Resize frame to target height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / max(h, 1)
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _pad_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
    """Pad frame to target width with black bars."""
    h, w = frame.shape[:2]
    if w >= target_w:
        return frame
    pad = target_w - w
    return cv2.copyMakeBorder(frame, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _draw_label(frame: np.ndarray, label: str, frame_idx: int, offset: int = 0) -> None:
    """Draw camera label and frame info on frame."""
    text = f"{label} | f={frame_idx} | off={offset:+d}"
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _collect_frame_hands(pipeline: HandPipeline, frame_idx: int, width: int, height: int) -> list:
    """Collect hand data for a frame with visibility scoring."""
    hands = []
    for hand_idx in range(pipeline.num_hands):
        joints = pipeline.get_joints(hand_idx, frame_idx)
        if joints is None:
            continue
        mano_idx = frame_idx - pipeline.start_frame
        if not (0 <= mano_idx < pipeline.T):
            continue
        hand_is_right = pipeline.is_right[hand_idx, mano_idx].item()
        handedness = "right" if hand_is_right > 0.5 else "left"
        joints_2d, valid = pipeline.project_to_2d(joints)
        xs = joints_2d[:, 0]
        ys = joints_2d[:, 1]
        on = valid & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        visibility = float(on.sum() / len(joints_2d))
        hands.append({
            "hand_idx": hand_idx,
            "handedness": handedness,
            "joints_3d": joints.tolist(),
            "joints_2d": joints_2d.tolist(),
            "visibility": visibility,
        })
    return hands


def _fuse_best_view(cam_hands: dict) -> list:
    """Fuse hands from multiple cameras by selecting best visibility per hand."""
    fused = []
    for handedness in ("left", "right"):
        best = None
        for cam_name, hands in cam_hands.items():
            for hand in hands:
                if hand["handedness"] != handedness:
                    continue
                if best is None or hand["visibility"] > best["visibility"]:
                    best = {
                        "source_camera": cam_name,
                        **hand,
                    }
        if best is not None:
            fused.append(best)
    return fused


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_skeleton(args):
    """Render skeleton overlay."""
    pipeline = HandPipeline(args.params, args.cameras, args.device)
    pipeline.render_skeleton(args.input, args.output, thickness=args.thickness)

def cmd_mesh(args):
    """Render mesh overlay."""
    pipeline = HandPipeline(args.params, args.cameras, args.device)
    pipeline.render_mesh(args.input, args.output, alpha=args.alpha)

def cmd_export(args):
    """Export joint data to JSON."""
    pipeline = HandPipeline(args.params, args.cameras, args.device)
    pipeline.export_joints(args.output, include_names=not args.no_names)

def cmd_all(args):
    """Run full pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = HandPipeline(args.params, args.cameras, args.device)

    # Export joints
    pipeline.export_joints(str(output_dir / "joints.jsonl"))

    # Render skeleton
    if args.input:
        pipeline.render_skeleton(args.input, str(output_dir / "skeleton.mp4"))

        # Render mesh
        if not args.skip_mesh:
            pipeline.render_mesh(args.input, str(output_dir / "mesh.mp4"))

    print(f"\n✅ Pipeline complete! Output in {output_dir}/")


def cmd_sync(args):
    """Synchronize three cameras and render multi-view output."""
    print("\n📹 Multi-Camera Sync")
    print("=" * 50)

    # Load pipelines
    print("\nLoading pipelines...")
    head = HandPipeline(args.head_params, args.head_cameras, args.device)
    lwrist = HandPipeline(args.lwrist_params, args.lwrist_cameras, args.device)
    rwrist = HandPipeline(args.rwrist_params, args.rwrist_cameras, args.device)

    # Get video info
    head_info = _video_info(args.head_video)
    lw_info = _video_info(args.lwrist_video)
    rw_info = _video_info(args.rwrist_video)

    fps = head_info["fps"]
    if abs(lw_info["fps"] - fps) > 0.5 or abs(rw_info["fps"] - fps) > 0.5:
        print("WARNING: FPS mismatch across cameras; alignment assumes equal fps.")

    print(f"\nVideo info:")
    print(f"  Head:   {head_info['width']}x{head_info['height']} @ {head_info['fps']:.1f}fps, {head_info['frames']} frames")
    print(f"  LWrist: {lw_info['width']}x{lw_info['height']} @ {lw_info['fps']:.1f}fps, {lw_info['frames']} frames")
    print(f"  RWrist: {rw_info['width']}x{rw_info['height']} @ {rw_info['fps']:.1f}fps, {rw_info['frames']} frames")

    total_head = max(head_info["frames"], head.start_frame + head.T)
    total_lw = max(lw_info["frames"], lwrist.start_frame + lwrist.T)
    total_rw = max(rw_info["frames"], rwrist.start_frame + rwrist.T)

    # Build motion signals for sync
    print("\nBuilding motion signals...")
    head_signal = _build_motion_signal(head, total_head)
    lw_signal = _build_motion_signal(lwrist, total_lw)
    rw_signal = _build_motion_signal(rwrist, total_rw)

    head_signal = _smooth_signal(head_signal, args.smooth_window)
    lw_signal = _smooth_signal(lw_signal, args.smooth_window)
    rw_signal = _smooth_signal(rw_signal, args.smooth_window)

    # Estimate or use manual offsets
    if args.offset_lw is None:
        lag_lw, corr_lw = _estimate_lag(head_signal, lw_signal, args.max_offset_frames)
        print(f"  Estimated LW lag: {lag_lw:+d} frames (corr={corr_lw:.3f})" if corr_lw else f"  Estimated LW lag: {lag_lw:+d} frames (no correlation)")
    else:
        lag_lw = args.offset_lw
        print(f"  Using manual LW lag: {lag_lw:+d} frames")

    if args.offset_rw is None:
        lag_rw, corr_rw = _estimate_lag(head_signal, rw_signal, args.max_offset_frames)
        print(f"  Estimated RW lag: {lag_rw:+d} frames (corr={corr_rw:.3f})" if corr_rw else f"  Estimated RW lag: {lag_rw:+d} frames (no correlation)")
    else:
        lag_rw = args.offset_rw
        print(f"  Using manual RW lag: {lag_rw:+d} frames")

    # Compute overlapping frame range
    t_start = max(0, -lag_lw, -lag_rw)
    t_end = min(
        head_info["frames"] - 1,
        lw_info["frames"] - 1 - lag_lw,
        rw_info["frames"] - 1 - lag_rw,
    )
    if t_end < t_start:
        raise RuntimeError("No overlapping frames after alignment; check offsets.")

    total_frames = t_end - t_start + 1
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    print(f"\nSynchronized range: frames {t_start}-{t_start + total_frames - 1} ({total_frames} frames)")

    # Open video captures
    cap_head = cv2.VideoCapture(str(args.head_video))
    cap_lw = cv2.VideoCapture(str(args.lwrist_video))
    cap_rw = cv2.VideoCapture(str(args.rwrist_video))

    cap_head.set(cv2.CAP_PROP_POS_FRAMES, t_start)
    cap_lw.set(cv2.CAP_PROP_POS_FRAMES, t_start + lag_lw)
    cap_rw.set(cv2.CAP_PROP_POS_FRAMES, t_start + lag_rw)

    # Calculate output dimensions
    tile_h = args.tile_height if args.tile_height else min(
        head_info["height"], lw_info["height"], rw_info["height"]
    )

    head_w = int(round(head_info["width"] * (tile_h / head_info["height"])))
    lw_w = int(round(lw_info["width"] * (tile_h / lw_info["height"])))
    rw_w = int(round(rw_info["width"] * (tile_h / rw_info["height"])))

    if args.layout == "row":
        out_w = head_w + lw_w + rw_w
        out_h = tile_h
    else:  # grid
        tile_w = max(head_w, lw_w, rw_w)
        out_w = tile_w * 2
        out_h = tile_h * 2

    print(f"\nOutput: {out_w}x{out_h} @ {fps:.1f}fps ({args.layout} layout)")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output_video), fourcc, fps, (out_w, out_h))

    # Setup JSONL output
    jsonl = open(args.output_jsonl, 'w') if args.output_jsonl else None

    # Render synchronized frames
    for i in tqdm(range(total_frames), desc="Sync render"):
        ret_h, frame_h = cap_head.read()
        ret_lw, frame_lw = cap_lw.read()
        ret_rw, frame_rw = cap_rw.read()
        if not (ret_h and ret_lw and ret_rw):
            break

        head_idx = t_start + i
        lw_idx = t_start + i + lag_lw
        rw_idx = t_start + i + lag_rw

        # Draw skeletons if enabled
        if not args.no_skeleton:
            for hand_idx in range(head.num_hands):
                joints = head.get_joints(hand_idx, head_idx)
                if joints is not None:
                    joints_2d, valid = head.project_to_2d(joints)
                    head._draw_skeleton(frame_h, joints_2d, valid, HAPTICA_TEAL, thickness=2)
            for hand_idx in range(lwrist.num_hands):
                joints = lwrist.get_joints(hand_idx, lw_idx)
                if joints is not None:
                    joints_2d, valid = lwrist.project_to_2d(joints)
                    lwrist._draw_skeleton(frame_lw, joints_2d, valid, HAPTICA_TEAL, thickness=2)
            for hand_idx in range(rwrist.num_hands):
                joints = rwrist.get_joints(hand_idx, rw_idx)
                if joints is not None:
                    joints_2d, valid = rwrist.project_to_2d(joints)
                    rwrist._draw_skeleton(frame_rw, joints_2d, valid, HAPTICA_TEAL, thickness=2)

        # Draw labels
        _draw_label(frame_h, "HEAD", head_idx, 0)
        _draw_label(frame_lw, "LWRIST", lw_idx, lag_lw)
        _draw_label(frame_rw, "RWRIST", rw_idx, lag_rw)

        # Resize frames
        frame_h = _resize_to_height(frame_h, tile_h)
        frame_lw = _resize_to_height(frame_lw, tile_h)
        frame_rw = _resize_to_height(frame_rw, tile_h)

        # Stitch frames
        if args.layout == "row":
            stitched = np.hstack([frame_h, frame_lw, frame_rw])
        else:  # grid
            tile_w = max(frame_h.shape[1], frame_lw.shape[1], frame_rw.shape[1])
            frame_h = _pad_to_width(frame_h, tile_w)
            frame_lw = _pad_to_width(frame_lw, tile_w)
            frame_rw = _pad_to_width(frame_rw, tile_w)
            blank = np.zeros_like(frame_h)
            row1 = np.hstack([frame_h, frame_lw])
            row2 = np.hstack([frame_rw, blank])
            stitched = np.vstack([row1, row2])

        out.write(stitched)

        # Write JSONL
        if jsonl:
            cam_hands = {
                "head": _collect_frame_hands(head, head_idx, head_info["width"], head_info["height"]),
                "lwrist": _collect_frame_hands(lwrist, lw_idx, lw_info["width"], lw_info["height"]),
                "rwrist": _collect_frame_hands(rwrist, rw_idx, rw_info["width"], rw_info["height"]),
            }
            frame_data = {
                "frame": i,
                "head_frame": head_idx,
                "lwrist_frame": lw_idx,
                "rwrist_frame": rw_idx,
                "offsets": {"lwrist": lag_lw, "rwrist": lag_rw},
                "cameras": cam_hands,
            }
            if args.fuse_mode == "best_view":
                frame_data["fused"] = {"mode": "best_view", "hands": _fuse_best_view(cam_hands)}
            jsonl.write(json.dumps(frame_data) + "\n")

    # Cleanup
    cap_head.release()
    cap_lw.release()
    cap_rw.release()
    out.release()
    if jsonl:
        jsonl.close()

    print(f"\n✅ Sync output saved to {args.output_video}")
    if args.output_jsonl:
        print(f"✅ Sync JSONL saved to {args.output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Haptica Hand Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render skeleton overlay
  %(prog)s skeleton -i video.mp4 -p smooth_fit_params.pth -o skeleton.mp4

  # Export joint positions to JSON
  %(prog)s export -p smooth_fit_params.pth -o joints.jsonl

  # Run full pipeline
  %(prog)s all -i video.mp4 -p smooth_fit_params.pth -d ./output/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Skeleton command
    p_skel = subparsers.add_parser('skeleton', help='Render skeleton overlay')
    p_skel.add_argument('-i', '--input', required=True, help='Input video')
    p_skel.add_argument('-p', '--params', required=True, help='smooth_fit_params.pth')
    p_skel.add_argument('-o', '--output', required=True, help='Output video')
    p_skel.add_argument('-c', '--cameras', help='cameras.json path')
    p_skel.add_argument('--thickness', type=int, default=3, help='Line thickness')
    p_skel.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    p_skel.set_defaults(func=cmd_skeleton)

    # Mesh command
    p_mesh = subparsers.add_parser('mesh', help='Render mesh overlay')
    p_mesh.add_argument('-i', '--input', required=True, help='Input video')
    p_mesh.add_argument('-p', '--params', required=True, help='smooth_fit_params.pth')
    p_mesh.add_argument('-o', '--output', required=True, help='Output video')
    p_mesh.add_argument('-c', '--cameras', help='cameras.json path')
    p_mesh.add_argument('--alpha', type=float, default=0.5, help='Mesh opacity')
    p_mesh.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    p_mesh.set_defaults(func=cmd_mesh)

    # Export command
    p_export = subparsers.add_parser('export', help='Export joint data')
    p_export.add_argument('-p', '--params', required=True, help='smooth_fit_params.pth')
    p_export.add_argument('-o', '--output', required=True, help='Output JSONL file')
    p_export.add_argument('-c', '--cameras', help='cameras.json path')
    p_export.add_argument('--no-names', action='store_true', help='Exclude joint names')
    p_export.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    p_export.set_defaults(func=cmd_export)

    # All command
    p_all = subparsers.add_parser('all', help='Run full pipeline')
    p_all.add_argument('-i', '--input', help='Input video (optional for export-only)')
    p_all.add_argument('-p', '--params', required=True, help='smooth_fit_params.pth')
    p_all.add_argument('-d', '--output-dir', required=True, help='Output directory')
    p_all.add_argument('-c', '--cameras', help='cameras.json path')
    p_all.add_argument('--skip-mesh', action='store_true', help='Skip mesh rendering')
    p_all.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    p_all.set_defaults(func=cmd_all)

    # Sync command
    p_sync = subparsers.add_parser('sync', help='Synchronize multi-camera views',
        description='Synchronize head, left wrist, and right wrist cameras with automatic temporal alignment.')
    p_sync.add_argument('--head-video', required=True, help='Head camera video')
    p_sync.add_argument('--lwrist-video', required=True, help='Left wrist video')
    p_sync.add_argument('--rwrist-video', required=True, help='Right wrist video')
    p_sync.add_argument('--head-params', required=True, help='Head smooth_fit_params.pth')
    p_sync.add_argument('--lwrist-params', required=True, help='Left wrist smooth_fit_params.pth')
    p_sync.add_argument('--rwrist-params', required=True, help='Right wrist smooth_fit_params.pth')
    p_sync.add_argument('--head-cameras', help='Head cameras.json')
    p_sync.add_argument('--lwrist-cameras', help='Left wrist cameras.json')
    p_sync.add_argument('--rwrist-cameras', help='Right wrist cameras.json')
    p_sync.add_argument('-o', '--output-video', required=True, help='Output synced video')
    p_sync.add_argument('--output-jsonl', help='Output JSONL with multi-cam joints')
    p_sync.add_argument('--layout', choices=['row', 'grid'], default='row',
                        help='Layout: row (3 side-by-side) or grid (2x2)')
    p_sync.add_argument('--tile-height', type=int, help='Tile height in pixels')
    p_sync.add_argument('--max-offset-frames', type=int, default=180,
                        help='Max sync offset to search (frames, default: 180 = 3s at 60fps)')
    p_sync.add_argument('--smooth-window', type=int, default=7,
                        help='Smoothing window for motion signal')
    p_sync.add_argument('--offset-lw', type=int, help='Manual left-wrist offset (frames)')
    p_sync.add_argument('--offset-rw', type=int, help='Manual right-wrist offset (frames)')
    p_sync.add_argument('--max-frames', type=int, help='Limit output frames')
    p_sync.add_argument('--no-skeleton', action='store_true', help='Disable skeleton overlay')
    p_sync.add_argument('--fuse-mode', choices=['none', 'best_view'], default='best_view',
                        help='Fusion mode for JSONL: best_view selects most visible camera per hand')
    p_sync.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    p_sync.set_defaults(func=cmd_sync)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
