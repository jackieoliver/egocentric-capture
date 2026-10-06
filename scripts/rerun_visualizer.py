#!/usr/bin/env python3
"""Rerun visualization for multi-camera hand tracking datasets."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import rerun as rr
import rerun.blueprint as rrb

from recording import telemetry


CAMERAS = ("head", "palm_L", "palm_R", "3rd_person")
SENSOR_KEYS = {"ACCL": "accel", "GYRO": "gyro", "GRAV": "grav"}


@dataclass
class SampleSeries:
    times: List[Optional[float]]
    vectors: List[Tuple[float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize multi-camera sessions in Rerun.")
    parser.add_argument("--session", required=True, type=Path, help="Path to session folder")
    parser.add_argument(
        "--spawn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Spawn the Rerun viewer",
    )
    parser.add_argument("--save", type=Path, default=None, help="Save recording to .rrd")
    return parser.parse_args()


def find_video(session_dir: Path, camera_name: str) -> Optional[Path]:
    # Check synced_lite first (720p 15fps for fast loading)
    lite_dir = session_dir / "synced_lite"
    if lite_dir.exists():
        lite_candidates = sorted(
            [
                *lite_dir.glob(f"{camera_name}_synced_*.mp4"),
                *lite_dir.glob(f"{camera_name}_synced_*.MP4"),
            ]
        )
        if lite_candidates:
            return lite_candidates[0]

    # Check synced_sample (trimmed/smaller versions)
    sample_dir = session_dir / "synced_sample"
    if sample_dir.exists():
        sample_candidates = sorted(
            [
                *sample_dir.glob(f"{camera_name}_synced_*.mp4"),
                *sample_dir.glob(f"{camera_name}_synced_*.MP4"),
            ]
        )
        if sample_candidates:
            return sample_candidates[0]

    synced_dir = session_dir / "synced"
    if synced_dir.exists():
        synced_candidates = sorted(
            [
                *synced_dir.glob(f"{camera_name}_synced.mp4"),
                *synced_dir.glob(f"{camera_name}_synced.MP4"),
                *synced_dir.glob(f"{camera_name}_synced.m4v"),
                *synced_dir.glob(f"{camera_name}_synced.M4V"),
            ]
        )
        if synced_candidates:
            return synced_candidates[0]

    camera_dir = session_dir / camera_name
    if not camera_dir.exists():
        return None
    candidates = sorted(
        [
            *camera_dir.glob("*.mp4"),
            *camera_dir.glob("*.MP4"),
            *camera_dir.glob("*.m4v"),
            *camera_dir.glob("*.M4V"),
        ]
    )
    return candidates[0] if candidates else None


def find_raw_video(session_dir: Path, camera_name: str) -> Optional[Path]:
    """Find raw GoPro video (not synced) for telemetry extraction."""
    camera_dir = session_dir / camera_name
    if not camera_dir.exists():
        return None
    candidates = sorted(
        [
            *camera_dir.glob("*.mp4"),
            *camera_dir.glob("*.MP4"),
            *camera_dir.glob("*.m4v"),
            *camera_dir.glob("*.M4V"),
        ]
    )
    return candidates[0] if candidates else None


def read_jsonl(path: Path) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def normalize_time_seconds(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_vectors(values: object, samples: Optional[int]) -> List[Tuple[float, float, float]]:
    if values is None:
        return []

    if isinstance(values, dict):
        x = values.get("x")
        y = values.get("y")
        z = values.get("z")
        if x is None or y is None or z is None:
            return []
        return [(float(x), float(y), float(z))]

    if not isinstance(values, list):
        return []

    if not values:
        return []

    if all(isinstance(item, (list, tuple)) for item in values):
        vectors: List[Tuple[float, float, float]] = []
        for item in values:
            if len(item) < 3:
                continue
            try:
                vectors.append((float(item[0]), float(item[1]), float(item[2])))
            except (TypeError, ValueError):
                continue
        return vectors

    if all(isinstance(item, (int, float)) for item in values):
        if samples is None:
            samples = len(values) // 3
        vectors = []
        for idx in range(samples):
            offset = idx * 3
            if offset + 2 >= len(values):
                break
            vectors.append((float(values[offset]), float(values[offset + 1]), float(values[offset + 2])))
        if vectors:
            return vectors
        if len(values) >= 3:
            return [(float(values[0]), float(values[1]), float(values[2]))]

    return []


def build_series(entry: dict) -> SampleSeries:
    t_in = normalize_time_seconds(entry.get("t_in"))
    t_out = normalize_time_seconds(entry.get("t_out"))
    samples = entry.get("samples")
    try:
        samples = int(samples) if samples is not None else None
    except (TypeError, ValueError):
        samples = None

    vectors = to_vectors(entry.get("values"), samples)
    if not vectors:
        return SampleSeries([], [])

    count = samples or len(vectors)
    if count <= 0:
        return SampleSeries([], [])

    times: List[Optional[float]] = []
    if t_in is None:
        times = [None for _ in range(len(vectors))]
    else:
        if t_out is None:
            t_out = t_in
        if count <= 1:
            times = [t_in for _ in range(len(vectors))]
        else:
            dt = (t_out - t_in) / max(count - 1, 1)
            times = [t_in + dt * idx for idx in range(len(vectors))]

    return SampleSeries(times, vectors)


MAX_VIDEO_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB limit for Arrow int32


def log_video(camera_path: str, video_path: Path) -> bool:
    """Log video to Rerun. Returns True if successful, False if skipped."""
    entity = f"{camera_path}/video"

    # Check file size - Rerun's Arrow backend has 32-bit int limitation
    file_size = video_path.stat().st_size
    if file_size > MAX_VIDEO_SIZE_BYTES:
        print(f"Skipping video {video_path.name} ({file_size / 1e9:.1f}GB) - too large for Rerun", file=sys.stderr)
        return False

    try:
        video_asset = rr.AssetVideo(path=video_path)
        frame_times_ns = video_asset.read_frame_timestamps_nanos()
        rr.log(entity, video_asset, static=True)
    except Exception as e:
        print(f"Failed to load video {video_path.name}: {e}", file=sys.stderr)
        return False

    if len(frame_times_ns) == 0:
        rr.set_time("time", duration=0.0)
        rr.log(entity, rr.VideoFrameReference(seconds=0.0))
        return True

    # Use columnar API for efficient frame logging
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn("time", duration=1e-9 * frame_times_ns)],
        columns=rr.VideoFrameReference.columns_nanos(frame_times_ns),
    )
    return True


def log_imu(camera_path: str, jsonl_path: Path) -> None:
    sample_index = 0
    for entry in read_jsonl(jsonl_path):
        key = entry.get("key")
        sensor_name = SENSOR_KEYS.get(key)
        if sensor_name is None:
            continue

        series = build_series(entry)
        if not series.vectors:
            continue

        sensor_base = f"{camera_path}/imu/{sensor_name}"
        for idx, vector in enumerate(series.vectors):
            time_value = series.times[idx] if idx < len(series.times) else None
            if time_value is None:
                rr.set_time("sample", sequence=sample_index)
            else:
                rr.set_time("time", duration= time_value)
            rr.log(f"{sensor_base}/x", rr.Scalars(vector[0]))
            rr.log(f"{sensor_base}/y", rr.Scalars(vector[1]))
            rr.log(f"{sensor_base}/z", rr.Scalars(vector[2]))
            sample_index += 1


def telemetry_jsonl_from_video(video_path: Path, output_dir: Path) -> Optional[Path]:
    result = telemetry.extract_gopro_telemetry(video_path, output_dir)
    if not result.ok:
        if result.error:
            print(f"Telemetry extraction failed for {video_path.name}: {result.error}", file=sys.stderr)
    for path in result.exports:
        if path.suffix.lower() == ".jsonl":
            return path
    fallback = output_dir / f"{video_path.stem}.jsonl"
    return fallback if fallback.exists() else None


def create_blueprint(session_name: str, camera_names: List[str]) -> rrb.Blueprint:
    """Create a blueprint with 2D views for each camera video and time series for IMU."""
    video_views = []
    imu_views = []

    for camera_name in camera_names:
        camera_path = f"/session/{session_name}/cameras/{camera_name}"
        # Create a 2D spatial view for each video
        video_views.append(
            rrb.Spatial2DView(
                name=f"{camera_name} video",
                origin=f"{camera_path}/video",
            )
        )
        # Create time series views for IMU data
        imu_views.append(
            rrb.TimeSeriesView(
                name=f"{camera_name} IMU",
                origin=f"{camera_path}/imu",
            )
        )

    # Layout: videos in a 2x2 grid on top, IMU plots on bottom
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Grid(
                *video_views,
                grid_columns=2,
            ),
            rrb.Horizontal(*imu_views),
            row_shares=[3, 1],  # Videos get 3/4 of space, IMU gets 1/4
        ),
        collapse_panels=True,
    )


def main() -> None:
    args = parse_args()
    session_dir = args.session
    if not session_dir.exists():
        raise SystemExit(f"Session path does not exist: {session_dir}")

    session_name = session_dir.name
    rr.init(f"hand_tracking_{session_name}", spawn=args.spawn)
    if args.save:
        rr.save(args.save)

    # Collect which cameras actually have video
    loaded_cameras = []

    telemetry_dir = session_dir / "telemetry"
    for camera_name in CAMERAS:
        video_path = find_video(session_dir, camera_name)
        if video_path is None:
            print(f"Missing video for camera {camera_name} in {session_dir}", file=sys.stderr)
            continue

        camera_path = f"/session/{session_name}/cameras/{camera_name}"
        video_loaded = log_video(camera_path, video_path)
        if video_loaded:
            print(f"Loaded video: {camera_name}")
            loaded_cameras.append(camera_name)

        # For telemetry, prefer raw video (has GPMF) over synced (may not)
        raw_video = find_raw_video(session_dir, camera_name)
        telemetry_source = raw_video if raw_video else video_path
        jsonl_path = telemetry_jsonl_from_video(telemetry_source, telemetry_dir)
        if jsonl_path is None:
            print(f"No telemetry JSONL for {telemetry_source.name}", file=sys.stderr)
            continue
        log_imu(camera_path, jsonl_path)
        print(f"Loaded IMU: {camera_name}")

    # Send blueprint to configure the viewer layout
    if loaded_cameras:
        blueprint = create_blueprint(session_name, loaded_cameras)
        rr.send_blueprint(blueprint)


if __name__ == "__main__":
    main()
