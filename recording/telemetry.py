from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .gpmf_tool import ensure_gpmf_extract_tool


@dataclass(frozen=True)
class TelemetryExportResult:
    ok: bool
    tool: Optional[str]
    exports: list[Path]
    error: Optional[str] = None
    gpmf_path: Optional[Path] = None


def extract_gopro_telemetry(
    video_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    binary_path: Optional[Path] = None,
    keys: str = "ACCL,GYRO,GPS5,GRAV,CORI",
) -> TelemetryExportResult:
    """
    Extract telemetry derived artifacts for a GoPro MP4.

    - Always tries to preserve raw GPMF as `<stem>.gpmf.bin` (via ffmpeg).
    - Generates `<stem>.jsonl` via GoPro's official gpmf-parser (vendored submodule).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        return TelemetryExportResult(False, None, [], "video_missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    jsonl_path = output_dir / f"{stem}.jsonl"
    gpmf_out_path = output_dir / f"{stem}.gpmf.bin"

    existing = list(_collect_outputs(stem, [output_dir]))
    gpmf_existing = _resolve_existing_gpmf(binary_path, gpmf_out_path)
    if gpmf_existing and gpmf_existing not in existing:
        existing.append(gpmf_existing)
    if not overwrite and jsonl_path.exists() and gpmf_existing:
        return TelemetryExportResult(True, "gpmf-parser", sorted(existing), None, gpmf_path=gpmf_existing)

    gpmf_path = _resolve_existing_gpmf(binary_path, gpmf_out_path)
    if gpmf_path is None or overwrite:
        gpmf_path = _extract_gpmf_bin_from_mp4(video_path, gpmf_out_path, overwrite=overwrite) or gpmf_path

    try:
        tool = ensure_gpmf_extract_tool()
    except Exception as exc:
        return TelemetryExportResult(False, None, [], f"gpmf_tool_missing:{exc}", gpmf_path=gpmf_path)

    if (not jsonl_path.exists()) or overwrite:
        err = _run_gpmf_extract(tool.bin_path, video_path, jsonl_path, keys=keys)
        if err:
            return TelemetryExportResult(False, "gpmf-parser", [], err, gpmf_path=gpmf_path)

    exports = list(_collect_outputs(stem, [output_dir]))
    if gpmf_path and gpmf_path.exists() and gpmf_path not in exports:
        exports.append(gpmf_path)
    return TelemetryExportResult(True, "gpmf-parser", sorted(set(exports)), None, gpmf_path=gpmf_path)


def _resolve_existing_gpmf(binary_path: Optional[Path], gpmf_out_path: Path) -> Optional[Path]:
    if binary_path:
        candidate = Path(binary_path)
        if candidate.exists():
            return candidate
    if gpmf_out_path.exists():
        return gpmf_out_path
    return None


def _extract_gpmf_bin_from_mp4(video_path: Path, dest_path: Path, *, overwrite: bool = False) -> Optional[Path]:
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        return None
    dest_path = Path(dest_path)
    if dest_path.exists() and not overwrite:
        return dest_path

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-print_format",
            "json",
            "-i",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    stream_index = None
    for stream in info.get("streams", []):
        tag = str(stream.get("codec_tag_string") or "").lower()
        if tag == "gpmd":
            stream_index = stream.get("index")
            break
    if stream_index is None:
        return None

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-map",
            f"0:{stream_index}",
            "-codec",
            "copy",
            "-f",
            "rawvideo",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        return None
    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        tmp_path.replace(dest_path)
        return dest_path
    tmp_path.unlink(missing_ok=True)
    return None


def _run_gpmf_extract(bin_path: Path, video_path: Path, jsonl_path: Path, *, keys: str) -> Optional[str]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            str(bin_path),
            "--input",
            str(video_path),
            "--output",
            str(tmp_path),
            "--keys",
            keys,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        err = (result.stderr or result.stdout or "").strip()
        return err or f"gpmf_extract_failed:{result.returncode}"
    tmp_path.replace(jsonl_path)
    return None


def _collect_outputs(stem: str, dirs: Iterable[Path]) -> Iterable[Path]:
    for dir_path in dirs:
        if not dir_path or not Path(dir_path).exists():
            continue
        for path in Path(dir_path).glob(f"{stem}*"):
            if path.is_dir():
                continue
            suffix = path.suffix.lower()
            if suffix in {".jsonl", ".csv", ".gpx", ".kml", ".json", ".bin"}:
                yield path


def csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> None:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for row in reader:
                out.write(json.dumps(row, sort_keys=False) + "\n")
