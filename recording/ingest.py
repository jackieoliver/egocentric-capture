from __future__ import annotations

from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Dict, Optional

import hashlib

from .models import RecordingStatus, SessionManifest
from .offload import DeviceOffloader, OffloadRequest
from .telemetry import extract_gopro_telemetry
from .paths import sessions_root
from .store import SessionStore


@dataclass(frozen=True)
class IngestConfig:
    """
    Default ingest locations.

    `dest_root` is inside the canonical sessions directory by default.
    """

    dest_root: Path


def default_ingest_config() -> IngestConfig:
    return IngestConfig(dest_root=sessions_root())


class IngestManager:
    """
    Device-agnostic ingest/offload orchestrator.

    This decides *what* to fetch and updates the session manifest; the
    device-specific *how* is delegated to a `DeviceOffloader`.
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        offloaders: Dict[str, DeviceOffloader],
        config: Optional[IngestConfig] = None,
    ) -> None:
        self.store = store
        self.offloaders = offloaders
        self.config = config or default_ingest_config()

    def offload_session(
        self,
        session_id: str,
        *,
        kind: Optional[str] = None,
        extract_telemetry: bool = False,
        telemetry_force: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> SessionManifest:
        session = self.store.load(session_id)
        if not session:
            raise ValueError("session_not_found")

        def _cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        for take in session.takes:
            for rec in take.recordings:
                if _cancelled():
                    return session
                if kind and rec.kind != kind:
                    continue
                offloader = self.offloaders.get(rec.kind)
                should_offload = rec.status == RecordingStatus.LINKED and not rec.local_path

                dest_dir = (
                    self.config.dest_root
                    / session.session_id
                    / "takes"
                    / f"take_{take.take_number:03d}"
                    / "recordings"
                    / rec.device_id
                    / (rec.stream_id or "video_main")
                )
                req = OffloadRequest(
                    session_id=session.session_id,
                    take_number=take.take_number,
                    recording=rec,
                    dest_dir=str(dest_dir),
                )

                if should_offload:
                    if _cancelled():
                        return session
                    if not offloader:
                        self.store.add_alert(
                            session,
                            code="offload_no_offloader",
                            message=f"No offloader registered for kind={rec.kind}",
                            device_id=rec.device_id,
                        )
                        continue

                    result = offloader.offload(req)
                    if _cancelled():
                        return session
                    if not result.ok or not result.local_path:
                        self.store.add_alert(
                            session,
                            code="offload_failed",
                            message=result.error or "offload_failed",
                            device_id=rec.device_id,
                        )
                        continue

                    rec.local_path = result.local_path
                    rec.size_bytes = rec.size_bytes or result.size_bytes
                    rec.status = RecordingStatus.INGESTED
                    self.store.save(session)
                    self.store.append_event(
                        session.session_id,
                        "recording_ingested",
                        {
                            "take_number": take.take_number,
                            "device_id": rec.device_id,
                            "kind": rec.kind,
                            "local_path": rec.local_path,
                            "size_bytes": rec.size_bytes,
                        },
                    )

                telemetry_path: Optional[Path] = None
                telemetry_result = None
                if rec.kind == "gopro" and rec.folder and rec.file:
                    telemetry_path = (
                        self.config.dest_root
                        / session.session_id
                        / "takes"
                        / f"take_{take.take_number:03d}"
                        / "metadata"
                        / "telemetry"
                        / "gopro"
                        / rec.device_id
                        / f"{Path(rec.file).stem}.gpmf.bin"
                    )
                    telemetry_existing = telemetry_path.exists()
                    if rec.extra.get("telemetry", {}).get("path"):
                        existing_path = Path(str(rec.extra["telemetry"]["path"]))
                        if existing_path.exists():
                            telemetry_existing = True

                    should_fetch_telemetry = telemetry_force or (
                        not telemetry_existing and (should_offload or extract_telemetry)
                    )
                    if should_fetch_telemetry:
                        if _cancelled():
                            return session
                        if offloader and hasattr(offloader, "offload_telemetry"):
                            telemetry_result = offloader.offload_telemetry(req, dest_path=telemetry_path)
                        elif rec.status == RecordingStatus.LINKED:
                            self.store.add_alert(
                                session,
                                code="telemetry_no_offloader",
                                message=f"No telemetry offloader for kind={rec.kind}",
                                device_id=rec.device_id,
                            )

                if telemetry_result and telemetry_result.get("ok"):
                    rec.extra.setdefault("telemetry", {})
                    rec.extra["telemetry"]["path"] = str(telemetry_result.get("local_path") or telemetry_path)
                    if telemetry_result.get("size_bytes") is not None:
                        rec.extra["telemetry"]["size_bytes"] = int(telemetry_result["size_bytes"])
                    self.store.save(session)
                    self.store.append_event(
                        session.session_id,
                        "recording_telemetry_ingested",
                        {
                            "take_number": take.take_number,
                            "device_id": rec.device_id,
                            "path": str(telemetry_result.get("local_path") or telemetry_path),
                            "size_bytes": telemetry_result.get("size_bytes"),
                        },
                    )
                elif telemetry_result and telemetry_result.get("error") not in {
                    None,
                    "telemetry_not_supported",
                    "folder_file_required",
                }:
                    if not extract_telemetry or not rec.local_path:
                        self.store.add_alert(
                            session,
                            code="telemetry_failed",
                            message=str(telemetry_result.get("error") or "telemetry_failed"),
                            device_id=rec.device_id,
                        )

                if extract_telemetry and rec.kind == "gopro" and rec.local_path:
                    if _cancelled():
                        return session
                    output_dir = (
                        self.config.dest_root
                        / session.session_id
                        / "takes"
                        / f"take_{take.take_number:03d}"
                        / "metadata"
                        / "telemetry"
                        / "gopro"
                        / rec.device_id
                    )
                    telemetry_hint = rec.extra.get("telemetry", {}).get("path")
                    telemetry_hint_path = Path(str(telemetry_hint)) if telemetry_hint else telemetry_path
                    result = extract_gopro_telemetry(
                        Path(rec.local_path),
                        output_dir,
                        overwrite=telemetry_force,
                        binary_path=telemetry_hint_path,
                    )
                    if result.ok:
                        rec.extra.setdefault("telemetry", {})
                        if result.gpmf_path and not rec.extra["telemetry"].get("path"):
                            rec.extra["telemetry"]["path"] = str(result.gpmf_path)
                        rec.extra["telemetry"]["exports"] = [str(p) for p in result.exports]
                        rec.extra["telemetry"]["tool"] = result.tool
                        self.store.save(session)
                        self.store.append_event(
                            session.session_id,
                            "recording_telemetry_extracted",
                            {
                                "take_number": take.take_number,
                                "device_id": rec.device_id,
                                "tool": result.tool,
                                "exports": [str(p) for p in result.exports],
                            },
                        )
                    else:
                        tool_missing = (
                            result.error == "telemetry_tool_missing"
                            or str(result.error or "").startswith("gpmf_tool_missing:")
                        )
                        if result.error is not None and not tool_missing:
                            self.store.add_alert(
                                session,
                                code="telemetry_extract_failed",
                                message=str(result.error or "telemetry_extract_failed"),
                                device_id=rec.device_id,
                            )

        return session

    def verify_session(self, session_id: str, *, kind: Optional[str] = None) -> SessionManifest:
        session = self.store.load(session_id)
        if not session:
            raise ValueError("session_not_found")

        for take in session.takes:
            for rec in take.recordings:
                if kind and rec.kind != kind:
                    continue
                if rec.status not in (RecordingStatus.INGESTED, RecordingStatus.VERIFIED):
                    continue
                if not rec.local_path:
                    continue
                path = Path(rec.local_path)
                if not path.exists():
                    self.store.add_alert(
                        session,
                        code="verify_missing_file",
                        message=f"Missing file: {rec.local_path}",
                        device_id=rec.device_id,
                    )
                    continue
                rec.sha256 = _sha256(path)
                rec.status = RecordingStatus.VERIFIED
                self.store.save(session)
                self.store.append_event(
                    session.session_id,
                    "recording_verified",
                    {
                        "take_number": take.take_number,
                        "device_id": rec.device_id,
                        "sha256": rec.sha256,
                    },
                )
        return session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
