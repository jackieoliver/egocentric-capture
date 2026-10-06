#!/usr/bin/env python3
"""
Device-agnostic recording CLI (daemon-first).

This CLI manages sessions/takes in `recording/` and uses device daemons
for hardware control (gopro or dummy).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from recording.identity import load_identity, save_identity
from recording.models import DeviceState, SessionState
from recording.paths import sessions_root
from recording.registry import RegistryStore
from recording.store import get_session_store

from gopro.sd import format_sd_from_extra_rich
from recording.recorder import Recorder, _device_kind
from recording.uploader import DriveUploader
from recording.ingest import IngestManager
from recording.offload import DeviceOffloader
from recording.telemetry import extract_gopro_telemetry
from gopro.offload import GoProDaemonOffloader
from dummy_camera.offload import DummyDaemonOffloader

console = Console()


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def cmd_session_init(args: argparse.Namespace) -> None:
    store = get_session_store()
    recorder = Recorder(device=args.device, store=store)
    rig_spec = None
    registry = None
    rig_id = getattr(args, "rig", None)
    scene_id = getattr(args, "scene", None)
    tactition_id = getattr(args, "tactition", None)
    if rig_id or scene_id or tactition_id:
        try:
            registry = RegistryStore()
        except RuntimeError:
            console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
            return
    if registry and rig_id:
        rig = registry.get_rig(rig_id)
        if not rig:
            console.print(f"[red]Unknown rig:[/red] {rig_id}")
            return
        rig_spec = registry.rig_spec_from_registry(rig)
    if scene_id:
        if not registry or not registry.get_scene(scene_id):
            console.print(f"[red]Unknown scene:[/red] {scene_id}")
            return
    if tactition_id:
        if not registry or not registry.get_tactition(tactition_id):
            console.print(f"[red]Unknown tactition:[/red] {tactition_id}")
            return
    try:
        session = recorder.create_session(
            name=args.name,
            participant=args.participant,
            cameras=getattr(args, "cameras", None),
            rig=getattr(args, "rig", None),
            rig_spec=rig_spec,
        )
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("rig_missing:"):
            missing = msg.split(":", 1)[-1]
            console.print(f"[red]Missing required rig roles:[/red] {missing}")
        elif msg == "rig_device_mismatch":
            console.print("[red]Rig device_kind does not match selected device backend.[/red]")
        elif msg.startswith("unknown_rig:"):
            rig_name = msg.split(":", 1)[-1]
            console.print(f"[red]Unknown rig:[/red] {rig_name}")
        elif msg.startswith("cameras_not_found:"):
            missing = msg.split(":", 1)[-1]
            console.print(f"[red]Requested cameras not found:[/red] {missing}")
        else:
            console.print("[red]No cameras available from daemon status.[/red]")
        return
    if registry:
        session.registry_revision = registry.registry_revision()
    if rig_id:
        session.rig = rig_id
    if scene_id:
        session.scene_id = scene_id
    if tactition_id:
        session.tactition_id = tactition_id
    store.save(session)
    console.print(f"\n[bold green]Created session: {session.session_id}[/bold green]")
    for dev in session.devices:
        if dev.kind != _device_kind(args.device):
            continue
        role = dev.role or dev.device_id[-4:]
        if dev.state == DeviceState.UNREACHABLE:
            console.print(f"  {role}: [red]unreachable[/red]")
        else:
            console.print(f"  {role}: [green]connected[/green]")


def cmd_session_status(args: argparse.Namespace) -> None:
    store = get_session_store()
    session = store.get_active()
    if not session:
        console.print("[yellow]No active session.[/yellow]")
        return

    if args.json:
        print(json.dumps(session.to_dict(), indent=2))
        return

    recorder = Recorder(device=args.device, store=store)
    try:
        recorder.refresh_session(session)
    except Exception:
        pass

    table = Table(title=f"Session {session.session_id}")
    table.add_column("Role")
    table.add_column("State")
    table.add_column("Batt")
    table.add_column("Storage")
    table.add_column("Conn")
    for dev in session.devices:
        if dev.kind != _device_kind(args.device):
            continue
        role = dev.role or dev.device_id[-4:]
        batt = f"{dev.battery_pct}%" if dev.battery_pct is not None else "??"
        storage = format_sd_from_extra_rich(dev.extra or {}, fallback_remaining_gib=dev.storage_gb)
        table.add_row(role, dev.state.value, batt, storage, dev.connection or "-")
    console.print(table)


def cmd_session_list(args: argparse.Namespace) -> None:
    store = get_session_store()
    sessions = store.list_sessions(limit=50)
    if not sessions:
        console.print("[dim]No sessions found[/dim]")
        return
    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Created")
    table.add_column("Participant")
    table.add_column("Devices")
    table.add_column("Status")
    active_id = store.get_active_id()
    for sid in sessions[:20]:
        session = store.load(sid)
        if not session:
            continue
        is_active = "[bold cyan]ACTIVE[/bold cyan]" if sid == active_id else ""
        status = "closed" if session.closed_at else is_active or "open"
        created = session.created_at[:16] if session.created_at else "-"
        table.add_row(session.session_id, created, session.participant or "-", str(len(session.devices)), status)
    console.print(table)


def cmd_session_use(args: argparse.Namespace) -> None:
    store = get_session_store()
    session = store.load(args.session_id)
    if not session:
        console.print(f"[red]Session not found:[/red] {args.session_id}")
        return
    if session.state == SessionState.CLOSED or session.closed_at:
        console.print(f"[red]Session is closed:[/red] {args.session_id}")
        return
    store.set_active(args.session_id)
    console.print(f"[green]Switched to session:[/green] {args.session_id}")


def cmd_session_close(args: argparse.Namespace) -> None:
    store = get_session_store()
    session = store.get_active()
    if not session:
        console.print("[yellow]No active session.[/yellow]")
        return
    recording = [d for d in session.devices if d.state == DeviceState.RECORDING]
    if recording and not args.force:
        console.print("[yellow]Some devices are still recording. Use --force to close anyway.[/yellow]")
        return
    store.close(session)
    console.print(f"[green]Closed session:[/green] {session.session_id}")


def cmd_take_start(args: argparse.Namespace) -> None:
    store = get_session_store()
    session = store.get_active()
    if not session:
        console.print("[red]No active session.[/red]")
        return
    if session.current_take:
        console.print("[red]Take already recording.[/red]")
        return

    recorder = Recorder(device=args.device, store=store)
    try:
        take = recorder.start_take(session, label=args.label)
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("devices_not_ready:"):
            roles = msg.split(":", 1)[-1]
            console.print(f"[red]Devices not ready:[/red] {roles}")
        elif msg == "session_closed":
            console.print("[red]Session is closed; cannot start a take.[/red]")
        elif msg == "partial_start":
            console.print("[red]Partial start detected; take aborted.[/red]")
        else:
            console.print("[red]Failed to start take on all devices.[/red]")
        return
    if not take:
        console.print("[red]No devices started the take.[/red]")
        return
    console.print(f"[bold cyan]Take {take.take_number}[/bold cyan]" + (f" ({args.label})" if args.label else ""))


def cmd_take_stop(args: argparse.Namespace) -> None:
    store = get_session_store()
    session = store.get_active()
    if not session:
        console.print("[red]No active session.[/red]")
        return
    take = session.current_take
    if not take:
        console.print("[yellow]No take currently recording.[/yellow]")
        return

    console.print(f"[bold]Stopping take {take.take_number}...[/bold]")
    recorder = Recorder(device=args.device, store=store)
    try:
        recorder.stop_take(session)
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("devices_failed_stop:"):
            roles = msg.split(":", 1)[-1]
            console.print(f"[red]Failed to stop devices:[/red] {roles}")
            return
        console.print("[red]Failed to stop take.[/red]")
        return
    console.print(f"[bold green]Take {take.take_number} stopped[/bold green]")


def cmd_take_list(args: argparse.Namespace) -> None:
    store = get_session_store()
    session = store.get_active()
    if not session:
        console.print("[yellow]No active session.[/yellow]")
        return
    if not session.takes:
        console.print("[dim]No takes recorded yet[/dim]")
        return
    table = Table(title=f"Takes in {session.session_id}")
    table.add_column("#", style="bold")
    table.add_column("Label")
    table.add_column("Started")
    table.add_column("Duration")
    table.add_column("Recordings")
    table.add_column("Status")
    for take in session.takes:
        started = take.started_at[11:19] if take.started_at else "-"
        duration = f"{take.duration_sec}s" if take.duration_sec is not None else "-"
        status = "recording" if take.is_recording else "done"
        table.add_row(str(take.take_number), take.label or "-", started, duration, str(len(take.recordings)), status)
    console.print(table)


def cmd_registry_sync(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    registry.pull(force=args.force)
    console.print("[green]Registry synced from Drive.[/green]")


def cmd_registry_list(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    registry.pull()
    kind = args.kind
    if kind == "rigs":
        rows = registry.list_rigs()
        title = "Rigs"
        columns = ("ID", "Name", "Device", "Roles")
        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for rig in rows:
            table.add_row(
                str(rig.get("rig_id") or ""),
                str(rig.get("name") or ""),
                str(rig.get("device_kind") or ""),
                ",".join(rig.get("roles") or []),
            )
        console.print(table)
        return
    if kind == "tactitions":
        rows = registry.list_tactitions()
        title = "Tactitions"
        columns = ("ID", "Name", "Email")
        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for tact in rows:
            table.add_row(
                str(tact.get("tactition_id") or ""),
                str(tact.get("name") or tact.get("display_name") or ""),
                str(tact.get("email") or "-"),
            )
        console.print(table)
        return
    if kind == "scenes":
        rows = registry.list_scenes()
        title = "Scenes"
        columns = ("ID", "Name", "Tags")
        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for scene in rows:
            tags = scene.get("tags") or []
            table.add_row(
                str(scene.get("scene_id") or ""),
                str(scene.get("name") or ""),
                ",".join(tags),
            )
        console.print(table)
        return
    console.print(f"[red]Unknown registry kind:[/red] {kind}")


def cmd_registry_push(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    registry.push()
    console.print("[green]Registry pushed to Drive.[/green]")


def cmd_registry_add_rig(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    registry.pull()
    data = registry.load()
    if any(rig.get("rig_id") == args.rig_id for rig in data.rigs):
        console.print(f"[red]Rig already exists:[/red] {args.rig_id}")
        return
    roles = [r.strip().upper() for r in (args.roles.split(",") if args.roles else []) if r.strip()]
    data.rigs.append(
        {
            "rig_id": args.rig_id,
            "name": args.name,
            "device_kind": args.device_kind,
            "roles": roles,
        }
    )
    registry.save(data)
    registry.push()
    console.print(f"[green]Rig added:[/green] {args.rig_id}")


def cmd_registry_add_scene(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    registry.pull()
    data = registry.load()
    if any(scene.get("scene_id") == args.scene_id for scene in data.scenes):
        console.print(f"[red]Scene already exists:[/red] {args.scene_id}")
        return
    tags = [t.strip() for t in (args.tags.split(",") if args.tags else []) if t.strip()]
    data.scenes.append(
        {
            "scene_id": args.scene_id,
            "name": args.name,
            "tags": tags,
        }
    )
    registry.save(data)
    registry.push()
    console.print(f"[green]Scene added:[/green] {args.scene_id}")


def cmd_registry_add_tactition(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    registry.pull()
    data = registry.load()
    if any(tact.get("tactition_id") == args.tactition_id for tact in data.tactitions):
        console.print(f"[red]Tactition already exists:[/red] {args.tactition_id}")
        return
    data.tactitions.append(
        {
            "tactition_id": args.tactition_id,
            "name": args.name,
            "email": args.email,
        }
    )
    registry.save(data)
    registry.push()
    console.print(f"[green]Tactition added:[/green] {args.tactition_id}")


def cmd_identity_show(args: argparse.Namespace) -> None:
    identity = load_identity()
    table = Table(title="Recorder Identity")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in identity.to_dict().items():
        table.add_row(key, str(value))
    console.print(table)


def cmd_identity_set(args: argparse.Namespace) -> None:
    identity = load_identity()
    if args.operator_id:
        identity.operator_id = args.operator_id
    if args.operator_name:
        identity.operator_name = args.operator_name
    save_identity(identity)
    console.print("[green]Recorder identity updated.[/green]")


def cmd_upload_session(args: argparse.Namespace) -> None:
    store = get_session_store()
    session_id = args.session_id or store.get_active_id()
    if not session_id:
        console.print("[red]No active session.[/red]")
        return
    try:
        uploader = DriveUploader(store=store, format_version=args.format_version)
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    try:
        summary = uploader.upload_session(session_id, include_recordings=not args.no_recordings)
    except ValueError as exc:
        console.print(f"[red]Upload failed:[/red] {exc}")
        return
    console.print(
        f"[green]Uploaded session[/green] {summary.session_id} -> {summary.drive_folder_id} "
        f"({summary.file_count} files, {summary.total_bytes} bytes)"
    )


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    if value < 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MB"
    return f"{value / (1024 * 1024 * 1024):.2f} GB"


def cmd_upload_list(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    records = registry.load_uploads(force=args.force)
    if args.session_id:
        records = [r for r in records if r.get("session_id") == args.session_id or r.get("session_uid") == args.session_id]
    if not records:
        console.print("[dim]No uploads found[/dim]")
        return
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        sid = str(record.get("session_id") or "")
        size = int(record.get("size_bytes") or 0)
        summary.setdefault(sid, {"files": 0, "bytes": 0})
        summary[sid]["files"] += 1
        summary[sid]["bytes"] += size
    table = Table(title="Uploaded Sessions")
    table.add_column("Session")
    table.add_column("Files")
    table.add_column("Total Size")
    for sid, stats in summary.items():
        table.add_row(sid, str(stats["files"]), _format_bytes(stats["bytes"]))
    console.print(table)


def cmd_upload_show(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    records = registry.load_uploads(force=args.force)
    records = [r for r in records if r.get("session_id") == args.session_id or r.get("session_uid") == args.session_id]
    if not records:
        console.print("[dim]No uploads found[/dim]")
        return
    table = Table(title=f"Uploads for {args.session_id}")
    table.add_column("Path")
    table.add_column("Size")
    table.add_column("File ID")
    for record in records:
        table.add_row(
            str(record.get("relative_path") or ""),
            _format_bytes(int(record.get("size_bytes") or 0)),
            str(record.get("drive_file_id") or ""),
        )
    console.print(table)


def cmd_upload_stats(args: argparse.Namespace) -> None:
    try:
        registry = RegistryStore()
    except RuntimeError:
        console.print("[red]Drive registry not configured (missing credentials or root id).[/red]")
        return
    records = registry.load_uploads(force=args.force)
    if not records:
        console.print("[dim]No uploads found[/dim]")
        return
    key = args.by
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        if key == "session":
            group = str(record.get("session_id") or "")
        elif key == "operator":
            op_id = str(record.get("operator_id") or "")
            op_name = str(record.get("operator_name") or "")
            group = f"{op_id} {op_name}".strip() or "unknown"
        elif key == "recorder":
            group = str(record.get("recorder_id") or "unknown")
        elif key == "scene":
            group = str(record.get("scene_id") or "unknown")
        elif key == "rig":
            group = str(record.get("rig") or "unknown")
        else:
            group = "unknown"
        size = int(record.get("size_bytes") or 0)
        summary.setdefault(group, {"files": 0, "bytes": 0})
        summary[group]["files"] += 1
        summary[group]["bytes"] += size
    table = Table(title=f"Upload Stats by {key}")
    table.add_column("Group")
    table.add_column("Files")
    table.add_column("Total Size")
    for group, stats in sorted(summary.items()):
        table.add_row(group, str(stats["files"]), _format_bytes(stats["bytes"]))
    console.print(table)


def _offloaders_for(kind: Optional[str], *, allow_usb: bool = False) -> dict[str, DeviceOffloader]:
    offloaders: dict[str, DeviceOffloader] = {}
    if not kind or kind == "gopro":
        offloaders["gopro"] = GoProDaemonOffloader(allow_usb=allow_usb)
    if not kind or kind == "dummy_camera":
        offloaders["dummy_camera"] = DummyDaemonOffloader()
    return offloaders


def cmd_ingest_session(args: argparse.Namespace) -> None:
    store = get_session_store()
    session_id = args.session_id or store.get_active_id()
    if not session_id:
        console.print("[red]No active session.[/red]")
        return
    kind = args.kind
    offloaders = _offloaders_for(kind, allow_usb=args.usb)
    ingest = IngestManager(store=store, offloaders=offloaders)
    try:
        ingest.offload_session(
            session_id,
            kind=kind,
            extract_telemetry=bool(args.telemetry),
            telemetry_force=bool(args.telemetry_force),
        )
        if args.verify:
            ingest.verify_session(session_id, kind=kind)
    except ValueError as exc:
        console.print(f"[red]Ingest failed:[/red] {exc}")
        return
    console.print(f"[green]Ingest complete:[/green] {session_id}")


def cmd_ingest_verify(args: argparse.Namespace) -> None:
    store = get_session_store()
    session_id = args.session_id or store.get_active_id()
    if not session_id:
        console.print("[red]No active session.[/red]")
        return
    kind = args.kind
    ingest = IngestManager(store=store, offloaders={})
    try:
        ingest.verify_session(session_id, kind=kind)
    except ValueError as exc:
        console.print(f"[red]Verify failed:[/red] {exc}")
        return
    console.print(f"[green]Verify complete:[/green] {session_id}")


def cmd_telemetry_extract(args: argparse.Namespace) -> None:
    store = get_session_store()
    session_id = args.session_id or store.get_active_id()
    if not session_id:
        console.print("[red]No active session.[/red]")
        return
    session = store.load(session_id)
    if not session:
        console.print(f"[red]Session not found:[/red] {session_id}")
        return

    extracted = 0
    for take in session.takes:
        for rec in take.recordings:
            if rec.kind != "gopro":
                continue
            if not rec.local_path:
                continue
            output_dir = (
                sessions_root()
                / session.session_id
                / "takes"
                / f"take_{take.take_number:03d}"
                / "metadata"
                / "telemetry"
                / "gopro"
                / rec.device_id
            )
            telemetry_hint = rec.extra.get("telemetry", {}).get("path")
            telemetry_hint_path = Path(str(telemetry_hint)) if telemetry_hint else None
            result = extract_gopro_telemetry(
                Path(rec.local_path),
                output_dir,
                overwrite=bool(args.force),
                binary_path=telemetry_hint_path,
            )
            if not result.ok:
                if result.error == "telemetry_tool_missing" or str(result.error or "").startswith("gpmf_tool_missing:"):
                    console.print(
                        "[red]Telemetry tool not available.[/red] Initialize the `vendor/gopro/gpmf-parser` submodule "
                        "and rebuild by rerunning this command."
                    )
                    return
                console.print(f"[yellow]Telemetry extract failed:[/yellow] {result.error}")
                continue
            rec.extra.setdefault("telemetry", {})
            if result.gpmf_path and not rec.extra["telemetry"].get("path"):
                rec.extra["telemetry"]["path"] = str(result.gpmf_path)
            rec.extra["telemetry"]["exports"] = [str(p) for p in result.exports]
            rec.extra["telemetry"]["tool"] = result.tool
            store.save(session)
            store.append_event(
                session.session_id,
                "recording_telemetry_extracted",
                {
                    "take_number": take.take_number,
                    "device_id": rec.device_id,
                    "tool": result.tool,
                    "exports": [str(p) for p in result.exports],
                },
            )
            extracted += 1

    console.print(f"[green]Telemetry extraction complete:[/green] {extracted} recordings")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcriptions recorder CLI (daemon-first)")
    sub = parser.add_subparsers(dest="command", required=True)

    device_parent = argparse.ArgumentParser(add_help=False)
    device_parent.add_argument(
        "--device",
        choices=["gopro", "dummy"],
        default="gopro",
        help="Device backend to use (default: gopro)",
    )

    session = sub.add_parser("session", help="Session management", parents=[device_parent])
    session_sub = session.add_subparsers(dest="session_command", required=True)

    init = session_sub.add_parser("init", help="Create a new session")
    init.add_argument("--name", "-n", type=str, help="Session name")
    init.add_argument("--participant", "-p", type=str, help="Participant id")
    init.add_argument("--cameras", "-c", type=str, help="Comma-separated roles/ids")
    init.add_argument("--rig", "-r", type=str, help="Rig id (from registry)")
    init.add_argument("--scene", type=str, help="Scene id")
    init.add_argument("--tactition", type=str, help="Tactition id")
    init.set_defaults(func=cmd_session_init)

    status = session_sub.add_parser("status", help="Show session status")
    status.add_argument("--json", action="store_true", help="JSON output")
    status.set_defaults(func=cmd_session_status)

    list_cmd = session_sub.add_parser("list", help="List sessions")
    list_cmd.set_defaults(func=cmd_session_list)

    use = session_sub.add_parser("use", help="Switch active session")
    use.add_argument("session_id", help="Session id")
    use.set_defaults(func=cmd_session_use)

    close = session_sub.add_parser("close", help="Close active session")
    close.add_argument("--force", action="store_true", help="Close even if recording")
    close.set_defaults(func=cmd_session_close)

    take = sub.add_parser("take", help="Take management", parents=[device_parent])
    take_sub = take.add_subparsers(dest="take_command", required=True)

    take_start = take_sub.add_parser("start", help="Start a take")
    take_start.add_argument("--label", "-l", type=str, help="Take label")
    take_start.set_defaults(func=cmd_take_start)

    take_stop = take_sub.add_parser("stop", help="Stop the current take")
    take_stop.set_defaults(func=cmd_take_stop)

    take_list = take_sub.add_parser("list", help="List takes")
    take_list.set_defaults(func=cmd_take_list)

    registry = sub.add_parser("registry", help="Registry sync/list (Drive-backed)")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    registry_sync = registry_sub.add_parser("sync", help="Sync registry from Drive")
    registry_sync.add_argument("--force", action="store_true", help="Force re-download")
    registry_sync.set_defaults(func=cmd_registry_sync)
    registry_list = registry_sub.add_parser("list", help="List registry entries")
    registry_list.add_argument("kind", choices=["rigs", "tactitions", "scenes"])
    registry_list.set_defaults(func=cmd_registry_list)
    registry_push = registry_sub.add_parser("push", help="Push registry to Drive")
    registry_push.set_defaults(func=cmd_registry_push)
    registry_add = registry_sub.add_parser("add", help="Add registry entries")
    registry_add_sub = registry_add.add_subparsers(dest="registry_add_command", required=True)
    add_rig = registry_add_sub.add_parser("rig", help="Add a rig")
    add_rig.add_argument("rig_id", type=str, help="Rig id")
    add_rig.add_argument("--name", required=True, help="Rig name")
    add_rig.add_argument("--device-kind", required=True, help="Device kind (e.g., gopro, dummy_camera)")
    add_rig.add_argument("--roles", default="", help="Comma-separated roles")
    add_rig.set_defaults(func=cmd_registry_add_rig)
    add_scene = registry_add_sub.add_parser("scene", help="Add a scene")
    add_scene.add_argument("scene_id", type=str, help="Scene id")
    add_scene.add_argument("--name", required=True, help="Scene name")
    add_scene.add_argument("--tags", default="", help="Comma-separated tags")
    add_scene.set_defaults(func=cmd_registry_add_scene)
    add_tactition = registry_add_sub.add_parser("tactition", help="Add a tactition")
    add_tactition.add_argument("tactition_id", type=str, help="Tactition id")
    add_tactition.add_argument("--name", required=True, help="Tactition name")
    add_tactition.add_argument("--email", default="", help="Email")
    add_tactition.set_defaults(func=cmd_registry_add_tactition)

    identity = sub.add_parser("identity", help="Recorder identity")
    identity_sub = identity.add_subparsers(dest="identity_command", required=True)
    identity_show = identity_sub.add_parser("show", help="Show recorder identity")
    identity_show.set_defaults(func=cmd_identity_show)
    identity_set = identity_sub.add_parser("set", help="Update recorder identity")
    identity_set.add_argument("--operator-id", type=str, help="Operator id")
    identity_set.add_argument("--operator-name", type=str, help="Operator name")
    identity_set.set_defaults(func=cmd_identity_set)

    upload = sub.add_parser("upload", help="Upload sessions to Drive")
    upload_sub = upload.add_subparsers(dest="upload_command", required=True)
    upload_session = upload_sub.add_parser("session", help="Upload a session folder")
    upload_session.add_argument("--session-id", type=str, help="Session id (default: active)")
    upload_session.add_argument("--no-recordings", action="store_true", help="Skip take recordings/metadata")
    upload_session.add_argument("--format-version", type=str, default="v1", help="Format version folder (default: v1)")
    upload_session.set_defaults(func=cmd_upload_session)
    upload_list = upload_sub.add_parser("list", help="List uploaded sessions from registry")
    upload_list.add_argument("--session-id", type=str, help="Filter by session id or uid")
    upload_list.add_argument("--force", action="store_true", help="Force re-download uploads.jsonl")
    upload_list.set_defaults(func=cmd_upload_list)
    upload_show = upload_sub.add_parser("show", help="Show uploaded files for a session")
    upload_show.add_argument("session_id", type=str, help="Session id or uid")
    upload_show.add_argument("--force", action="store_true", help="Force re-download uploads.jsonl")
    upload_show.set_defaults(func=cmd_upload_show)
    upload_stats = upload_sub.add_parser("stats", help="Aggregate upload stats")
    upload_stats.add_argument("--by", choices=["session", "operator", "recorder", "scene", "rig"], default="session")
    upload_stats.add_argument("--force", action="store_true", help="Force re-download uploads.jsonl")
    upload_stats.set_defaults(func=cmd_upload_stats)

    ingest = sub.add_parser("ingest", help="Ingest media into session folders")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)
    ingest_session = ingest_sub.add_parser("session", help="Ingest a session's linked recordings")
    ingest_session.add_argument("--session-id", type=str, help="Session id (default: active)")
    ingest_session.add_argument("--kind", choices=["gopro", "dummy_camera"], help="Filter by device kind")
    ingest_session.add_argument("--usb", action="store_true", help="Allow USB HTTP for GoPro offload")
    ingest_session.add_argument("--verify", action="store_true", help="Verify sha256 after ingest")
    ingest_session.add_argument("--telemetry", action="store_true", help="Extract telemetry exports after ingest")
    ingest_session.add_argument("--telemetry-force", action="store_true", help="Re-extract telemetry exports")
    ingest_session.set_defaults(func=cmd_ingest_session)
    ingest_verify = ingest_sub.add_parser("verify", help="Verify ingested recordings")
    ingest_verify.add_argument("--session-id", type=str, help="Session id (default: active)")
    ingest_verify.add_argument("--kind", choices=["gopro", "dummy_camera"], help="Filter by device kind")
    ingest_verify.set_defaults(func=cmd_ingest_verify)

    telemetry = sub.add_parser("telemetry", help="Telemetry utilities")
    telemetry_sub = telemetry.add_subparsers(dest="telemetry_command", required=True)
    telemetry_extract = telemetry_sub.add_parser("extract", help="Extract telemetry exports (GoPro)")
    telemetry_extract.add_argument("--session-id", type=str, help="Session id (default: active)")
    telemetry_extract.add_argument("--force", action="store_true", help="Re-extract telemetry exports")
    telemetry_extract.set_defaults(func=cmd_telemetry_extract)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
