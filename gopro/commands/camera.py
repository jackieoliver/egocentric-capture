#!/usr/bin/env python3
"""
Per-camera control commands.

Handles individual camera operations: status, wake, reconnect, settings.
"""

from __future__ import annotations

import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from rich.console import Console

from recording.models import DeviceState
from recording.store import get_session_store

from ..state import get_inventory
from .. import http as gopro_http
from ..sd import format_sd_card_status_rich, sd_card_status_from_values
from ..ble_daemon import DEFAULT_SOCKET_PATH, request_daemon, start_daemon
from ..gopro_daemon import (
    DEFAULT_SOCKET_PATH as GOPRO_DAEMON_SOCKET_PATH,
    request_gopro_daemon,
    start_gopro_daemon,
)
from ..preview import run_preview, run_pose_preview, resolve_vision_detector_path

console = Console()


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def _ensure_daemon() -> bool:
    try:
        status = request_daemon(method="status", socket_path=DEFAULT_SOCKET_PATH)
        return bool(status.get("running"))
    except Exception:
        pass
    console.print("[yellow]BLE daemon not running; starting...[/yellow]")
    start_daemon()
    time.sleep(1.0)
    try:
        status = request_daemon(method="status", socket_path=DEFAULT_SOCKET_PATH)
        return bool(status.get("running"))
    except Exception:
        return False


def _resolve_camera(identifier: str):
    """Resolve camera by serial, short_id, or role."""
    inventory = get_inventory()

    # Try by role first
    camera = inventory.get_by_role(identifier)
    if camera:
        return camera

    # Try by serial/short_id
    camera = inventory.get(identifier)
    return camera


def cmd_status(args: argparse.Namespace) -> None:
    """Show detailed status for a single camera."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    console.print(f"\n[bold]Camera {camera.short_id}[/bold]")
    console.print(f"  Serial: {camera.serial_number}")
    console.print(f"  Name: {camera.camera_name or 'unknown'}")
    console.print(f"  Role: {camera.role or 'unassigned'}")
    console.print(f"  Profile: {camera.profile or 'none'}")

    # Try to get live status
    device, transport = gopro_http.query_camera(
        camera,
        prefer_usb=False,
        allow_usb=getattr(args, "usb", False),
    )
    if device:
        console.print(f"\n[bold]Live Status ({transport})[/bold]")
        sd = sd_card_status_from_values(present=device.sd_card_present, remaining_kib=device.sd_space_remaining_kb)
        sd_cell = format_sd_card_status_rich(sd)
        console.print(f"  Battery: {device.battery_percent}%")
        console.print(f"  SD Card: {sd_cell}")
        console.print(f"  Recording: {'[red]YES[/red]' if device.is_recording else 'no'}")
        console.print(f"  Video remaining: {device.video_minutes_remaining} min")
        console.print(f"  Photos remaining: {device.photos_remaining}")
        console.print(f"  GPS lock: {'yes' if device.gps_lock else 'no'}")
    else:
        if getattr(args, "usb", False):
            console.print("\n[yellow]Camera not responding (tried COHN + USB)[/yellow]")
        else:
            console.print("\n[yellow]Camera not responding (COHN only; add --usb to allow USB)[/yellow]")

    # Connection status
    console.print(f"\n[bold]Connections[/bold]")
    console.print(f"  USB: {camera.usb.ip if camera.usb.available else 'not available'}")
    console.print(f"  BLE: {'paired' if camera.provisioning.ble_paired else 'not paired'}")
    console.print(f"  WiFi/COHN: {'enabled' if camera.provisioning.cohn_enabled else 'not enabled'}")


def cmd_identify(args: argparse.Namespace) -> None:
    """Flash/beep a camera for physical identification."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    console.print(f"[bold]Identifying {camera.role or camera.short_id}...[/bold]")

    # Try HTTP query to wake screen
    device, transport = gopro_http.query_camera(
        camera,
        prefer_usb=False,
        allow_usb=getattr(args, "usb", False),
    )
    if device:
        console.print(f"\n[green]Camera should show info on screen ({transport})[/green]")
        console.print(f"  Serial: {camera.serial_number}")
        console.print(f"  Name: {device.camera_name}")
        console.print(f"  Battery: {device.battery_percent}%")
        return

    # Try BLE locate if HTTP failed
    if camera.provisioning.ble_paired and camera.ble.address:
        console.print("[yellow]Attempting BLE locate...[/yellow]")
        # BLE locate command would go here
        console.print("[dim]BLE locate not yet implemented[/dim]")
        return

    console.print("[red]No connection available to identify camera[/red]")


def cmd_wake(args: argparse.Namespace) -> None:
    """Wake a sleeping camera via BLE."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    if not camera.provisioning.ble_paired:
        console.print("[red]Camera not BLE paired. Run 'gopro provision start' first.[/red]")
        return

    console.print(f"[bold]Waking {camera.role or camera.short_id} via BLE...[/bold]")

    if not _ensure_daemon():
        console.print("[red]BLE daemon unavailable. Try again after 'gopro provision start'.[/red]")
        return

    try:
        result = request_daemon(
            method="wake",
            params={"camera": camera.serial_number},
            socket_path=DEFAULT_SOCKET_PATH,
        )
        success = bool(result.get("connected"))
    except Exception as e:
        console.print(f"[red]BLE error: {e}[/red]")
        success = False

    if success:
        console.print("[green]Wake signal sent[/green]")

        # Update session state if active
        store = get_session_store()
        session = store.get_active()
        if session:
            store.update_device_state(session, device_id=camera.serial_number, state=DeviceState.WAKING)

        console.print("[dim]Camera may take a few seconds to wake. Check status.[/dim]")
    else:
        console.print("[red]Failed to wake camera[/red]")


def cmd_reconnect(args: argparse.Namespace) -> None:
    """Attempt to reconnect to a camera."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    console.print(f"[bold]Reconnecting to {camera.role or camera.short_id}...[/bold]")

    connected = False
    connection_type = None

    # Try HTTP (COHN by default; USB opt-in)
    console.print(f"  Trying HTTP... ", end="")
    device, transport = gopro_http.query_camera(
        camera,
        prefer_usb=False,
        allow_usb=getattr(args, "usb", False),
    )
    if device:
        console.print(f"[green]connected ({transport})[/green]")
        connected = True
        connection_type = transport

        # Update inventory
        inventory = get_inventory()
        if transport == "usb":
            camera.usb.available = True
        camera.last_seen = _now()
        inventory.upsert(camera)
        inventory.save()
    else:
        console.print("[red]failed[/red]")

    # Try BLE wake if HTTP failed
    if not connected and camera.provisioning.ble_paired:
        console.print("  Trying BLE wake... ", end="")
        woke = False
        if _ensure_daemon():
            try:
                result = request_daemon(
                    method="wake",
                    params={"camera": camera.serial_number},
                    socket_path=DEFAULT_SOCKET_PATH,
                )
                woke = bool(result.get("connected"))
            except Exception:
                woke = False
        if woke:
            console.print("[green]wake sent[/green]")
            # Wait a bit and retry HTTP
            time.sleep(2)
            device, transport = gopro_http.query_camera(
                camera,
                prefer_usb=False,
                allow_usb=getattr(args, "usb", False),
            )
            if device:
                connected = True
                connection_type = transport
                console.print(f"  HTTP now [green]connected ({transport})[/green]")
        else:
            console.print("[red]failed[/red]")

    # Update session state
    store = get_session_store()
    session = store.get_active()
    if session:
        store.update_device_state(
            session,
            device_id=camera.serial_number,
            state=DeviceState.IDLE if connected else DeviceState.UNREACHABLE,
            connection=connection_type,
        )

    if connected:
        console.print(f"\n[green]Reconnected via {connection_type}[/green]")
    else:
        console.print(f"\n[red]Could not reconnect. Check camera manually.[/red]")


def cmd_settings(args: argparse.Namespace) -> None:
    """Show current camera settings."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    console.print(f"[bold]Settings for {camera.role or camera.short_id}[/bold]")

    # Get HTTP connection
    client, base_url, transport = gopro_http.create_http_client(
        camera,
        prefer_usb=False,
        allow_usb=getattr(args, "usb", False),
    )
    if not client:
        console.print("[red]No HTTP connection available[/red]")
        return

    console.print(f"[dim]via {transport}[/dim]")

    try:
        resp = client.get(f"{base_url}/gopro/camera/state")
        if resp.status_code == 200:
            state = resp.json()
            settings = state.get("settings", {})

            # Show key settings
            key_settings = {
                "2": "Resolution",
                "3": "FPS",
                "121": "Lens",
                "135": "HyperSmooth",
                "183": "Bit Depth",
                "232": "Framing",
            }

            console.print("\n[bold]Key Settings:[/bold]")
            for setting_id, name in key_settings.items():
                value = settings.get(setting_id, "?")
                console.print(f"  {name}: {value}")

            if args.all:
                console.print("\n[bold]All Settings:[/bold]")
                for k, v in sorted(settings.items(), key=lambda x: int(x[0])):
                    console.print(f"  {k}: {v}")
        else:
            console.print(f"[red]Failed to get settings: {resp.status_code}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        gopro_http.close_http_client(client)


def cmd_record_start(args: argparse.Namespace) -> None:
    """Start recording on a camera (via gopro-daemon)."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    data = request_gopro_daemon(
        method="record_start",
        params={
            "camera": camera.serial_number,
            "session_id": args.session_id,
            "take_id": args.take_id,
            "output_root": args.output_root,
        },
        socket_path=GOPRO_DAEMON_SOCKET_PATH,
    )
    if args.json:
        import json
        print(json.dumps(data, indent=2))
        return

    if data.get("ok"):
        console.print(f"[green]recording[/green] {camera.role or camera.short_id} ({data.get('transport')})")
        if data.get("recording_id"):
            console.print(f"[dim]recording_id={data.get('recording_id')}[/dim]")
    else:
        console.print(f"[red]failed[/red] {camera.role or camera.short_id} ({data.get('msg')})")


def cmd_record_stop(args: argparse.Namespace) -> None:
    """Stop recording on a camera (via gopro-daemon)."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    data = request_gopro_daemon(
        method="record_stop",
        params={
            "camera": camera.serial_number,
            "wait_seconds": args.wait,
            "recording_id": args.recording_id,
            "session_id": args.session_id,
            "take_id": args.take_id,
            "output_root": args.output_root,
        },
        socket_path=GOPRO_DAEMON_SOCKET_PATH,
    )
    if args.json:
        import json
        print(json.dumps(data, indent=2))
        return

    if not data.get("ok"):
        console.print(f"[red]failed[/red] {camera.role or camera.short_id} ({data.get('msg')})")
        return

    recording_id = data.get("recording_id")
    link_status = data.get("link_status")
    media = data.get("media") or {}
    folder = media.get("folder")
    file = media.get("file")
    gumi = media.get("gumi")
    info = media.get("info") or {}
    size_bytes = info.get("s")
    if link_status == "linked" and folder and file:
        if size_bytes:
            size_mb = float(size_bytes) / (1024 * 1024)
            console.print(
                f"[green]stopped[/green] {camera.role or camera.short_id} → {folder}/{file} ({size_mb:.1f} MB)"
            )
        else:
            console.print(f"[green]stopped[/green] {camera.role or camera.short_id} → {folder}/{file}")
        if gumi:
            console.print(f"[dim]gumi={gumi}[/dim]")
    elif link_status == "pending":
        console.print(f"[green]stopped[/green] {camera.role or camera.short_id} (link pending)")
    elif link_status == "untracked":
        console.print(f"[yellow]stopped[/yellow] {camera.role or camera.short_id} (untracked)")
    else:
        console.print(f"[green]stopped[/green] {camera.role or camera.short_id}")
    if recording_id:
        console.print(f"[dim]recording_id={recording_id}[/dim]")


def cmd_record_link(args: argparse.Namespace) -> None:
    """Fetch linked media for a recording id (via gopro-daemon)."""
    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    data = request_gopro_daemon(
        method="record_link_status",
        params={
            "recording_id": args.recording_id,
            "session_id": args.session_id,
            "take_id": args.take_id,
            "output_root": args.output_root,
        },
        socket_path=GOPRO_DAEMON_SOCKET_PATH,
    )
    if args.json:
        import json
        print(json.dumps(data, indent=2))
        return
    if not data.get("ok"):
        console.print(f"[red]link status failed[/red] ({data.get('msg')})")
        return
    status = data.get("status")
    media = data.get("media") or {}
    folder = media.get("folder")
    file = media.get("file")
    if status == "linked" and folder and file:
        console.print(f"[green]linked[/green] {folder}/{file}")
    else:
        console.print(f"[yellow]{status}[/yellow] {args.recording_id}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a media file via gopro-daemon."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    folder = args.folder
    file = args.file

    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    if args.last:
        last = request_gopro_daemon(
            method="last_captured",
            params={"camera": camera.serial_number},
            socket_path=GOPRO_DAEMON_SOCKET_PATH,
        )
        folder = folder or last.get("folder")
        file = file or last.get("file")

    if not folder or not file:
        console.print("[red]folder/file required (or use --last)[/red]")
        return

    dest_dir = Path(args.dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file

    data = request_gopro_daemon(
        method="download_media",
        params={
            "camera": camera.serial_number,
            "folder": folder,
            "file": file,
            "dest_path": str(dest_path),
            "prefer_usb": False,
            "allow_usb": False,
        },
        socket_path=GOPRO_DAEMON_SOCKET_PATH,
    )

    if args.json:
        import json
        print(json.dumps(data, indent=2))
        return

    if data.get("ok"):
        size_bytes = data.get("size_bytes")
        if size_bytes:
            size_mb = float(size_bytes) / (1024 * 1024)
            console.print(f"[green]downloaded[/green] {dest_path} ({size_mb:.1f} MB)")
        else:
            console.print(f"[green]downloaded[/green] {dest_path}")
    else:
        console.print(f"[red]download failed[/red] {data.get('error')}")


def cmd_stream_start(args: argparse.Namespace) -> None:
    """Start preview stream on a camera (via gopro-daemon)."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    data = request_gopro_daemon(
        method="stream_start",
        params={
            "camera": camera.serial_number,
            "port": args.port if args.port is not None else (camera.udp_preview_port or 8554),
            "watchdog": bool(args.watchdog),
            "stall_timeout": float(args.stall_timeout),
        },
        socket_path=GOPRO_DAEMON_SOCKET_PATH,
    )

    if args.json:
        import json
        print(json.dumps(data, indent=2))
        return

    if not data.get("ok"):
        msg = data.get("msg") or "failed"
        console.print(f"[red]stream start failed[/red] {camera.role or camera.short_id} ({msg})")
        return
    port = data.get("port") or args.port or camera.udp_preview_port or 8554
    console.print(f"[green]streaming[/green] {camera.role or camera.short_id} udp://@:{port}")


def cmd_stream_stop(args: argparse.Namespace) -> None:
    """Stop preview stream on a camera (via gopro-daemon)."""
    camera = _resolve_camera(args.camera)
    if not camera:
        console.print(f"[red]Camera '{args.camera}' not found[/red]")
        return

    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    data = request_gopro_daemon(
        method="stream_stop",
        params={"camera": camera.serial_number},
        socket_path=GOPRO_DAEMON_SOCKET_PATH,
    )

    if args.json:
        import json
        print(json.dumps(data, indent=2))
        return

    if data.get("ok"):
        console.print(f"[green]stream stopped[/green] {camera.role or camera.short_id}")
    else:
        console.print(f"[red]stream stop failed[/red] {camera.role or camera.short_id} ({data.get('msg')})")


def _parse_ports(value: Optional[str]) -> list[int]:
    if not value:
        return []
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in parts]


def _parse_size(value: Optional[str]) -> Optional[tuple[int, int]]:
    if not value:
        return None
    text = value.lower().replace("x", " ")
    parts = [p for p in text.split() if p]
    if len(parts) != 2:
        raise ValueError("size must be WxH")
    return int(parts[0]), int(parts[1])


def _normalize_take_id(take_id: Optional[str]) -> Optional[str]:
    if not take_id:
        return None
    if take_id.startswith("take_"):
        return take_id
    if take_id.isdigit():
        return f"take_{int(take_id):03d}"
    return take_id


def _parse_take_number(take_id: Optional[str]) -> Optional[int]:
    if not take_id:
        return None
    if take_id.isdigit():
        return int(take_id)
    if take_id.startswith("take_"):
        tail = take_id.split("take_", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def _safe_component(value: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value.strip())
    return safe or "camera"


def _resolve_pose_output_root(
    *,
    output_root: Optional[str],
    session_id: Optional[str],
    take_id: Optional[str],
) -> Path:
    if output_root:
        return Path(output_root)
    if session_id and take_id:
        store = get_session_store()
        take_slug = _normalize_take_id(take_id) or take_id
        return store.session_dir(session_id) / "takes" / (take_slug or "take_unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("gopro") / "tmp" / "pose" / timestamp


def _emit_pose_events(
    *,
    session_id: Optional[str],
    take_id: Optional[str],
    output_root: Path,
    outputs,
    camera_meta: dict[str, Any],
) -> None:
    if not session_id or not take_id:
        return
    take_number = _parse_take_number(take_id)
    store = get_session_store()

    def _emit(kind: str, camera_id: str, path: Optional[Path], drops: Optional[int] = None) -> None:
        if not path:
            return
        size_bytes = None
        if path.exists():
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None
        try:
            relpath = str(path.relative_to(output_root))
        except ValueError:
            relpath = str(path)
        meta = camera_meta.get(camera_id)
        extra = {}
        if meta:
            extra = {"role": meta.role, "serial": meta.serial_number}
        if drops is not None:
            extra["record_drops"] = drops
        payload = {
            "device_id": camera_id,
            "kind": kind,
            "file": relpath,
        }
        if take_number is not None:
            payload["take_number"] = take_number
        if size_bytes is not None:
            payload["size_bytes"] = size_bytes
        if extra:
            payload["extra"] = extra
        if take_number is None:
            payload["take_id"] = take_id
        store.append_event(session_id, "recording_linked", payload, source="gopro.preview")

    for output in outputs:
        _emit("preview_overlay", output.camera_id, output.overlay_path, drops=output.record_drops)
        _emit("vision_landmarks", output.camera_id, output.landmarks_path)
        _emit("vision_stats", output.camera_id, output.stats_path)


def _check_udp_port(port: int) -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        return f"UDP port {port} is already in use ({exc})"
    finally:
        sock.close()
    return None


def cmd_preview(args: argparse.Namespace) -> None:
    """Open a low-latency preview viewer for one or more cameras."""
    cameras = []
    for ident in args.camera:
        cam = _resolve_camera(ident)
        if not cam:
            console.print(f"[red]Camera '{ident}' not found[/red]")
            return
        cameras.append(cam)

    ports = _parse_ports(args.ports)
    if ports and len(ports) != len(cameras):
        console.print("[red]ports count must match cameras[/red]")
        return

    resolved_ports: list[int] = []
    for idx, cam in enumerate(cameras):
        if ports:
            resolved_ports.append(ports[idx])
        elif args.port is not None and len(cameras) == 1:
            resolved_ports.append(int(args.port))
        else:
            resolved_ports.append(int(cam.udp_preview_port or 8554))

    for port in resolved_ports:
        err = _check_udp_port(port)
        if err:
            console.print(f"[red]{err}[/red]")
            return

    start_gopro_daemon(socket_path=GOPRO_DAEMON_SOCKET_PATH)
    def _start_streams() -> None:
        if args.no_start:
            return
        def _start_one(cam, port) -> None:
            request_gopro_daemon(
                method="stream_start",
                params={
                    "camera": cam.serial_number,
                    "port": port,
                    "watchdog": not args.no_watchdog,
                    "stall_timeout": float(args.stall_timeout),
                },
                socket_path=GOPRO_DAEMON_SOCKET_PATH,
            )

        max_workers = min(4, len(cameras)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_start_one, cam, port) for cam, port in zip(cameras, resolved_ports)]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    console.print(f"[red]stream-start failed: {exc}[/red]")

    size = None
    if args.size:
        try:
            size = _parse_size(args.size)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return

    backend = args.backend
    if backend == "auto":
        backend = "ffmpeg" if len(cameras) > 1 else "opencv"

    if size is None and (args.pose or backend == "ffmpeg"):
        size = (640, 360) if len(cameras) > 1 else (1280, 720)

    labels = [cam.role or cam.short_id for cam in cameras]
    camera_ids = [cam.role or cam.short_id or cam.serial_number for cam in cameras]

    if args.pose:
        try:
            vision_bin = resolve_vision_detector_path(args.vision_bin)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        output_root = _resolve_pose_output_root(
            output_root=args.output_root,
            session_id=args.session_id,
            take_id=args.take_id,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]pose outputs -> {output_root}[/dim]")

        landmarks_paths: list[Optional[Path]] = []
        stats_paths: list[Optional[Path]] = []
        overlay_paths: list[Optional[Path]] = []
        for cam_id in camera_ids:
            safe_id = _safe_component(cam_id)
            landmarks_path = output_root / "metadata" / "vision" / "gopro" / safe_id / "landmarks.jsonl"
            stats_path = output_root / "metadata" / "vision" / "gopro" / safe_id / "stats.jsonl"
            landmarks_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            landmarks_paths.append(landmarks_path)
            stats_paths.append(stats_path)
            if args.no_record_overlay:
                overlay_paths.append(None)
            else:
                overlay_path = (
                    output_root
                    / "recordings"
                    / "preview_overlay"
                    / safe_id
                    / "preview_overlay.mp4"
                )
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_paths.append(overlay_path)

        try:
            outputs = run_pose_preview(
                resolved_ports,
                labels,
                camera_ids,
                size=size,
                vision_bin=vision_bin,
                landmarks_paths=landmarks_paths,
                stats_paths=stats_paths,
                overlay_paths=overlay_paths,
                stats_overlay=not args.no_stats_overlay,
                async_detection=not args.pose_sync,
                metal=not args.no_pose_metal,
                target_fps=int(args.pose_fps),
                on_start=_start_streams,
            )
        except KeyboardInterrupt:
            outputs = []
        finally:
            if not args.no_stop:
                for cam in cameras:
                    request_gopro_daemon(
                        method="stream_stop",
                        params={"camera": cam.serial_number},
                        socket_path=GOPRO_DAEMON_SOCKET_PATH,
                    )

        camera_meta = {cam_id: cam for cam_id, cam in zip(camera_ids, cameras)}
        _emit_pose_events(
            session_id=args.session_id,
            take_id=args.take_id,
            output_root=output_root,
            outputs=outputs,
            camera_meta=camera_meta,
        )
        return

    try:
        run_preview(
            resolved_ports,
            labels,
            backend=backend,
            size=size,
            on_start=_start_streams,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_stop:
            for cam in cameras:
                request_gopro_daemon(
                    method="stream_stop",
                    params={"camera": cam.serial_number},
                    socket_path=GOPRO_DAEMON_SOCKET_PATH,
                )


def add_camera_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add camera subcommand to parser."""
    camera = subparsers.add_parser("camera", help="Per-camera operations")
    camera_sub = camera.add_subparsers(dest="camera_command", required=True)

    # status
    status = camera_sub.add_parser("status", help="Show camera status")
    status.add_argument("camera", help="Camera serial, short_id, or role")
    status.add_argument("--usb", action="store_true", help="Allow USB HTTP (opt-in)")
    status.set_defaults(func=cmd_status)

    # identify
    identify = camera_sub.add_parser("identify", help="Flash/beep camera")
    identify.add_argument("camera", help="Camera serial, short_id, or role")
    identify.add_argument("--usb", action="store_true", help="Allow USB HTTP (opt-in)")
    identify.set_defaults(func=cmd_identify)

    # wake
    wake = camera_sub.add_parser("wake", help="Wake sleeping camera via BLE")
    wake.add_argument("camera", help="Camera serial, short_id, or role")
    wake.set_defaults(func=cmd_wake)

    # reconnect
    reconnect = camera_sub.add_parser("reconnect", help="Reconnect to camera")
    reconnect.add_argument("camera", help="Camera serial, short_id, or role")
    reconnect.add_argument("--usb", action="store_true", help="Allow USB HTTP (opt-in)")
    reconnect.set_defaults(func=cmd_reconnect)

    # settings
    settings = camera_sub.add_parser("settings", help="Show camera settings")
    settings.add_argument("camera", help="Camera serial, short_id, or role")
    settings.add_argument("--all", "-a", action="store_true", help="Show all settings")
    settings.add_argument("--usb", action="store_true", help="Allow USB HTTP (opt-in)")
    settings.set_defaults(func=cmd_settings)

    record_start = camera_sub.add_parser("record-start", help="Start recording (via gopro-daemon)")
    record_start.add_argument("camera", help="Camera serial, short_id, or role")
    record_start.add_argument("--session-id", type=str, help="Session id for fetch job output")
    record_start.add_argument("--take-id", type=str, help="Take id for fetch job output")
    record_start.add_argument("--output-root", type=str, help="Take directory root for fetch job output")
    record_start.add_argument("--json", action="store_true", help="JSON output")
    record_start.set_defaults(func=cmd_record_start)

    record_stop = camera_sub.add_parser("record-stop", help="Stop recording (via gopro-daemon)")
    record_stop.add_argument("camera", help="Camera serial, short_id, or role")
    record_stop.add_argument("--wait", type=float, default=1.0, help="Seconds to wait for media list refresh")
    record_stop.add_argument("--recording-id", type=str, help="Recording id from record-start")
    record_stop.add_argument("--session-id", type=str, help="Session id for fetch job output")
    record_stop.add_argument("--take-id", type=str, help="Take id for fetch job output")
    record_stop.add_argument("--output-root", type=str, help="Take directory root for fetch job output")
    record_stop.add_argument("--json", action="store_true", help="JSON output")
    record_stop.set_defaults(func=cmd_record_stop)

    record_link = camera_sub.add_parser("record-link", help="Fetch linked media for a recording id")
    record_link.add_argument("recording_id", help="Recording id from record-start")
    record_link.add_argument("--session-id", type=str, help="Session id for fetch job output")
    record_link.add_argument("--take-id", type=str, help="Take id for fetch job output")
    record_link.add_argument("--output-root", type=str, help="Take directory root for fetch job output")
    record_link.add_argument("--json", action="store_true", help="JSON output")
    record_link.set_defaults(func=cmd_record_link)

    download = camera_sub.add_parser("download", help="Download a media file (via gopro-daemon)")
    download.add_argument("camera", help="Camera serial, short_id, or role")
    download.add_argument("--folder", type=str, help="Media folder (e.g. 100GOPRO)")
    download.add_argument("--file", type=str, help="Media file (e.g. GX010001.MP4)")
    download.add_argument("--last", action="store_true", help="Use last captured file")
    download.add_argument("--dest-dir", type=str, default="gopro/tmp", help="Download directory")
    download.add_argument("--json", action="store_true", help="JSON output")
    download.set_defaults(func=cmd_download)

    stream_start = camera_sub.add_parser("stream-start", help="Start preview stream (via gopro-daemon)")
    stream_start.add_argument("camera", help="Camera serial, short_id, or role")
    stream_start.add_argument(
        "--port",
        type=int,
        help="UDP port for preview stream (defaults to inventory udp_preview_port or 8554)",
    )
    stream_start.add_argument(
        "--no-watchdog",
        dest="watchdog",
        action="store_false",
        help="Disable daemon watchdog restart",
    )
    stream_start.set_defaults(watchdog=True)
    stream_start.add_argument("--stall-timeout", type=float, default=3.0, help="Seconds without packets before restart")
    stream_start.add_argument("--json", action="store_true", help="JSON output")
    stream_start.set_defaults(func=cmd_stream_start)

    stream_stop = camera_sub.add_parser("stream-stop", help="Stop preview stream (via gopro-daemon)")
    stream_stop.add_argument("camera", help="Camera serial, short_id, or role")
    stream_stop.add_argument("--json", action="store_true", help="JSON output")
    stream_stop.set_defaults(func=cmd_stream_stop)

    preview = camera_sub.add_parser("preview", help="Low-latency preview viewer")
    preview.add_argument("camera", nargs="+", help="Camera serial, short_id, or role")
    preview.add_argument("--port", type=int, help="UDP port override (single camera)")
    preview.add_argument("--ports", type=str, help="Comma-separated UDP ports (multi-camera)")
    preview.add_argument(
        "--backend",
        choices=["auto", "opencv", "ffmpeg"],
        default="auto",
        help="Preview backend (auto selects ffmpeg for multi-cam)",
    )
    preview.add_argument("--size", type=str, help="Force WxH output size (defaults vary by backend)")
    preview.add_argument("--pose", action="store_true", help="Enable Vision pose overlay pipeline")
    preview.add_argument("--vision-bin", type=str, help="Path to vision-detector binary")
    preview.add_argument("--pose-fps", type=int, default=30, help="Target FPS for pose pipeline")
    preview.add_argument("--pose-sync", action="store_true", help="Run pose detection synchronously")
    preview.add_argument("--no-pose-metal", action="store_true", help="Disable Metal overlay rendering")
    preview.add_argument("--no-stats-overlay", action="store_true", help="Disable stats overlay drawing")
    preview.add_argument("--no-record-overlay", action="store_true", help="Disable overlay MP4 recording")
    preview.add_argument("--session-id", type=str, help="Session id for pose output paths/events")
    preview.add_argument("--take-id", type=str, help="Take id for pose output paths/events")
    preview.add_argument("--output-root", type=str, help="Take directory root for pose outputs")
    preview.add_argument("--no-start", action="store_true", help="Do not start stream via daemon")
    preview.add_argument("--no-stop", action="store_true", help="Do not stop stream on exit")
    preview.add_argument("--no-watchdog", action="store_true", help="Disable daemon watchdog restart")
    preview.add_argument("--stall-timeout", type=float, default=3.0, help="Seconds before watchdog restart")
    preview.set_defaults(no_watchdog=True)
    preview.set_defaults(func=cmd_preview)
