#!/usr/bin/env python3
"""
Provisioning commands (daemon-first, minimal surface).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .. import http as gopro_http
from ..ble import GoProBleConnection, scan_ble_devices
from ..ble_daemon import DEFAULT_SOCKET_PATH, request_daemon, start_daemon, stop_daemon
from ..config import load_profile
from ..labs import build_profile_labs_command
from ..qr import render_qr
from ..sd import sd_card_status_from_values
from ..state import (
    CameraInventoryRecord,
    ConnectionInfo,
    ProvisioningInfo,
    ProvisioningState,
    get_inventory,
)

console = Console()


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def _daemon_running() -> bool:
    try:
        status = request_daemon(method="status", socket_path=DEFAULT_SOCKET_PATH)
    except Exception:
        return False
    return bool(status.get("running"))


def _ensure_daemon() -> bool:
    if _daemon_running():
        return True
    console.print("[yellow]BLE daemon not running; starting...[/yellow]")
    start_daemon()
    time.sleep(1.0)
    return _daemon_running()


def _resolve_cameras(identifier: Optional[str]) -> list[CameraInventoryRecord]:
    inventory = get_inventory()
    if not identifier:
        return list(inventory.cameras)
    camera = inventory.get(identifier) or inventory.get_by_role(identifier)
    if not camera:
        console.print("[red]Camera not found[/red]")
        return []
    return [camera]


def _match_inventory(
    inventory: list[CameraInventoryRecord],
    *,
    serial_suffix: Optional[str],
    ap_mac_suffix: Optional[str],
) -> Optional[CameraInventoryRecord]:
    if serial_suffix:
        for cam in inventory:
            if cam.serial_number.endswith(serial_suffix):
                return cam
    if ap_mac_suffix:
        suffix = ap_mac_suffix.lower()
        for cam in inventory:
            if cam.ap_mac_addr and cam.ap_mac_addr.lower().endswith(suffix):
                return cam
    return None


def cmd_start(args: argparse.Namespace) -> None:
    """Start daemon, sync inventory, connect all cameras, show summary."""
    if not _ensure_daemon():
        console.print("[red]BLE daemon failed to start[/red]")
        return

    try:
        request_daemon(method="sync_inventory", socket_path=DEFAULT_SOCKET_PATH)
    except Exception as exc:
        console.print(f"[yellow]BLE daemon sync failed: {exc}[/yellow]")

    inventory = get_inventory()
    scan_results = asyncio.run(scan_ble_devices(timeout=args.timeout))
    scan_by_suffix = {d.serial_suffix: d for d in scan_results if d.serial_suffix}

    table = Table(title="Provisioning Start")
    table.add_column("Serial")
    table.add_column("Role")
    table.add_column("UDP")
    table.add_column("Paired")
    table.add_column("Connected")
    table.add_column("Pairing")
    table.add_column("COHN")

    for cam in inventory.cameras:
        try:
            result = request_daemon(
                method="connect",
                params={
                    "camera": cam.serial_number,
                    "finish_pairing": not cam.provisioning.ble_paired,
                },
                socket_path=DEFAULT_SOCKET_PATH,
            )
            connected = "[green]Y[/green]" if result.get("connected") else "[yellow]?[/yellow]"
        except Exception:
            connected = "[red]x[/red]"

        scan = scan_by_suffix.get(cam.short_id)
        pairing = "-"
        if scan and scan.pairing is not None:
            pairing = "yes" if scan.pairing else "no"

        paired = "[green]Y[/green]" if cam.provisioning.ble_paired else "[yellow]?[/yellow]"
        cohn = "[green]Y[/green]" if cam.provisioning.cohn_enabled and cam.wifi.ip else "[dim]-[/dim]"

        udp_port = str(cam.udp_preview_port) if cam.udp_preview_port else "-"
        table.add_row(cam.serial_number, cam.role or "-", udp_port, paired, connected, pairing, cohn)

    console.print(table)


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan BLE and show pairing/known state."""
    inventory = get_inventory()
    results = asyncio.run(scan_ble_devices(timeout=args.timeout))

    table = Table(title="BLE Scan")
    table.add_column("Name")
    table.add_column("Suffix")
    table.add_column("Address")
    table.add_column("Pairing")
    table.add_column("Known")
    table.add_column("Role")
    table.add_column("Profile")

    for result in results:
        if result.pairing is not True:
            continue
        match = _match_inventory(
            inventory.cameras,
            serial_suffix=result.serial_suffix,
            ap_mac_suffix=result.ap_mac_suffix,
        )
        known = "[green]yes[/green]" if match else "[yellow]no[/yellow]"
        role = match.role if match else "-"
        profile = match.profile if match else "-"
        addr = result.address[:8] + "..." if result.address else "-"
        pairing = "yes" if result.pairing else "no"
        table.add_row(result.name or "-", result.serial_suffix or "-", addr, pairing, known, role, profile)

    console.print(table)


def cmd_adopt(args: argparse.Namespace) -> None:
    """Adopt a pairing-mode camera into inventory."""
    inventory = get_inventory()
    results = asyncio.run(scan_ble_devices(timeout=args.timeout))

    target = None
    for result in results:
        if result.serial_suffix and result.serial_suffix.endswith(args.suffix):
            target = result
            break

    if not target:
        console.print("[red]No matching camera found in pairing mode[/red]")
        return

    match = _match_inventory(
        inventory.cameras,
        serial_suffix=target.serial_suffix,
        ap_mac_suffix=target.ap_mac_suffix,
    )
    if match:
        record = match
    else:
        record = CameraInventoryRecord(
            serial_number=target.serial_suffix,
            camera_name=target.name or None,
            provisioning=ProvisioningInfo(
                state=ProvisioningState.UNKNOWN,
                discovered_at=_now(),
                discovered_via="ble",
            ),
        )

    record.ble = ConnectionInfo(available=True, address=target.address)
    record.last_seen = _now()
    if args.role:
        record.role = args.role
    if args.profile:
        record.profile = args.profile
        profile = load_profile(args.profile)
        if profile and profile.get("udp_preview_port"):
            record.udp_preview_port = int(profile["udp_preview_port"])
    if target.name and not record.camera_name:
        record.camera_name = target.name

    inventory.upsert(record)
    inventory.save()
    console.print(f"[green]Adopted {record.short_id}[/green]")


def cmd_connect(args: argparse.Namespace) -> None:
    """Connect BLE daemon to all inventory cameras."""
    if not _ensure_daemon():
        console.print("[red]BLE daemon failed to start[/red]")
        return

    try:
        request_daemon(method="sync_inventory", socket_path=DEFAULT_SOCKET_PATH)
    except Exception as exc:
        console.print(f"[yellow]BLE daemon sync failed: {exc}[/yellow]")

    cameras = _resolve_cameras(None)

    table = Table(title="BLE Connect")
    table.add_column("Serial")
    table.add_column("Role")
    table.add_column("Connected")
    table.add_column("Error")

    for cam in cameras:
        try:
            result = request_daemon(
                method="connect",
                params={
                    "camera": cam.serial_number,
                    "finish_pairing": not cam.provisioning.ble_paired,
                },
                socket_path=DEFAULT_SOCKET_PATH,
            )
            connected = "[green]Y[/green]" if result.get("connected") else "[yellow]?[/yellow]"
            error = result.get("last_error") or "-"
        except Exception as exc:
            connected = "[red]x[/red]"
            error = str(exc)
        table.add_row(cam.serial_number, cam.role or "-", connected, error)

    console.print(table)


def cmd_health(args: argparse.Namespace) -> None:
    """Fetch COHN HTTP health (battery/storage/recording)."""
    inventory = get_inventory()
    cameras = _resolve_cameras(args.serial)
    if not cameras:
        return

    if args.wake and _ensure_daemon():
        for cam in cameras:
            try:
                request_daemon(
                    method="wake",
                    params={"camera": cam.serial_number},
                    socket_path=DEFAULT_SOCKET_PATH,
                )
            except Exception:
                pass
        time.sleep(max(0.5, float(args.wake_delay)))

    table = Table(title="Camera Health")
    table.add_column("Serial")
    table.add_column("Role")
    table.add_column("Battery")
    table.add_column("SD")
    table.add_column("Remaining")
    table.add_column("Rec")
    table.add_column("Transport")

    for cam in cameras:
        device, transport = gopro_http.query_camera(
            cam,
            prefer_usb=False,
            allow_usb=False,
        )
        if not device:
            table.add_row(cam.serial_number, cam.role or "-", "-", "-", "-", "-", "-")
            continue
        sd = sd_card_status_from_values(present=device.sd_card_present, remaining_kib=device.sd_space_remaining_kb)
        if sd.status == "missing":
            sd_status = "missing"
            remaining_cell = "-"
        elif sd.status == "full":
            sd_status = "FULL"
            remaining_cell = f"{sd.remaining_gib:.1f} GiB" if sd.remaining_gib is not None else "-"
        elif sd.status == "ok":
            sd_status = "yes"
            remaining_cell = f"{sd.remaining_gib:.1f} GiB" if sd.remaining_gib is not None else "-"
        else:
            sd_status = "-"
            remaining_cell = "-"
        table.add_row(
            cam.serial_number,
            cam.role or "-",
            f"{device.battery_percent}%",
            sd_status,
            remaining_cell,
            "yes" if device.is_recording else "no",
            transport or "-",
        )

    console.print(table)


def _apply_one(camera: CameraInventoryRecord, args: argparse.Namespace, inventory) -> None:
    if not args.force and camera.provisioning.settings_applied and camera.provisioning.labs_applied:
        console.print(f"[dim]{camera.short_id}: already applied[/dim]")
        return

    if args.role:
        camera.role = args.role
    if args.profile:
        camera.profile = args.profile
    if not camera.profile:
        console.print(f"[yellow]Skipping {camera.short_id}: no profile assigned[/yellow]")
        return

    profile = load_profile(camera.profile)
    if not profile:
        console.print(f"[red]Profile '{camera.profile}' not found for {camera.short_id}[/red]")
        return
    if profile.get("udp_preview_port"):
        camera.udp_preview_port = int(profile["udp_preview_port"])

    apply_settings = args.force or not camera.provisioning.settings_applied
    if apply_settings:
        applied, skipped, transport = gopro_http.apply_profile_settings_camera(
            camera,
            profile,
            prefer_usb=False,
            allow_usb=False,
        )
        if not transport:
            console.print(f"[red]{camera.short_id}: no HTTP connection available[/red]")
            return

        console.print(f"[green]{camera.short_id}: applied via {transport}[/green]")
        if applied:
            console.print(f"  Applied: {', '.join(applied)}")
        if skipped:
            console.print(f"  Skipped: {', '.join(skipped)}")

        camera.provisioning.settings_applied = True
        camera.provisioning.state = ProvisioningState.SETTINGS_APPLIED
    else:
        console.print(f"[dim]{camera.short_id}: settings already applied[/dim]")

    command, labels = build_profile_labs_command(
        profile,
        profile_name=camera.profile,
        handle_id=camera.handle_id,
    )
    apply_labs = args.force or not camera.provisioning.labs_applied
    if command and apply_labs:
        output_path = Path(args.qr_output) if args.qr_output else None
        result = render_qr(
            command,
            title=f"Labs Settings ({camera.short_id})",
            output_path=output_path,
            write_latest=not args.qr_no_latest,
            ascii_output=args.qr_ascii,
            terminal_image_output=args.qr_terminal_image,
            terminal_protocol=args.qr_terminal_protocol,
        )
        if result.get("terminal_image"):
            print(result["terminal_image"])
        if result.get("terminal_image_error"):
            console.print(f"[yellow]{result['terminal_image_error']}[/yellow]")
        if result.get("ascii"):
            print(result["ascii"])
        if result.get("output_path"):
            console.print(f"[yellow]Labs QR saved to {result['output_path']}[/yellow]")
        if result.get("latest_path") and result.get("latest_path") != result.get("output_path"):
            console.print(f"[yellow]Latest: {result['latest_path']}[/yellow]")
        if labels:
            console.print(f"  Includes: {', '.join(labels)}")
        console.print(f"[dim]Hint: gopro labs apply --profile {camera.profile} --camera {camera.short_id}[/dim]")
        waited = False
        if not args.no_wait_labs:
            input("  Scan the QR code on the camera, then press Enter to continue...")
            waited = True
        if waited or args.mark_labs:
            camera.provisioning.labs_applied = True
            camera.provisioning.labs_applied_at = _now()
            camera.provisioning.state = ProvisioningState.LABS_APPLIED
        else:
            console.print("[dim]Labs not marked applied (use --mark-labs to force)[/dim]")
    elif command:
        console.print(f"[dim]{camera.short_id}: labs already applied[/dim]")

    inventory.upsert(camera)
    inventory.save()


def cmd_apply(args: argparse.Namespace) -> None:
    """Apply profile settings and generate Labs QR if needed."""
    inventory = get_inventory()
    if args.all:
        if args.role or args.profile:
            console.print("[yellow]--all ignores --role/--profile; using inventory profiles[/yellow]")
        for camera in inventory.cameras:
            _apply_one(camera, args, inventory)
        return

    cameras = _resolve_cameras(args.serial)
    if not cameras:
        return
    _apply_one(cameras[0], args, inventory)


def cmd_rename(args: argparse.Namespace) -> None:
    """Set the camera's advertised name over BLE and update inventory."""
    inventory = get_inventory()
    camera = inventory.get(args.camera) or inventory.get_by_role(args.camera)
    if not camera:
        console.print("[red]Camera not found[/red]")
        return

    if not _ensure_daemon():
        console.print("[red]BLE daemon failed to start[/red]")
        return

    try:
        request_daemon(method="sync_inventory", socket_path=DEFAULT_SOCKET_PATH)
    except Exception as exc:
        console.print(f"[yellow]BLE daemon sync failed: {exc}[/yellow]")

    def _send_rename() -> dict:
        return request_daemon(
            method="set_camera_name",
            params={
                "camera": camera.serial_number,
                "name": args.name,
                "claim_control": not args.no_claim,
                "timeout": args.timeout,
            },
            socket_path=DEFAULT_SOCKET_PATH,
            timeout=args.timeout + 5.0,
        )

    try:
        result = _send_rename()
    except Exception as exc:
        result = {}
        if "unknown_method:set_camera_name" in str(exc):
            console.print("[yellow]BLE daemon is outdated; restarting...[/yellow]")
            stop_daemon()
            if not _ensure_daemon():
                console.print("[red]BLE daemon failed to restart[/red]")
                return
            try:
                result = _send_rename()
            except Exception as retry_exc:
                exc = retry_exc

        if not result:
            console.print(f"[yellow]Rename via daemon failed: {exc}[/yellow]")
            stop_daemon()
            try:
                ok = asyncio.run(
                    _rename_direct_ble(
                        camera,
                        args.name,
                        claim_control=not args.no_claim,
                        timeout=args.timeout,
                    )
                )
            except Exception as direct_exc:
                console.print(f"[red]Rename failed: {direct_exc}[/red]")
                _ensure_daemon()
                return
            _ensure_daemon()
            if ok:
                camera.camera_name = args.name
                inventory.upsert(camera)
                inventory.save()
                console.print(f"[green]{camera.short_id}: camera name set to '{args.name}'[/green]")
                console.print(f"[dim]updated {inventory.path}[/dim]")
            else:
                console.print(f"[red]{camera.short_id}: rename failed[/red]")
            return

    if not result.get("ok"):
        console.print(f"[red]{camera.short_id}: rename failed[/red]")
        return

    updated = get_inventory().get(camera.serial_number)
    updated_name = updated.camera_name if updated else args.name
    console.print(f"[green]{camera.short_id}: camera name set to '{updated_name}'[/green]")
    if result.get("previous_name") and result.get("previous_name") != updated_name:
        console.print(f"[dim]was '{result.get('previous_name')}'[/dim]")
    if result.get("inventory_updated"):
        console.print(f"[dim]updated {inventory.path}[/dim]")


async def _rename_direct_ble(
    camera: CameraInventoryRecord,
    name: str,
    *,
    claim_control: bool,
    timeout: float,
) -> bool:
    conn = GoProBleConnection(
        serial=camera.serial_number,
        identifier=camera.short_id,
        camera_name=camera.camera_name,
        address=None,
    )
    try:
        await asyncio.wait_for(conn.connect(timeout=timeout, retries=1), timeout=timeout + 5.0)
        await asyncio.wait_for(conn.wait_ready(), timeout=timeout)
        if claim_control:
            await asyncio.wait_for(conn.claim_control(), timeout=timeout)
        return await asyncio.wait_for(conn.set_camera_name(name), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("timeout") from exc
    finally:
        try:
            await conn.disconnect()
        except Exception:
            pass


def add_provision_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add provision subcommand to parser."""
    provision = subparsers.add_parser("provision", help="Provisioning (daemon-first)")
    provision_sub = provision.add_subparsers(dest="provision_command", required=True)

    start = provision_sub.add_parser("start", help="Start daemon and connect all cameras")
    start.add_argument("--timeout", type=float, default=3.0)
    start.set_defaults(func=cmd_start)

    scan = provision_sub.add_parser("scan", help="Scan for pairing cameras")
    scan.add_argument("--timeout", type=float, default=3.0)
    scan.set_defaults(func=cmd_scan)

    adopt = provision_sub.add_parser("adopt", help="Adopt a pairing camera by suffix")
    adopt.add_argument("suffix", help="Serial suffix (last 4 digits)")
    adopt.add_argument("--role", type=str, help="Assign role")
    adopt.add_argument("--profile", type=str, help="Assign profile")
    adopt.add_argument("--timeout", type=float, default=3.0)
    adopt.set_defaults(func=cmd_adopt)

    connect = provision_sub.add_parser("connect", help="Connect daemon to all cameras")
    connect.add_argument("--all", action="store_true", help="Connect all inventory cameras (default)")
    connect.set_defaults(func=cmd_connect)

    health = provision_sub.add_parser("health", help="COHN health check")
    health.add_argument("serial", nargs="?", help="Camera serial or role")
    health.add_argument("--wake", action="store_true", help="Wake via BLE before HTTP")
    health.add_argument("--wake-delay", type=float, default=2.0)
    health.set_defaults(func=cmd_health)

    apply_cmd = provision_sub.add_parser("apply", help="Apply settings and Labs")
    apply_cmd.add_argument("serial", nargs="?", help="Camera serial or role")
    apply_cmd.add_argument("--role", type=str, help="Assign role")
    apply_cmd.add_argument("--profile", type=str, help="Settings profile")
    apply_cmd.add_argument("--no-wait-labs", action="store_true")
    apply_cmd.add_argument("--mark-labs", action="store_true", help="Mark labs applied without waiting")
    apply_cmd.add_argument("--qr-output", type=str, help="Save Labs QR to a specific path")
    apply_cmd.add_argument("--qr-ascii", action="store_true", help="Print Labs QR in terminal")
    apply_cmd.add_argument("--qr-no-latest", action="store_true", help="Do not write latest_qr.png")
    apply_cmd.add_argument("--qr-terminal-image", action="store_true", help="Render Labs QR as terminal image")
    apply_cmd.add_argument("--qr-terminal-protocol", choices=["auto", "kitty", "iterm2"], default="auto")
    apply_cmd.add_argument("--force", action="store_true", help="Reapply settings/labs")
    apply_cmd.add_argument("--all", action="store_true", help="Apply to all inventory cameras")
    apply_cmd.set_defaults(func=cmd_apply)

    rename = provision_sub.add_parser("rename", help="Set camera name over BLE and update inventory")
    rename.add_argument("camera", help="Camera serial, short id, or role")
    rename.add_argument("name", help="New camera name")
    rename.add_argument("--no-claim", action="store_true", help="Do not claim external control before renaming")
    rename.add_argument("--timeout", type=float, default=20.0, help="Timeout (seconds) for BLE operations")
    rename.set_defaults(func=cmd_rename)
