#!/usr/bin/env python3
"""
GoPro Labs command utilities.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..config import load_profile
from ..labs import HAPTICA_NAMES, SETUP_FEATURES, LabsFeature, build_profile_labs_command
from ..qr import render_qr
from ..state import ProvisioningState, get_inventory

console = Console()

NO_WAIT_ENV_VARS = ("HAPTICA_LABS_NO_WAIT", "HAPTICA_LABS_NONBLOCKING")


def _should_wait(args: argparse.Namespace) -> bool:
    if args.no_wait:
        return False
    for key in NO_WAIT_ENV_VARS:
        value = os.getenv(key, "")
        if value.lower() in {"1", "true", "yes", "y"}:
            return False
    return True


def _mark_labs_applied(camera_id: Optional[str]) -> None:
    if not camera_id:
        return
    inventory = get_inventory()
    camera = inventory.get(camera_id) or inventory.get_by_role(camera_id)
    if not camera:
        console.print(f"[yellow]Camera '{camera_id}' not found in inventory; not marking labs applied[/yellow]")
        return
    camera.provisioning.labs_applied = True
    camera.provisioning.labs_applied_at = camera.provisioning.labs_applied_at or _now()
    if camera.provisioning.state == ProvisioningState.SETTINGS_APPLIED:
        camera.provisioning.state = ProvisioningState.LABS_APPLIED
    inventory.upsert(camera)
    inventory.save()


def _now() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="microseconds")


def _render_and_print(
    command: str,
    *,
    title: Optional[str],
    output: Optional[str],
    write_latest: bool,
    ascii_output: bool,
    terminal_image_output: bool,
    terminal_protocol: str | None,
) -> None:
    out_path = Path(output) if output else None
    result = render_qr(
        command,
        title=title,
        output_path=out_path,
        write_latest=write_latest,
        ascii_output=ascii_output,
        terminal_image_output=terminal_image_output,
        terminal_protocol=terminal_protocol,
    )
    if result.get("terminal_image"):
        print(result["terminal_image"])
    if result.get("terminal_image_error"):
        print(result["terminal_image_error"])
    if result.get("ascii"):
        print(result["ascii"])
    if result.get("output_path"):
        print(f"Saved to: {result['output_path']}")
    if result.get("latest_path") and result.get("latest_path") != result.get("output_path"):
        print(f"Latest: {result['latest_path']}")


def _output_dir(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def _maybe_wait(args: argparse.Namespace) -> bool:
    if not _should_wait(args):
        return False
    input("Scan the QR code on the camera, then press Enter to continue...")
    return True


def cmd_list(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("GOPRO LABS FEATURES")
    print("=" * 60)

    print("\n[SETUP FEATURES]")
    for feature in SETUP_FEATURES:
        marker = " [CRITICAL]" if feature.category == "critical" else ""
        print(f"  {feature.command:15} - {feature.name}{marker}")
        print(f"                    {feature.description}")

    print("\n[CAMERA HANDLES]")
    print("  *HNDL=1..31      - Assign camera ID for multi-camera setups")

    print("\n[WIFI PROVISIONING]")
    print("  !MJOIN=\"SSID:pass\" - Store WiFi credentials permanently")
    print("  !W               - Join stored WiFi network (STA mode)")
    print("  Combined:        !MJOIN=\"SSID:pass\"!W")

    print("\n[CAMERA NAMING (Haptica)]")
    print("  !OWNR=\"name\"     - Set owner name displayed on camera LCD")
    print("  Predefined positions:")
    for pos, name in HAPTICA_NAMES.items():
        print(f"    {pos}: {name}")

    print("\n[COMMON COMMANDS]")
    print("  !PA              - Start BLE pairing (Quik)")
    print("  !PR              - Start BLE remote pairing")
    print("  !PS              - Cancel BLE pairing")
    print("  !S               - Start recording")
    print("  !E               - Stop recording")
    print("  !MSYNC           - Sync time to GPS")
    print("  mV               - Video mode")
    print("  mP               - Photo mode")
    print("  r4               - 4K resolution")
    print("  p60              - 60fps")

    print()


def cmd_setup(args: argparse.Namespace) -> None:
    features = SETUP_FEATURES.copy()

    if args.handle:
        handle_num = min(max(args.handle, 1), 31)
        features.insert(0, LabsFeature(
            name=f"Camera Handle {handle_num}",
            command=f"*HNDL={handle_num}",
            description=f"Assign camera ID {handle_num} for multi-camera identification.",
            category="handle",
        ))

    if args.category:
        features = [f for f in features if f.category == args.category or f.category == "critical"]

    if args.combined:
        combined_cmd = "".join(f.command for f in features)
        title = args.title or "GoPro Labs Setup"
        print("\n" + "=" * 60)
        print("COMBINED SETUP QR CODE")
        print("=" * 60)
        print(f"Command: {combined_cmd}\n")

        _render_and_print(
            combined_cmd,
            title=title,
            output=args.output,
            write_latest=not args.no_latest,
            ascii_output=args.ascii,
            terminal_image_output=args.terminal_image,
            terminal_protocol=args.terminal_protocol,
        )
        waited = _maybe_wait(args)
        if waited or args.mark:
            _mark_labs_applied(args.camera)
        return

    print("\n" + "=" * 60)
    print("GOPRO LABS SETUP QR CODES")
    print("=" * 60)
    print("\nScan these QR codes to configure your GoPro for egocentric recording.\n")

    out_dir = _output_dir(args.output)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, feature in enumerate(features, 1):
        category_marker = "[CRITICAL]" if feature.category == "critical" else ""
        print(f"\n{'-' * 60}")
        print(f"[{i}] {feature.name} {category_marker}")
        print(f"    Command: {feature.command}")
        print(f"    {feature.description}\n")

        out_path = None
        if out_dir:
            safe_name = feature.name.lower().replace(" ", "_").replace("-", "_")[:30]
            out_path = str(out_dir / f"setup_{i:02d}_{safe_name}.png")
        _render_and_print(
            feature.command,
            title=feature.name,
            output=out_path,
            write_latest=not args.no_latest,
            ascii_output=args.ascii,
            terminal_image_output=args.terminal_image,
            terminal_protocol=args.terminal_protocol,
        )

    print(f"\n{'=' * 60}")
    print("TIP: Use --combined to generate a single QR with all settings")
    print("=" * 60 + "\n")

    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_handles(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("CAMERA HANDLE QR CODES")
    print("=" * 60)
    print("\nUse these to assign unique IDs to cameras in multi-camera setups.\n")

    max_handle = args.count if args.count else 8
    out_dir = _output_dir(args.output)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, max_handle + 1):
        command = f"*HNDL={i}"
        title = f"Camera Handle {i}"
        print(f"\n{'-' * 60}")
        print(f"Camera {i}: {command}\n")

        out_path = str(out_dir / f"handle_{i:02d}.png") if out_dir else None
        _render_and_print(
            command,
            title=title,
            output=out_path,
            write_latest=not args.no_latest,
            ascii_output=args.ascii,
            terminal_image_output=args.terminal_image,
            terminal_protocol=args.terminal_protocol,
        )

    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_wifi(args: argparse.Namespace) -> None:
    ssid = args.ssid
    password = args.password
    command = f'!MJOIN="{ssid}:{password}"!W'
    title = args.title or f"WiFi: Join {ssid}"

    print("\n[WIFI PROVISIONING QR]")
    print(f"  SSID: {ssid}")
    print(f"  Command: {command}\n")

    _render_and_print(
        command,
        title=title,
        output=args.output,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_owner(args: argparse.Namespace) -> None:
    name = args.name

    if name.lower() in HAPTICA_NAMES:
        display_name = HAPTICA_NAMES[name.lower()]
    else:
        display_name = name

    if len(display_name) > 63:
        display_name = display_name[:63]
        console.print(f"[yellow]Name truncated to 63 chars: {display_name}[/yellow]")

    command = f'*OWNR="{display_name}"'
    title = args.title or f"Set Camera Name: {display_name}"

    print("\n[CAMERA NAME QR]")
    print(f"  Name: {display_name}")
    print(f"  Command: {command}\n")

    _render_and_print(
        command,
        title=title,
        output=args.output,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )

    print("\nPredefined mount positions:")
    for pos, haptica_name in HAPTICA_NAMES.items():
        print(f"  {pos}: {haptica_name}")

    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_command(args: argparse.Namespace) -> None:
    command = args.command
    title = args.title or "GoPro Labs Command"

    _render_and_print(
        command,
        title=title,
        output=args.output,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_feature(args: argparse.Namespace) -> None:
    token = args.feature.strip().lower()
    feature = None
    for candidate in SETUP_FEATURES:
        if candidate.command.lower() == token:
            feature = candidate
            break
        if candidate.name.lower() == token:
            feature = candidate
            break
    if not feature:
        console.print(f"[red]Labs feature '{args.feature}' not found[/red]")
        return

    _render_and_print(
        feature.command,
        title=feature.name,
        output=args.output,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_apply(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    if not profile:
        console.print(f"[red]Profile '{args.profile}' not found[/red]")
        return

    command, labels = build_profile_labs_command(
        profile,
        profile_name=args.profile,
        handle_id=args.handle,
    )
    if not command:
        console.print("[yellow]No Labs settings found in profile[/yellow]")
        return

    title = args.title or f"Labs Settings ({args.profile})"
    _render_and_print(
        command,
        title=title,
        output=args.output,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    if labels:
        console.print(f"[dim]Includes: {', '.join(labels)}[/dim]")

    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_timesync(args: argparse.Namespace) -> None:
    command = "!MSYNC"
    title = args.title or "GPS Time Sync"
    _render_and_print(
        command,
        title=title,
        output=args.output,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    waited = _maybe_wait(args)
    if waited or args.mark:
        _mark_labs_applied(args.camera)


def cmd_confirm(args: argparse.Namespace) -> None:
    _mark_labs_applied(args.camera)
    console.print(f"[green]Marked labs applied for {args.camera}[/green]")


def add_labs_parser(subparsers: argparse._SubParsersAction) -> None:
    labs = subparsers.add_parser("labs", help="GoPro Labs helpers (QR output)")
    labs_sub = labs.add_subparsers(dest="labs_command", required=True)

    def _add_common(parser: argparse.ArgumentParser, *, output_help: str = "Save QR code to PNG file") -> None:
        parser.add_argument("--output", "-o", help=output_help)
        parser.add_argument("--ascii", action="store_true", help="Print QR in terminal")
        parser.add_argument("--terminal-image", action="store_true", help="Render QR as terminal image")
        parser.add_argument("--terminal-protocol", choices=["auto", "kitty", "iterm2"], default="auto")
        parser.add_argument("--no-latest", action="store_true", help="Do not write latest_qr.png")
        parser.add_argument("--no-wait", action="store_true", help="Do not wait for operator confirmation")
        parser.add_argument("--camera", type=str, help="Camera serial or role to mark in inventory")
        parser.add_argument("--mark", action="store_true", help="Mark labs_applied without waiting")
        parser.add_argument("--title", type=str, help="Caption title")

    list_cmd = labs_sub.add_parser("list", help="List Labs features")
    list_cmd.set_defaults(func=cmd_list)

    setup_cmd = labs_sub.add_parser("setup", help="Generate setup QR codes")
    setup_cmd.add_argument("--handle", "-H", type=int, help="Include camera handle (1-31)")
    setup_cmd.add_argument(
        "--category",
        "-c",
        choices=["critical", "sync", "power", "ble"],
        help="Filter by category",
    )
    setup_cmd.add_argument("--combined", "-C", action="store_true", help="Generate single combined QR code")
    _add_common(setup_cmd, output_help="Output path (file for --combined, directory for multi)")
    setup_cmd.set_defaults(func=cmd_setup)

    handles_cmd = labs_sub.add_parser("handles", help="Generate camera handle QR codes")
    handles_cmd.add_argument("--count", "-n", type=int, default=8, help="Number of handles (default: 8)")
    _add_common(handles_cmd, output_help="Output directory for PNG files")
    handles_cmd.set_defaults(func=cmd_handles)

    wifi_cmd = labs_sub.add_parser("wifi", help="Generate WiFi provisioning QR")
    wifi_cmd.add_argument("ssid", help="WiFi network SSID")
    wifi_cmd.add_argument("password", help="WiFi network password")
    _add_common(wifi_cmd)
    wifi_cmd.set_defaults(func=cmd_wifi)

    owner_cmd = labs_sub.add_parser("owner", help="Set camera owner name via QR")
    owner_cmd.add_argument("name", help="Camera name or mount position (head, wrist_left, wrist_right, etc.)")
    _add_common(owner_cmd)
    owner_cmd.set_defaults(func=cmd_owner)

    cmd_cmd = labs_sub.add_parser("command", help="Generate Labs QR for a raw command")
    cmd_cmd.add_argument("command", help="Labs command string")
    _add_common(cmd_cmd)
    cmd_cmd.set_defaults(func=cmd_command)

    feature_cmd = labs_sub.add_parser("feature", help="Generate Labs QR by feature name or command")
    feature_cmd.add_argument("feature", help="Feature name or command (e.g. '!MSYNC')")
    _add_common(feature_cmd)
    feature_cmd.set_defaults(func=cmd_feature)

    apply_cmd = labs_sub.add_parser("apply", help="Generate Labs QR from a profile")
    apply_cmd.add_argument("--profile", required=True, help="Profile name (e.g. head, wrist_left)")
    apply_cmd.add_argument("--handle", type=int, help="Handle ID (1-31)")
    _add_common(apply_cmd)
    apply_cmd.set_defaults(func=cmd_apply)

    sync_cmd = labs_sub.add_parser("timesync", help="Generate GPS time sync QR (!MSYNC)")
    _add_common(sync_cmd)
    sync_cmd.set_defaults(func=cmd_timesync)

    confirm_cmd = labs_sub.add_parser("confirm", help="Mark Labs as applied in inventory")
    confirm_cmd.add_argument("camera", type=str, help="Camera serial or role")
    confirm_cmd.set_defaults(func=cmd_confirm)
