#!/usr/bin/env python3
"""
GoPro daemon CLI.

This is the daemon-first entrypoint for:
- keeping BLE connected (via ble-daemon)
- polling COHN health (battery/storage/recording)
- providing a single status surface for the rest of the program
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from ..gopro_daemon import (
    DEFAULT_LOG_PATH,
    DEFAULT_PID_PATH,
    DEFAULT_SOCKET_PATH,
    request_gopro_daemon,
    start_gopro_daemon,
    stop_gopro_daemon,
)
from ..sd import format_sd_health_cell_rich
from ..state import get_inventory

console = Console()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _print_status(data: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(data, indent=2))
        return

    inventory = get_inventory()
    udp_ports = {cam.serial_number: cam.udp_preview_port for cam in inventory.cameras if cam.udp_preview_port}

    ble = data.get("ble_daemon") or {}
    header = f"GoPro Daemon (started_at={data.get('started_at')})"
    table = Table(title=header)
    table.add_column("Role")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("BLE")
    table.add_column("Pairing")
    table.add_column("Sleep")
    table.add_column("WiFi/COHN")
    table.add_column("Batt")
    table.add_column("SD")
    table.add_column("Rec")
    table.add_column("Stream")
    table.add_column("HTTP")
    table.add_column("Err")

    now = datetime.now()
    for cam in data.get("cameras", []) or []:
        role = cam.get("role") or "-"
        serial = cam.get("serial") or ""
        short_id = cam.get("short_id") or (serial[-4:] if serial else "-")
        name = cam.get("name") or "-"
        ble_info = cam.get("ble") or {}
        wifi_info = cam.get("wifi") or {}
        wifi_connected = wifi_info.get("connected")
        wifi_ip = wifi_info.get("ip") or wifi_info.get("ip_cached")
        health = cam.get("health") or {}

        ble_connected = "Y" if ble_info.get("connected") else "-"
        ble_paired = "Y" if ble_info.get("paired") else "-"
        ble_cell = f"{ble_connected}/{ble_paired}"

        pairing_mode = ble_info.get("pairing_mode")
        if pairing_mode is True:
            pairing_cell = "yes"
        elif pairing_mode is False:
            pairing_cell = "no"
        else:
            pairing_cell = "-"

        sleeping_cell = "yes" if cam.get("sleeping") else "-"

        wifi_available = wifi_info.get("available")
        if wifi_connected is True:
            cohn = "Y" if wifi_info.get("cohn_enabled") and wifi_ip else "-"
            wifi_cell = f"{cohn} {wifi_ip or ''}".strip()
        elif wifi_connected is False:
            wifi_cell = "off"
        elif wifi_available is True:
            cohn = "Y" if wifi_info.get("cohn_enabled") and wifi_ip else "-"
            wifi_cell = f"{cohn} {wifi_ip or ''}".strip()
        elif wifi_available is False and wifi_ip:
            wifi_cell = "off"
        else:
            wifi_cell = "-"

        batt = health.get("battery_percent")
        batt_cell = f"{batt}%" if isinstance(batt, int) and batt >= 0 else "-"
        sd_cell = format_sd_health_cell_rich(health)
        rec = health.get("is_recording")
        rec_cell = "yes" if rec is True else ("no" if rec is False else "-")
        stream = cam.get("stream") or {}
        stream_port = stream.get("port") or udp_ports.get(serial)
        stream_active = bool(stream.get("active"))
        age_ms = stream.get("age_ms")
        first_packet_ms = stream.get("first_packet_ms")
        if stream_active:
            state_label = "active"
            if stream.get("watchdog_enabled") and not stream.get("watchdog_active"):
                state_label = "active?"
            last_packet = _parse_iso(stream.get("last_packet_at"))
            stall_timeout = stream.get("stall_timeout_seconds")
            if last_packet and isinstance(stall_timeout, (int, float)):
                if (now - last_packet).total_seconds() > float(stall_timeout):
                    state_label = "stalled"
            suffix = []
            if isinstance(age_ms, (int, float)):
                suffix.append(f"{age_ms:.0f}ms")
            if isinstance(first_packet_ms, (int, float)):
                suffix.append(f"first {first_packet_ms:.0f}ms")
            extra = f" {' '.join(suffix)}" if suffix else ""
            stream_cell = f"{stream_port or '-'} {state_label}{extra}"
        elif stream_port:
            stream_cell = f"{stream_port} idle"
        else:
            stream_cell = "-"
        http_cell = health.get("transport") or "-"
        err_cell = health.get("last_error") or (ble_info.get("last_error") or "-")

        table.add_row(
            role,
            short_id,
            name,
            ble_cell,
            pairing_cell,
            sleeping_cell,
            wifi_cell,
            batt_cell,
            sd_cell,
            rec_cell,
            stream_cell,
            http_cell,
            str(err_cell)[:80],
        )

    console.print(
        f"[dim]ble-daemon running={bool(ble.get('running'))} socket={ble.get('socket') or '-'}[/dim]"
    )
    console.print(table)


def cmd_start(args: argparse.Namespace) -> None:
    pid = start_gopro_daemon(
        socket_path=DEFAULT_SOCKET_PATH,
        pid_path=DEFAULT_PID_PATH,
        log_path=DEFAULT_LOG_PATH,
    )
    console.print(f"[green]Started gopro-daemon pid={pid}[/green]")


def cmd_stop(args: argparse.Namespace) -> None:
    ok = stop_gopro_daemon(socket_path=DEFAULT_SOCKET_PATH, pid_path=DEFAULT_PID_PATH)
    if ok:
        console.print("[green]Stopped gopro-daemon[/green]")
        return
    console.print("[yellow]Stop request sent; daemon may still be shutting down[/yellow]")


def cmd_status(args: argparse.Namespace) -> None:
    data = request_gopro_daemon(method="status", socket_path=DEFAULT_SOCKET_PATH)
    _print_status(data, json_output=args.json)


def cmd_stream_health(args: argparse.Namespace) -> None:
    data = request_gopro_daemon(method="stream_health", socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return

    table = Table(title=f"Stream Health (updated_at={data.get('updated_at')})")
    table.add_column("Role")
    table.add_column("ID")
    table.add_column("Port")
    table.add_column("State")
    table.add_column("Age")
    table.add_column("First")
    table.add_column("Last Packet")
    table.add_column("Restart")
    table.add_column("Err")

    for stream in data.get("streams", []) or []:
        role = stream.get("role") or "-"
        serial = stream.get("serial") or ""
        short_id = stream.get("short_id") or (serial[-4:] if serial else "-")
        port = stream.get("port") or "-"
        state = stream.get("status") or "-"
        age_ms = stream.get("age_ms")
        age_cell = f"{age_ms:.0f}ms" if isinstance(age_ms, (int, float)) else "-"
        first_ms = stream.get("first_packet_ms")
        first_cell = f"{first_ms:.0f}ms" if isinstance(first_ms, (int, float)) else "-"
        last_packet = stream.get("last_packet_at") or "-"
        last_restart = stream.get("last_restart_at") or "-"
        err = stream.get("last_error") or "-"

        table.add_row(
            role,
            short_id,
            str(port),
            state,
            age_cell,
            first_cell,
            str(last_packet),
            str(last_restart),
            str(err)[:80],
        )

    summary = data.get("summary") or {}
    summary_bits = ", ".join(f"{key}={summary.get(key)}" for key in sorted(summary))
    if summary_bits:
        console.print(f"[dim]{summary_bits}[/dim]")
    console.print(table)


def cmd_refresh(args: argparse.Namespace) -> None:
    data = request_gopro_daemon(method="refresh", socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    console.print(f"[green]Refreshed[/green] {data.get('refreshed_at')}")


def cmd_connect(args: argparse.Namespace) -> None:
    data = request_gopro_daemon(method="connect_all", socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    console.print(f"[green]Connect requested[/green] {data.get('requested_at')}")


def cmd_locate(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "camera": args.camera,
        "duration": float(args.duration),
    }
    if args.on:
        payload["on"] = True
    if args.off:
        payload["on"] = False
        payload["duration"] = 0.0
    if args.no_volume:
        payload["beep_volume"] = None
    else:
        payload["beep_volume"] = int(args.beep_volume)

    data = request_gopro_daemon(method="locate", params=payload, socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    cam = data.get("camera") or args.camera
    console.print(f"[green]Locate[/green] {cam} transport={data.get('transport')}")


def cmd_arm(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "camera": args.camera,
        "all": bool(args.all),
        "bounce_ap": bool(args.bounce_ap),
        "skip_wifi": bool(args.skip_wifi),
        "skip_cohn": bool(args.skip_cohn),
        "no_cohn_enable": bool(args.no_cohn_enable),
    }
    data = request_gopro_daemon(method="arm", params=payload, socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    console.print("[green]Armed[/green]")


def cmd_disarm(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "camera": args.camera,
        "all": bool(args.all),
        "keep_ap": bool(args.keep_ap),
    }
    data = request_gopro_daemon(method="disarm", params=payload, socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    console.print("[green]Disarmed[/green]")


def cmd_wake(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {"camera": args.camera}
    data = request_gopro_daemon(method="wake", params=payload, socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    console.print(f"[green]Wake[/green] {data.get('camera') or args.camera}")


def cmd_sleep(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {"camera": args.camera}
    data = request_gopro_daemon(method="sleep", params=payload, socket_path=DEFAULT_SOCKET_PATH)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    console.print(f"[green]Sleep[/green] {data.get('camera') or args.camera}")


def cmd_watch(args: argparse.Namespace) -> None:
    interval = float(args.interval)
    while True:
        try:
            data = request_gopro_daemon(method="status", socket_path=DEFAULT_SOCKET_PATH)
        except Exception as exc:
            console.print(f"[red]daemon unavailable[/red] {exc}")
            time.sleep(interval)
            continue

        console.clear()
        _print_status(data, json_output=False)
        time.sleep(interval)


def add_daemon_parser(subparsers: argparse._SubParsersAction) -> None:
    daemon = subparsers.add_parser("daemon", help="GoPro manager daemon (entrypoint)")
    sub = daemon.add_subparsers(dest="daemon_command", required=True)

    start = sub.add_parser("start", help="Start daemon in background")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="Stop daemon")
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status", help="Show daemon status")
    status.add_argument("--json", action="store_true", help="JSON output")
    status.set_defaults(func=cmd_status)

    stream_health = sub.add_parser("stream-health", help="Show preview stream health")
    stream_health.add_argument("--json", action="store_true", help="JSON output")
    stream_health.set_defaults(func=cmd_stream_health)

    refresh = sub.add_parser("refresh", help="Force a health refresh now")
    refresh.add_argument("--json", action="store_true", help="JSON output")
    refresh.set_defaults(func=cmd_refresh)

    connect = sub.add_parser("connect", help="Request connect-all via ble-daemon")
    connect.add_argument("--json", action="store_true", help="JSON output")
    connect.set_defaults(func=cmd_connect)

    locate = sub.add_parser("locate", help="Beep/locate a camera (legacy gpControl endpoint)")
    locate.add_argument("camera", help="Camera serial, short_id, or role")
    locate.add_argument("--duration", type=float, default=8.0, help="Seconds to beep before stopping")
    locate.add_argument("--on", action="store_true", help="Start locate (do not auto-stop)")
    locate.add_argument("--off", action="store_true", help="Stop locate")
    locate.add_argument("--beep-volume", type=int, default=100, help="Beep volume option (216): 70/85/100")
    locate.add_argument("--no-volume", action="store_true", help="Skip setting beep volume")
    locate.add_argument("--json", action="store_true", help="JSON output")
    locate.set_defaults(func=cmd_locate)

    arm = sub.add_parser("arm", help="Prepare Wi-Fi/COHN for use (optionally all cameras)")
    arm.add_argument("camera", nargs="?", help="Camera serial, short_id, or role")
    arm.add_argument("--all", action="store_true", help="Apply to all inventory cameras")
    arm.add_argument("--bounce-ap", action="store_true", help="Bounce Wi-Fi AP before joining")
    arm.add_argument("--skip-wifi", action="store_true", help="Skip Wi-Fi join step")
    arm.add_argument("--skip-cohn", action="store_true", help="Skip COHN status/enable")
    arm.add_argument("--no-cohn-enable", action="store_true", help="Do not create COHN cert if missing")
    arm.add_argument("--json", action="store_true", help="JSON output")
    arm.set_defaults(func=cmd_arm)

    disarm = sub.add_parser("disarm", help="Turn off Wi-Fi/COHN while keeping BLE connected")
    disarm.add_argument("camera", nargs="?", help="Camera serial, short_id, or role")
    disarm.add_argument("--all", action="store_true", help="Apply to all inventory cameras")
    disarm.add_argument("--keep-ap", action="store_true", help="Keep the camera AP enabled")
    disarm.add_argument("--json", action="store_true", help="JSON output")
    disarm.set_defaults(func=cmd_disarm)

    wake = sub.add_parser("wake", help="Wake a camera via BLE (soft on)")
    wake.add_argument("camera", help="Camera serial, short_id, or role")
    wake.add_argument("--json", action="store_true", help="JSON output")
    wake.set_defaults(func=cmd_wake)

    sleep = sub.add_parser("sleep", help="Sleep a camera via BLE (soft off)")
    sleep.add_argument("camera", help="Camera serial, short_id, or role")
    sleep.add_argument("--json", action="store_true", help="JSON output")
    sleep.set_defaults(func=cmd_sleep)

    watch = sub.add_parser("watch", help="Auto-refresh terminal status view")
    watch.add_argument("--interval", type=float, default=1.0, help="Refresh interval seconds")
    watch.set_defaults(func=cmd_watch)
