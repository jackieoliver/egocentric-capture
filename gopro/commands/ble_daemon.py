#!/usr/bin/env python3
"""
BLE daemon CLI commands.

Thin client around gopro.ble_daemon for RPC operations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from ..ble import DEFAULT_STATUS_IDS
from ..ble_daemon import (
    DEFAULT_LOG_PATH,
    DEFAULT_PID_PATH,
    DEFAULT_SOCKET_PATH,
    BleDaemon,
    request_daemon,
    start_daemon,
    stop_daemon,
)


def _socket_path(value: Optional[str]) -> Path:
    return Path(value) if value else DEFAULT_SOCKET_PATH


def cmd_ble_daemon_run(args: argparse.Namespace) -> None:
    daemon = BleDaemon(
        socket_path=_socket_path(args.socket),
        pid_path=Path(args.pid) if args.pid else DEFAULT_PID_PATH,
        maintenance_interval=args.maintenance_interval,
        default_phone_name=args.phone_name,
    )
    asyncio.run(daemon.run())


def cmd_ble_daemon_start(args: argparse.Namespace) -> None:
    pid = start_daemon(
        socket_path=_socket_path(args.socket),
        pid_path=Path(args.pid) if args.pid else DEFAULT_PID_PATH,
        log_path=Path(args.log) if args.log else DEFAULT_LOG_PATH,
        phone_name=args.phone_name,
    )
    print(f"BLE daemon started pid={pid}")


def cmd_ble_daemon_stop(args: argparse.Namespace) -> None:
    ok = stop_daemon(socket_path=_socket_path(args.socket), pid_path=Path(args.pid) if args.pid else DEFAULT_PID_PATH)
    print("BLE daemon stopped" if ok else "BLE daemon stop requested (check status)")


def cmd_ble_daemon_status(args: argparse.Namespace) -> None:
    status = request_daemon(method="status", socket_path=_socket_path(args.socket))
    print(json.dumps(status, indent=2))


def cmd_ble_daemon_sync(args: argparse.Namespace) -> None:
    status = request_daemon(method="sync_inventory", socket_path=_socket_path(args.socket))
    print(json.dumps(status, indent=2))


def cmd_ble_daemon_connect(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {
        "camera": args.camera,
        "finish_pairing": args.finish_pairing,
        "claim_control": args.claim,
        "phone_name": args.phone_name,
    }
    if args.address:
        payload["address"] = args.address
    result = request_daemon(method="connect", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_disconnect(args: argparse.Namespace) -> None:
    payload = {"camera": args.camera}
    result = request_daemon(method="disconnect", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_wake(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera, "phone_name": args.phone_name}
    if args.address:
        payload["address"] = args.address
    result = request_daemon(method="wake", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_get_status(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera, "status_ids": args.status_ids}
    if args.address:
        payload["address"] = args.address
    result = request_daemon(method="get_status", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_cohn_status(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera}
    if args.address:
        payload["address"] = args.address
    result = request_daemon(method="cohn_status", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_wifi_join(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera}
    if args.address:
        payload["address"] = args.address
    if args.ssid:
        payload["ssid"] = args.ssid
    if args.password:
        payload["password"] = args.password
    result = request_daemon(method="wifi_join", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_sleep(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera}
    if args.address:
        payload["address"] = args.address
    if args.keep_managed:
        payload["keep_managed"] = True
    if args.no_claim:
        payload["claim_control"] = False
    result = request_daemon(method="sleep", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_wifi_ap(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera, "mode": args.mode}
    if args.address:
        payload["address"] = args.address
    result = request_daemon(method="wifi_ap", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_wifi_release(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera}
    if args.address:
        payload["address"] = args.address
    result = request_daemon(method="wifi_release", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def cmd_ble_daemon_smoke(args: argparse.Namespace) -> None:
    payload: dict[str, object] = {"camera": args.camera}
    if args.address:
        payload["address"] = args.address
    if args.status_ids:
        payload["status_ids"] = args.status_ids
    result = request_daemon(method="smoke", params=payload, socket_path=_socket_path(args.socket))
    print(json.dumps(result, indent=2))


def add_ble_daemon_parser(subparsers: argparse._SubParsersAction) -> None:
    ble_daemon = subparsers.add_parser("ble-daemon", help="Run/control long-lived BLE daemon")
    ble_daemon_sub = ble_daemon.add_subparsers(dest="ble_daemon_cmd", required=True)

    ble_daemon_run = ble_daemon_sub.add_parser("run", help="Run BLE daemon in foreground")
    ble_daemon_run.add_argument("--socket", type=str, default=None, help=f"Daemon socket path (default: {DEFAULT_SOCKET_PATH})")
    ble_daemon_run.add_argument("--pid", type=str, default=None, help=f"Daemon pid path (default: {DEFAULT_PID_PATH})")
    ble_daemon_run.add_argument("--phone-name", type=str, default="HapticaProvisioner")
    ble_daemon_run.add_argument("--maintenance-interval", type=float, default=2.0)
    ble_daemon_run.set_defaults(func=cmd_ble_daemon_run)

    ble_daemon_start = ble_daemon_sub.add_parser("start", help="Start BLE daemon in background")
    ble_daemon_start.add_argument("--socket", type=str, default=None, help=f"Daemon socket path (default: {DEFAULT_SOCKET_PATH})")
    ble_daemon_start.add_argument("--pid", type=str, default=None, help=f"Daemon pid path (default: {DEFAULT_PID_PATH})")
    ble_daemon_start.add_argument("--log", type=str, default=None, help=f"Daemon log path (default: {DEFAULT_LOG_PATH})")
    ble_daemon_start.add_argument("--phone-name", type=str, default="HapticaProvisioner")
    ble_daemon_start.set_defaults(func=cmd_ble_daemon_start)

    ble_daemon_stop = ble_daemon_sub.add_parser("stop", help="Stop BLE daemon")
    ble_daemon_stop.add_argument("--socket", type=str, default=None, help=f"Daemon socket path (default: {DEFAULT_SOCKET_PATH})")
    ble_daemon_stop.add_argument("--pid", type=str, default=None, help=f"Daemon pid path (default: {DEFAULT_PID_PATH})")
    ble_daemon_stop.set_defaults(func=cmd_ble_daemon_stop)

    ble_daemon_status = ble_daemon_sub.add_parser("status", help="Show BLE daemon status")
    ble_daemon_status.add_argument("--socket", type=str, default=None, help=f"Daemon socket path (default: {DEFAULT_SOCKET_PATH})")
    ble_daemon_status.set_defaults(func=cmd_ble_daemon_status)

    ble_daemon_sync = ble_daemon_sub.add_parser("sync", help="Sync BLE daemon with inventory")
    ble_daemon_sync.add_argument("--socket", type=str, default=None, help=f"Daemon socket path (default: {DEFAULT_SOCKET_PATH})")
    ble_daemon_sync.set_defaults(func=cmd_ble_daemon_sync)

    ble_daemon_connect = ble_daemon_sub.add_parser("connect", help="Connect to a camera and keep BLE alive")
    ble_daemon_connect.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_connect.add_argument("--socket", type=str, default=None)
    ble_daemon_connect.add_argument("--address", type=str, default=None)
    ble_daemon_connect.add_argument("--finish-pairing", action="store_true")
    ble_daemon_connect.add_argument("--claim", action="store_true")
    ble_daemon_connect.add_argument("--phone-name", type=str, default="HapticaProvisioner")
    ble_daemon_connect.set_defaults(func=cmd_ble_daemon_connect)

    ble_daemon_disconnect = ble_daemon_sub.add_parser("disconnect", help="Disconnect camera or all")
    ble_daemon_disconnect.add_argument("camera", type=str, help="Role, serial, short id, or 'all'")
    ble_daemon_disconnect.add_argument("--socket", type=str, default=None)
    ble_daemon_disconnect.set_defaults(func=cmd_ble_daemon_disconnect)

    ble_daemon_wake = ble_daemon_sub.add_parser("wake", help="Wake a camera via daemon (claim control)")
    ble_daemon_wake.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_wake.add_argument("--socket", type=str, default=None)
    ble_daemon_wake.add_argument("--address", type=str, default=None)
    ble_daemon_wake.add_argument("--phone-name", type=str, default="HapticaProvisioner")
    ble_daemon_wake.set_defaults(func=cmd_ble_daemon_wake)

    ble_daemon_get_status = ble_daemon_sub.add_parser("get-status", help="Read status values via daemon")
    ble_daemon_get_status.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_get_status.add_argument("--socket", type=str, default=None)
    ble_daemon_get_status.add_argument("--address", type=str, default=None)
    ble_daemon_get_status.add_argument("--status-id", dest="status_ids", action="append", type=int, default=list(DEFAULT_STATUS_IDS))
    ble_daemon_get_status.set_defaults(func=cmd_ble_daemon_get_status)

    ble_daemon_cohn_status = ble_daemon_sub.add_parser("cohn-status", help="Read COHN status via daemon")
    ble_daemon_cohn_status.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_cohn_status.add_argument("--socket", type=str, default=None)
    ble_daemon_cohn_status.add_argument("--address", type=str, default=None)
    ble_daemon_cohn_status.set_defaults(func=cmd_ble_daemon_cohn_status)

    ble_daemon_wifi_join = ble_daemon_sub.add_parser("wifi-join", help="Join WiFi over BLE via daemon")
    ble_daemon_wifi_join.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_wifi_join.add_argument("--socket", type=str, default=None)
    ble_daemon_wifi_join.add_argument("--address", type=str, default=None)
    ble_daemon_wifi_join.add_argument("--ssid", type=str, default=None)
    ble_daemon_wifi_join.add_argument("--password", type=str, default=None)
    ble_daemon_wifi_join.set_defaults(func=cmd_ble_daemon_wifi_join)

    ble_daemon_wifi_ap = ble_daemon_sub.add_parser("wifi-ap", help="Control Wi-Fi AP over BLE (0=off,1=on,2=bounce)")
    ble_daemon_wifi_ap.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_wifi_ap.add_argument("--socket", type=str, default=None)
    ble_daemon_wifi_ap.add_argument("--address", type=str, default=None)
    ble_daemon_wifi_ap.add_argument("--mode", type=str, default="bounce", help="disable|enable|bounce or 0|1|2")
    ble_daemon_wifi_ap.set_defaults(func=cmd_ble_daemon_wifi_ap)

    ble_daemon_wifi_release = ble_daemon_sub.add_parser(
        "wifi-release",
        help="Disconnect STA Wi-Fi so the camera returns to AP mode",
    )
    ble_daemon_wifi_release.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_wifi_release.add_argument("--socket", type=str, default=None)
    ble_daemon_wifi_release.add_argument("--address", type=str, default=None)
    ble_daemon_wifi_release.set_defaults(func=cmd_ble_daemon_wifi_release)

    ble_daemon_sleep = ble_daemon_sub.add_parser("sleep", help="Put camera into sleep (soft off)")
    ble_daemon_sleep.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_sleep.add_argument("--socket", type=str, default=None)
    ble_daemon_sleep.add_argument("--address", type=str, default=None)
    ble_daemon_sleep.add_argument("--keep-managed", action="store_true", help="Keep camera desired after sleep")
    ble_daemon_sleep.add_argument("--no-claim", action="store_true", help="Do not claim external control before sleep")
    ble_daemon_sleep.set_defaults(func=cmd_ble_daemon_sleep)

    ble_daemon_smoke = ble_daemon_sub.add_parser("smoke", help="Run BLE smoke test via daemon")
    ble_daemon_smoke.add_argument("camera", type=str, help="Role, serial, or short id")
    ble_daemon_smoke.add_argument("--socket", type=str, default=None)
    ble_daemon_smoke.add_argument("--address", type=str, default=None)
    ble_daemon_smoke.add_argument("--status-id", dest="status_ids", action="append", type=int, default=[])
    ble_daemon_smoke.set_defaults(func=cmd_ble_daemon_smoke)
