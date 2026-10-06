#!/usr/bin/env python3
"""
Inventory utilities.

Manual updates for inventory records when auto-discovery is incomplete.
"""

from __future__ import annotations

import argparse

from rich.console import Console

from ..state import ProvisioningState, get_inventory

console = Console()


def cmd_set(args: argparse.Namespace) -> None:
    inventory = get_inventory()
    camera = inventory.get(args.serial)
    if not camera:
        console.print(f"[red]Camera {args.serial} not found in inventory[/red]")
        return

    updated = False

    if args.role:
        camera.role = args.role.upper()
        updated = True
    if args.profile:
        camera.profile = args.profile
        updated = True
    if args.handle is not None:
        camera.handle_id = args.handle
        updated = True
    if args.udp_preview_port is not None:
        camera.udp_preview_port = args.udp_preview_port
        updated = True
    if args.camera_name:
        camera.camera_name = args.camera_name
        updated = True
    if args.model_name:
        camera.model_name = args.model_name
        updated = True
    if args.model_number:
        camera.model_number = args.model_number
        updated = True
    if args.firmware_version:
        camera.firmware_version = args.firmware_version
        updated = True
    if args.ap_ssid:
        camera.ap_ssid = args.ap_ssid
        updated = True
    if args.ap_mac_addr:
        camera.ap_mac_addr = args.ap_mac_addr
        updated = True

    if args.ble_address:
        camera.ble.address = args.ble_address
        camera.ble.available = True
        updated = True
    if args.ble_available is not None:
        camera.ble.available = args.ble_available
        updated = True

    if args.usb_ip:
        camera.usb.ip = args.usb_ip
        camera.usb.available = True
        updated = True
    if args.usb_available is not None:
        camera.usb.available = args.usb_available
        updated = True

    if args.wifi_ip:
        camera.wifi.ip = args.wifi_ip
        camera.wifi.available = True
        updated = True
    if args.wifi_port is not None:
        camera.wifi.port = args.wifi_port
        updated = True
    if args.wifi_available is not None:
        camera.wifi.available = args.wifi_available
        updated = True
    if args.wifi_ssid:
        camera.provisioning.wifi_ssid = args.wifi_ssid
        camera.provisioning.wifi_joined = True
        updated = True
    if args.wifi_joined is not None:
        camera.provisioning.wifi_joined = args.wifi_joined
        updated = True

    if args.ble_paired is not None:
        camera.provisioning.ble_paired = args.ble_paired
        updated = True
    if args.cohn_enabled is not None:
        camera.provisioning.cohn_enabled = args.cohn_enabled
        updated = True
    if args.cohn_credential_id:
        camera.provisioning.cohn_credential_id = args.cohn_credential_id
        updated = True

    if args.state:
        camera.provisioning.state = ProvisioningState(args.state)
        updated = True

    if not updated:
        console.print("[yellow]No changes provided.[/yellow]")
        return

    inventory.upsert(camera)
    inventory.save()
    console.print(f"[green]Updated {camera.short_id}[/green]")
    console.print(f"[dim]wrote {inventory.path}[/dim]")


def add_inventory_parser(subparsers: argparse._SubParsersAction) -> None:
    inventory = subparsers.add_parser("inventory", help="Inventory utilities")
    inventory_sub = inventory.add_subparsers(dest="inventory_command", required=True)

    set_cmd = inventory_sub.add_parser("set", help="Manually update inventory record")
    set_cmd.add_argument("serial", help="Camera serial or short id")
    set_cmd.add_argument("--role", type=str, help="Set role (HEAD, LWRIST, RWRIST, etc)")
    set_cmd.add_argument("--profile", type=str, help="Set profile name")
    set_cmd.add_argument("--handle", type=int, help="Set handle ID (1-31)")
    set_cmd.add_argument("--udp-preview-port", type=int, help="Set UDP preview port for streaming")
    set_cmd.add_argument("--camera-name", type=str, help="Set camera name")
    set_cmd.add_argument("--model-name", type=str, help="Set model name")
    set_cmd.add_argument("--model-number", type=str, help="Set model number")
    set_cmd.add_argument("--firmware-version", type=str, help="Set firmware version")
    set_cmd.add_argument("--ap-ssid", type=str, help="Set AP SSID")
    set_cmd.add_argument("--ap-mac-addr", type=str, help="Set AP MAC address")
    set_cmd.add_argument("--ble-address", type=str, help="Set BLE address")
    set_cmd.add_argument("--usb-ip", type=str, help="Set USB IP")
    set_cmd.add_argument("--wifi-ip", type=str, help="Set WiFi IP (COHN)")
    set_cmd.add_argument("--wifi-port", type=int, help="Set WiFi/COHN port")
    set_cmd.add_argument("--wifi-ssid", type=str, help="Set WiFi SSID")
    set_cmd.add_argument("--cohn-credential-id", type=str, help="Set COHN credential id")
    set_cmd.add_argument(
        "--state",
        type=str,
        choices=[state.value for state in ProvisioningState],
        help="Set provisioning state",
    )

    ble_group = set_cmd.add_mutually_exclusive_group()
    ble_group.add_argument("--ble-available", dest="ble_available", action="store_true")
    ble_group.add_argument("--ble-unavailable", dest="ble_available", action="store_false")
    set_cmd.set_defaults(ble_available=None)

    usb_group = set_cmd.add_mutually_exclusive_group()
    usb_group.add_argument("--usb-available", dest="usb_available", action="store_true")
    usb_group.add_argument("--usb-unavailable", dest="usb_available", action="store_false")
    set_cmd.set_defaults(usb_available=None)

    wifi_group = set_cmd.add_mutually_exclusive_group()
    wifi_group.add_argument("--wifi-available", dest="wifi_available", action="store_true")
    wifi_group.add_argument("--wifi-unavailable", dest="wifi_available", action="store_false")
    set_cmd.set_defaults(wifi_available=None)

    wifi_joined_group = set_cmd.add_mutually_exclusive_group()
    wifi_joined_group.add_argument("--wifi-joined", dest="wifi_joined", action="store_true")
    wifi_joined_group.add_argument("--wifi-not-joined", dest="wifi_joined", action="store_false")
    set_cmd.set_defaults(wifi_joined=None)

    ble_paired_group = set_cmd.add_mutually_exclusive_group()
    ble_paired_group.add_argument("--ble-paired", dest="ble_paired", action="store_true")
    ble_paired_group.add_argument("--ble-not-paired", dest="ble_paired", action="store_false")
    set_cmd.set_defaults(ble_paired=None)

    cohn_group = set_cmd.add_mutually_exclusive_group()
    cohn_group.add_argument("--cohn-enabled", dest="cohn_enabled", action="store_true")
    cohn_group.add_argument("--cohn-disabled", dest="cohn_enabled", action="store_false")
    set_cmd.set_defaults(cohn_enabled=None)

    set_cmd.set_defaults(func=cmd_set)
