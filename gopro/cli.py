#!/usr/bin/env python3
"""
Terminal-first control surface for Haptica GoPro workflows.

Keeps YAML state in state/inventory.yaml and delegates device IO to modules in
this package.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GoPro control CLI",
        epilog="""Examples:
  gopro provision scan                  # Scan pairing cameras
  gopro provision adopt 8308 --role HEAD --profile head
  gopro provision start                 # Start daemon + connect all
  gopro provision rename HEAD "Haptica-Head"
  gopro provision apply --all           # Apply settings + Labs
  gopro camera wake LWRIST              # Wake sleeping camera
  gopro daemon start                    # Start gopro-daemon
  gopro daemon arm --all                # Prep Wi-Fi/COHN for all cameras
  gopro labs setup --combined           # Generate setup QR""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import and add new command modules
    from .commands.provision import add_provision_parser
    from .commands.inventory import add_inventory_parser
    from .commands.camera import add_camera_parser
    from .commands.ble_daemon import add_ble_daemon_parser
    from .commands.qr import add_qr_parser
    from .commands.labs import add_labs_parser
    from .commands.daemon import add_daemon_parser

    add_provision_parser(subparsers)
    add_inventory_parser(subparsers)
    add_camera_parser(subparsers)
    add_ble_daemon_parser(subparsers)
    add_daemon_parser(subparsers)
    add_qr_parser(subparsers)
    add_labs_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
