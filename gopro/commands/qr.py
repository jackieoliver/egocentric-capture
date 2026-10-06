#!/usr/bin/env python3
"""
QR CLI commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..qr import render_qr


def _print_result(result: dict) -> None:
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


def cmd_qr(args: argparse.Namespace) -> None:
    command = args.command
    title = args.title or "QR Command"
    out_path = Path(args.output) if args.output else None
    result = render_qr(
        command,
        title=title,
        output_path=out_path,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    _print_result(result)


def cmd_task(args: argparse.Namespace) -> None:
    label = args.label
    title = args.title or "Task Marker"
    out_path = Path(args.output) if args.output else None
    print(f"\n[TASK MARKER] {label}")
    result = render_qr(
        label,
        title=title,
        output_path=out_path,
        write_latest=not args.no_latest,
        ascii_output=args.ascii,
        terminal_image_output=args.terminal_image,
        terminal_protocol=args.terminal_protocol,
    )
    _print_result(result)


def add_qr_parser(subparsers: argparse._SubParsersAction) -> None:
    qr = subparsers.add_parser("qr", help="QR utilities")
    qr_sub = qr.add_subparsers(dest="qr_command", required=True)

    qr_custom = qr_sub.add_parser("qr", help="Generate QR for a raw command")
    qr_custom.add_argument("command", help="Command string to encode")
    qr_custom.add_argument("--title", type=str, help="Caption title")
    qr_custom.add_argument("--output", "-o", help="Save QR code to PNG file")
    qr_custom.add_argument("--ascii", action="store_true", help="Print QR in terminal")
    qr_custom.add_argument("--terminal-image", action="store_true", help="Render QR as terminal image")
    qr_custom.add_argument("--terminal-protocol", choices=["auto", "kitty", "iterm2"], default="auto")
    qr_custom.add_argument("--no-latest", action="store_true", help="Do not write latest_qr.png")
    qr_custom.set_defaults(func=cmd_qr)

    qr_task = qr_sub.add_parser("task", help="Generate task marker QR code")
    qr_task.add_argument("label", help="Task label (e.g. 'wash_tomato', 'cut_onion')")
    qr_task.add_argument("--title", type=str, help="Caption title")
    qr_task.add_argument("--output", "-o", help="Save QR code to PNG file")
    qr_task.add_argument("--ascii", action="store_true", help="Print QR in terminal")
    qr_task.add_argument("--terminal-image", action="store_true", help="Render QR as terminal image")
    qr_task.add_argument("--terminal-protocol", choices=["auto", "kitty", "iterm2"], default="auto")
    qr_task.add_argument("--no-latest", action="store_true", help="Do not write latest_qr.png")
    qr_task.set_defaults(func=cmd_task)
