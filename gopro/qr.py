#!/usr/bin/env python3
"""
QR Code Utilities

Generate captioned QR codes for arbitrary command strings.
"""

import base64
import io
import os

from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_M

from .state import QR_STATE_DIR, STATE_DIR

LEGACY_QR_DIR = STATE_DIR
LEGACY_QR_PATH = LEGACY_QR_DIR / "latest_qr.png"
LATEST_QR_PATH = QR_STATE_DIR / "latest_qr.png"


# =============================================================================
# QR Code Generation
# =============================================================================
def _default_output() -> Path:
    _migrate_legacy_qr_outputs()
    LATEST_QR_PATH.parent.mkdir(parents=True, exist_ok=True)
    return LATEST_QR_PATH


def _migrate_legacy_qr_outputs() -> None:
    QR_STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LEGACY_QR_PATH.exists() and not LATEST_QR_PATH.exists():
        try:
            LEGACY_QR_PATH.rename(LATEST_QR_PATH)
        except OSError:
            pass
    for legacy_path in LEGACY_QR_DIR.glob("labs_*.png"):
        dest = QR_STATE_DIR / legacy_path.name
        if dest.exists():
            continue
        try:
            legacy_path.rename(dest)
        except OSError:
            pass


def generate_qr(command: str, box_size: int = 10, border: int = 2) -> qrcode.QRCode:
    """Generate a QR code for a command string."""
    qr = qrcode.QRCode(
        version=None,  # Auto-size
        error_correction=ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(command)
    qr.make(fit=True)
    return qr


def qr_to_ascii(qr: qrcode.QRCode) -> str:
    """Convert QR code to ASCII art for terminal display."""
    # Use the built-in ASCII method but capture it
    f = io.StringIO()
    qr.print_ascii(out=f, invert=True)
    return f.getvalue()


def qr_to_image(qr: qrcode.QRCode, fill_color: str = "black", back_color: str = "white"):
    """Convert QR code to PIL Image."""
    return qr.make_image(fill_color=fill_color, back_color=back_color)


def save_qr_image(
    qr: qrcode.QRCode,
    path: Path,
    fill_color: str = "black",
    back_color: str = "white",
    title: str | None = None,
    command: str | None = None,
) -> None:
    """Save QR code as PNG image with optional caption.

    Args:
        qr: The QR code object
        path: Output file path
        fill_color: QR code foreground color
        back_color: QR code background color
        title: Optional title to display above the command
        command: Optional command text to display below the QR code
    """
    from PIL import Image, ImageDraw, ImageFont

    _migrate_legacy_qr_outputs()
    path.parent.mkdir(parents=True, exist_ok=True)
    qr_img = qr_to_image(qr, fill_color, back_color)
    qr_img = qr_img.convert("RGB")
    qr_width, qr_height = qr_img.size

    # If no caption needed, just save the QR
    if not title and not command:
        qr_img.save(str(path))
        return

    # Calculate caption area height
    padding = 20
    line_height = 24
    caption_lines = []

    if title:
        caption_lines.append(("title", title))
    if command:
        # Wrap long commands
        max_chars = max(40, qr_width // 10)
        if len(command) > max_chars:
            # Split into multiple lines
            words = command
            while len(words) > max_chars:
                caption_lines.append(("command", words[:max_chars]))
                words = words[max_chars:]
            if words:
                caption_lines.append(("command", words))
        else:
            caption_lines.append(("command", command))

    caption_height = padding * 2 + len(caption_lines) * line_height

    # Create new image with space for caption
    total_height = qr_height + caption_height
    final_img = Image.new("RGB", (qr_width, total_height), "white")

    # Paste QR code at top
    final_img.paste(qr_img, (0, 0))

    # Draw caption text
    draw = ImageDraw.Draw(final_img)

    # Try to use a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
            title_font = font
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

    y = qr_height + padding
    for line_type, text in caption_lines:
        if line_type == "title":
            # Center the title, bold
            bbox = draw.textbbox((0, 0), text, font=title_font)
            text_width = bbox[2] - bbox[0]
            x = (qr_width - text_width) // 2
            draw.text((x, y), text, fill="black", font=title_font)
        else:
            # Command text - smaller, monospace, gray
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (qr_width - text_width) // 2
            draw.text((x, y), text, fill="#555555", font=font)
        y += line_height

    final_img.save(str(path))


def qr_to_png_bytes(qr: qrcode.QRCode) -> bytes:
    """Render QR to PNG bytes."""
    img = qr_to_image(qr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _terminal_protocol(protocol: str | None) -> str | None:
    if protocol and protocol != "auto":
        return protocol
    term_program = os.getenv("TERM_PROGRAM", "").lower()
    if "iterm" in term_program:
        return "iterm2"
    if "ghostty" in term_program:
        return "kitty"
    if os.getenv("KITTY_WINDOW_ID") or os.getenv("TERM", "").endswith("kitty"):
        return "kitty"
    return None


def terminal_image(png_bytes: bytes, *, filename: str = "qr.png", protocol: str | None = None) -> str | None:
    """Return terminal image escape sequence for iTerm2/kitty."""
    chosen = _terminal_protocol(protocol)
    if not chosen:
        return None
    payload = base64.b64encode(png_bytes).decode("ascii")
    if chosen == "iterm2":
        name = base64.b64encode(filename.encode("utf-8")).decode("ascii")
        return f"\x1b]1337;File=name={name};inline=1:{payload}\x07"
    if chosen == "kitty":
        return f"\x1b_Gf=100,a=T;{payload}\x1b\\"
    return None


def render_qr(
    command: str,
    *,
    title: str | None = None,
    output_path: Path | None = None,
    write_latest: bool = True,
    ascii_output: bool = False,
    terminal_image_output: bool = False,
    terminal_protocol: str | None = None,
) -> dict:
    """Render a QR code and optionally save/return it."""
    qr = generate_qr(command)
    ascii_art = qr_to_ascii(qr) if ascii_output else None
    terminal_art = None
    terminal_error = None
    if terminal_image_output:
        terminal_art = terminal_image(qr_to_png_bytes(qr), protocol=terminal_protocol)
        if terminal_art is None:
            terminal_error = (
                "Terminal image output not supported. Try kitty/ghostty or iTerm2 "
                "(use --terminal-protocol=kitty|iterm2)."
            )

    saved_path = None
    latest_path = None

    if output_path:
        save_qr_image(qr, output_path, title=title, command=command)
        saved_path = output_path

    if write_latest or not output_path:
        latest_path = _default_output()
        save_qr_image(qr, latest_path, title=title, command=command)

    return {
        "command": command,
        "ascii": ascii_art,
        "terminal_image": terminal_art,
        "terminal_image_error": terminal_error,
        "output_path": str(saved_path) if saved_path else None,
        "latest_path": str(latest_path) if latest_path else None,
    }
