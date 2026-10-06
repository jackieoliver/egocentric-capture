#!/usr/bin/env python3
"""
Estimate 3D camera trajectory from GoPro GPMF telemetry data.

Uses CORI quaternions for orientation, ACCL for acceleration, and GRAV for gravity removal.
Applies Zero-Velocity Updates (ZUPT) for basic drift correction.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class TelemetryBlock:
    t_in: float
    t_out: float
    key: str
    values: np.ndarray  # shape (samples, elements)


def _load_jsonl(path: Path):
    blocks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            vals = np.asarray(obj.get("values", []), dtype=np.float64)
            if vals.size == 0:
                continue
            blocks.append(
                TelemetryBlock(
                    t_in=float(obj.get("t_in")),
                    t_out=float(obj.get("t_out")),
                    key=key,
                    values=vals,
                )
            )
    return blocks


def _stack_key(blocks, key):
    """Concatenate samples for a telemetry key, building per-sample timestamps."""
    t_list = []
    v_list = []
    for b in blocks:
        if b.key != key:
            continue
        n = b.values.shape[0]
        # linspace over [t_in, t_out) with n samples
        # Use endpoint False to avoid duplicate t_out when multiple blocks abut
        if n == 1:
            ts = np.array([b.t_in], dtype=np.float64)
        else:
            ts = np.linspace(b.t_in, b.t_out, n, endpoint=False, dtype=np.float64)
        t_list.append(ts)
        v_list.append(b.values.astype(np.float64))
    if not t_list:
        return None, None
    t = np.concatenate(t_list, axis=0)
    v = np.concatenate(v_list, axis=0)
    # Ensure sorted by time
    order = np.argsort(t)
    return t[order], v[order]


def _interp_vec(t_src, v_src, t_dst, kind="linear"):
    if t_src is None:
        return None
    f = interp1d(t_src, v_src, axis=0, kind=kind, bounds_error=False, fill_value="extrapolate")
    return f(t_dst)


def _interp_quat(t_src, q_src_wxyz, t_dst):
    if t_src is None:
        return None
    # Convert wxyz -> xyzw for scipy
    q_xyzw = np.column_stack([q_src_wxyz[:, 1], q_src_wxyz[:, 2], q_src_wxyz[:, 3], q_src_wxyz[:, 0]])
    r = R.from_quat(q_xyzw)
    # Use Slerp for smooth rotation interpolation
    from scipy.spatial.transform import Slerp

    slerp = Slerp(t_src, r)
    r_i = slerp(t_dst)
    q_i_xyzw = r_i.as_quat()
    # Back to wxyz
    q_i_wxyz = np.column_stack([q_i_xyzw[:, 3], q_i_xyzw[:, 0], q_i_xyzw[:, 1], q_i_xyzw[:, 2]])
    return q_i_wxyz


def _estimate_trajectory(t, acc_world, stationary_thresh=0.15, zupt_window=0.2):
    """
    Simple integration with ZUPT:
    - integrate acceleration to velocity
    - detect stationary by acc magnitude < threshold (m/s^2)
    - zero velocity in stationary windows
    - integrate velocity to position
    """
    dt = np.diff(t, prepend=t[0])
    vel = np.zeros_like(acc_world)
    pos = np.zeros_like(acc_world)

    for i in range(1, len(t)):
        vel[i] = vel[i - 1] + acc_world[i] * dt[i]
        pos[i] = pos[i - 1] + vel[i] * dt[i]

    # ZUPT: detect stationary segments via moving RMS of acceleration
    if len(t) > 1:
        # window size in samples
        win = max(1, int(round(zupt_window / np.median(dt[1:]))))
        acc_mag = np.linalg.norm(acc_world, axis=1)
        # moving average
        kernel = np.ones(win, dtype=np.float64) / win
        acc_smooth = np.convolve(acc_mag, kernel, mode="same")
        stationary = acc_smooth < stationary_thresh

        # Zero velocities in stationary intervals and re-integrate positions
        vel2 = vel.copy()
        vel2[stationary] = 0.0
        pos2 = np.zeros_like(pos)
        for i in range(1, len(t)):
            pos2[i] = pos2[i - 1] + vel2[i] * dt[i]
        return pos2, vel2, stationary

    return pos, vel, np.zeros(len(t), dtype=bool)


def main():
    p = argparse.ArgumentParser(description="Estimate 3D camera trajectory from GoPro GPMF JSONL telemetry.")
    p.add_argument("--input", required=True, help="Path to GPMF JSONL telemetry")
    p.add_argument("--output", required=True, help="Path to output NPZ")
    p.add_argument("--plot", action="store_true", help="Plot 3D trajectory")
    p.add_argument("--rate", type=float, default=200.0, help="Common sample rate (Hz)")
    p.add_argument("--stationary-thresh", type=float, default=0.15, help="Stationary accel threshold (m/s^2)")
    p.add_argument("--zupt-window", type=float, default=0.2, help="ZUPT window seconds")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    blocks = _load_jsonl(in_path)
    print(f"Loaded {len(blocks)} telemetry blocks from {in_path}")

    # Load each telemetry stream
    t_cori, q_cori = _stack_key(blocks, "CORI")
    t_grav, g_grav = _stack_key(blocks, "GRAV")
    t_accl, a_accl = _stack_key(blocks, "ACCL")

    if t_cori is None or t_accl is None or t_grav is None:
        print("Missing required telemetry keys (CORI, GRAV, ACCL).", file=sys.stderr)
        sys.exit(1)

    print(f"CORI: {len(t_cori)} samples, {t_cori[0]:.2f}s - {t_cori[-1]:.2f}s")
    print(f"GRAV: {len(t_grav)} samples, {t_grav[0]:.2f}s - {t_grav[-1]:.2f}s")
    print(f"ACCL: {len(t_accl)} samples, {t_accl[0]:.2f}s - {t_accl[-1]:.2f}s")

    # Common timestamps
    t_start = max(t_cori[0], t_grav[0], t_accl[0])
    t_end = min(t_cori[-1], t_grav[-1], t_accl[-1])
    if t_end <= t_start:
        print("No overlapping time range among sensors.", file=sys.stderr)
        sys.exit(1)

    dt = 1.0 / args.rate
    t = np.arange(t_start, t_end, dt, dtype=np.float64)
    print(f"Common timeline: {len(t)} samples at {args.rate}Hz ({t_end - t_start:.2f}s)")

    # Interpolate streams
    print("Interpolating sensor streams...")
    q_i = _interp_quat(t_cori, q_cori, t)
    g_i = _interp_vec(t_grav, g_grav, t)
    a_i = _interp_vec(t_accl, a_accl, t)

    # Build rotation objects for body -> world transform
    print("Building rotation transforms...")
    q_i_xyzw = np.column_stack([q_i[:, 1], q_i[:, 2], q_i[:, 3], q_i[:, 0]])
    rot = R.from_quat(q_i_xyzw)

    # Gravity removal
    # NOTE: GoPro HERO8+ has different axis conventions for GRAV vs ACCL
    # GRAV X and Y are swapped relative to ACCL frame
    # Swap GRAV axes to match ACCL body frame before scaling
    g_body_corrected = np.column_stack([g_i[:, 1], g_i[:, 0], g_i[:, 2]]) * 9.81

    # Remove gravity in body frame BEFORE rotating to world
    a_lin_body = a_i - g_body_corrected

    # Now rotate linear acceleration to world frame
    a_lin = rot.apply(a_lin_body)

    print(f"Linear acceleration stats: mean={np.mean(np.linalg.norm(a_lin, axis=1)):.3f} m/s²")

    # Integrate + ZUPT
    print("Integrating trajectory with ZUPT...")
    pos, vel, stationary = _estimate_trajectory(
        t, a_lin, stationary_thresh=args.stationary_thresh, zupt_window=args.zupt_window
    )

    stationary_pct = np.mean(stationary) * 100
    print(f"Stationary frames: {stationary_pct:.1f}%")
    print(f"Final position: [{pos[-1, 0]:.3f}, {pos[-1, 1]:.3f}, {pos[-1, 2]:.3f}] m")
    print(f"Total distance: {np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)):.3f} m")

    out_path = Path(args.output)
    np.savez(
        out_path,
        timestamps=t,
        positions=pos,
        velocities=vel,
        orientations=q_i,  # wxyz
        stationary=stationary.astype(np.uint8),
    )

    print(f"Saved trajectory to {out_path}")

    if args.plot:
        if plt is None:
            print("matplotlib not available; skipping plot.", file=sys.stderr)
            return
        fig = plt.figure(figsize=(12, 5))

        # 3D trajectory
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=1.5)
        ax1.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], c='green', s=50, label='Start')
        ax1.scatter([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], c='red', s=50, label='End')
        ax1.set_title("Estimated 3D Trajectory")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.legend()

        # Position over time
        ax2 = fig.add_subplot(122)
        ax2.plot(t - t[0], pos[:, 0], label='X')
        ax2.plot(t - t[0], pos[:, 1], label='Y')
        ax2.plot(t - t[0], pos[:, 2], label='Z')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (m)")
        ax2.set_title("Position Components")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
