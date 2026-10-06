#!/usr/bin/env python3
"""
Post-Smoothing Script for Dyn-HaMR Output
=========================================

Applies two-pass RTS (Rauch-Tung-Striebel) smoothing to MANO parameters
for "buttery" motion with zero lag (offline processing).

Smooths:
- Translation (t): Linear RTS on R³
- Rotation (R): Smooth in SO(3) tangent space (axis-angle)
- Pose (θ): RTS on joint angles
- Shape (β): Usually constant, light smoothing if needed

Usage:
    python 03_smooth_params.py <input_params> [output_params]
    python 03_smooth_params.py raw_params.npz smoothed_params.npz
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pickle

# =============================================================================
# RTS Smoother Implementation
# =============================================================================

class RTSSmoother:
    """
    Rauch-Tung-Striebel (RTS) two-pass smoother.

    Forward pass: Kalman filter
    Backward pass: RTS smoothing (uses future information)
    Result: Optimal smoothed estimates with zero lag.
    """

    def __init__(self,
                 process_noise: float = 0.01,
                 measurement_noise: float = 1.0,
                 model: str = 'constant_velocity'):
        """
        Args:
            process_noise: Q - higher = more responsive, lower = smoother
            measurement_noise: R - higher = smoother, lower = trust data more
            model: 'constant_velocity' or 'constant_acceleration'
        """
        self.q = process_noise
        self.r = measurement_noise
        self.model = model

    def smooth(self, measurements: np.ndarray, return_velocity: bool = False):
        """
        Smooth a sequence of measurements.

        Args:
            measurements: (T, D) array of measurements over T timesteps
            return_velocity: If True, also return velocity estimates

        Returns:
            If return_velocity=False: smoothed (T, D) array
            If return_velocity=True: (smoothed (T, D), velocities (T, D))
        """
        T, D = measurements.shape

        # State dimension: [x, v] for constant velocity
        if self.model == 'constant_velocity':
            state_dim = 2
        else:  # constant_acceleration
            state_dim = 3

        # Process each dimension independently
        smoothed = np.zeros_like(measurements)
        velocities = np.zeros_like(measurements) if return_velocity else None

        for d in range(D):
            z = measurements[:, d]
            if return_velocity:
                smoothed[:, d], velocities[:, d] = self._smooth_1d(z, state_dim, return_velocity=True)
            else:
                smoothed[:, d] = self._smooth_1d(z, state_dim, return_velocity=False)

        if return_velocity:
            return smoothed, velocities
        return smoothed

    def _smooth_1d(self, z: np.ndarray, state_dim: int, return_velocity: bool = False):
        """Smooth a 1D signal using RTS.

        Args:
            z: 1D signal to smooth
            state_dim: 2 for constant_velocity, 3 for constant_acceleration
            return_velocity: If True, return (position, velocity) tuple

        Returns:
            If return_velocity=False: smoothed positions (T,)
            If return_velocity=True: (positions (T,), velocities (T,))
        """
        T = len(z)

        # State transition matrix (constant velocity model)
        dt = 1.0  # Normalized time step
        if state_dim == 2:
            F = np.array([[1, dt],
                          [0, 1]])
            H = np.array([[1, 0]])
            Q = self.q * np.array([[dt**3/3, dt**2/2],
                                   [dt**2/2, dt]])
        else:  # constant_acceleration
            F = np.array([[1, dt, 0.5*dt**2],
                          [0, 1, dt],
                          [0, 0, 1]])
            H = np.array([[1, 0, 0]])
            Q = self.q * np.eye(3)

        R = np.array([[self.r]])

        # Initialize
        x = np.zeros((T, state_dim))
        P = np.zeros((T, state_dim, state_dim))

        x[0, 0] = z[0]
        P[0] = np.eye(state_dim) * 1.0

        # Forward pass (Kalman filter)
        x_pred = np.zeros((T, state_dim))
        P_pred = np.zeros((T, state_dim, state_dim))

        for t in range(1, T):
            # Predict
            x_pred[t] = F @ x[t-1]
            P_pred[t] = F @ P[t-1] @ F.T + Q

            # Update
            y = z[t] - H @ x_pred[t]
            S = H @ P_pred[t] @ H.T + R
            K = P_pred[t] @ H.T @ np.linalg.inv(S)

            x[t] = x_pred[t] + K @ y
            P[t] = (np.eye(state_dim) - K @ H) @ P_pred[t]

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((T, state_dim))
        x_smooth[-1] = x[-1]

        for t in range(T-2, -1, -1):
            G = P[t] @ F.T @ np.linalg.pinv(P_pred[t+1])
            x_smooth[t] = x[t] + G @ (x_smooth[t+1] - x_pred[t+1])

        if return_velocity:
            return x_smooth[:, 0], x_smooth[:, 1]
        return x_smooth[:, 0]


# =============================================================================
# Rotation Smoothing (SO(3))
# =============================================================================

def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2,1] - R[1,2],
                     R[0,2] - R[2,0],
                     R[1,0] - R[0,1]]) / (2 * np.sin(angle))
    return axis * angle


def smooth_rotations(rotations: np.ndarray, smoother: RTSSmoother) -> np.ndarray:
    """
    Smooth rotations in tangent space to avoid gimbal lock / quaternion flips.

    Args:
        rotations: (T, 3) axis-angle representations
        smoother: RTSSmoother instance

    Returns:
        smoothed: (T, 3) smoothed axis-angle representations
    """
    T = len(rotations)

    # Convert to incremental rotations (tangent space)
    increments = np.zeros((T, 3))
    R_prev = np.eye(3)

    for t in range(T):
        R_curr = axis_angle_to_matrix(rotations[t])
        dR = R_prev.T @ R_curr
        increments[t] = matrix_to_axis_angle(dR)
        R_prev = R_curr

    # Smooth increments
    smoothed_increments = smoother.smooth(increments)

    # Reconstruct rotations
    smoothed = np.zeros((T, 3))
    R_curr = axis_angle_to_matrix(rotations[0])  # Start from first frame

    for t in range(T):
        if t > 0:
            dR = axis_angle_to_matrix(smoothed_increments[t])
            R_curr = R_curr @ dR
        smoothed[t] = matrix_to_axis_angle(R_curr)

    return smoothed


# =============================================================================
# Main Processing
# =============================================================================

def load_params(filepath: Path) -> dict:
    """Load parameters from .npz or .pkl file."""
    if filepath.suffix == '.npz':
        data = dict(np.load(filepath, allow_pickle=True))
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")
    return data


def save_params(data: dict, filepath: Path):
    """Save parameters to .npz or .pkl file."""
    if filepath.suffix == '.npz':
        np.savez(filepath, **data)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")


def smooth_mano_params(params: dict,
                       translation_noise: float = 1.0,
                       rotation_noise: float = 2.0,
                       pose_noise: float = 1.5,
                       process_noise: float = 0.01,
                       save_velocities: bool = False) -> dict:
    """
    Apply RTS smoothing to MANO parameters.

    Args:
        params: Dictionary containing MANO parameters
        translation_noise: Measurement noise for translation (higher = smoother)
        rotation_noise: Measurement noise for rotation
        pose_noise: Measurement noise for pose/joint angles
        process_noise: Process noise (lower = smoother)
        save_velocities: If True, also save velocity estimates from the smoother

    Returns:
        smoothed_params: Dictionary with smoothed parameters
                         If save_velocities=True, includes velocity keys like 'trans_velocity'
    """
    smoothed = {}

    # Copy non-smoothed parameters
    for key, value in params.items():
        smoothed[key] = value

    # Common parameter names in MANO outputs
    translation_keys = ['trans', 'translation', 'global_t', 't', 'transl']
    rotation_keys = ['global_orient', 'root_orient', 'global_r', 'rot', 'orient']
    pose_keys = ['hand_pose', 'pose', 'body_pose', 'theta']
    shape_keys = ['betas', 'shape', 'beta']

    # Find and smooth translation
    for key in translation_keys:
        if key in params:
            print(f"  Smoothing translation: {key}")
            smoother = RTSSmoother(process_noise=process_noise,
                                   measurement_noise=translation_noise)
            data = np.array(params[key])
            if data.ndim == 1:
                data = data.reshape(-1, 3)
            if save_velocities:
                smoothed[key], smoothed[f'{key}_velocity'] = smoother.smooth(data, return_velocity=True)
                print(f"    -> Saved {key}_velocity")
            else:
                smoothed[key] = smoother.smooth(data)
            break

    # Find and smooth rotation
    for key in rotation_keys:
        if key in params:
            print(f"  Smoothing rotation: {key}")
            smoother = RTSSmoother(process_noise=process_noise,
                                   measurement_noise=rotation_noise)
            data = np.array(params[key])
            if data.ndim == 1:
                data = data.reshape(-1, 3)
            # Note: smooth_rotations doesn't support velocity output yet (rotation velocity is complex)
            smoothed[key] = smooth_rotations(data, smoother)
            break

    # Find and smooth pose
    for key in pose_keys:
        if key in params:
            print(f"  Smoothing pose: {key}")
            smoother = RTSSmoother(process_noise=process_noise,
                                   measurement_noise=pose_noise)
            data = np.array(params[key])
            original_shape = data.shape
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            if save_velocities:
                smoothed_flat, velocity_flat = smoother.smooth(data, return_velocity=True)
                smoothed[key] = smoothed_flat.reshape(original_shape)
                smoothed[f'{key}_velocity'] = velocity_flat.reshape(original_shape)
                print(f"    -> Saved {key}_velocity")
            else:
                smoothed[key] = smoother.smooth(data).reshape(original_shape)
            break

    # Shape is usually constant, but light smoothing if needed
    for key in shape_keys:
        if key in params:
            data = np.array(params[key])
            if data.ndim > 1 and data.shape[0] > 1:
                print(f"  Light smoothing shape: {key}")
                smoother = RTSSmoother(process_noise=0.001,
                                       measurement_noise=5.0)
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                smoothed[key] = smoother.smooth(data)
            break

    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description='Post-smooth Dyn-HaMR MANO parameters for buttery motion')
    parser.add_argument('input', type=Path,
                        help='Input parameter file (.npz or .pkl)')
    parser.add_argument('output', type=Path, nargs='?',
                        help='Output parameter file (default: input_smoothed.npz)')
    parser.add_argument('--translation-noise', type=float, default=1.0,
                        help='Translation smoothing (higher=smoother, default: 1.0)')
    parser.add_argument('--rotation-noise', type=float, default=2.0,
                        help='Rotation smoothing (higher=smoother, default: 2.0)')
    parser.add_argument('--pose-noise', type=float, default=1.5,
                        help='Pose/joint smoothing (higher=smoother, default: 1.5)')
    parser.add_argument('--process-noise', type=float, default=0.01,
                        help='Process noise (lower=smoother, default: 0.01)')
    parser.add_argument('--save-velocities', action='store_true',
                        help='Save velocity estimates from the RTS smoother (adds *_velocity keys)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_smoothed{args.input.suffix}"

    print("========================================")
    print("MANO Parameter Post-Smoothing")
    print("========================================")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Settings:")
    print(f"  Translation noise: {args.translation_noise}")
    print(f"  Rotation noise:    {args.rotation_noise}")
    print(f"  Pose noise:        {args.pose_noise}")
    print(f"  Process noise:     {args.process_noise}")
    print(f"  Save velocities:   {args.save_velocities}")
    print("")

    # Load parameters
    print("Loading parameters...")
    params = load_params(args.input)
    print(f"  Found keys: {list(params.keys())}")
    print("")

    # Smooth
    print("Applying RTS smoothing...")
    smoothed = smooth_mano_params(
        params,
        translation_noise=args.translation_noise,
        rotation_noise=args.rotation_noise,
        pose_noise=args.pose_noise,
        process_noise=args.process_noise,
        save_velocities=args.save_velocities
    )
    print("")

    # Save
    print(f"Saving smoothed parameters to {args.output}...")
    save_params(smoothed, args.output)

    if args.save_velocities:
        velocity_keys = [k for k in smoothed.keys() if '_velocity' in k]
        print(f"  Velocity keys saved: {velocity_keys}")

    print("")
    print("========================================")
    print("Smoothing complete!")
    print("========================================")
    print("")
    print("Next: Re-render with smoothed params using Dyn-HaMR visualization")
    print("========================================")


if __name__ == '__main__':
    main()
