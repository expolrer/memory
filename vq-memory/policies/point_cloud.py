import math
from typing import Tuple

import numpy as np


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _sample_box_surface(center, half_size, rot, count: int, rng: np.random.Generator, part_id: float):
    center = np.asarray(center, dtype=np.float32)
    half = np.asarray(half_size, dtype=np.float32)
    face = rng.integers(0, 6, size=count)
    pts = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32) * half
    axis = face // 2
    sign = (face % 2) * 2 - 1
    pts[np.arange(count), axis] = sign.astype(np.float32) * half[axis]
    pts = pts @ rot.T + center
    # Simple semantic features mimic colored point clouds: normalized part id and constant occupancy.
    feats = np.zeros((count, 3), dtype=np.float32)
    feats[:, 0] = part_id
    feats[:, 1] = 1.0 - part_id
    feats[:, 2] = 1.0
    return np.concatenate([pts, feats], axis=1).astype(np.float32)


def primitive_safe_point_cloud(joint_state: np.ndarray, point_count: int = 256, seed: int = 0) -> np.ndarray:
    """Generate an analytic point cloud from the primitive safe joint vector.

    The point layout is compatible with DP3-style point observations: [N, 6]
    with xyz plus three lightweight semantic/color channels. It avoids SAPIEN
    rendering, which is unstable in the current container.
    """
    joint_state = np.asarray(joint_state, dtype=np.float32)
    rng = np.random.default_rng(seed)
    knob_angle = float(joint_state[0]) * math.radians(90.0)
    handle_angle = float(joint_state[1]) * math.radians(55.0)
    door_angle = float(joint_state[2]) * math.radians(75.0)
    door_center = [float(joint_state[10]), float(joint_state[11]), 0.35]

    counts = np.full(4, point_count // 4, dtype=np.int64)
    counts[: point_count % 4] += 1
    parts = [
        _sample_box_surface([0.0, 0.0, 0.35], [0.32, 0.08, 0.24], np.eye(3, dtype=np.float32), int(counts[0]), rng, 0.10),
        _sample_box_surface(door_center, [0.30, 0.025, 0.22], _rot_z(door_angle), int(counts[1]), rng, 0.35),
        _sample_box_surface([float(joint_state[6]), -0.125, float(joint_state[7])], [0.045, 0.025, 0.045], _rot_z(knob_angle), int(counts[2]), rng, 0.65),
        _sample_box_surface([float(joint_state[8]), -0.125, float(joint_state[9])], [0.025, 0.025, 0.12], _rot_x(handle_angle), int(counts[3]), rng, 0.90),
    ]
    pc = np.concatenate(parts, axis=0)
    if pc.shape[0] != point_count:
        pc = pc[:point_count]
    return pc.astype(np.float32)
