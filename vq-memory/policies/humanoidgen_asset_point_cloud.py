import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from policies.point_cloud import primitive_safe_point_cloud


DEFAULT_DOOR_ASSET = Path("third_party/HumanoidGen/assets/objects/articulated_objs/door/8877")


def _axis_angle_rotate(points: np.ndarray, origin: np.ndarray, direction: np.ndarray, angle: float) -> np.ndarray:
    direction = direction.astype(np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    shifted = points - origin[None, :]
    c = math.cos(angle)
    s = math.sin(angle)
    cross = np.cross(direction[None, :], shifted)
    dot = shifted @ direction
    rotated = shifted * c + cross * s + direction[None, :] * dot[:, None] * (1.0 - c)
    return rotated + origin[None, :]


@lru_cache(maxsize=8)
def _load_door_asset(asset_root_str: str):
    asset_root = Path(asset_root_str)
    points = np.loadtxt(asset_root / "point_sample" / "sample-points-all-pts-nor-rgba-10000.txt").astype(np.float32)
    labels = np.loadtxt(asset_root / "point_sample" / "sample-points-all-label-10000.txt").astype(np.int64)
    with open(asset_root / "mobility_v2.json", "r", encoding="utf-8") as f:
        mobility = json.load(f)
    joint_by_label = {}
    for joint in mobility:
        joint_data = joint.get("jointData") or {}
        axis = joint_data.get("axis")
        if not axis:
            continue
        for part in joint.get("parts", []):
            joint_by_label[int(part["id"])] = {
                "origin": np.asarray(axis["origin"], dtype=np.float32),
                "direction": np.asarray(axis["direction"], dtype=np.float32),
                "name": joint.get("name", ""),
            }
    xyz = points[:, :3]
    rgb = np.clip(points[:, 6:9] / 255.0, 0.0, 1.0)
    return xyz, rgb, labels, joint_by_label


def humanoidgen_door_point_cloud(
    joint_state: np.ndarray,
    point_count: int = 256,
    seed: int = 0,
    asset_root: Path = DEFAULT_DOOR_ASSET,
    asset_fraction: float = 0.75,
) -> np.ndarray:
    """Generate a DP3-style point cloud from real HumanoidGen door assets.

    The SAPIEN visual path currently segfaults in this container, so this uses
    HumanoidGen's pre-sampled articulated door point cloud and applies a simple
    hinge transform to moving door panels. Primitive knob/handle samples are
    appended so RuleSafe lock states remain observable.
    """
    joint_state = np.asarray(joint_state, dtype=np.float32)
    rng = np.random.default_rng(seed)
    xyz, rgb, labels, joint_by_label = _load_door_asset(str(asset_root))

    asset_count = int(round(point_count * asset_fraction))
    asset_count = max(1, min(point_count, asset_count))
    primitive_count = point_count - asset_count
    idx = rng.choice(len(xyz), size=asset_count, replace=len(xyz) < asset_count)
    pts = xyz[idx].copy()
    colors = rgb[idx].copy()
    part_labels = labels[idx]

    door_angle = float(joint_state[2]) * math.radians(75.0)
    for label, joint in joint_by_label.items():
        mask = part_labels == label
        if not np.any(mask):
            continue
        sign = 1.0 if label == 7 else -1.0
        pts[mask] = _axis_angle_rotate(pts[mask], joint["origin"], joint["direction"], sign * door_angle)

    # Normalize and place the asset door near the primitive safe coordinate frame.
    pts = pts.astype(np.float32)
    pts[:, 0] *= 0.55
    pts[:, 1] *= 0.35
    pts[:, 2] = pts[:, 2] * 0.8 + 0.35

    asset_pc = np.concatenate([pts, colors.astype(np.float32)], axis=1)
    if primitive_count > 0:
        primitive = primitive_safe_point_cloud(joint_state, point_count=primitive_count, seed=seed + 17)
        # Keep mostly knob/handle/door-lock cues from the primitive cloud. Its last
        # channels are semantic colors and are already in [0, 1].
        pc = np.concatenate([asset_pc, primitive], axis=0)
    else:
        pc = asset_pc
    if pc.shape[0] != point_count:
        pc = pc[:point_count]
    return pc.astype(np.float32)
