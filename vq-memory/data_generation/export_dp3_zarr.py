import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr
from numcodecs import Blosc

from policies.point_cloud import primitive_safe_point_cloud


def _select_episodes(total: int, max_episodes: Optional[int]) -> np.ndarray:
    if max_episodes is None or max_episodes >= total:
        return np.arange(total, dtype=np.int64)
    if max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided")
    return np.arange(max_episodes, dtype=np.int64)


def _as_text_list(values: Iterable[object]) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _as_scalar_text(value: object) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.ndim == 0 else str(value)


def export_dp3_zarr(
    npz_path: Path,
    out_path: Path,
    point_count: int = 1024,
    max_episodes: Optional[int] = None,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_path)

    with np.load(npz_path, allow_pickle=True) as payload:
        joint_states = np.asarray(payload["joint_states"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.float32)
        rule_ids = _as_text_list(payload["rule_ids"]) if "rule_ids" in payload else ["unknown"] * len(joint_states)
        success = np.asarray(payload["success"], dtype=np.bool_) if "success" in payload else np.ones(len(joint_states), dtype=np.bool_)
        process_scores = np.asarray(payload["process_scores"], dtype=np.float32) if "process_scores" in payload else np.ones(len(joint_states), dtype=np.float32)
        joint_source = _as_scalar_text(payload["joint_source"]) if "joint_source" in payload else "unknown"
        rule_profile = _as_scalar_text(payload["rule_profile"]) if "rule_profile" in payload else "historical"

    if joint_states.ndim != 3:
        raise ValueError(f"joint_states must have shape [episodes, T, D], got {joint_states.shape}")
    if actions.shape != joint_states.shape:
        raise ValueError(f"actions shape {actions.shape} must match joint_states shape {joint_states.shape}")
    if not np.isfinite(joint_states).all() or not np.isfinite(actions).all():
        raise ValueError("joint_states/actions contain non-finite values")

    episode_indices = _select_episodes(joint_states.shape[0], max_episodes)
    num_episodes = int(len(episode_indices))
    horizon = int(joint_states.shape[1])
    state_dim = int(joint_states.shape[2])
    total_steps = int(num_episodes * horizon)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(out_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    time_chunk = max(1, min(horizon, 128))
    state_arr = data.zeros(
        "state",
        shape=(total_steps, state_dim),
        chunks=(time_chunk, state_dim),
        dtype="f4",
        compressor=compressor,
    )
    action_arr = data.zeros(
        "action",
        shape=(total_steps, state_dim),
        chunks=(time_chunk, state_dim),
        dtype="f4",
        compressor=compressor,
    )
    pc_arr = data.zeros(
        "point_cloud",
        shape=(total_steps, point_count, 6),
        chunks=(max(1, min(time_chunk, 64)), point_count, 6),
        dtype="f4",
        compressor=compressor,
    )

    episode_ends = np.arange(horizon, total_steps + 1, horizon, dtype=np.int64)
    meta.array("episode_ends", episode_ends, dtype="i8", compressor=None)
    meta.array("source_episode_index", episode_indices, dtype="i8", compressor=None)
    meta.array("success", success[episode_indices].astype(np.bool_), dtype="bool", compressor=None)
    meta.array("process_score", process_scores[episode_indices].astype(np.float32), dtype="f4", compressor=None)

    selected_rule_ids = [rule_ids[int(i)] for i in episode_indices]
    unique_rules = sorted(set(selected_rule_ids))
    rule_to_index = {rule: idx for idx, rule in enumerate(unique_rules)}
    rule_index = np.asarray([rule_to_index[rule] for rule in selected_rule_ids], dtype=np.int64)
    meta.array("rule_index", rule_index, dtype="i8", compressor=None)

    root.attrs["source_npz"] = str(npz_path)
    root.attrs["joint_source"] = joint_source
    root.attrs["rule_profile"] = rule_profile
    root.attrs["point_count"] = int(point_count)
    root.attrs["state_dim"] = int(state_dim)
    root.attrs["action_dim"] = int(state_dim)
    root.attrs["trajectory_length"] = int(horizon)
    root.attrs["rule_id_to_index"] = json.dumps(rule_to_index, sort_keys=True)
    root.attrs["format"] = "dp3_metaworld_replay_buffer"

    for out_episode_idx, source_episode_idx in enumerate(episode_indices):
        start = out_episode_idx * horizon
        end = start + horizon
        states = joint_states[int(source_episode_idx)]
        state_arr[start:end] = states
        action_arr[start:end] = actions[int(source_episode_idx)]
        point_cloud = np.empty((horizon, point_count, 6), dtype=np.float32)
        base_seed = int(seed + int(source_episode_idx) * 1000003)
        for t in range(horizon):
            point_cloud[t] = primitive_safe_point_cloud(states[t], point_count=point_count, seed=base_seed + t)
        pc_arr[start:end] = point_cloud

    summary = {
        "out_path": str(out_path),
        "num_episodes": num_episodes,
        "trajectory_length": horizon,
        "total_steps": total_steps,
        "state_shape": list(state_arr.shape),
        "action_shape": list(action_arr.shape),
        "point_cloud_shape": list(pc_arr.shape),
        "point_count": point_count,
        "rules": unique_rules,
        "joint_source": joint_source,
        "rule_profile": rule_profile,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RuleSafe NPZ demonstrations to a DP3-compatible zarr replay buffer.")
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = export_dp3_zarr(
        npz_path=args.npz,
        out_path=args.out,
        point_count=args.point_count,
        max_episodes=args.max_episodes,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
