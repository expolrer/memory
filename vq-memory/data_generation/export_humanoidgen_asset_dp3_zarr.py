import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from numcodecs import Blosc

from policies.humanoidgen_asset_point_cloud import humanoidgen_door_point_cloud


def export_asset_zarr(npz_path: Path, out_path: Path, point_count: int = 256, max_episodes: Optional[int] = None, overwrite: bool = False) -> dict:
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite")
        shutil.rmtree(out_path)
    with np.load(npz_path, allow_pickle=True) as payload:
        joint_states = np.asarray(payload["joint_states"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.float32)
        rule_ids = [str(x) for x in payload["rule_ids"]] if "rule_ids" in payload else ["unknown"] * len(joint_states)
        success = np.asarray(payload["success"], dtype=np.bool_) if "success" in payload else np.ones(len(joint_states), dtype=np.bool_)
        process_scores = np.asarray(payload["process_scores"], dtype=np.float32) if "process_scores" in payload else np.ones(len(joint_states), dtype=np.float32)
        joint_source = str(payload["joint_source"]) if "joint_source" in payload else "unknown"
    if max_episodes is None or max_episodes >= len(joint_states):
        episode_indices = np.arange(len(joint_states), dtype=np.int64)
    else:
        episode_indices = np.unique(np.linspace(0, len(joint_states) - 1, num=max_episodes, dtype=np.int64))
    n_ep = int(len(episode_indices))
    horizon = int(joint_states.shape[1])
    state_dim = int(joint_states.shape[2])
    total = n_ep * horizon
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(out_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.zeros("state", shape=(total, state_dim), chunks=(min(128, horizon), state_dim), dtype="f4", compressor=compressor)
    data.zeros("action", shape=(total, state_dim), chunks=(min(128, horizon), state_dim), dtype="f4", compressor=compressor)
    data.zeros("point_cloud", shape=(total, point_count, 6), chunks=(min(64, horizon), point_count, 6), dtype="f4", compressor=compressor)
    meta.array("episode_ends", np.arange(horizon, total + 1, horizon, dtype=np.int64), dtype="i8", compressor=None)
    meta.array("source_episode_index", episode_indices, dtype="i8", compressor=None)
    meta.array("success", success[episode_indices].astype(np.bool_), dtype="bool", compressor=None)
    meta.array("process_score", process_scores[episode_indices].astype(np.float32), dtype="f4", compressor=None)
    selected_rules = [rule_ids[int(i)] for i in episode_indices]
    rule_to_index = {rule: idx for idx, rule in enumerate(sorted(set(selected_rules)))}
    meta.array("rule_index", np.asarray([rule_to_index[r] for r in selected_rules], dtype=np.int64), dtype="i8", compressor=None)
    root.attrs["source_npz"] = str(npz_path)
    root.attrs["joint_source"] = joint_source
    root.attrs["point_count"] = int(point_count)
    root.attrs["state_dim"] = state_dim
    root.attrs["action_dim"] = state_dim
    root.attrs["trajectory_length"] = horizon
    root.attrs["rule_id_to_index"] = json.dumps(rule_to_index, sort_keys=True)
    root.attrs["format"] = "dp3_metaworld_replay_buffer"
    root.attrs["point_cloud_source"] = "humanoidgen_door_8877_point_sample_plus_primitive_lock_parts"
    for out_ep, src_ep in enumerate(episode_indices):
        start = out_ep * horizon
        end = start + horizon
        states = joint_states[int(src_ep)]
        data["state"][start:end] = states
        data["action"][start:end] = actions[int(src_ep)]
        pcs = np.empty((horizon, point_count, 6), dtype=np.float32)
        for t in range(horizon):
            pcs[t] = humanoidgen_door_point_cloud(states[t], point_count=point_count, seed=int(src_ep) * 1000003 + t)
        data["point_cloud"][start:end] = pcs
    return {
        "out_path": str(out_path),
        "episodes": n_ep,
        "total_steps": total,
        "state_shape": list(data["state"].shape),
        "action_shape": list(data["action"].shape),
        "point_cloud_shape": list(data["point_cloud"].shape),
        "rules": sorted(set(selected_rules)),
        "point_cloud_source": root.attrs["point_cloud_source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RuleSafe NPZ to DP3 zarr using HumanoidGen asset-backed point clouds.")
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--point-count", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_asset_zarr(args.npz, args.out, args.point_count, args.max_episodes, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
