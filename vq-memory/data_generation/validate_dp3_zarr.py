import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr


def validate_zarr(zarr_path: Path, dp3_root: Path | None = None, sample_index: int = 0) -> dict:
    root = zarr.open_group(str(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("zarr root must contain data and meta groups")
    data = root["data"]
    meta = root["meta"]
    required = ["state", "action", "point_cloud"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing /data keys: {missing}")
    if "episode_ends" not in meta:
        raise ValueError("missing /meta/episode_ends")

    episode_ends = meta["episode_ends"][:]
    if episode_ends.ndim != 1 or len(episode_ends) == 0:
        raise ValueError("episode_ends must be a non-empty 1D array")
    if not np.all(np.diff(episode_ends) > 0):
        raise ValueError("episode_ends must be strictly increasing")
    total_steps = int(episode_ends[-1])
    for key in required:
        arr = data[key]
        if arr.shape[0] != total_steps:
            raise ValueError(f"/data/{key} first dimension {arr.shape[0]} != episode_ends[-1] {total_steps}")
        probe = arr[min(sample_index, arr.shape[0] - 1)]
        if not np.isfinite(probe).all():
            raise ValueError(f"/data/{key} sample contains non-finite values")

    summary = {
        "zarr_path": str(zarr_path),
        "episodes": int(len(episode_ends)),
        "total_steps": total_steps,
        "state_shape": list(data["state"].shape),
        "action_shape": list(data["action"].shape),
        "point_cloud_shape": list(data["point_cloud"].shape),
        "attrs": dict(root.attrs),
    }

    if dp3_root is not None:
        sys.path.insert(0, str(dp3_root))
        from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
        from diffusion_policy_3d.dataset.metaworld_dataset import MetaworldDataset

        replay_buffer = ReplayBuffer.copy_from_path(str(zarr_path), keys=required)
        dataset = MetaworldDataset(
            zarr_path=str(zarr_path),
            horizon=4,
            pad_before=0,
            pad_after=0,
            val_ratio=0.0,
        )
        item = dataset[min(sample_index, len(dataset) - 1)]
        summary["dp3_replay_buffer"] = {
            "n_episodes": int(replay_buffer.n_episodes),
            "n_steps": int(replay_buffer.n_steps),
        }
        summary["dp3_dataset_sample"] = {
            "obs.point_cloud": list(item["obs"]["point_cloud"].shape),
            "obs.agent_pos": list(item["obs"]["agent_pos"].shape),
            "action": list(item["action"].shape),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DP3/MetaworldDataset-compatible zarr replay buffer.")
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--dp3-root", type=Path, default=None, help="Path containing the diffusion_policy_3d package.")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(validate_zarr(args.zarr, args.dp3_root, args.sample_index), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
