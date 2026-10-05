import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc


def _rule_mapping(root) -> dict:
    raw = root.attrs.get("rule_id_to_index", "{}")
    mapping = {str(rule_id): int(index) for rule_id, index in json.loads(raw).items()}
    if not mapping:
        raise KeyError("zarr attrs rule_id_to_index is required")
    expected = list(range(len(mapping)))
    if sorted(mapping.values()) != expected:
        raise ValueError(f"rule indices must be contiguous from zero, got {sorted(mapping.values())}")
    return mapping


def add_rule_condition(zarr_path: Path) -> dict:
    root = zarr.open_group(str(zarr_path), mode="a")
    if "episode_ends" not in root["meta"] or "rule_index" not in root["meta"]:
        raise KeyError("/meta/episode_ends and /meta/rule_index are required")

    rule_to_index = _rule_mapping(root)
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    episode_rule_indices = np.asarray(root["meta/rule_index"][:], dtype=np.int64)
    if len(episode_ends) != len(episode_rule_indices):
        raise ValueError("episode_ends and rule_index must have the same length")

    total_steps = int(episode_ends[-1])
    condition_dim = len(rule_to_index)
    data = root["data"]
    if "rule_onehot" in data:
        del data["rule_onehot"]
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    condition = data.zeros(
        "rule_onehot",
        shape=(total_steps, condition_dim),
        chunks=(min(1024, total_steps), condition_dim),
        dtype="f4",
        compressor=compressor,
    )

    start = 0
    for rule_index, end in zip(episode_rule_indices, episode_ends):
        index = int(rule_index)
        if index < 0 or index >= condition_dim:
            raise ValueError(f"rule index {index} is outside [0, {condition_dim})")
        onehot = np.zeros((condition_dim,), dtype=np.float32)
        onehot[index] = 1.0
        episode_length = int(end) - start
        condition[start : int(end)] = np.repeat(onehot[None, :], episode_length, axis=0)
        start = int(end)

    root.attrs["rule_condition"] = "onehot"
    root.attrs["rule_condition_dim"] = condition_dim
    return {
        "zarr_path": str(zarr_path),
        "rule_onehot_shape": list(condition.shape),
        "rule_condition_dim": condition_dim,
        "rules": sorted(rule_to_index),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add per-timestep RuleSafe one-hot task conditioning to a DP3 zarr.")
    parser.add_argument("--zarr", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(add_rule_condition(args.zarr), indent=2))


if __name__ == "__main__":
    main()
