import json

import numpy as np
import zarr

from data_generation.add_rulesafe_rule_condition_to_dp3_zarr import add_rule_condition


def test_add_rule_condition_writes_onehot_per_episode(tmp_path):
    path = tmp_path / "tiny.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_group("data")
    meta = root.create_group("meta")
    meta.array("episode_ends", np.asarray([2, 5], dtype=np.int64))
    meta.array("rule_index", np.asarray([0, 1], dtype=np.int64))
    root.attrs["rule_id_to_index"] = json.dumps({"rule001": 0, "rule020": 1})

    summary = add_rule_condition(path)

    root = zarr.open_group(str(path), mode="r")
    values = root["data/rule_onehot"][:]
    assert values.shape == (5, 2)
    assert values.tolist() == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert summary["rule_condition_dim"] == 2
    assert root.attrs["rule_condition"] == "onehot"
    assert int(root.attrs["rule_condition_dim"]) == 2
