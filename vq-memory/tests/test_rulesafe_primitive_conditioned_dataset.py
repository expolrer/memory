from pathlib import Path
import sys

import pytest
import zarr


DP3_ROOT = Path("third_party/3D-Diffusion-Policy/3D-Diffusion-Policy")
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.dataset.rulesafe_conditioned_dataset import RuleSafeConditionedDataset


PRIMITIVE_ZARR = Path("data/dp3_rulesafe_primitive/rulesafe_20rules_1000eps_256pts.zarr")


def test_primitive_zarr_contains_all_policy_conditions():
    if not PRIMITIVE_ZARR.exists():
        pytest.skip(f"primitive zarr is not available: {PRIMITIVE_ZARR}")
    root = zarr.open_group(str(PRIMITIVE_ZARR), mode="r")
    total_steps = root["data/state"].shape[0]
    assert root["data/memory_tokens"].shape == (total_steps, 40)
    assert root["data/rule_onehot"].shape == (total_steps, 20)
    assert root["data/plan_tokens"].shape == (total_steps, 8)
    assert root["data/next_plan_token"].shape == (total_steps, 1)
    assert root.attrs["joint_source"] == "primitive"
    assert root.attrs["rule_condition"] == "onehot"


def test_conditioned_dataset_reads_primitive_zarr():
    if not PRIMITIVE_ZARR.exists():
        pytest.skip(f"primitive zarr is not available: {PRIMITIVE_ZARR}")
    dataset = RuleSafeConditionedDataset(
        zarr_path=str(PRIMITIVE_ZARR),
        horizon=4,
        pad_before=1,
        pad_after=1,
        val_ratio=0.02,
        use_memory=True,
        use_plan=True,
        use_next_plan=True,
    )
    sample = dataset[0]
    assert sample["obs"]["agent_pos"].shape == (4, 13)
    assert sample["obs"]["point_cloud"].shape == (4, 256, 6)
    assert sample["obs"]["memory_tokens"].shape == (4, 40)
    assert sample["obs"]["rule_onehot"].shape == (4, 20)
    assert sample["obs"]["plan_tokens"].shape == (4, 8)
    assert sample["obs"]["next_plan_token"].shape == (4, 1)
    assert sample["action"].shape == (4, 13)
    assert sample["obs"]["rule_onehot"][0].tolist() == [1.0] + [0.0] * 19
    assert sample["obs"]["plan_tokens"][0].tolist() == [2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert sample["obs"]["next_plan_token"][0].tolist() == [2.0]
