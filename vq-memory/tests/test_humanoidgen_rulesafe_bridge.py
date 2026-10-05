from pathlib import Path

import pytest

from rulesafe.envs import RuleSafeSapienEnv
from scripts.build_humanoidgen_rulesafe_bridge import build_bridge
from scripts.inventory_humanoidgen_safe_assets import parse_safe_urdf


OFFICIAL_SOURCE = Path(
    "third_party/HumanoidGen/humanoidgen/assets/objects/articulated_objs/"
    "safe_rotate_new/101612"
)
BRIDGE_URDF = Path("artifacts/humanoidgen_rulesafe_bridge/safe_101612/mobility.urdf")


def test_build_bridge_promotes_three_semantic_joints(tmp_path):
    if not OFFICIAL_SOURCE.is_dir():
        pytest.skip(f"official HumanoidGen source asset not available: {OFFICIAL_SOURCE}")
    output = tmp_path / "safe_101612"
    manifest = build_bridge(OFFICIAL_SOURCE, output)
    parsed = parse_safe_urdf(output / "mobility.urdf")
    assert manifest["status"] == "reconstructed_not_official_rulesafe"
    assert parsed["movable_joint_count"] == 3
    assert [joint["name"] for joint in parsed["movable_joints"]] == [
        "door_joint",
        "knob_joint",
        "handle_joint",
    ]


def test_reconstructed_bridge_runs_paper_rule001_in_sapien():
    if not BRIDGE_URDF.is_file():
        pytest.skip(f"reconstructed bridge not available: {BRIDGE_URDF}")
    env = RuleSafeSapienEnv(
        "rule001",
        rule_profile="paper_fidelity",
        asset_urdf=str(BRIDGE_URDF),
    )
    env.apply_symbolic_event("knob", True)
    env.apply_symbolic_event("handle", True)
    env.apply_symbolic_event("door", True)
    obs = env.observe()
    joint_state = env.get_joint_state()
    assert obs["door_open"]
    assert obs["rule_profile"] == "paper_fidelity"
    assert obs["asset_urdf"] == str(BRIDGE_URDF)
    assert joint_state.shape == (13,)
    assert joint_state[:4].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=0.01)
