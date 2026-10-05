from pathlib import Path

import pytest

from scripts.inventory_humanoidgen_safe_assets import inventory_safe_assets, parse_safe_urdf


OFFICIAL_ASSET_ROOT = Path(
    "third_party/HumanoidGen/humanoidgen/assets/objects/articulated_objs"
)


def test_parse_safe_urdf_detects_components_and_movable_joints(tmp_path):
    urdf = tmp_path / "mobility.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="test-safe">
  <link name="base"><visual name="safe-door"/></link>
  <link name="knob"><visual name="knob"/></link>
  <link name="handle"><visual name="handle"/></link>
  <link name="door"><visual name="door"/></link>
  <joint name="knob_joint" type="revolute"><parent link="base"/><child link="knob"/><axis xyz="1 0 0"/><limit lower="0" upper="1"/></joint>
  <joint name="handle_joint" type="revolute"><parent link="base"/><child link="handle"/><axis xyz="0 1 0"/><limit lower="0" upper="1"/></joint>
  <joint name="door_joint" type="revolute"><parent link="base"/><child link="door"/><axis xyz="0 0 1"/><limit lower="0" upper="2"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    result = parse_safe_urdf(urdf)
    assert result["robot_name"] == "test-safe"
    assert result["movable_joint_count"] == 3
    assert result["component_tags"] == ["door", "handle", "knob"]
    assert result["paper_rulesafe_ready"]


def test_official_humanoidgen_safe_asset_inventory_when_available():
    if not OFFICIAL_ASSET_ROOT.is_dir():
        pytest.skip(f"official HumanoidGen assets not available: {OFFICIAL_ASSET_ROOT}")
    result = inventory_safe_assets(OFFICIAL_ASSET_ROOT)
    assert result["asset_entries"] == 18
    assert result["unique_asset_id_count"] == 16
    assert result["group_counts"] == {
        "safe_move": 2,
        "safe_rotate": 15,
        "safe_rotate_new": 1,
    }
