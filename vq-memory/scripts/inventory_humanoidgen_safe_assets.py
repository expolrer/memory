import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


SAFE_GROUPS = ("safe_move", "safe_rotate", "safe_rotate_new")
COMPONENT_NAMES = ("door", "handle", "knob", "lock", "button")


def _xyz(element) -> list[float] | None:
    if element is None or "xyz" not in element.attrib:
        return None
    return [float(value) for value in element.attrib["xyz"].split()]


def _link_ref(element) -> str:
    return "" if element is None else element.attrib.get("link", "")


def parse_safe_urdf(urdf_path: Path) -> dict:
    root = ET.parse(urdf_path).getroot()
    component_tags = set()
    links = []
    for link in root.findall("link"):
        visual_names = [visual.attrib.get("name", "") for visual in link.findall("visual")]
        for visual_name in visual_names:
            lowered = visual_name.lower()
            component_tags.update(name for name in COMPONENT_NAMES if name in lowered)
        links.append({"name": link.attrib.get("name", ""), "visual_names": visual_names})

    joints = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        joint_info = {
            "name": joint.attrib.get("name", ""),
            "type": joint.attrib.get("type", ""),
            "parent": _link_ref(joint.find("parent")),
            "child": _link_ref(joint.find("child")),
            "axis": _xyz(joint.find("axis")),
            "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        }
        joints.append(joint_info)

    movable_joints = [joint for joint in joints if joint["type"] not in {"fixed", "floating"}]
    has_door_handle_lock = "door" in component_tags and "handle" in component_tags and bool(
        {"knob", "lock"} & component_tags
    )
    return {
        "urdf": str(urdf_path),
        "robot_name": root.attrib.get("name", ""),
        "link_count": len(links),
        "joint_count": len(joints),
        "movable_joint_count": len(movable_joints),
        "movable_joints": movable_joints,
        "component_tags": sorted(component_tags),
        "has_door_handle_lock_visuals": has_door_handle_lock,
        "has_three_movable_joints": len(movable_joints) >= 3,
        "paper_rulesafe_ready": has_door_handle_lock and len(movable_joints) >= 3,
    }


def inventory_safe_assets(articulated_root: Path) -> dict:
    assets = []
    for group in SAFE_GROUPS:
        group_dir = articulated_root / group
        if not group_dir.is_dir():
            continue
        for asset_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            urdf_path = asset_dir / "mobility.urdf"
            if not urdf_path.is_file():
                continue
            record = parse_safe_urdf(urdf_path)
            record.update({"group": group, "asset_id": asset_dir.name})
            assets.append(record)

    group_counts = Counter(record["group"] for record in assets)
    movable_joint_histogram = Counter(record["movable_joint_count"] for record in assets)
    return {
        "articulated_root": str(articulated_root),
        "asset_entries": len(assets),
        "unique_asset_ids": sorted({record["asset_id"] for record in assets}),
        "unique_asset_id_count": len({record["asset_id"] for record in assets}),
        "group_counts": dict(sorted(group_counts.items())),
        "movable_joint_histogram": {
            str(count): occurrences for count, occurrences in sorted(movable_joint_histogram.items())
        },
        "paper_rulesafe_ready_entries": [
            f"{record['group']}/{record['asset_id']}" for record in assets if record["paper_rulesafe_ready"]
        ],
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory official HumanoidGen safe assets and URDF joints.")
    parser.add_argument(
        "--articulated-root",
        type=Path,
        default=Path(
            "third_party/HumanoidGen/humanoidgen/assets/objects/articulated_objs"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = inventory_safe_assets(args.articulated_root)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
