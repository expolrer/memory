import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_SOURCE = Path(
    "third_party/HumanoidGen/humanoidgen/assets/objects/articulated_objs/"
    "safe_rotate_new/101612"
)
DEFAULT_OUTPUT = Path("artifacts/humanoidgen_rulesafe_bridge/safe_101612")

JOINT_RECONSTRUCTION = {
    "joint_0": {
        "semantic_name": "door_joint",
        "type": "revolute",
        "lower": 0.0,
        "upper": math.pi,
        "open_endpoint": "upper",
    },
    "joint_1": {
        "semantic_name": "knob_joint",
        "type": "revolute",
        "lower": 0.0,
        "upper": math.pi / 2.0,
        "open_endpoint": "upper",
    },
    "joint_2": {
        "semantic_name": "handle_joint",
        "type": "revolute",
        "lower": -math.pi / 2.0,
        "upper": 0.0,
        "open_endpoint": "lower",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bridge(source_dir: Path, output_dir: Path) -> dict:
    source_urdf = source_dir / "mobility.urdf"
    if not source_urdf.is_file():
        raise FileNotFoundError(source_urdf)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    output_urdf = output_dir / "mobility.urdf"
    tree = ET.parse(output_urdf)
    root = tree.getroot()
    joints = {joint.attrib.get("name", ""): joint for joint in root.findall("joint")}
    missing = sorted(set(JOINT_RECONSTRUCTION) - set(joints))
    if missing:
        raise KeyError(f"source URDF is missing expected joints: {missing}")

    modifications = []
    for original_name, spec in JOINT_RECONSTRUCTION.items():
        joint = joints[original_name]
        original_type = joint.attrib.get("type", "")
        joint.attrib["name"] = spec["semantic_name"]
        joint.attrib["type"] = spec["type"]
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.attrib["lower"] = repr(float(spec["lower"]))
        limit.attrib["upper"] = repr(float(spec["upper"]))
        modifications.append(
            {
                "source_joint": original_name,
                "output_joint": spec["semantic_name"],
                "source_type": original_type,
                "output_type": spec["type"],
                "lower": spec["lower"],
                "upper": spec["upper"],
                "open_endpoint": spec["open_endpoint"],
            }
        )

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    manifest = {
        "status": "reconstructed_not_official_rulesafe",
        "source": str(source_dir),
        "source_urdf_sha256": sha256_file(source_urdf),
        "output": str(output_dir),
        "output_urdf_sha256": sha256_file(output_urdf),
        "reason": (
            "The public HumanoidGen asset has separate door, lock, and handle links, "
            "but only the door is movable. The unpublished RuleSafe asset mapping is unavailable."
        ),
        "semantic_mapping": {
            "door": "door_joint",
            "knob": "knob_joint",
            "handle": "handle_joint",
        },
        "modifications": modifications,
    }
    manifest_path = output_dir / "rulesafe_bridge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a transparent three-joint RuleSafe bridge from a HumanoidGen safe asset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_bridge(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
