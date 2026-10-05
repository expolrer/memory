import argparse
import json
import math
from pathlib import Path


DEFAULT_URDF = Path(
    "third_party/HumanoidGen/humanoidgen/assets/objects/articulated_objs/"
    "safe_rotate/101564/mobility.urdf"
)


def load_and_step(urdf_path: Path, steps: int = 10, timestep: float = 0.01) -> dict:
    import sapien
    from sapien import physx, render

    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if steps < 1:
        raise ValueError("steps must be positive")

    scene = sapien.Scene([physx.PhysxCpuSystem(), render.RenderSystem()])
    scene.set_timestep(float(timestep))
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(str(urdf_path))
    if articulation is None:
        raise RuntimeError(f"SAPIEN failed to load {urdf_path}")

    for _ in range(steps):
        scene.step()

    qpos = [float(value) for value in articulation.get_qpos()]
    if not all(math.isfinite(value) for value in qpos):
        raise ValueError(f"non-finite qpos after stepping: {qpos}")
    active_joints = articulation.get_active_joints()
    return {
        "urdf": str(urdf_path),
        "steps": int(steps),
        "timestep": float(timestep),
        "link_count": len(articulation.get_links()),
        "active_joint_count": len(active_joints),
        "active_joint_names": [joint.get_name() for joint in active_joints],
        "qpos": qpos,
        "finite_qpos": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an official HumanoidGen safe URDF in SAPIEN and step PhysX.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = load_and_step(args.urdf, steps=args.steps, timestep=args.timestep)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
