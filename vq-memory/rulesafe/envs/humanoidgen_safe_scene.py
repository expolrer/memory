from pathlib import Path

import numpy as np


class HumanoidGenSafeScene:
    """SAPIEN adapter for the explicitly reconstructed three-joint safe URDF."""

    JOINT_NAMES = ("knob_joint", "handle_joint", "door_joint")
    OPEN_ENDPOINT = {
        "knob_joint": "upper",
        "handle_joint": "lower",
        "door_joint": "upper",
    }

    def __init__(self, scene, urdf_path: Path):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        self.articulation = loader.load(str(self.urdf_path))
        if self.articulation is None:
            raise RuntimeError(f"SAPIEN failed to load {self.urdf_path}")

        active_joints = self.articulation.get_active_joints()
        self.active_joints = {joint.get_name(): joint for joint in active_joints}
        self.joint_index = {joint.get_name(): idx for idx, joint in enumerate(active_joints)}
        missing = sorted(set(self.JOINT_NAMES) - set(self.joint_index))
        if missing:
            raise KeyError(f"asset bridge is missing active joints: {missing}")
        self.joint_limits = {}
        for joint in active_joints:
            limits = np.asarray(joint.get_limits(), dtype=np.float32).reshape(-1, 2)[0]
            self.joint_limits[joint.get_name()] = (float(limits[0]), float(limits[1]))
            joint.set_drive_properties(
                stiffness=10000.0,
                damping=500.0,
                force_limit=1e8,
                mode="acceleration",
            )

    def _phase_value(self, joint_name: str, opened: bool) -> float:
        lower, upper = self.joint_limits[joint_name]
        open_value = upper if self.OPEN_ENDPOINT[joint_name] == "upper" else lower
        closed_value = lower if self.OPEN_ENDPOINT[joint_name] == "upper" else upper
        return open_value if opened else closed_value

    def _phase(self, joint_name: str, value: float) -> float:
        opened = self._phase_value(joint_name, True)
        closed = self._phase_value(joint_name, False)
        return float(np.clip((value - closed) / (opened - closed), 0.0, 1.0))

    def apply_phases(self, knob_open: bool, handle_open: bool, door_open: bool, door_locked: bool) -> None:
        qpos = np.asarray(self.articulation.get_qpos(), dtype=np.float32)
        phases = {
            "knob_joint": knob_open,
            "handle_joint": handle_open,
            "door_joint": door_open,
        }
        for joint_name, opened in phases.items():
            target = self._phase_value(joint_name, bool(opened))
            qpos[self.joint_index[joint_name]] = target
            self.active_joints[joint_name].set_drive_target(target)
        self.articulation.set_qpos(qpos)
        self.articulation.set_qvel(np.zeros_like(qpos))

    def read_joint_state(self, process_score: float, recording: float, password_ok: float, door_locked: bool) -> np.ndarray:
        qpos = np.asarray(self.articulation.get_qpos(), dtype=np.float32)
        knob_q = float(qpos[self.joint_index["knob_joint"]])
        handle_q = float(qpos[self.joint_index["handle_joint"]])
        door_q = float(qpos[self.joint_index["door_joint"]])
        knob_phase = self._phase("knob_joint", knob_q)
        handle_phase = self._phase("handle_joint", handle_q)
        door_phase = self._phase("door_joint", door_q)
        return np.asarray(
            [
                knob_phase,
                handle_phase,
                door_phase,
                float(not door_locked),
                recording,
                password_ok,
                (knob_phase + handle_phase) / 2.0,
                process_score,
                knob_q,
                handle_q,
                door_q,
                1.0,
                process_score,
            ],
            dtype=np.float32,
        )
