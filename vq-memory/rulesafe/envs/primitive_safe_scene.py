import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import sapien


@dataclass
class PrimitiveSafeConfig:
    knob_open_angle: float = math.radians(90.0)
    handle_open_angle: float = math.radians(55.0)
    door_open_angle: float = math.radians(75.0)


def quat_axis_angle(axis, angle: float):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle * 0.5
    s = math.sin(half)
    return [math.cos(half), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s)]


def quat_angle_about_axis(q, axis_index: int) -> float:
    # For our primitive actors, each quaternion is created around a single axis.
    w = float(q[0])
    v = float(q[axis_index + 1])
    return 2.0 * math.atan2(v, w)


class PrimitiveSafeScene:
    """Collision-only SAPIEN safe made of body, knob, handle, and door actors.

    Visual components are intentionally not added yet. On the current container,
    SAPIEN render components can segfault because the Vulkan/driver stack is not
    fully usable, while collision-only CPU PhysX actors are stable.
    """

    def __init__(self, scene: sapien.Scene, config: PrimitiveSafeConfig = None):
        self.scene = scene
        self.config = config or PrimitiveSafeConfig()
        self.knob_angle = 0.0
        self.handle_angle = 0.0
        self.door_angle = 0.0
        self.actors = self._build_actors()
        self.apply_phases(False, False, False, True)

    def _box_actor(self, name: str, half_size, pose, kinematic: bool = True):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        actor = builder.build_kinematic(name=name) if kinematic else builder.build_static(name=name)
        actor.set_pose(pose)
        return actor

    def _build_actors(self) -> Dict[str, object]:
        return {
            "body": self._box_actor("safe_body", [0.32, 0.08, 0.24], sapien.Pose([0.0, 0.0, 0.35]), kinematic=False),
            "door": self._box_actor("safe_door", [0.30, 0.025, 0.22], sapien.Pose([0.0, -0.085, 0.35])),
            "knob": self._box_actor("safe_knob", [0.045, 0.025, 0.045], sapien.Pose([-0.10, -0.125, 0.40])),
            "handle": self._box_actor("safe_handle", [0.025, 0.025, 0.12], sapien.Pose([0.11, -0.125, 0.35])),
        }

    def apply_phases(self, knob_open: bool, handle_open: bool, door_open: bool, door_locked: bool):
        self.knob_angle = self.config.knob_open_angle if knob_open else 0.0
        self.handle_angle = self.config.handle_open_angle if handle_open else 0.0
        self.door_angle = self.config.door_open_angle if door_open else 0.0

        self.actors["knob"].set_pose(
            sapien.Pose([-0.10, -0.125, 0.40], quat_axis_angle([0, 0, 1], self.knob_angle))
        )
        self.actors["handle"].set_pose(
            sapien.Pose([0.11, -0.125, 0.35], quat_axis_angle([1, 0, 0], self.handle_angle))
        )
        hinge_x = -0.31
        door_center_x = hinge_x + 0.30 * math.cos(self.door_angle)
        door_center_y = -0.085 - 0.30 * math.sin(self.door_angle)
        self.actors["door"].set_pose(
            sapien.Pose([door_center_x, door_center_y, 0.35], quat_axis_angle([0, 0, 1], self.door_angle))
        )
        self.door_locked = bool(door_locked)

    def read_joint_state(self, process_score: float = 0.0, recording: float = 0.0, password_ok: float = 0.0):
        knob_pose = self.actors["knob"].get_pose()
        handle_pose = self.actors["handle"].get_pose()
        door_pose = self.actors["door"].get_pose()
        knob_angle = quat_angle_about_axis(knob_pose.q, 2) / max(self.config.knob_open_angle, 1e-6)
        handle_angle = quat_angle_about_axis(handle_pose.q, 0) / max(self.config.handle_open_angle, 1e-6)
        door_angle = quat_angle_about_axis(door_pose.q, 2) / max(self.config.door_open_angle, 1e-6)
        joint = np.array(
            [
                knob_angle,
                handle_angle,
                door_angle,
                float(not self.door_locked),
                float(recording),
                float(password_ok),
                float(knob_pose.p[0]),
                float(knob_pose.p[2]),
                float(handle_pose.p[0]),
                float(handle_pose.p[2]),
                float(door_pose.p[0]),
                float(door_pose.p[1]),
                float(process_score),
            ],
            dtype=np.float32,
        )
        return joint
