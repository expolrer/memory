from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import sapien
from sapien import physx

from rulesafe.rules import RULE_PROFILES, SafeState
from .humanoidgen_safe_scene import HumanoidGenSafeScene
from .primitive_safe_scene import PrimitiveSafeScene


class RuleSafeSapienEnv:
    """Minimal CPU-PhysX RuleSafe environment scaffold.

    This environment can run in two modes:
      - symbolic-only: stable state machine plus SAPIEN stepping shell.
      - primitive scene: adds collision-only knob/handle/door actors and reads
        a 13-D joint-state vector from actor poses for VQ-Memory data.
    """

    def __init__(
        self,
        rule_id: str = "rule001",
        sim_frequency_hz: int = 100,
        log_frequency_hz: int = 10,
        primitive_scene: bool = False,
        rule_profile: str = "historical",
        asset_urdf: str = None,
    ):
        if rule_profile not in RULE_PROFILES:
            raise KeyError(f"unknown rule_profile={rule_profile!r}; available={sorted(RULE_PROFILES)}")
        rules = RULE_PROFILES[rule_profile]
        if rule_id not in rules:
            raise KeyError(f"unknown rule_id={rule_id!r}; available={sorted(rules)}")
        self.rule_profile = rule_profile
        self.rule = rules[rule_id]()
        self.sim_frequency_hz = sim_frequency_hz
        self.log_frequency_hz = log_frequency_hz
        if primitive_scene and asset_urdf is not None:
            raise ValueError("primitive_scene and asset_urdf are mutually exclusive")
        systems = [physx.PhysxCpuSystem()]
        if asset_urdf is not None:
            from sapien import render

            systems.append(render.RenderSystem())
        self.scene = sapien.Scene(systems)
        self.scene.set_timestep(1.0 / sim_frequency_hz)
        self.use_primitive_scene = bool(primitive_scene)
        self.primitive_safe = PrimitiveSafeScene(self.scene) if self.use_primitive_scene else None
        self.asset_urdf = str(Path(asset_urdf)) if asset_urdf is not None else None
        self.humanoidgen_safe = (
            HumanoidGenSafeScene(self.scene, Path(asset_urdf)) if asset_urdf is not None else None
        )
        self.state = SafeState()
        self.sim_step = 0
        self.reset()

    def reset(self):
        self.state = SafeState()
        self.rule.reset(self.state)
        self.sim_step = 0
        self._sync_safe_scene()
        return self.observe()

    def _sync_safe_scene(self):
        phases = {
            "knob_open": self.state.knob_open,
            "handle_open": self.state.handle_open,
            "door_open": self.state.door_open,
            "door_locked": self.state.door_locked,
        }
        if self.primitive_safe is not None:
            self.primitive_safe.apply_phases(**phases)
        if self.humanoidgen_safe is not None:
            self.humanoidgen_safe.apply_phases(**phases)

    def observe(self):
        obs = self.state.snapshot()
        obs.update(
            {
                "rule_id": self.rule.rule_id,
                "rule_profile": self.rule_profile,
                "asset_urdf": self.asset_urdf,
                "sim_step": self.sim_step,
                "process_score": self.rule.process_score(self.state),
            }
        )
        return obs

    def get_joint_state(self):
        obs = self.observe()
        if self.primitive_safe is not None:
            metadata = obs.get("metadata", {})
            return self.primitive_safe.read_joint_state(
                process_score=float(obs["process_score"]),
                recording=float(metadata.get("recording", 0)),
                password_ok=float(metadata.get("password_ok", 0)),
            )
        if self.humanoidgen_safe is not None:
            metadata = obs.get("metadata", {})
            return self.humanoidgen_safe.read_joint_state(
                process_score=float(obs["process_score"]),
                recording=float(metadata.get("recording", 0)),
                password_ok=float(metadata.get("password_ok", 0)),
                door_locked=bool(obs["door_locked"]),
            )
        return np.array(
            [
                float(obs["knob_open"]),
                float(obs["handle_open"]),
                float(obs["door_open"]),
                float(not obs["door_locked"]),
                float(obs.get("metadata", {}).get("recording", 0)),
                float(obs.get("metadata", {}).get("password_ok", 0)),
                float(len(obs.get("input_buffer", []))) / 2.0,
                float(obs["process_score"]),
                self.sim_step / max(1, self.sim_frequency_hz),
                0.0,
                0.0,
                0.0,
                float(obs["process_score"]),
            ],
            dtype=np.float32,
        )

    def step_physics(self, steps: int = 1):
        for _ in range(steps):
            self.scene.step()
            self.sim_step += 1
        return self.observe()

    def apply_symbolic_event(self, component: str, opened: bool = True):
        if component == "knob":
            self.rule.on_knob_change(self.state, opened)
        elif component == "handle":
            self.rule.on_handle_change(self.state, opened)
        elif component == "door":
            self.rule.on_door_attempt(self.state)
        else:
            raise ValueError(f"unknown component={component!r}")
        self._sync_safe_scene()
        self.step_physics(max(1, self.sim_frequency_hz // self.log_frequency_hz))
        return self.observe()

    def run_symbolic_plan(self, plan: Iterable[Tuple[str, bool]]):
        observations = [self.observe()]
        for component, opened in plan:
            observations.append(self.apply_symbolic_event(component, opened))
        return observations
