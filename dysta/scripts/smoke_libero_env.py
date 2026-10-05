"""Reset, render, and step a real LIBERO task without loading a policy."""

from __future__ import annotations

import json
import os

import numpy as np


def main() -> None:
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl for headless rendering")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.seed(0)
        env.reset()
        initial_states = suite.get_task_init_states(0)
        obs = env.set_init_state(initial_states[0])
        for _ in range(5):
            obs, reward, done, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        camera_shapes = {
            "agentview_image": list(obs["agentview_image"].shape),
            "robot0_eye_in_hand_image": list(obs["robot0_eye_in_hand_image"].shape),
        }
        if camera_shapes != {
            "agentview_image": [256, 256, 3],
            "robot0_eye_in_hand_image": [256, 256, 3],
        }:
            raise AssertionError(f"unexpected camera shapes: {camera_shapes}")
        if np.std(obs["agentview_image"]) <= 1.0:
            raise AssertionError("agent-view render appears blank")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "task_language": task.language,
                    "bddl_file": bddl_file,
                    "initial_states": int(len(initial_states)),
                    "steps": 5,
                    "last_reward": float(reward),
                    "last_done": bool(done),
                    "camera_shapes": camera_shapes,
                    "agentview_mean": float(np.mean(obs["agentview_image"])),
                    "agentview_std": float(np.std(obs["agentview_image"])),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
