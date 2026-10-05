import numpy as np

from rulesafe.envs import RuleSafeSapienEnv


def test_primitive_scene_joint_state_rule001():
    env = RuleSafeSapienEnv("rule001", primitive_scene=True)
    initial = env.get_joint_state()
    assert initial.shape == (13,)
    assert np.isclose(initial[0], 0.0)
    env.apply_symbolic_event("knob", True)
    env.apply_symbolic_event("handle", True)
    obs = env.apply_symbolic_event("door", True)
    joint = env.get_joint_state()
    assert obs["door_open"]
    assert joint.shape == (13,)
    assert joint[0] > 0.9
    assert joint[1] > 0.9
    assert joint[2] > 0.9
    assert joint[3] == 1.0


def test_primitive_scene_joint_state_rule020():
    env = RuleSafeSapienEnv("rule020", primitive_scene=True)
    plan = [
        ("handle", True),
        ("knob", True),
        ("handle", False),
        ("knob", False),
        ("handle", True),
        ("knob", True),
        ("handle", False),
        ("door", True),
    ]
    for event in plan:
        env.apply_symbolic_event(*event)
    joint = env.get_joint_state()
    assert joint.shape == (13,)
    assert joint[2] > 0.9
    assert joint[3] == 1.0
    assert joint[5] == 1.0
    assert joint[-1] == 1.0
