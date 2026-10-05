from rulesafe.envs import RuleSafeSapienEnv


def test_sapien_rulesafe_rule001_scaffold():
    env = RuleSafeSapienEnv("rule001")
    env.apply_symbolic_event("knob", True)
    env.apply_symbolic_event("handle", True)
    obs = env.apply_symbolic_event("door", True)
    assert obs["door_open"]
    assert obs["process_score"] == 1.0
    assert obs["sim_step"] > 0


def test_sapien_rulesafe_rule020_scaffold():
    env = RuleSafeSapienEnv("rule020")
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
    obs = env.run_symbolic_plan(plan)[-1]
    assert obs["input_buffer"] == ["1", "1"]
    assert obs["door_open"]
    assert obs["process_score"] == 1.0
