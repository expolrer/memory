from data_generation.collect_symbolic_demos import RULE_PLANS
from rulesafe.envs import RuleSafeSapienEnv
from rulesafe.rules import RULES


def test_all_rules_registered_and_have_plans():
    assert len(RULES) == 20
    assert set(RULES) == set(RULE_PLANS)


def test_all_rule_success_plans_symbolic():
    for rule_id, plan in sorted(RULE_PLANS.items()):
        env = RuleSafeSapienEnv(rule_id)
        for component, opened in plan:
            env.apply_symbolic_event(component, opened)
        obs = env.observe()
        assert obs["door_open"], (rule_id, obs)
        assert obs["process_score"] == 1.0, (rule_id, obs)


def test_all_rule_success_plans_primitive():
    for rule_id, plan in sorted(RULE_PLANS.items()):
        env = RuleSafeSapienEnv(rule_id, primitive_scene=True)
        for component, opened in plan:
            env.apply_symbolic_event(component, opened)
        obs = env.observe()
        joint = env.get_joint_state()
        assert obs["door_open"], (rule_id, obs)
        assert joint.shape == (13,)
