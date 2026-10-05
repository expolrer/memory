from data_generation.add_rulesafe_plan_to_dp3_zarr import plan_tokens_for_rule
from data_generation.collect_symbolic_demos import PAPER_RULE_PLANS, RULE_PLANS
from rulesafe.rules import PAPER_RULES, RULES, SafeState


CORRECTED_RULE_IDS = ("rule006", "rule009", "rule010", "rule011")


def run_plan(rule_cls, plan):
    state = SafeState()
    rule = rule_cls()
    rule.reset(state)
    for component, opened in plan:
        if component == "knob":
            rule.on_knob_change(state, opened)
        elif component == "handle":
            rule.on_handle_change(state, opened)
        elif component == "door":
            rule.on_door_attempt(state)
        else:
            raise AssertionError(component)
    return state, rule


def test_paper_profile_covers_all_twenty_rules_without_mutating_history():
    assert set(PAPER_RULES) == set(RULES) == set(PAPER_RULE_PLANS) == set(RULE_PLANS)
    assert len(PAPER_RULES) == 20
    for rule_id in CORRECTED_RULE_IDS:
        assert PAPER_RULES[rule_id] is not RULES[rule_id]
        assert PAPER_RULE_PLANS[rule_id] != RULE_PLANS[rule_id]
    for rule_id in set(RULES) - set(CORRECTED_RULE_IDS):
        assert PAPER_RULES[rule_id] is RULES[rule_id]
        assert PAPER_RULE_PLANS[rule_id] == RULE_PLANS[rule_id]


def test_all_paper_fidelity_plans_unlock_and_open_the_door():
    for rule_id, plan in sorted(PAPER_RULE_PLANS.items()):
        state, rule = run_plan(PAPER_RULES[rule_id], plan)
        assert state.door_open, (rule_id, state)
        assert rule.process_score(state) == 1.0, (rule_id, state)


def test_corrected_table5_password_buffers():
    expected = {
        "rule006": ["0", "1", "0"],
        "rule009": ["1", "0", "0", "1"],
        "rule010": ["0", "1", "1", "0"],
        "rule011": ["1", "0", "1", "0"],
    }
    for rule_id, password in expected.items():
        plan_without_door = [event for event in PAPER_RULE_PLANS[rule_id] if event[0] != "door"]
        state, _rule = run_plan(PAPER_RULES[rule_id], plan_without_door)
        assert state.input_buffer == password, (rule_id, state.input_buffer)
        assert not state.door_locked, (rule_id, state)


def test_plan_tokens_are_profile_specific_for_corrected_rules():
    for rule_id in CORRECTED_RULE_IDS:
        historical = plan_tokens_for_rule(rule_id, 8, rule_profile="historical")
        paper = plan_tokens_for_rule(rule_id, 8, rule_profile="paper_fidelity")
        assert historical.tolist() != paper.tolist(), rule_id
