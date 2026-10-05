from rulesafe.rules import Rule001, Rule020, SafeState


def run_rule001():
    state = SafeState()
    rule = Rule001()
    rule.reset(state)
    rule.on_knob_change(state, True)
    rule.on_handle_change(state, True)
    success = rule.on_door_attempt(state)
    return success, rule.process_score(state), state.snapshot()


def run_rule020():
    state = SafeState()
    rule = Rule020()
    rule.reset(state)
    rule.on_handle_change(state, True)
    rule.on_knob_change(state, True)
    rule.on_handle_change(state, False)
    rule.on_knob_change(state, False)
    rule.on_handle_change(state, True)
    rule.on_knob_change(state, True)
    rule.on_handle_change(state, False)
    success = rule.on_door_attempt(state)
    return success, rule.process_score(state), state.snapshot()


if __name__ == "__main__":
    for name, fn in [("rule001", run_rule001), ("rule020", run_rule020)]:
        success, score, snapshot = fn()
        print(name, "success=", success, "process_score=", round(score, 3), snapshot)
