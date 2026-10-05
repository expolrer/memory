from rulesafe.rules.demo_rules import run_rule001, run_rule020


def test_rule001_success_path():
    success, score, snapshot = run_rule001()
    assert success
    assert score == 1.0
    assert snapshot["door_open"]


def test_rule020_success_path():
    success, score, snapshot = run_rule020()
    assert success
    assert score == 1.0
    assert snapshot["input_buffer"] == ["1", "1"]
