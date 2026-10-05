import json
import subprocess
import sys


def test_analyze_rulesafe_rollout_events_cli(tmp_path):
    rollout = {
        "checkpoint": "dummy.pt",
        "rollouts": [
            {
                "rule_id": "rule001",
                "success": True,
                "unlocked": True,
                "process_score": 1.0,
                "steps": 10,
                "events": [[1, 0, "knob", True], [2, 0, "handle", True], [3, 0, "door", True]],
            },
            {
                "rule_id": "rule013",
                "success": False,
                "unlocked": False,
                "process_score": 0.166,
                "steps": 20,
                "events": [[1, 0, "door", False], [2, 0, "handle", True]],
            },
            {
                "rule_id": "rule020",
                "success": False,
                "unlocked": True,
                "process_score": 0.9,
                "steps": 20,
                "events": [
                    [1, 0, "handle", True],
                    [2, 0, "knob", True],
                    [3, 0, "handle", False],
                    [4, 0, "knob", False],
                    [5, 0, "handle", True],
                    [6, 0, "knob", True],
                    [7, 0, "handle", False],
                ],
            },
        ],
    }
    rollout_path = tmp_path / "rollout.json"
    output_path = tmp_path / "analysis.json"
    markdown_path = tmp_path / "analysis.md"
    rollout_path.write_text(json.dumps(rollout), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_rulesafe_rollout_events.py",
            "--rollout",
            str(rollout_path),
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
        ],
        check=True,
    )
    analysis = json.loads(output_path.read_text(encoding="utf-8"))
    assert analysis["episodes"] == 3
    assert analysis["by_rule"]["rule001"]["success_rate"] == 1.0
    assert analysis["rollouts"][1]["failure_type"] == "starts_with_door_attempt"
    assert analysis["rollouts"][2]["failure_type"] == "unlocked_no_success"
    assert "rule020" in markdown_path.read_text(encoding="utf-8")
