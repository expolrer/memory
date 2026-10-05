from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "summarize_libero_eval.py"
SPEC = spec_from_file_location("summarize_libero_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_completed_log(tmp_path: Path) -> None:
    log_path = tmp_path / "EVAL-libero_spatial-openvla--dysta-task3-gpu6.txt"
    log_path.write_text(
        "Task: move the bowl\n"
        "Final results:\n"
        "Total episodes: 50\n"
        "Total successes: 47\n"
        "Overall success rate: 0.9400 (94.0%)\n",
        encoding="utf-8",
    )

    result = MODULE.parse_log(log_path)

    assert result is not None
    assert result.task_suite == "libero_spatial"
    assert result.task_id == 3
    assert result.gpu_id == 6
    assert result.task_name == "move the bowl"
    assert result.episodes == 50
    assert result.successes == 47
    assert result.success_rate == 0.94


def test_incomplete_log_is_ignored(tmp_path: Path) -> None:
    log_path = tmp_path / "EVAL-libero_spatial-openvla--dysta-task0-gpu6.txt"
    log_path.write_text("Task: move the bowl\nStarting episode 8...\n", encoding="utf-8")

    assert MODULE.parse_log(log_path) is None


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = MODULE.wilson_interval(47, 50)

    assert 0.83 < low < 0.94
    assert 0.94 < high < 1.0
