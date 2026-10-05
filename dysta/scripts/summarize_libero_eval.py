#!/usr/bin/env python3
"""Summarize completed per-task LIBERO evaluation logs into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RUN_RE = re.compile(
    r"^EVAL-(?P<task_suite>libero_[a-z0-9_]+)-.*--dysta-"
    r"(?:libero_[a-z0-9_]+-)?task(?P<task_id>\d+)-gpu(?P<gpu_id>\d+)\.txt$"
)
TASK_RE = re.compile(r"^Task:\s*(.+)$", re.MULTILINE)
EPISODES_RE = re.compile(r"^Total episodes:\s*(\d+)\s*$", re.MULTILINE)
SUCCESSES_RE = re.compile(r"^Total successes:\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TaskResult:
    task_suite: str
    task_id: int
    gpu_id: int
    task_name: str
    episodes: int
    successes: int
    success_rate: float
    source_log: str
    source_mtime_utc: str


def parse_task_ids(value: str) -> list[int]:
    task_ids = [int(item) for item in value.split(",") if item]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise argparse.ArgumentTypeError("task IDs must be a non-empty unique comma-separated list")
    return task_ids


def parse_log(path: Path) -> TaskResult | None:
    run_match = RUN_RE.search(path.name)
    if run_match is None:
        return None

    text = path.read_text(encoding="utf-8", errors="replace")
    task_match = TASK_RE.search(text)
    episode_matches = EPISODES_RE.findall(text)
    success_matches = SUCCESSES_RE.findall(text)
    if task_match is None or not episode_matches or not success_matches:
        return None

    episodes = int(episode_matches[-1])
    successes = int(success_matches[-1])
    if episodes <= 0 or successes < 0 or successes > episodes:
        raise ValueError(f"invalid totals in {path}: successes={successes}, episodes={episodes}")

    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return TaskResult(
        task_suite=run_match.group("task_suite"),
        task_id=int(run_match.group("task_id")),
        gpu_id=int(run_match.group("gpu_id")),
        task_name=task_match.group(1).strip(),
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        source_log=str(path),
        source_mtime_utc=mtime.isoformat(),
    )


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes * episodes))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def select_results(
    paths: Iterable[Path], expected_task_suite: str, expected_task_ids: list[int], expected_trials: int
) -> tuple[list[TaskResult], list[int]]:
    selected: dict[int, TaskResult] = {}
    for path in paths:
        result = parse_log(path)
        if (
            result is None
            or result.task_suite != expected_task_suite
            or result.task_id not in expected_task_ids
            or result.episodes != expected_trials
        ):
            continue
        previous = selected.get(result.task_id)
        if previous is None or result.source_mtime_utc > previous.source_mtime_utc:
            selected[result.task_id] = result

    missing = sorted(set(expected_task_ids) - set(selected))
    return [selected[task_id] for task_id in sorted(selected)], missing


def write_csv(path: Path, results: list[TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--expected-task-suite", default="libero_spatial")
    parser.add_argument("--expected-task-ids", type=parse_task_ids, default=parse_task_ids("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--expected-trials", type=int, default=50)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.expected_trials <= 0:
        parser.error("--expected-trials must be positive")

    results, missing = select_results(
        args.log_dir.glob("EVAL-*.txt"),
        args.expected_task_suite,
        args.expected_task_ids,
        args.expected_trials,
    )
    if not results:
        raise SystemExit("no completed evaluation logs matched the requested task IDs and trial count")

    total_episodes = sum(result.episodes for result in results)
    total_successes = sum(result.successes for result in results)
    ci_low, ci_high = wilson_interval(total_successes, total_episodes)
    summary = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "expected_task_suite": args.expected_task_suite,
        "expected_task_ids": args.expected_task_ids,
        "expected_trials_per_task": args.expected_trials,
        "missing_task_ids": missing,
        "tasks": [asdict(result) for result in results],
        "overall": {
            "tasks_completed": len(results),
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / total_episodes,
            "wilson_95_ci": [ci_low, ci_high],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, results)
    print(json.dumps(summary["overall"], indent=2))

    if args.strict and missing:
        print(f"missing completed task logs: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
