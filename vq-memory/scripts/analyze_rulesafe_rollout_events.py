import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.collect_symbolic_demos import RULE_PLANS

EventToken = Tuple[str, bool]


def event_token(event: Sequence[object]) -> EventToken:
    if len(event) < 4:
        raise ValueError(f"rollout event must have [step, chunk, part, opened], got: {event}")
    return str(event[2]), bool(event[3])


def token_label(token: EventToken) -> str:
    part, opened = token
    return f"{part}:{'open' if opened else 'closed'}"


def longest_exact_prefix(plan: Sequence[EventToken], actual: Sequence[EventToken]) -> int:
    count = 0
    for expected, observed in zip(plan, actual):
        if expected != observed:
            break
        count += 1
    return count


def longest_ordered_plan_prefix(plan: Sequence[EventToken], actual: Sequence[EventToken]) -> int:
    idx = 0
    for token in actual:
        if idx < len(plan) and token == plan[idx]:
            idx += 1
    return idx


def classify_rollout(plan: Sequence[EventToken], actual: Sequence[EventToken], rollout: Dict[str, object], prefix_len: int, subseq_len: int) -> str:
    if rollout.get("success"):
        return "success"
    if not actual:
        return "no_events"
    if rollout.get("unlocked"):
        return "unlocked_no_success"
    if subseq_len >= len(plan) - 1:
        return "missing_or_failed_final_door"
    if prefix_len == 0:
        if actual[0][0] == "door":
            return "starts_with_door_attempt"
        return "wrong_first_event"
    if subseq_len > prefix_len:
        return "right_events_with_interleaving_noise"
    return "partial_prefix_then_wrong_event"


def analyze_rollout(rollout: Dict[str, object]) -> Dict[str, object]:
    rule_id = str(rollout["rule_id"])
    if rule_id not in RULE_PLANS:
        raise KeyError(f"unknown rule_id={rule_id}")
    plan = [(str(part), bool(opened)) for part, opened in RULE_PLANS[rule_id]]
    actual = [event_token(event) for event in rollout.get("events", [])]
    prefix_len = longest_exact_prefix(plan, actual)
    subseq_len = longest_ordered_plan_prefix(plan, actual)
    next_prefix_expected = plan[prefix_len] if prefix_len < len(plan) else None
    next_subseq_expected = plan[subseq_len] if subseq_len < len(plan) else None
    first_wrong = actual[prefix_len] if prefix_len < min(len(plan), len(actual)) else None
    door_attempts = [token for token in actual if token[0] == "door"]
    part_counts = Counter(part for part, _opened in actual)
    transition_counts = Counter(token_label(token) for token in actual)
    return {
        "rule_id": rule_id,
        "success": bool(rollout.get("success", False)),
        "unlocked": bool(rollout.get("unlocked", False)),
        "process_score": float(rollout.get("process_score", 0.0)),
        "steps": int(rollout.get("steps", 0)),
        "event_count": len(actual),
        "plan": [token_label(token) for token in plan],
        "actual_prefix": [token_label(token) for token in actual[: min(12, len(actual))]],
        "exact_prefix_len": prefix_len,
        "ordered_prefix_len": subseq_len,
        "plan_len": len(plan),
        "exact_prefix_ratio": prefix_len / max(1, len(plan)),
        "ordered_prefix_ratio": subseq_len / max(1, len(plan)),
        "next_exact_expected": token_label(next_prefix_expected) if next_prefix_expected is not None else None,
        "next_ordered_expected": token_label(next_subseq_expected) if next_subseq_expected is not None else None,
        "first_wrong_event": token_label(first_wrong) if first_wrong is not None else None,
        "door_attempts": len(door_attempts),
        "successful_door_events": sum(1 for token in door_attempts if token[1]),
        "failed_door_events": sum(1 for token in door_attempts if not token[1]),
        "dominant_part": part_counts.most_common(1)[0][0] if part_counts else None,
        "top_transitions": transition_counts.most_common(5),
        "failure_type": classify_rollout(plan, actual, rollout, prefix_len, subseq_len),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def aggregate(items: List[Dict[str, object]]) -> Dict[str, object]:
    by_rule = defaultdict(list)
    for item in items:
        by_rule[item["rule_id"]].append(item)
    out = {}
    for rule_id, rule_items in sorted(by_rule.items()):
        failure_types = Counter(str(item["failure_type"]) for item in rule_items)
        next_expected = Counter(str(item["next_ordered_expected"]) for item in rule_items if item.get("next_ordered_expected") is not None)
        first_wrong = Counter(str(item["first_wrong_event"]) for item in rule_items if item.get("first_wrong_event") is not None)
        out[rule_id] = {
            "episodes": len(rule_items),
            "success_rate": mean(bool(item["success"]) for item in rule_items),
            "unlock_rate": mean(bool(item["unlocked"]) for item in rule_items),
            "process_score": mean(float(item["process_score"]) for item in rule_items),
            "exact_prefix_ratio": mean(float(item["exact_prefix_ratio"]) for item in rule_items),
            "ordered_prefix_ratio": mean(float(item["ordered_prefix_ratio"]) for item in rule_items),
            "avg_events": mean(int(item["event_count"]) for item in rule_items),
            "avg_failed_door_events": mean(int(item["failed_door_events"]) for item in rule_items),
            "failure_types": dict(failure_types),
            "most_common_next_expected": next_expected.most_common(3),
            "most_common_first_wrong": first_wrong.most_common(3),
        }
    return out


def markdown_table(by_rule: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "| rule | SR | PS | exact prefix | ordered prefix | avg events | failed door | top failure | next expected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rule_id, item in sorted(by_rule.items()):
        failures = Counter(item.get("failure_types", {}))
        top_failure = failures.most_common(1)[0][0] if failures else "none"
        next_expected = item.get("most_common_next_expected", [])
        next_label = next_expected[0][0] if next_expected else "none"
        lines.append(
            f"| `{rule_id}` | {float(item['success_rate']):.3f} | {float(item['process_score']):.3f} | "
            f"{float(item['exact_prefix_ratio']):.3f} | {float(item['ordered_prefix_ratio']):.3f} | "
            f"{float(item['avg_events']):.1f} | {float(item['avg_failed_door_events']):.1f} | {top_failure} | {next_label} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RuleSafe rollout event traces against symbolic success plans.")
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.rollout.read_text())
    analyses = [analyze_rollout(item) for item in data.get("rollouts", [])]
    by_rule = aggregate(analyses)
    summary = {
        "rollout": str(args.rollout),
        "checkpoint": data.get("checkpoint"),
        "episodes": len(analyses),
        "rules": sorted(by_rule),
        "overall_success_rate": mean(bool(item["success"]) for item in analyses),
        "overall_process_score": mean(float(item["process_score"]) for item in analyses),
        "overall_ordered_prefix_ratio": mean(float(item["ordered_prefix_ratio"]) for item in analyses),
        "by_rule": by_rule,
        "rollouts": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = markdown_table(by_rule)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
