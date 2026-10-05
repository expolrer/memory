import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.collect_symbolic_demos import RULE_PLAN_PROFILES, RULE_PLANS

PLAN_TOKEN_VOCAB = {
    "pad": 0,
    "knob:closed": 1,
    "knob:open": 2,
    "handle:closed": 3,
    "handle:open": 4,
    "door:closed": 5,
    "door:open": 6,
}


def event_token(part: str, opened: bool) -> int:
    label = f"{part}:{'open' if opened else 'closed'}"
    return int(PLAN_TOKEN_VOCAB[label])


def _plans_for_profile(rule_profile: str):
    if rule_profile not in RULE_PLAN_PROFILES:
        raise KeyError(f"unknown rule_profile={rule_profile!r}; available={sorted(RULE_PLAN_PROFILES)}")
    return RULE_PLAN_PROFILES[rule_profile]


def plan_tokens_for_rule(rule_id: str, plan_length: int, rule_profile: str = "historical") -> np.ndarray:
    plans = _plans_for_profile(rule_profile)
    if rule_id not in plans:
        raise KeyError(f"unknown rule_id={rule_id}")
    tokens = np.zeros((plan_length,), dtype=np.int64)
    plan = plans[rule_id]
    if len(plan) > plan_length:
        raise ValueError(f"plan_length={plan_length} is shorter than {rule_id} plan length {len(plan)}")
    for idx, (part, opened) in enumerate(plan):
        tokens[idx] = event_token(part, opened)
    return tokens


def next_plan_tokens_for_rule(rule_id: str, episode_length: int, rule_profile: str = "historical") -> np.ndarray:
    plans = _plans_for_profile(rule_profile)
    if rule_id not in plans:
        raise KeyError(f"unknown rule_id={rule_id}")
    plan = plans[rule_id]
    tokens = np.zeros((int(episode_length), 1), dtype=np.int64)
    if not plan or episode_length <= 0:
        return tokens
    event_steps = np.linspace(max(5, int(episode_length) // 10), int(episode_length) - 5, num=len(plan), dtype=int)
    for t in range(int(episode_length)):
        next_idx = int(np.searchsorted(event_steps, t, side="right"))
        if next_idx < len(plan):
            tokens[t, 0] = event_token(*plan[next_idx])
    return tokens


def _index_to_rule_map(root) -> dict:
    raw = root.attrs.get("rule_id_to_index", "{}")
    rule_to_index = {str(k): int(v) for k, v in json.loads(raw).items()}
    return {idx: rule for rule, idx in rule_to_index.items()}


def add_plan_tokens(zarr_path: Path, plan_length: int = None, rule_profile: str = None) -> dict:
    root = zarr.open_group(str(zarr_path), mode="a")
    if "rule_index" not in root["meta"]:
        raise KeyError("/meta/rule_index is required to add plan_tokens")
    index_to_rule = _index_to_rule_map(root)
    if not index_to_rule:
        raise KeyError("zarr attrs rule_id_to_index is required to add plan_tokens")
    stored_profile = str(root.attrs.get("rule_profile", "historical"))
    if rule_profile is not None and rule_profile != stored_profile:
        raise ValueError(
            f"requested rule_profile={rule_profile!r} does not match zarr rule_profile={stored_profile!r}"
        )
    rule_profile = stored_profile if rule_profile is None else rule_profile
    plans = _plans_for_profile(rule_profile)
    if plan_length is None:
        plan_length = max(len(plan) for plan in plans.values())
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    episode_rule_index = np.asarray(root["meta/rule_index"][:], dtype=np.int64)
    total_steps = int(episode_ends[-1])
    data = root["data"]
    if "plan_tokens" in data:
        del data["plan_tokens"]
    if "next_plan_token" in data:
        del data["next_plan_token"]
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    plan_arr = data.zeros(
        "plan_tokens",
        shape=(total_steps, int(plan_length)),
        chunks=(min(1024, max(1, int(plan_length) * 128)), int(plan_length)),
        dtype="i8",
        compressor=compressor,
    )
    next_plan_arr = data.zeros(
        "next_plan_token",
        shape=(total_steps, 1),
        chunks=(min(1024, total_steps), 1),
        dtype="i8",
        compressor=compressor,
    )
    start = 0
    for episode_idx, end in enumerate(episode_ends):
        rule_id = index_to_rule[int(episode_rule_index[episode_idx])]
        episode_len = int(end) - start
        tokens = plan_tokens_for_rule(rule_id, int(plan_length), rule_profile=rule_profile)
        plan_arr[start:int(end)] = np.repeat(tokens[None, :], episode_len, axis=0)
        next_plan_arr[start:int(end)] = next_plan_tokens_for_rule(
            rule_id,
            episode_len,
            rule_profile=rule_profile,
        )
        start = int(end)
    profile_prefix = "rulesafe" if rule_profile == "historical" else f"rulesafe_{rule_profile}"
    root.attrs["plan_condition"] = f"{profile_prefix}_event_tokens"
    root.attrs["next_plan_condition"] = f"{profile_prefix}_next_event_token"
    root.attrs["plan_token_length"] = int(plan_length)
    root.attrs["next_plan_token_length"] = 1
    root.attrs["plan_token_vocab_size"] = int(max(PLAN_TOKEN_VOCAB.values()) + 1)
    root.attrs["plan_token_vocab"] = json.dumps(PLAN_TOKEN_VOCAB, sort_keys=True)
    return {
        "zarr_path": str(zarr_path),
        "plan_tokens_shape": list(plan_arr.shape),
        "next_plan_token_shape": list(next_plan_arr.shape),
        "plan_token_length": int(plan_length),
        "next_plan_token_length": 1,
        "plan_token_vocab_size": int(root.attrs["plan_token_vocab_size"]),
        "rule_profile": rule_profile,
        "rules": sorted(index_to_rule.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add RuleSafe symbolic plan token conditioning to a DP3 zarr replay buffer.")
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--plan-length", type=int, default=None)
    parser.add_argument("--rule-profile", choices=sorted(RULE_PLAN_PROFILES), default=None)
    args = parser.parse_args()
    print(json.dumps(add_plan_tokens(args.zarr, args.plan_length, args.rule_profile), indent=2))


if __name__ == "__main__":
    main()
