import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rulesafe.envs import RuleSafeSapienEnv
from rulesafe.rules import RULE_DESCRIPTION_PROFILES, RULES


RULE_PLANS: Dict[str, List[Tuple[str, bool]]] = {
    "rule001": [("knob", True), ("handle", True), ("door", True)],
    "rule002": [("knob", True), ("door", True)],
    "rule003": [("handle", True), ("door", True)],
    "rule004": [("handle", True), ("knob", True), ("door", True)],
    "rule005": [("knob", True), ("knob", False), ("handle", True), ("door", True)],
    "rule006": [("knob", True), ("handle", True), ("knob", False), ("handle", False), ("knob", True), ("door", True)],
    "rule007": [("knob", True), ("handle", True), ("handle", False), ("door", True)],
    "rule008": [("knob", True), ("handle", True), ("handle", False), ("door", True)],
    "rule009": [("handle", True), ("knob", True), ("knob", False), ("handle", False), ("door", True)],
    "rule010": [("handle", True), ("knob", True), ("handle", False), ("handle", True), ("knob", False), ("handle", False), ("door", True)],
    "rule011": [("knob", True), ("knob", False), ("knob", True), ("knob", False), ("door", True)],
    "rule012": [("knob", True), ("handle", True), ("knob", False), ("door", True)],
    "rule013": [("handle", True), ("knob", True), ("knob", False), ("knob", True), ("door", True)],
    "rule014": [("knob", True), ("handle", True), ("knob", False), ("door", True)],
    "rule015": [("knob", True), ("knob", False), ("knob", True), ("handle", True), ("handle", False), ("handle", True), ("handle", False), ("door", True)],
    "rule016": [("knob", True), ("knob", False), ("knob", True), ("knob", False), ("knob", True), ("knob", False), ("handle", True), ("door", True)],
    "rule017": [("knob", True), ("knob", False), ("handle", True), ("handle", False), ("door", True)],
    "rule018": [("handle", True), ("knob", True), ("handle", False), ("knob", False), ("knob", True), ("handle", True), ("door", True)],
    "rule019": [("knob", True), ("handle", True), ("knob", False), ("knob", True), ("handle", False), ("door", True)],
    "rule020": [("handle", True), ("knob", True), ("handle", False), ("knob", False), ("handle", True), ("knob", True), ("handle", False), ("door", True)],
}

PAPER_RULE_PLANS: Dict[str, List[Tuple[str, bool]]] = dict(RULE_PLANS)
PAPER_RULE_PLANS.update(
    {
        "rule006": [("handle", True), ("knob", True), ("handle", False), ("knob", False), ("handle", True), ("knob", True), ("door", True)],
        "rule009": [("knob", True), ("handle", True), ("handle", False), ("knob", False), ("door", True)],
        "rule010": [("knob", True), ("handle", True), ("knob", False), ("handle", False), ("handle", True), ("knob", True), ("handle", False), ("door", True)],
        "rule011": [("knob", True), ("knob", False), ("knob", True), ("knob", False), ("knob", True), ("door", True)],
    }
)

RULE_PLAN_PROFILES = {
    "historical": RULE_PLANS,
    "paper_fidelity": PAPER_RULE_PLANS,
}

RULE_TEXT = {rule_id: RULES[rule_id].description for rule_id in RULE_PLANS}
RULE_TEXT_PROFILES = {
    profile: {rule_id: descriptions[rule_id] for rule_id in RULE_PLAN_PROFILES[profile]}
    for profile, descriptions in RULE_DESCRIPTION_PROFILES.items()
}


def phase_state_to_joint(obs: Dict[str, object], t: int, total_steps: int, rng: np.random.Generator, demo_idx: int) -> np.ndarray:
    progress = t / max(1, total_steps - 1)
    buffer_value = 0.0
    for bit in obs.get("input_buffer", []):
        try:
            buffer_value = buffer_value * 4.0 + float(bit)
        except ValueError:
            buffer_value = buffer_value * 4.0
    metadata = obs.get("metadata", {})
    joint = np.array(
        [
            float(obs["knob_open"]),
            float(obs["handle_open"]),
            float(obs["door_open"]),
            float(not obs["door_locked"]),
            float(metadata.get("recording", 0)),
            float(metadata.get("password_ok", 0)),
            min(buffer_value / 64.0, 1.0),
            min(float(len(obs.get("input_buffer", []))) / 4.0, 1.0),
            progress,
            np.sin(progress * np.pi * 2.0),
            np.cos(progress * np.pi * 2.0),
            (demo_idx % 10) / 10.0,
            float(obs["process_score"]),
        ],
        dtype=np.float32,
    )
    return joint


def read_joint_target(env: RuleSafeSapienEnv, obs: Dict[str, object], t: int, total_steps: int, rng: np.random.Generator, demo_idx: int, joint_source: str):
    if joint_source == "phase":
        joint = phase_state_to_joint(obs, t, total_steps, rng, demo_idx)
    elif joint_source in {"primitive", "asset"}:
        joint = env.get_joint_state().astype(np.float32)
        joint[12] = float(obs["process_score"])
    else:
        raise ValueError(f"unknown joint_source={joint_source!r}")
    phase_noise = rng.normal(0.0, 0.01, size=13).astype(np.float32)
    phase_noise[3:6] *= 0.25
    phase_noise[12] *= 0.25
    return (joint + phase_noise).astype(np.float32)


def interpolate(prev: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return prev * (1.0 - alpha) + target * alpha


def generate_one(
    rule_id: str,
    trajectory_length: int,
    rng: np.random.Generator,
    demo_idx: int,
    joint_source: str,
    rule_profile: str = "historical",
    asset_urdf: str = None,
):
    if rule_profile not in RULE_PLAN_PROFILES:
        raise KeyError(f"unknown rule_profile={rule_profile!r}; available={sorted(RULE_PLAN_PROFILES)}")
    plans = RULE_PLAN_PROFILES[rule_profile]
    env = RuleSafeSapienEnv(
        rule_id,
        primitive_scene=(joint_source == "primitive"),
        rule_profile=rule_profile,
        asset_urdf=asset_urdf if joint_source == "asset" else None,
    )
    plan = plans[rule_id]
    event_steps = np.linspace(max(5, trajectory_length // 10), trajectory_length - 5, num=len(plan), dtype=int)
    observations = []
    current_obs = env.observe()
    event_ptr = 0
    prev_target = read_joint_target(env, current_obs, 0, trajectory_length, rng, demo_idx, joint_source)
    next_target = prev_target.copy()
    last_event_step = 0

    joints = []
    process_scores = []
    for t in range(trajectory_length):
        if event_ptr < len(plan) and t >= event_steps[event_ptr]:
            component, opened = plan[event_ptr]
            current_obs = env.apply_symbolic_event(component, opened)
            prev_target = next_target
            next_target = read_joint_target(env, current_obs, t, trajectory_length, rng, demo_idx, joint_source)
            last_event_step = t
            event_ptr += 1
        denom = max(1, trajectory_length // (len(plan) + 2))
        alpha = (t - last_event_step) / denom
        joint = interpolate(prev_target, next_target, alpha)
        if joint_source == "phase":
            joint[8] = t / max(1, trajectory_length - 1)
        joints.append(joint.astype(np.float32))
        process_scores.append(float(current_obs["process_score"]))
        observations.append(current_obs)

    joints = np.stack(joints, axis=0).astype(np.float32)
    actions = np.zeros_like(joints)
    actions[:-1] = joints[1:] - joints[:-1]
    actions[-1] = actions[-2]
    final_obs = observations[-1]
    return joints, actions, bool(final_obs["door_open"]), float(process_scores[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/rulesafe_primitive/rulesafe.npz")
    parser.add_argument("--rules", nargs="+", default=sorted(RULE_PLANS))
    parser.add_argument("--demos-per-rule", type=int, default=50)
    parser.add_argument("--trajectory-length", type=int, default=220)
    parser.add_argument("--joint-source", choices=["phase", "primitive", "asset"], default="primitive")
    parser.add_argument(
        "--asset-urdf",
        default="artifacts/humanoidgen_rulesafe_bridge/safe_101612/mobility.urdf",
    )
    parser.add_argument("--rule-profile", choices=sorted(RULE_PLAN_PROFILES), default="historical")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plans = RULE_PLAN_PROFILES[args.rule_profile]
    rule_text = RULE_TEXT_PROFILES[args.rule_profile]
    if args.joint_source == "asset" and not Path(args.asset_urdf).is_file():
        raise FileNotFoundError(f"asset joint source requires --asset-urdf: {args.asset_urdf}")
    rng = np.random.default_rng(args.seed)
    joint_states = []
    actions = []
    rule_ids = []
    instructions = []
    success = []
    process_scores = []
    for rule_id in args.rules:
        if rule_id not in plans:
            raise KeyError(f"No {args.rule_profile!r} symbolic plan registered for {rule_id}")
        for _local_idx in range(args.demos_per_rule):
            demo_idx = len(joint_states)
            joints, acts, ok, score = generate_one(
                rule_id,
                args.trajectory_length,
                rng,
                demo_idx,
                args.joint_source,
                rule_profile=args.rule_profile,
                asset_urdf=args.asset_urdf,
            )
            joint_states.append(joints)
            actions.append(acts)
            rule_ids.append(rule_id)
            instructions.append(rule_text[rule_id])
            success.append(ok)
            process_scores.append(score)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        joint_states=np.stack(joint_states, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        rule_ids=np.array(rule_ids),
        instructions=np.array(instructions),
        success=np.array(success, dtype=np.bool_),
        process_scores=np.array(process_scores, dtype=np.float32),
        joint_source=np.array(args.joint_source),
        rule_profile=np.array(args.rule_profile),
        asset_urdf=np.array(args.asset_urdf if args.joint_source == "asset" else ""),
    )
    print("saved", out)
    print(
        "joint_states",
        np.stack(joint_states).shape,
        "success_rate",
        float(np.mean(success)),
        "joint_source",
        args.joint_source,
        "rule_profile",
        args.rule_profile,
    )
    print("rules", len(set(rule_ids)), sorted(set(rule_ids))[:3], sorted(set(rule_ids))[-3:])


if __name__ == "__main__":
    main()
