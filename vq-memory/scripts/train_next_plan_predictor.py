import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policies.next_plan_predictor import NextPlanTokenPredictor


def _episode_indices(episode_ends, episode_ids):
    pieces = []
    for episode_idx in episode_ids:
        start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        end = int(episode_ends[episode_idx])
        pieces.append(np.arange(start, end, dtype=np.int64))
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=np.int64)


def _batch(arrays, indices, device):
    state, rule, memory, plan, labels, state_mean, state_std = arrays
    state_t = torch.from_numpy(state[indices]).to(device=device, dtype=torch.float32)
    state_t = (state_t - state_mean) / state_std.clamp_min(1e-6)
    return (
        state_t,
        torch.from_numpy(rule[indices]).to(device=device, dtype=torch.float32),
        torch.from_numpy(memory[indices]).to(device=device, dtype=torch.float32),
        torch.from_numpy(plan[indices]).to(device=device, dtype=torch.float32),
        torch.from_numpy(labels[indices]).to(device=device, dtype=torch.long),
    )


@torch.no_grad()
def _evaluate(model, arrays, indices, device, batch_size):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    per_class_total = np.zeros((model.output_dim,), dtype=np.int64)
    per_class_correct = np.zeros((model.output_dim,), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) == 0:
            continue
        state, rule, memory, plan, labels = _batch(arrays, batch_idx, device)
        logits = model(state, rule, memory, plan)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        pred = torch.argmax(logits, dim=-1)
        total_loss += float(loss.detach().cpu())
        total += int(labels.numel())
        correct += int((pred == labels).sum().detach().cpu())
        labels_np = labels.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        for cls in range(model.output_dim):
            mask = labels_np == cls
            per_class_total[cls] += int(mask.sum())
            per_class_correct[cls] += int((pred_np[mask] == cls).sum())
    model.train()
    per_class_acc = {
        str(cls): (float(per_class_correct[cls] / per_class_total[cls]) if per_class_total[cls] else None)
        for cls in range(model.output_dim)
    }
    return {
        'loss': total_loss / max(1, total),
        'accuracy': correct / max(1, total),
        'examples': total,
        'per_class_accuracy': per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Train a closed-loop next-event token predictor for the RuleSafe next-plan scaffold.')
    parser.add_argument('--zarr', type=Path, default=Path('data/dp3_rulesafe_asset/rulesafe_20rules_1000eps_humanoidgen_door_256pts.zarr'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--eval-batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    root = zarr.open_group(str(args.zarr), mode='r')
    state = np.asarray(root['data/state'][:], dtype=np.float32)
    rule = np.asarray(root['data/rule_onehot'][:], dtype=np.float32)
    memory = np.asarray(root['data/memory_tokens'][:], dtype=np.int64)
    plan = np.asarray(root['data/plan_tokens'][:], dtype=np.int64)
    labels = np.asarray(root['data/next_plan_token'][:, 0], dtype=np.int64)
    episode_ends = np.asarray(root['meta/episode_ends'][:], dtype=np.int64)

    episode_ids = np.arange(len(episode_ends), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episode_ids)
    n_val = max(1, int(round(len(episode_ids) * args.val_ratio)))
    val_episode_ids = np.sort(episode_ids[:n_val])
    train_episode_ids = np.sort(episode_ids[n_val:])
    train_indices = _episode_indices(episode_ends, train_episode_ids)
    val_indices = _episode_indices(episode_ends, val_episode_ids)

    state_mean_np = state[train_indices].mean(axis=0).astype(np.float32)
    state_std_np = state[train_indices].std(axis=0).astype(np.float32)
    state_std_np[state_std_np < 1e-6] = 1.0
    state_mean = torch.from_numpy(state_mean_np).to(device=device, dtype=torch.float32).reshape(1, -1)
    state_std = torch.from_numpy(state_std_np).to(device=device, dtype=torch.float32).reshape(1, -1)
    arrays = (state, rule, memory, plan, labels, state_mean, state_std)

    config = {
        'state_dim': int(state.shape[-1]),
        'rule_dim': int(rule.shape[-1]),
        'memory_vocab_size': int(root.attrs.get('memory_token_vocab_size', 4)),
        'memory_length': int(memory.shape[-1]),
        'plan_vocab_size': int(root.attrs.get('plan_token_vocab_size', 7)),
        'plan_length': int(plan.shape[-1]),
        'embed_dim': 16,
        'token_feature_dim': 32,
        'hidden_dim': 128,
        'output_dim': int(root.attrs.get('plan_token_vocab_size', 7)),
    }
    model = NextPlanTokenPredictor(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    losses = []
    model.train()
    for _ in trange(args.steps, desc='train_next_plan_predictor'):
        batch_idx = rng.choice(train_indices, size=args.batch_size, replace=True)
        state_b, rule_b, memory_b, plan_b, labels_b = _batch(arrays, batch_idx, device)
        logits = model(state_b, rule_b, memory_b, plan_b)
        loss = F.cross_entropy(logits, labels_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    train_eval_idx = rng.choice(train_indices, size=min(len(train_indices), 50000), replace=False)
    val_eval = _evaluate(model, arrays, val_indices, device, args.eval_batch_size)
    train_eval = _evaluate(model, arrays, train_eval_idx, device, args.eval_batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / 'checkpoint.pt'
    torch.save({
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'state_mean': torch.from_numpy(state_mean_np),
        'state_std': torch.from_numpy(state_std_np),
        'config': config,
        'zarr': str(args.zarr),
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'train_episodes': int(len(train_episode_ids)),
        'val_episodes': int(len(val_episode_ids)),
    }, checkpoint_path)

    metrics = {
        'checkpoint': str(checkpoint_path),
        'device': str(device),
        'steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'train_examples': int(len(train_indices)),
        'val_examples': int(len(val_indices)),
        'label_histogram': {str(i): int((labels == i).sum()) for i in range(config['output_dim'])},
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'tail_mean_loss': float(np.mean(losses[-min(200, len(losses)):])) if losses else None,
        'train_eval': train_eval,
        'val_eval': val_eval,
    }
    with open(args.output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
