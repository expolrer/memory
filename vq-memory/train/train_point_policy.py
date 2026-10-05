import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from policies import PointActionPolicy, PointPolicyFeatureBuilder
from policies.point_policy_dataset import PointPolicyDataset


def prepare_memory(batch, builder: PointPolicyFeatureBuilder, device):
    current = batch["current_state"].to(device)
    raw_memory = batch.get("raw_memory")
    vq_tokens = batch.get("vq_tokens")
    if raw_memory is not None:
        raw_memory = raw_memory.to(device)
    if vq_tokens is not None:
        vq_tokens = vq_tokens.to(device)
    return current, builder.make_memory_features(current, raw_memory=raw_memory, vq_tokens=vq_tokens)


@torch.no_grad()
def evaluate(model, loader, builder, device, max_batches=10):
    model.eval()
    total = 0.0
    count = 0
    for idx, batch in enumerate(loader):
        points = batch["points"].to(device).float()
        current, memory = prepare_memory(batch, builder, device)
        target = batch["action"].to(device)
        pred = model(points, current, memory)
        loss = F.mse_loss(pred, target, reduction="sum")
        total += float(loss.detach().cpu())
        count += target.numel()
        if idx + 1 >= max_batches:
            break
    model.train()
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_point.yaml")
    parser.add_argument("--data", default="data/rulesafe_multitask/rulesafe_20rules_50x.npz")
    parser.add_argument("--vq-artifact", default="artifacts/vq_memory_multitask/vq_memory_tokenizer.pt")
    parser.add_argument("--memory-mode", choices=["none", "raw", "vq"], default="none")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--point-count", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(int(cfg.get("seed", 42)))
    point_count = int(args.point_count or cfg["point_count"])
    batch_size = int(args.batch_size or cfg["batch_size"])

    ds_kwargs = dict(
        memory_mode=args.memory_mode,
        raw_memory_steps=cfg["raw_memory_steps"],
        vq_artifact=args.vq_artifact if args.memory_mode == "vq" else None,
        vq_memory_length=cfg["vq_memory_length"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        point_count=point_count,
    )
    train_ds = PointPolicyDataset(args.data, split="train", **ds_kwargs)
    val_ds = PointPolicyDataset(args.data, split="val", **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    builder = PointPolicyFeatureBuilder(
        point_feature_dim=cfg["point_feature_dim"],
        memory_mode=args.memory_mode,
        raw_memory_steps=cfg["raw_memory_steps"],
        joint_dim=cfg["joint_dim"],
        vq_memory_length=cfg["vq_memory_length"],
        vq_vocab_size=cfg["vq_vocab_size"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointActionPolicy(builder, point_dim=cfg["point_dim"], hidden_dim=cfg["hidden_dim"], action_dim=cfg["action_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    max_steps = int(args.steps or cfg["max_steps"])

    initial_val_mse = evaluate(model, val_loader, builder, device)
    progress = tqdm(total=max_steps, desc=f"train_point_{args.memory_mode}")
    step = 0
    last_train_mse = None
    while step < max_steps:
        for batch in train_loader:
            points = batch["points"].to(device).float()
            current, memory = prepare_memory(batch, builder, device)
            target = batch["action"].to(device)
            pred = model(points, current, memory)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            last_train_mse = float(loss.detach().cpu())
            progress.set_postfix(train_mse=f"{last_train_mse:.6f}")
            progress.update(1)
            if step >= max_steps:
                break
    progress.close()
    final_val_mse = evaluate(model, val_loader, builder, device)

    out_dir = Path(args.output_dir or f"artifacts/point_policy_{args.memory_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "memory_mode": args.memory_mode,
        "steps": step,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "point_count": point_count,
        "point_feature_dim": cfg["point_feature_dim"],
        "input_dim": builder.input_dim,
        "initial_val_mse": initial_val_mse,
        "final_val_mse": final_val_mse,
        "last_train_mse": last_train_mse,
        "data": args.data,
        "vq_artifact": args.vq_artifact if args.memory_mode == "vq" else None,
    }
    torch.save({"model_state_dict": model.cpu().state_dict(), "config": cfg, "metrics": metrics}, out_dir / "point_policy.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
