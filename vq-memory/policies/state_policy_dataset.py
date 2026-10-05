from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from vq_memory import JointStateVQVAE


class StatePolicyDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        memory_mode: str = "none",
        raw_memory_steps: int = 40,
        vq_artifact: Optional[str] = None,
        vq_memory_length: int = 40,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=True)
        self.joint_states = data["joint_states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.rule_ids = data["rule_ids"]
        self.memory_mode = memory_mode
        self.raw_memory_steps = int(raw_memory_steps)
        self.vq_memory_length = int(vq_memory_length)
        if memory_mode not in {"none", "raw", "vq"}:
            raise ValueError(f"unknown memory_mode={memory_mode!r}")

        rng = np.random.default_rng(seed)
        traj_indices = np.arange(self.joint_states.shape[0])
        rng.shuffle(traj_indices)
        split_at = max(1, int(round(len(traj_indices) * (1.0 - val_fraction))))
        selected = traj_indices[:split_at] if split == "train" else traj_indices[split_at:]
        if len(selected) == 0:
            selected = traj_indices[-1:]

        min_t = 1 if memory_mode == "none" else max(1, self.raw_memory_steps)
        self.index = []
        for traj_idx in selected:
            for t in range(min_t, self.joint_states.shape[1] - 1):
                self.index.append((int(traj_idx), int(t)))

        self.code_to_cluster = None
        self.vqvae = None
        if memory_mode == "vq":
            if vq_artifact is None:
                raise ValueError("vq_artifact is required for vq mode")
            artifact = torch.load(vq_artifact, map_location="cpu")
            cfg = artifact["config"]
            self.vqvae = JointStateVQVAE(
                joint_dim=cfg["joint_dim"],
                window_size=cfg["window_size"],
                hidden_dim=cfg["hidden_dim"],
                latent_dim=cfg["latent_dim"],
                codebook_size=cfg["codebook_size"],
                commitment_cost=cfg["commitment_cost"],
            )
            self.vqvae.load_state_dict(artifact["model_state_dict"])
            self.vqvae.eval()
            self.vq_window_size = int(cfg["window_size"])
            self.vq_stride = int(cfg["stride"])
            self.code_to_cluster = artifact["code_to_cluster"].long()
            self.vq_vocab_size = int(cfg["cluster_size"])
        else:
            self.vq_vocab_size = 4

    def __len__(self) -> int:
        return len(self.index)

    def _raw_memory(self, traj_idx: int, t: int) -> torch.Tensor:
        start = max(0, t - self.raw_memory_steps)
        mem = self.joint_states[traj_idx, start:t]
        if mem.shape[0] < self.raw_memory_steps:
            pad = np.repeat(mem[:1], self.raw_memory_steps - mem.shape[0], axis=0)
            mem = np.concatenate([pad, mem], axis=0)
        return torch.from_numpy(mem.astype(np.float32))

    @torch.no_grad()
    def _vq_tokens(self, traj_idx: int, t: int) -> torch.Tensor:
        hist = torch.from_numpy(self.joint_states[traj_idx, :t + 1].astype(np.float32))
        tokens = []
        if hist.shape[0] >= self.vq_window_size:
            windows = []
            for end in range(self.vq_window_size, hist.shape[0] + 1, self.vq_stride):
                windows.append(hist[end - self.vq_window_size:end])
            if windows:
                batch = torch.stack(windows, dim=0)
                out = self.vqvae(batch)
                codes = out["indices"].reshape(-1)
                tokens = self.code_to_cluster[codes].long()
        if len(tokens) == 0:
            tokens = torch.zeros(0, dtype=torch.long)
        if tokens.numel() > self.vq_memory_length:
            tokens = tokens[-self.vq_memory_length:]
        if tokens.numel() < self.vq_memory_length:
            pad = torch.zeros(self.vq_memory_length - tokens.numel(), dtype=torch.long)
            tokens = torch.cat([pad, tokens.long()], dim=0)
        return tokens.long()

    def __getitem__(self, idx: int):
        traj_idx, t = self.index[idx]
        current = torch.from_numpy(self.joint_states[traj_idx, t].astype(np.float32))
        action = torch.from_numpy(self.actions[traj_idx, t].astype(np.float32))
        item = {"current_state": current, "action": action, "traj_idx": traj_idx, "t": t}
        if self.memory_mode == "raw":
            item["raw_memory"] = self._raw_memory(traj_idx, t)
        elif self.memory_mode == "vq":
            item["vq_tokens"] = self._vq_tokens(traj_idx, t)
        return item
