from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class JointWindowDataset(Dataset):
    """Windowed joint-state dataset for VQ-Memory training.

    Expected NPZ schema:
      - joint_states: float32 array [num_trajectories, timesteps, joint_dim]
      - actions: float32 array [num_trajectories, timesteps, joint_dim]
      - rule_ids: string array [num_trajectories]
    """

    def __init__(self, npz_path: str, window_size: int = 50, stride: int = 20):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=True)
        self.joint_states = data["joint_states"].astype(np.float32)
        self.rule_ids = data.get("rule_ids", np.array(["unknown"] * len(self.joint_states)))
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.index: List[Tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.joint_states):
            if traj.shape[0] < self.window_size:
                continue
            for start in range(0, traj.shape[0] - self.window_size + 1, self.stride):
                self.index.append((traj_idx, start))
        if not self.index:
            raise ValueError(f"no windows found in {npz_path}; check trajectory length/window_size")

    @property
    def joint_dim(self) -> int:
        return int(self.joint_states.shape[-1])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        traj_idx, start = self.index[idx]
        window = self.joint_states[traj_idx, start:start + self.window_size]
        return torch.from_numpy(window)
