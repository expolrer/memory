from typing import Optional

import torch

from .point_cloud import primitive_safe_point_cloud
from .state_policy_dataset import StatePolicyDataset


class PointPolicyDataset(StatePolicyDataset):
    def __init__(self, *args, point_count: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_count = int(point_count)

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        traj_idx = int(item["traj_idx"])
        t = int(item["t"])
        seed = traj_idx * 1000003 + t
        points = primitive_safe_point_cloud(self.joint_states[traj_idx, t], point_count=self.point_count, seed=seed)
        item["points"] = torch.from_numpy(points)
        return item
