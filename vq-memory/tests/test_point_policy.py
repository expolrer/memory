import subprocess
import sys

import numpy as np
import torch

from policies.point_cloud import primitive_safe_point_cloud
from policies.point_policy import PointActionPolicy, PointPolicyFeatureBuilder
from policies.point_policy_dataset import PointPolicyDataset


def test_primitive_point_cloud_shape():
    joint = np.zeros(13, dtype=np.float32)
    joint[6:12] = [-0.1, 0.4, 0.11, 0.35, -0.01, -0.08]
    pc = primitive_safe_point_cloud(joint, point_count=128, seed=0)
    assert pc.shape == (128, 6)
    assert np.isfinite(pc).all()


def test_point_policy_forward_on_generated_data(tmp_path):
    data_path = tmp_path / 'rulesafe_tiny.npz'
    subprocess.check_call([
        sys.executable,
        '-m',
        'data_generation.collect_symbolic_demos',
        '--out',
        str(data_path),
        '--rules',
        'rule001',
        'rule020',
        '--demos-per-rule',
        '1',
        '--trajectory-length',
        '80',
    ])
    ds = PointPolicyDataset(
        str(data_path),
        memory_mode='none',
        point_count=64,
        split='train',
    )
    item = ds[0]
    builder = PointPolicyFeatureBuilder(memory_mode='none')
    model = PointActionPolicy(builder)
    points = item['points'].unsqueeze(0).float()
    current = item['current_state'].unsqueeze(0).float()
    memory = torch.zeros(1, builder.memory_dim)
    out = model(points, current, memory)
    assert out.shape == (1, 13)
