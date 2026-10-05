import subprocess
import sys
from pathlib import Path

import numpy as np

from vq_memory.dataset import JointWindowDataset


def test_symbolic_generation_and_dataset(tmp_path):
    out = tmp_path / "tiny.npz"
    subprocess.check_call([
        sys.executable,
        "-m",
        "data_generation.collect_symbolic_demos",
        "--out",
        str(out),
        "--demos-per-rule",
        "2",
        "--trajectory-length",
        "80",
    ])
    data = np.load(out, allow_pickle=True)
    assert data["joint_states"].shape == (40, 80, 13)
    assert data["actions"].shape == (40, 80, 13)
    assert len(set(data["rule_ids"].tolist())) == 20
    assert data["success"].all()
    ds = JointWindowDataset(str(out), window_size=20, stride=10)
    assert len(ds) > 0
    assert tuple(ds[0].shape) == (20, 13)
