import subprocess
import sys
from pathlib import Path

import torch


def test_train_vqvae_smoke(tmp_path):
    data = tmp_path / "tiny.npz"
    out_dir = tmp_path / "artifact"
    subprocess.check_call([
        sys.executable,
        "-m",
        "data_generation.collect_symbolic_demos",
        "--out",
        str(data),
        "--demos-per-rule",
        "2",
        "--trajectory-length",
        "90",
    ])
    subprocess.check_call([
        sys.executable,
        "-m",
        "train.train_vqvae",
        "--config",
        "configs/vq_memory.yaml",
        "--data",
        str(data),
        "--output-dir",
        str(out_dir),
        "--steps",
        "2",
        "--batch-size",
        "4",
    ])
    artifact = torch.load(out_dir / "vq_memory_tokenizer.pt", map_location="cpu")
    assert artifact["cluster_centroids"].shape[0] == 4
    assert artifact["code_to_cluster"].shape[0] == 256
    assert artifact["steps"] == 2
