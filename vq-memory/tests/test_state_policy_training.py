import json
import subprocess
import sys
from pathlib import Path


def test_state_policy_train_none_and_vq(tmp_path):
    data = tmp_path / "tiny.npz"
    vq_out = tmp_path / "vq"
    subprocess.check_call([
        sys.executable,
        "-m",
        "data_generation.collect_symbolic_demos",
        "--out",
        str(data),
        "--demos-per-rule",
        "3",
        "--trajectory-length",
        "90",
        "--joint-source",
        "primitive",
    ])
    subprocess.check_call([
        sys.executable,
        "-m",
        "train.train_vqvae",
        "--data",
        str(data),
        "--output-dir",
        str(vq_out),
        "--steps",
        "2",
        "--batch-size",
        "4",
    ])
    for mode in ["none", "vq"]:
        out = tmp_path / f"policy_{mode}"
        subprocess.check_call([
            sys.executable,
            "-m",
            "train.train_state_policy",
            "--data",
            str(data),
            "--vq-artifact",
            str(vq_out / "vq_memory_tokenizer.pt"),
            "--memory-mode",
            mode,
            "--output-dir",
            str(out),
            "--steps",
            "2",
            "--batch-size",
            "8",
        ])
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["memory_mode"] == mode
        assert metrics["steps"] == 2
        assert metrics["final_val_mse"] >= 0.0
