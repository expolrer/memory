import json
import subprocess
import sys


def test_point_policy_train_smoke(tmp_path):
    data = tmp_path / "tiny.npz"
    vq_out = tmp_path / "vq"
    subprocess.check_call([
        sys.executable, "-m", "data_generation.collect_symbolic_demos",
        "--out", str(data), "--rules", "rule001", "rule020",
        "--demos-per-rule", "2", "--trajectory-length", "90", "--joint-source", "primitive",
    ])
    subprocess.check_call([
        sys.executable, "-m", "train.train_vqvae",
        "--data", str(data), "--output-dir", str(vq_out), "--steps", "2", "--batch-size", "4",
    ])
    out = tmp_path / "point_vq"
    subprocess.check_call([
        sys.executable, "-m", "train.train_point_policy",
        "--data", str(data), "--vq-artifact", str(vq_out / "vq_memory_tokenizer.pt"),
        "--memory-mode", "vq", "--output-dir", str(out), "--steps", "2", "--batch-size", "4", "--point-count", "32",
    ])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["memory_mode"] == "vq"
    assert metrics["steps"] == 2
    assert metrics["point_count"] == 32
    assert metrics["final_val_mse"] >= 0.0
