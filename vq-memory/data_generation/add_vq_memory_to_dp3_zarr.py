import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from vq_memory import JointStateVQVAE


def _load_model(tokenizer_path: Path, device: torch.device):
    checkpoint = torch.load(tokenizer_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = JointStateVQVAE(
        joint_dim=cfg["joint_dim"],
        window_size=cfg["window_size"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        codebook_size=cfg["codebook_size"],
        commitment_cost=cfg["commitment_cost"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    code_to_cluster = checkpoint.get("code_to_cluster")
    if code_to_cluster is not None:
        code_to_cluster = code_to_cluster.to(device)
    return model, cfg, code_to_cluster


@torch.no_grad()
def _episode_memory_tokens(states: np.ndarray, model, cfg: dict, code_to_cluster, device: torch.device) -> np.ndarray:
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    memory_length = int(cfg.get("memory_length", 40))
    trajectory_length = states.shape[0]
    memory = np.zeros((trajectory_length, memory_length), dtype=np.int64)
    ends = list(range(window_size, trajectory_length + 1, stride))
    if not ends:
        return memory
    windows = np.stack([states[end - window_size:end] for end in ends], axis=0).astype(np.float32)
    batch = torch.from_numpy(windows).to(device)
    output = model(batch)
    tokens = output["indices"].reshape(-1)
    if code_to_cluster is not None:
        tokens = code_to_cluster[tokens]
    tokens_np = tokens.detach().cpu().numpy().astype(np.int64)
    for t in range(trajectory_length):
        count = int(np.searchsorted(ends, t, side="right"))
        if count <= 0:
            continue
        selected = tokens_np[max(0, count - memory_length):count]
        memory[t, memory_length - len(selected):] = selected
    return memory


def add_vq_memory(npz_path: Path, zarr_path: Path, tokenizer_path: Path, device_name: str = "cuda") -> dict:
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    model, cfg, code_to_cluster = _load_model(tokenizer_path, device)
    with np.load(npz_path, allow_pickle=True) as payload:
        joint_states = np.asarray(payload["joint_states"], dtype=np.float32)

    root = zarr.open_group(str(zarr_path), mode="a")
    total_steps = int(root["meta"]["episode_ends"][-1])
    episode_ends = root["meta"]["episode_ends"][:]
    source_indices = root["meta/source_episode_index"][:] if "source_episode_index" in root["meta"] else np.arange(len(episode_ends))
    memory_length = int(cfg.get("memory_length", 40))
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    data = root["data"]
    if "memory_tokens" in data:
        del data["memory_tokens"]
    memory_arr = data.zeros(
        "memory_tokens",
        shape=(total_steps, memory_length),
        chunks=(min(1024, max(1, memory_length * 16)), memory_length),
        dtype="i8",
        compressor=compressor,
    )

    start = 0
    for episode_idx, end in enumerate(tqdm(episode_ends, desc="add_vq_memory")):
        source_idx = int(source_indices[episode_idx])
        memory = _episode_memory_tokens(joint_states[source_idx], model, cfg, code_to_cluster, device)
        memory_arr[start:int(end)] = memory
        start = int(end)

    root.attrs["memory_tokenizer"] = str(tokenizer_path)
    root.attrs["memory_length"] = memory_length
    root.attrs["memory_vocab_size"] = int(cfg.get("cluster_size", cfg.get("codebook_size", 0)))
    root.attrs["memory_window_size"] = int(cfg["window_size"])
    root.attrs["memory_stride"] = int(cfg["stride"])
    summary = {
        "zarr_path": str(zarr_path),
        "memory_tokens_shape": list(memory_arr.shape),
        "memory_length": memory_length,
        "memory_vocab_size": int(root.attrs["memory_vocab_size"]),
        "device": str(device),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add VQ-Memory token sequences to a DP3 zarr replay buffer.")
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--tokenizer", default="artifacts/vq_memory_multitask/vq_memory_tokenizer.pt", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(add_vq_memory(args.npz, args.zarr, args.tokenizer, args.device), indent=2))


if __name__ == "__main__":
    main()
