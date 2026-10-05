import torch


class VQMemoryTokenizer:
    def __init__(self, vqvae, code_to_cluster=None, memory_length=40, window_size=50, stride=20):
        self.vqvae = vqvae.eval()
        self.code_to_cluster = code_to_cluster
        self.memory_length = memory_length
        self.window_size = window_size
        self.stride = stride

    @torch.no_grad()
    def encode_sequence(self, joint_sequence: torch.Tensor):
        windows = []
        for end in range(self.window_size, joint_sequence.shape[0] + 1, self.stride):
            windows.append(joint_sequence[end - self.window_size:end])
        if not windows:
            return torch.empty(0, dtype=torch.long, device=joint_sequence.device)
        batch = torch.stack(windows, dim=0)
        output = self.vqvae(batch)
        tokens = output["indices"].reshape(-1)
        if self.code_to_cluster is not None:
            tokens = self.code_to_cluster.to(tokens.device)[tokens]
        if tokens.numel() > self.memory_length:
            tokens = tokens[-self.memory_length:]
        return tokens
