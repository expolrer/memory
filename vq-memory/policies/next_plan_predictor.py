from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


class NextPlanTokenPredictor(nn.Module):
    def __init__(
        self,
        state_dim: int = 13,
        rule_dim: int = 20,
        memory_vocab_size: int = 4,
        memory_length: int = 40,
        plan_vocab_size: int = 7,
        plan_length: int = 8,
        embed_dim: int = 16,
        token_feature_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 7,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.rule_dim = int(rule_dim)
        self.memory_vocab_size = int(memory_vocab_size)
        self.memory_length = int(memory_length)
        self.plan_vocab_size = int(plan_vocab_size)
        self.plan_length = int(plan_length)
        self.output_dim = int(output_dim)

        self.memory_embedding = nn.Embedding(self.memory_vocab_size, embed_dim)
        self.memory_encoder = nn.Sequential(
            nn.Conv1d(embed_dim, token_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(token_feature_dim, token_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.plan_embedding = nn.Embedding(self.plan_vocab_size, embed_dim)
        self.plan_encoder = nn.Sequential(
            nn.Conv1d(embed_dim, token_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(token_feature_dim, token_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        input_dim = self.state_dim + self.rule_dim + token_feature_dim * 2
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, state, rule_onehot, memory_tokens, plan_tokens):
        memory_tokens = memory_tokens.round().long().clamp(min=0, max=self.memory_vocab_size - 1)
        plan_tokens = plan_tokens.round().long().clamp(min=0, max=self.plan_vocab_size - 1)
        memory_feat = self.memory_encoder(self.memory_embedding(memory_tokens).transpose(1, 2))
        plan_feat = self.plan_encoder(self.plan_embedding(plan_tokens).transpose(1, 2))
        x = torch.cat([state.float(), rule_onehot.float(), memory_feat, plan_feat], dim=-1)
        return self.head(x)


@dataclass
class LoadedNextPlanPredictor:
    model: NextPlanTokenPredictor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    config: Dict

    @torch.no_grad()
    def predict_token(self, state, rule_onehot, memory_tokens, plan_tokens) -> int:
        device = self.state_mean.device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
        rule_t = torch.as_tensor(rule_onehot, dtype=torch.float32, device=device).reshape(1, -1)
        memory_t = torch.as_tensor(memory_tokens, dtype=torch.float32, device=device).reshape(1, -1)
        plan_t = torch.as_tensor(plan_tokens, dtype=torch.float32, device=device).reshape(1, -1)
        state_t = (state_t - self.state_mean) / self.state_std.clamp_min(1e-6)
        logits = self.model(state_t, rule_t, memory_t, plan_t)
        return int(torch.argmax(logits, dim=-1).item())


def load_next_plan_predictor(path: Path, device: torch.device) -> LoadedNextPlanPredictor:
    checkpoint = torch.load(path, map_location="cpu")
    config = dict(checkpoint["config"])
    model = NextPlanTokenPredictor(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    state_mean = checkpoint["state_mean"].to(device=device, dtype=torch.float32).reshape(1, -1)
    state_std = checkpoint["state_std"].to(device=device, dtype=torch.float32).reshape(1, -1)
    return LoadedNextPlanPredictor(model=model, state_mean=state_mean, state_std=state_std, config=config)
