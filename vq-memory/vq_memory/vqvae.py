import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float = 4.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        flat_z = z.reshape(-1, self.latent_dim)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view_as(z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = z + (quantized - z).detach()
        return quantized, indices.view(z.shape[:-1]), loss


class JointStateVQVAE(nn.Module):
    def __init__(self, joint_dim=13, window_size=50, hidden_dim=256, latent_dim=64, codebook_size=256, commitment_cost=4.0):
        super().__init__()
        input_dim = joint_dim * window_size
        self.joint_dim = joint_dim
        self.window_size = window_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, joint_windows):
        batch = joint_windows.shape[0]
        z = self.encoder(joint_windows.reshape(batch, -1))
        quantized, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized).reshape(batch, self.window_size, self.joint_dim)
        recon_loss = F.mse_loss(recon, joint_windows)
        return {
            "recon": recon,
            "indices": indices,
            "loss": recon_loss + vq_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }
