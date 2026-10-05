import torch
from vq_memory import JointStateVQVAE, kmeans_codebook, VQMemoryTokenizer


def test_vqvae_dry_components():
    model = JointStateVQVAE(joint_dim=13, window_size=50, codebook_size=16, latent_dim=8, hidden_dim=32)
    out = model(torch.randn(4, 50, 13))
    assert out["recon"].shape == (4, 50, 13)
    centroids, assignment = kmeans_codebook(model.quantizer.embedding.weight, clusters=4)
    assert centroids.shape == (4, 8)
    assert assignment.shape == (16,)


def test_tokenizer_sequence():
    model = JointStateVQVAE(joint_dim=13, window_size=10, codebook_size=16, latent_dim=8, hidden_dim=32)
    tokenizer = VQMemoryTokenizer(model, memory_length=5, window_size=10, stride=5)
    tokens = tokenizer.encode_sequence(torch.randn(40, 13))
    assert 0 < tokens.numel() <= 5
