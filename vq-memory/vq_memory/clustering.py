import torch


@torch.no_grad()
def kmeans_codebook(codebook: torch.Tensor, clusters: int = 4, iterations: int = 100, seed: int = 42):
    """Cluster a learned VQ codebook into fewer semantic memory tokens."""
    generator = torch.Generator(device=codebook.device).manual_seed(seed)
    perm = torch.randperm(codebook.shape[0], generator=generator, device=codebook.device)
    centroids = codebook[perm[:clusters]].clone()
    assignments = torch.zeros(codebook.shape[0], dtype=torch.long, device=codebook.device)
    for _ in range(iterations):
        distances = torch.cdist(codebook, centroids)
        new_assignments = distances.argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for idx in range(clusters):
            mask = assignments == idx
            if mask.any():
                centroids[idx] = codebook[mask].mean(dim=0)
    return centroids, assignments
