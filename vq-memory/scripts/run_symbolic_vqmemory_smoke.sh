#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/autodl-tmp/VQ-Memory
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT/.conda/vqmemory"
cd "$PROJECT"
python -m data_generation.collect_symbolic_demos \
  --out data/rulesafe_symbolic/rulesafe_symbolic_100x2.npz \
  --demos-per-rule 100 \
  --trajectory-length 220
python -m train.train_vqvae \
  --config configs/vq_memory.yaml \
  --data data/rulesafe_symbolic/rulesafe_symbolic_100x2.npz \
  --output-dir artifacts/vq_memory_symbolic \
  --steps 50 \
  --batch-size 128
python - <<'PY'
import torch
p='artifacts/vq_memory_symbolic/vq_memory_tokenizer.pt'
a=torch.load(p, map_location='cpu')
print('artifact', p)
print('steps', a['steps'], 'windows', a['num_windows'])
print('centroids', tuple(a['cluster_centroids'].shape), 'assignment', tuple(a['code_to_cluster'].shape))
print('last_losses', a['last_losses'])
PY
