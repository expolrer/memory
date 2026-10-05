#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/autodl-tmp/VQ-Memory
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT/.conda/vqmemory"
cd "$PROJECT"
DATA=data/rulesafe_primitive/rulesafe_primitive_100x2.npz
VQ=artifacts/vq_memory_primitive/vq_memory_tokenizer.pt
for mode in none raw vq; do
  python -m train.train_state_policy \
    --config configs/policy_state.yaml \
    --data "$DATA" \
    --vq-artifact "$VQ" \
    --memory-mode "$mode" \
    --output-dir "artifacts/state_policy_${mode}" \
    --steps 80 \
    --batch-size 256
done
python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('artifacts').glob('state_policy_*/metrics.json')):
    m=json.loads(p.read_text())
    print(p, 'mode=', m['memory_mode'], 'input_dim=', m['input_dim'], 'initial=', round(m['initial_val_mse'], 8), 'final=', round(m['final_val_mse'], 8))
PY
