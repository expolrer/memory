#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/autodl-tmp/VQ-Memory
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT/.conda/vqmemory"
python - <<'PY'
from pathlib import Path
import mplib
path = Path(mplib.planner.__file__)
text = path.read_text()
text = text.replace('n_init_qpos=20', 'n_init_qpos=50')
text = text.replace('n_init_qpos: int = 20', 'n_init_qpos: int = 50')
path.write_text(text)
print(path)
for i, line in enumerate(text.splitlines(), 1):
    if 'n_init_qpos' in line:
        print(f'{i}: {line}')
PY
