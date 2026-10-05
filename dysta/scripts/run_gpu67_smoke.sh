#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_gpu67.sh"
cd /ssd/DySta/dysta_reproduction
python scripts/check_environment.py
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/smoke_ddp.py
