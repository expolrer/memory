from __future__ import annotations

import json
import os
import platform
import sys

import torch


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": visible,
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if visible.replace(" ", "") != "6,7":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 6,7")
    if torch.cuda.device_count() != 2:
        raise SystemExit("expected exactly two visible CUDA devices")


if __name__ == "__main__":
    main()
