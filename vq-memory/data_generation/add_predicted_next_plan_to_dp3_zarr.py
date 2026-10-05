import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from numcodecs import Blosc
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policies.next_plan_predictor import load_next_plan_predictor


def _class_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    result = {}
    for cls in range(num_classes):
        mask = target == cls
        result[str(cls)] = None if int(mask.sum()) == 0 else float(np.mean(pred[mask] == target[mask]))
    return result


def add_predicted_next_plan_tokens(zarr_path: Path, predictor_path: Path, batch_size: int = 8192, device: str = 'cuda:0') -> dict:
    root = zarr.open_group(str(zarr_path), mode='a')
    data = root['data']
    required = ['state', 'rule_onehot', 'memory_tokens', 'plan_tokens', 'next_plan_token']
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f'missing required zarr data arrays: {missing}')

    torch_device = torch.device(device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu')
    predictor = load_next_plan_predictor(predictor_path, torch_device)
    model = predictor.model
    state_mean = predictor.state_mean
    state_std = predictor.state_std.clamp_min(1e-6)

    total = int(data['state'].shape[0])
    if 'predicted_next_plan_token' in data:
        del data['predicted_next_plan_token']
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    out = data.zeros(
        'predicted_next_plan_token',
        shape=(total, 1),
        chunks=(min(4096, total), 1),
        dtype='i8',
        compressor=compressor,
    )

    preds = np.zeros((total,), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, total, int(batch_size)), desc='predict_next_plan_tokens'):
            end = min(total, start + int(batch_size))
            state = torch.as_tensor(data['state'][start:end], dtype=torch.float32, device=torch_device)
            state = (state - state_mean) / state_std
            rule = torch.as_tensor(data['rule_onehot'][start:end], dtype=torch.float32, device=torch_device)
            memory = torch.as_tensor(data['memory_tokens'][start:end], dtype=torch.float32, device=torch_device)
            plan = torch.as_tensor(data['plan_tokens'][start:end], dtype=torch.float32, device=torch_device)
            logits = model(state, rule, memory, plan)
            batch_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
            preds[start:end] = batch_pred
            out[start:end, 0] = batch_pred

    target = np.asarray(data['next_plan_token'][:, 0], dtype=np.int64)
    num_classes = int(predictor.config.get('output_dim', int(max(target.max(), preds.max()) + 1)))
    summary = {
        'zarr_path': str(zarr_path),
        'predictor_path': str(predictor_path),
        'device': str(torch_device),
        'predicted_next_plan_token_shape': list(out.shape),
        'accuracy_vs_schedule': float(np.mean(preds == target)),
        'per_class_accuracy_vs_schedule': _class_metrics(preds, target, num_classes),
        'predicted_histogram': {str(i): int(np.sum(preds == i)) for i in range(num_classes)},
        'target_histogram': {str(i): int(np.sum(target == i)) for i in range(num_classes)},
    }
    root.attrs['predicted_next_plan_condition'] = 'next_plan_predictor_offline_tokens'
    root.attrs['predicted_next_plan_predictor'] = str(predictor_path)
    root.attrs['predicted_next_plan_accuracy_vs_schedule'] = summary['accuracy_vs_schedule']
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Add offline predictor-generated next_plan_token aliases to a RuleSafe DP3 zarr dataset.')
    parser.add_argument('--zarr', type=Path, default=Path('data/dp3_rulesafe_asset/rulesafe_20rules_1000eps_humanoidgen_door_256pts.zarr'))
    parser.add_argument('--predictor', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(add_predicted_next_plan_tokens(args.zarr, args.predictor, args.batch_size, args.device), indent=2))


if __name__ == '__main__':
    main()
