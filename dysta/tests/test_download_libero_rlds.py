from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "download_libero_rlds.py"
SPEC = spec_from_file_location("download_libero_rlds", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_dataset(tmp_path: Path, shard_count: int = 2) -> Path:
    dataset_dir = tmp_path / "libero_test_no_noops"
    version_dir = dataset_dir / "1.0.0"
    version_dir.mkdir(parents=True)
    (version_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "name": "libero_test",
                "version": "1.0.0",
                "splits": [
                    {
                        "name": "train",
                        "numBytes": "6",
                        "shardLengths": [1] * shard_count,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "features.json").write_text("{}\n", encoding="utf-8")
    for index in range(shard_count):
        (version_dir / f"libero_test-train.tfrecord-{index:05d}-of-{shard_count:05d}").write_bytes(b"abc")
    return dataset_dir


def test_validate_dataset_builds_file_manifest(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path)

    result = MODULE.validate_dataset(dataset_dir)

    assert result["dataset"] == "libero_test_no_noops"
    assert result["examples"] == 2
    assert result["tfrecord_shards"] == 2
    assert len(result["files"]) == 4
    assert all(len(file_info["sha256"]) == 64 for file_info in result["files"])


def test_validate_dataset_rejects_missing_shard(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path)
    next((dataset_dir / "1.0.0").glob("*.tfrecord-*")).unlink()

    with pytest.raises(ValueError, match="expected 2 TFRecord shards, found 1"):
        MODULE.validate_dataset(dataset_dir)
