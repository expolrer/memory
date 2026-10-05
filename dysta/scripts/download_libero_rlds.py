#!/usr/bin/env python3
"""Download selected official LIBERO RLDS directories and build a SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download
    except ImportError:
        workspace_root = Path(__file__).resolve().parents[2]
        vendored_packages = workspace_root / "tools" / "_hf_download"
        if not vendored_packages.is_dir():
            raise
        sys.path.insert(0, str(vendored_packages))
        from huggingface_hub import snapshot_download

        return snapshot_download


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(dataset_dir: Path) -> dict:
    version_dir = dataset_dir / "1.0.0"
    info_path = version_dir / "dataset_info.json"
    features_path = version_dir / "features.json"
    if not info_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(f"missing TFDS metadata under {version_dir}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    splits = info.get("splits", [])
    if len(splits) != 1 or splits[0].get("name") != "train":
        raise ValueError(f"unexpected split metadata in {info_path}")

    expected_shards = len(splits[0].get("shardLengths", []))
    tfrecords = sorted(version_dir.glob("*.tfrecord-*"))
    if len(tfrecords) != expected_shards:
        raise ValueError(
            f"{dataset_dir.name}: expected {expected_shards} TFRecord shards, found {len(tfrecords)}"
        )

    files = sorted(path for path in version_dir.iterdir() if path.is_file())
    return {
        "dataset": dataset_dir.name,
        "tfds_name": info.get("name"),
        "version": info.get("version"),
        "examples": sum(int(value) for value in splits[0]["shardLengths"]),
        "expected_data_bytes": int(splits[0]["numBytes"]),
        "tfrecord_shards": len(tfrecords),
        "files": [
            {
                "path": str(path.relative_to(dataset_dir.parent)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="openvla/modified_libero_rlds")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()

    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("--datasets must not contain duplicates")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    if not args.validate_only:
        snapshot_download = load_snapshot_download()
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=[f"{dataset}/**" for dataset in args.datasets],
            max_workers=args.max_workers,
        )

    datasets = [validate_dataset(args.local_dir / name) for name in args.datasets]
    manifest = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "local_dir": str(args.local_dir.resolve()),
        "datasets": datasets,
        "total_files": sum(len(dataset["files"]) for dataset in datasets),
        "total_bytes": sum(
            file_info["bytes"] for dataset in datasets for file_info in dataset["files"]
        ),
    }
    if args.expected_manifest is not None:
        expected = json.loads(args.expected_manifest.read_text(encoding="utf-8"))
        expected_content = {
            "repo_id": expected.get("repo_id"),
            "revision": expected.get("revision"),
            "datasets": expected.get("datasets"),
        }
        actual_content = {
            "repo_id": manifest["repo_id"],
            "revision": manifest["revision"],
            "datasets": manifest["datasets"],
        }
        if actual_content != expected_content:
            raise ValueError("validated RLDS content does not match --expected-manifest")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("revision", "total_files", "total_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
