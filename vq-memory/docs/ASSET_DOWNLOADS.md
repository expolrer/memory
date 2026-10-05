# Asset Downloads

HumanoidGen requires two asset archives from the dataset repository `TeleEmbodied/humanoidgen_dataset`:

- `assets/assets.zip`
- `assets/table_assets.zip`

Run:

```bash
cd /root/autodl-tmp/VQ-Memory
bash scripts/download_humanoidgen_assets.sh
```

If direct HuggingFace access is slow from the server, try:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_humanoidgen_assets.sh
```

Expected extracted directories:

- `/root/autodl-tmp/VQ-Memory/third_party/HumanoidGen/assets`
- `/root/autodl-tmp/VQ-Memory/third_party/HumanoidGen/humanoidgen/scene_builder/table/assets`

A live `list_repo_files` query timed out during setup, so the script uses the file paths from HumanoidGen's README and HuggingFace link structure.
