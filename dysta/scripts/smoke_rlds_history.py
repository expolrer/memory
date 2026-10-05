"""Regression checks for temporal RLDS slicing in the OpenVLA patch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from prismatic.vla.datasets.datasets import RLDSBatchTransform


class RecordingActionTokenizer:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, actions):
        value = np.asarray(actions).copy()
        self.calls.append(value)
        if value.ndim == 1:
            return "A"
        return ["A"] * value.shape[0]


class FakeTokenizer:
    def __call__(self, _prompt, add_special_tokens=True):
        assert add_special_tokens
        return SimpleNamespace(input_ids=list(range(32)))


class FakePromptBuilder:
    def __init__(self, _family) -> None:
        self.turns = []

    def add_turn(self, role, value) -> None:
        self.turns.append((role, value))

    def get_prompt(self) -> str:
        return " ".join(value for _, value in self.turns)


def fake_image_transform(image) -> torch.Tensor:
    value = float(np.asarray(image)[0, 0, 0])
    return torch.full((1, 2, 2), value)


def make_batch() -> dict:
    actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
    primary = np.stack([np.full((2, 2, 3), i, dtype=np.uint8) for i in range(12)])
    wrist = np.stack([np.full((2, 2, 3), i + 20, dtype=np.uint8) for i in range(12)])
    proprio = np.arange(12 * 8, dtype=np.float32).reshape(12, 8)
    return {
        "dataset_name": "synthetic",
        "action": actions,
        "task": {"language_instruction": b"move the bowl"},
        "observation": {
            "image_primary": primary,
            "image_wrist": wrist,
            "proprio": proprio,
        },
    }


def main() -> None:
    batch = make_batch()
    action_tokenizer = RecordingActionTokenizer()
    transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=FakeTokenizer(),
        image_transform=fake_image_transform,
        prompt_builder_fn=FakePromptBuilder,
        history_frames=3,
        history_interval=2,
    )
    output = transform(batch)
    np.testing.assert_array_equal(output["actions"], batch["action"][4:])
    np.testing.assert_array_equal(action_tokenizer.calls[0], batch["action"][5:])
    np.testing.assert_array_equal(action_tokenizer.calls[1], batch["action"][4])
    assert output["pixel_values"][:, 0, 0].tolist() == [0.0, 2.0, 4.0]

    optional_transform = RLDSBatchTransform(
        action_tokenizer=RecordingActionTokenizer(),
        base_tokenizer=FakeTokenizer(),
        image_transform=fake_image_transform,
        prompt_builder_fn=FakePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    optional_output = optional_transform(batch)
    assert optional_output["pixel_values_wrist"][0, 0, 0].item() == 20.0
    np.testing.assert_array_equal(optional_output["proprio"], batch["observation"]["proprio"][0])

    print(
        json.dumps(
            {
                "status": "ok",
                "history_indices": [0, 2, 4],
                "current_action_index": 4,
                "first_future_action_index": 5,
                "optional_inputs_reachable": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
