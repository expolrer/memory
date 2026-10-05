import inspect

import torch
from omegaconf import OmegaConf

from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3


def pointnet_config(out_channels=64):
    return OmegaConf.create(
        {
            "in_channels": 3,
            "out_channels": out_channels,
            "use_layernorm": True,
            "final_norm": "layernorm",
        }
    )


def test_dp3_encoder_combines_all_rulesafe_conditions():
    encoder = DP3Encoder(
        observation_space={
            "point_cloud": (32, 3),
            "agent_pos": (13,),
            "memory_tokens": (40,),
            "rule_onehot": (20,),
            "plan_tokens": (8,),
            "next_plan_token": (1,),
        },
        out_channel=64,
        state_mlp_size=(32,),
        pointcloud_encoder_cfg=pointnet_config(),
        memory_vocab_size=4,
        memory_output_dim=16,
        rule_output_dim=8,
        plan_vocab_size=7,
        plan_output_dim=8,
        next_plan_vocab_size=7,
        next_plan_output_dim=4,
    )
    observations = {
        "point_cloud": torch.randn(2, 32, 3),
        "agent_pos": torch.randn(2, 13),
        "memory_tokens": torch.randint(0, 4, (2, 40)).float(),
        "rule_onehot": torch.nn.functional.one_hot(
            torch.tensor([0, 19]), num_classes=20
        ).float(),
        "plan_tokens": torch.randint(0, 7, (2, 8)).float(),
        "next_plan_token": torch.randint(0, 7, (2, 1)).float(),
    }

    features = encoder(observations)

    assert encoder.output_shape() == 64 + 32 + 16 + 8 + 8 + 4
    assert features.shape == (2, encoder.output_shape())
    features.square().mean().backward()
    assert encoder.memory_encoder.embedding.weight.grad is not None
    assert encoder.rule_encoder[0].weight.grad is not None
    assert encoder.plan_encoder.embedding.weight.grad is not None
    assert encoder.next_plan_encoder.embedding.weight.grad is not None


def test_simple_dp3_exposes_every_condition_encoder_parameter():
    parameters = inspect.signature(SimpleDP3.__init__).parameters
    expected = {
        "memory_vocab_size",
        "memory_length",
        "memory_embed_dim",
        "memory_output_dim",
        "rule_output_dim",
        "plan_vocab_size",
        "plan_length",
        "plan_embed_dim",
        "plan_output_dim",
        "next_plan_vocab_size",
        "next_plan_embed_dim",
        "next_plan_output_dim",
    }

    assert expected <= parameters.keys()
