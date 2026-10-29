from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.dense_vision import (
    DenseDinoVisionEncoder,
    GatedDenseCrossAttention,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.relative_pose import (
    JC_CANONICAL_ROTATION6D_IO,
    RELATIVE_POSE_CONVENTION,
    REQUIRED_COMPACT_NAMES,
)


class _FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def load_state_dict(self, _state_dict, strict=True):
        assert strict
        return [], []

    def get_intermediate_layers(self, image, n=1):
        assert n == 1
        pooled = image.mean(dim=(1, 2, 3), keepdim=True).reshape(image.shape[0], 1, 1)
        return [pooled.expand(image.shape[0], 37 * 37, 768) * self.scale]


def _make_encoder(tmp_path, monkeypatch):
    repository = tmp_path / "dinov2"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# pinned test source\n")
    weights = tmp_path / "dinov2_vitb14_pretrain.pth"
    weights.write_bytes(b"pinned test weights")
    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: _FakeDino())
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "lerobot.policies.smolvla.dense_vision.sha256_source_tree", lambda _path: "a" * 64
    )
    monkeypatch.setattr("lerobot.policies.smolvla.dense_vision.sha256_file", lambda _path: "b" * 64)
    return DenseDinoVisionEncoder(
        repository=str(repository),
        repository_sha256="a" * 64,
        weights_path=str(weights),
        weights_sha256="b" * 64,
        model_type="dinov2_vitb14",
        roi_xyxy=(480, 0, 1200, 720),
        image_size=518,
        patch_size=14,
        output_dim=32,
        adapter_dropout=0.0,
        freeze_backbone=True,
        backbone_dtype="bfloat16",
        color_jitter=False,
    )


def test_dense_encoder_uses_same_rgb_frame_and_returns_all_1369_tokens(tmp_path, monkeypatch):
    encoder = _make_encoder(tmp_path, monkeypatch)
    image = torch.rand(2, 3, 720, 1280)

    tokens = encoder(image, training=False)

    assert tokens.shape == (2, 1369, 32)
    assert all(not parameter.requires_grad for parameter in encoder.dino.parameters())
    assert any(parameter.requires_grad for parameter in encoder.adapter.parameters())


def test_dense_encoder_fails_closed_on_wrong_range_or_roi(tmp_path, monkeypatch):
    encoder = _make_encoder(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        encoder(torch.full((1, 3, 720, 1280), 255.0), training=False)
    with pytest.raises(ValueError, match="outside input image"):
        encoder(torch.rand(1, 3, 480, 640), training=False)


def test_zero_gate_preserves_base_action_tokens_exactly_and_then_learns():
    fusion = GatedDenseCrossAttention(
        hidden_dim=32,
        num_heads=4,
        dropout=0.0,
        gate_init=0.0,
    )
    action_tokens = torch.randn(2, 16, 32, requires_grad=True)
    dense_tokens = torch.randn(2, 1369, 32)

    initial = fusion(action_tokens, dense_tokens)
    assert torch.equal(initial, action_tokens)

    initial.square().mean().backward()
    assert fusion.gate.grad is not None
    assert torch.isfinite(fusion.gate.grad).all()

    with torch.no_grad():
        fusion.gate.fill_(0.1)
    fused = fusion(action_tokens.detach(), dense_tokens)
    assert fused.shape == action_tokens.shape
    assert not torch.equal(fused, action_tokens.detach())


def test_policy_extracts_original_frame_without_touching_public_state_action_contract():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        dense_vision_enabled=True,
        dense_vision_image_key="observation.images.head",
        n_obs_steps=1,
    )
    image = torch.rand(2, 1, 3, 720, 1280)

    extracted = policy.prepare_dense_image({"observation.images.head": image})

    assert extracted.data_ptr() == image[:, -1].data_ptr()
    assert extracted.shape == (2, 3, 720, 1280)


def test_dense_path_is_absent_for_existing_smolvla_configs():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace()
    assert policy.prepare_dense_image({}) is None


def _dense_config(**overrides):
    compact_names = sorted(REQUIRED_COMPACT_NAMES)
    kwargs = {
        "device": "cpu",
        "input_features": {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(25,)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        },
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(25,))},
        "normalization_mapping": {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        "chunk_size": 16,
        "n_action_steps": 6,
        "max_state_dim": 25,
        "max_action_dim": 25,
        "relative_pose_rotation6d_enabled": True,
        "relative_pose_convention": RELATIVE_POSE_CONVENTION,
        "relative_pose_inference_io_mode": JC_CANONICAL_ROTATION6D_IO,
        "relative_pose_state_feature_names": compact_names,
        "relative_pose_action_feature_names": compact_names,
        "relative_pose_normalization_mode": "limits",
        "relative_pose_state_min": [-2.0] * 25,
        "relative_pose_state_max": [2.0] * 25,
        "relative_pose_action_min": [-2.0] * 25,
        "relative_pose_action_max": [2.0] * 25,
        "relative_pose_stats_manifest_sha256": "c" * 64,
        "dense_vision_enabled": True,
        "dense_vision_contract": "nero_smolvla_dino_dense_jc_parity_v1",
        "dense_vision_repository": "/pinned/dinov2",
        "dense_vision_repository_sha256": "a" * 64,
        "dense_vision_weights_path": "/pinned/dinov2_vitb14_pretrain.pth",
        "dense_vision_weights_sha256": "b" * 64,
        "dense_vision_backbone_dtype": "bfloat16",
    }
    kwargs.update(overrides)
    return SmolVLAConfig(**kwargs)


def test_dense_config_pins_existing_rotation6d_io_contract():
    config = _dense_config()
    config.validate_features()
    assert config.chunk_size == 16
    assert config.n_action_steps == 6
    assert config.max_action_dim == 25
    assert config.relative_pose_inference_io_mode == JC_CANONICAL_ROTATION6D_IO


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"chunk_size": 17}, "state/action25, chunk16"),
        ({"relative_pose_inference_io_mode": "compact_quaternion"}, "canonical rotation6D"),
        ({"dense_vision_gate_init": 0.01}, "zero-initialized"),
        ({"dense_vision_roi_xyxy": (0, 0, 720, 720)}, "audited workspace ROI"),
    ),
)
def test_dense_config_rejects_contract_drift(override, message):
    with pytest.raises(ValueError, match=message):
        _dense_config(**override)
