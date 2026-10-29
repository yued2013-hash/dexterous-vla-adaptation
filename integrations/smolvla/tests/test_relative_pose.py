#!/usr/bin/env python

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    _ddp_global_mean_from_local_sum,
)
from lerobot.policies.smolvla.relative_pose import (
    CANONICAL_HAND_SOURCE_NAMES,
    JC_CANONICAL_ROTATION6D_IO,
    RELATIVE_POSE_CONVENTION,
    REQUIRED_COMPACT_NAMES,
    absolute_internal_to_compact,
    compact_to_absolute_internal,
    compute_world_pose_delta,
    matrix_to_rotation6d,
    quaternion_wxyz_to_matrix,
    quaternion_wxyz_to_rotation6d,
    reconstruct_absolute_action,
    rotation6d_to_matrix,
    rotation6d_to_quaternion_wxyz,
    validate_canonical_internal,
    validate_compact_feature_names,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

SOURCE_NAMES = (
    "ee_valid",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qw",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "hand_valid",
    "hand_thumb_base",
    "hand_index_base",
    "hand_middle_base",
    "hand_ring_base",
    "hand_pinky_base",
    "hand_thumb_abd",
    "hand_index_abd",
    "hand_middle_abd",
    "hand_ring_abd",
    "hand_pinky_abd",
    "hand_thumb_horiz_abd",
    "hand_thumb_tip",
    "hand_index_tip",
    "hand_middle_tip",
    "hand_ring_tip",
    "hand_pinky_tip",
)


def _random_quaternion(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
    quaternion = torch.randn(*shape, 4, dtype=torch.float64, generator=generator)
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)


def _make_compact(
    xyz: torch.Tensor,
    quaternion: torch.Tensor,
    hand: torch.Tensor,
    names: tuple[str, ...] = SOURCE_NAMES,
) -> torch.Tensor:
    values = torch.zeros(*xyz.shape[:-1], 25, dtype=xyz.dtype)
    values[..., names.index("ee_valid")] = 1
    values[..., names.index("hand_valid")] = 1
    values[..., [names.index(name) for name in ("ee_x", "ee_y", "ee_z")]] = xyz
    values[..., [names.index(name) for name in ("ee_qw", "ee_qx", "ee_qy", "ee_qz")]] = quaternion
    values[..., [names.index(name) for name in CANONICAL_HAND_SOURCE_NAMES]] = hand
    return values


def test_compact_schema_is_exact_and_rejects_drift():
    assert RELATIVE_POSE_CONVENTION == "jc_world_fixed_anchor_rotation6d_v1"
    assert set(SOURCE_NAMES) == REQUIRED_COMPACT_NAMES
    assert validate_compact_feature_names(SOURCE_NAMES) == SOURCE_NAMES
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_compact_feature_names((*SOURCE_NAMES[:-1], "unexpected"))
    with pytest.raises(ValueError, match="unique"):
        validate_compact_feature_names((*SOURCE_NAMES[:-1], SOURCE_NAMES[-2]))


def test_compact_conversion_drops_validity_and_reorders_hand_by_name():
    shuffled_names = tuple(reversed(SOURCE_NAMES))
    values_by_name = {name: float(index + 10) for index, name in enumerate(CANONICAL_HAND_SOURCE_NAMES)}
    values_by_name.update(
        {
            "ee_valid": 1.0,
            "hand_valid": 1.0,
            "ee_x": 1.0,
            "ee_y": 2.0,
            "ee_z": 3.0,
            "ee_qw": 1.0,
            "ee_qx": 0.0,
            "ee_qy": 0.0,
            "ee_qz": 0.0,
        }
    )
    compact = torch.tensor([values_by_name[name] for name in shuffled_names])

    internal = compact_to_absolute_internal(compact, shuffled_names, validate_values=True)

    torch.testing.assert_close(internal[:3], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(internal[3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(
        internal[9:], torch.tensor([values_by_name[name] for name in CANONICAL_HAND_SOURCE_NAMES])
    )


def test_compact_conversion_rejects_invalid_values_when_requested():
    compact = _make_compact(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.zeros(16))
    compact[SOURCE_NAMES.index("hand_valid")] = 0
    with pytest.raises(ValueError, match="invalid hand"):
        compact_to_absolute_internal(compact, SOURCE_NAMES, validate_values=True)


@pytest.mark.parametrize(
    "quaternion",
    (
        torch.zeros(4),
        torch.tensor([5e-9, 0.0, 0.0, 0.0]),
    ),
)
def test_compact_conversion_rejects_quaternion_at_or_below_normalization_epsilon(quaternion):
    compact = _make_compact(torch.zeros(3), quaternion, torch.zeros(16))

    with pytest.raises(ValueError, match="near-zero quaternion"):
        compact_to_absolute_internal(
            compact,
            SOURCE_NAMES,
            validate_values=True,
            eps=1e-8,
        )


def test_compact_conversion_accepts_non_unit_quaternion_with_reliable_direction():
    compact = _make_compact(
        torch.zeros(3),
        torch.tensor([0.9, 0.0, 0.0, 0.0]),
        torch.zeros(16),
    )

    internal = compact_to_absolute_internal(
        compact,
        SOURCE_NAMES,
        validate_values=True,
        eps=1e-8,
    )

    torch.testing.assert_close(internal[3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_compact_conversion_can_enforce_audited_unit_quaternion_tolerance():
    compact = _make_compact(
        torch.zeros(3),
        torch.tensor([0.9, 0.0, 0.0, 0.0]),
        torch.zeros(16),
    )

    with pytest.raises(ValueError, match="non-unit quaternion"):
        compact_to_absolute_internal(
            compact,
            SOURCE_NAMES,
            validate_values=True,
            quaternion_norm_tolerance=1e-5,
        )


def test_canonical_policy_boundary_rejects_degenerate_rotation6d():
    canonical = torch.zeros(25)
    with pytest.raises(ValueError, match="degenerate rotation6D"):
        validate_canonical_internal(canonical)


def test_canonical_policy_boundary_rejects_nonorthonormal_compact_layout_values():
    compact = _make_compact(
        torch.tensor([0.1, -0.2, 0.4]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.zeros(16),
    )

    with pytest.raises(ValueError, match="non-orthonormal rotation6D"):
        validate_canonical_internal(compact)


def test_real_nero_episode_tail_marks_only_current_action_valid(monkeypatch):
    dataset_root = Path(os.environ.get("LEROBOT_NERO_10HZ_DATASET_ROOT", "__dataset_not_configured__"))
    if not (dataset_root / "meta" / "info.json").is_file():
        pytest.skip(f"Nero 10Hz integration dataset is not available at {dataset_root}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    chunk_size = 16
    fps = 10
    dataset = LeRobotDataset(
        repo_id="local/nero-aligned-10hz",
        root=dataset_root,
        episodes=None,
        delta_timestamps={"action": [horizon / fps for horizon in range(chunk_size)]},
        tolerance_s=1e-4,
        revision="local",
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    episode = dataset.meta.episodes[0]
    episode_last_absolute_index = int(episode["dataset_to_index"]) - 1

    # Exercise the real DatasetReader.get_item path without paying for a video
    # decode; video content is irrelevant to the action window and padding mask.
    monkeypatch.setattr(dataset.reader, "_query_videos", lambda *_args, **_kwargs: {})
    sample = dataset.reader.get_item(episode_last_absolute_index)

    expected_padding = torch.tensor([False, *([True] * (chunk_size - 1))])
    torch.testing.assert_close(sample["action_is_pad"], expected_padding)
    assert sample["action"].shape == (chunk_size, 25)
    torch.testing.assert_close(sample["action"], sample["action"][:1].expand_as(sample["action"]))


def test_quaternion_rotation6d_round_trip_is_rotation_equivalent_and_batched():
    generator = torch.Generator().manual_seed(7)
    quaternion = _random_quaternion((2, 3), generator=generator)

    rotation6d = quaternion_wxyz_to_rotation6d(quaternion)
    recovered = rotation6d_to_quaternion_wxyz(rotation6d)

    assert rotation6d.shape == (2, 3, 6)
    torch.testing.assert_close(
        quaternion_wxyz_to_matrix(recovered), quaternion_wxyz_to_matrix(quaternion), atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        quaternion_wxyz_to_rotation6d(torch.tensor([1.0, 0.0, 0.0, 0.0])),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_rotation6d_degenerate_finite_input_projects_to_valid_rotation():
    degenerate = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    matrix = rotation6d_to_matrix(degenerate)
    identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)

    assert torch.isfinite(matrix).all()
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, identity, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(2, dtype=torch.float64))


def test_world_delta_matches_formula_and_round_trips_for_one_anchor_per_chunk():
    generator = torch.Generator().manual_seed(11)
    anchor_quaternion = _random_quaternion((2,), generator=generator)
    target_quaternion = _random_quaternion((2, 5), generator=generator)
    anchor = torch.cat(
        (
            torch.randn(2, 3, dtype=torch.float64, generator=generator),
            quaternion_wxyz_to_rotation6d(anchor_quaternion),
            torch.randn(2, 16, dtype=torch.float64, generator=generator),
        ),
        dim=-1,
    )
    target = torch.cat(
        (
            torch.randn(2, 5, 3, dtype=torch.float64, generator=generator),
            quaternion_wxyz_to_rotation6d(target_quaternion),
            torch.randn(2, 5, 16, dtype=torch.float64, generator=generator),
        ),
        dim=-1,
    )

    delta = compute_world_pose_delta(anchor, target)
    recovered = reconstruct_absolute_action(anchor, delta)
    expected_delta_matrix = quaternion_wxyz_to_matrix(target_quaternion) @ quaternion_wxyz_to_matrix(
        anchor_quaternion
    ).unsqueeze(1).transpose(-1, -2)

    torch.testing.assert_close(delta[..., :3], target[..., :3] - anchor[:, None, :3])
    torch.testing.assert_close(rotation6d_to_matrix(delta[..., 3:9]), expected_delta_matrix)
    torch.testing.assert_close(recovered[..., :3], target[..., :3], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        rotation6d_to_matrix(recovered[..., 3:9]),
        rotation6d_to_matrix(target[..., 3:9]),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(delta[..., 9:], target[..., 9:])
    torch.testing.assert_close(recovered[..., 9:], target[..., 9:])


def test_compact_pack_round_trip_preserves_named_values_and_aligns_quaternion_sign():
    generator = torch.Generator().manual_seed(19)
    quaternion = _random_quaternion((2,), generator=generator)
    compact = _make_compact(
        torch.randn(2, 3, dtype=torch.float64, generator=generator),
        quaternion,
        torch.randn(2, 16, dtype=torch.float64, generator=generator),
    )
    internal = compact_to_absolute_internal(compact, SOURCE_NAMES)
    packed = absolute_internal_to_compact(internal, SOURCE_NAMES, reference_quaternion_wxyz=-quaternion)

    torch.testing.assert_close(packed[..., 1:4], compact[..., 1:4])
    torch.testing.assert_close(
        quaternion_wxyz_to_matrix(packed[..., 4:8]), quaternion_wxyz_to_matrix(compact[..., 4:8])
    )
    assert ((packed[..., 4:8] * -quaternion).sum(dim=-1) >= 0).all()
    torch.testing.assert_close(packed[..., 9:], compact[..., 9:])
    assert (packed[..., [0, 8]] == 1).all()


def test_policy_internal_encode_decode_contract_without_loading_model_weights():
    generator = torch.Generator().manual_seed(23)
    state_quaternion = _random_quaternion((2,), generator=generator).float()
    action_quaternion = _random_quaternion((2, 4), generator=generator).float()
    state = _make_compact(
        torch.randn(2, 3, generator=generator),
        state_quaternion,
        torch.randn(2, 16, generator=generator),
    )
    actions = _make_compact(
        torch.randn(2, 4, 3, generator=generator),
        action_quaternion,
        torch.randn(2, 4, 16, generator=generator),
    )
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        relative_pose_state_feature_names=list(SOURCE_NAMES),
        relative_pose_action_feature_names=list(SOURCE_NAMES),
        relative_pose_validity_threshold=0.5,
        relative_pose_eps=1e-8,
        relative_pose_state_mean=[0.0] * 25,
        relative_pose_state_std=[1.0] * 25,
        relative_pose_action_mean=[0.0] * 25,
        relative_pose_action_std=[1.0] * 25,
    )
    policy.train()

    encoded_state = policy._encode_relative_pose_state(state)
    encoded_actions = policy._encode_relative_pose_actions(actions, state)
    decoded_actions = policy._decode_relative_pose_actions(encoded_actions, state)

    expected_state = compact_to_absolute_internal(state, SOURCE_NAMES)
    expected_actions = compact_to_absolute_internal(actions, SOURCE_NAMES)
    torch.testing.assert_close(encoded_state, expected_state)
    torch.testing.assert_close(decoded_actions[..., 1:4], actions[..., 1:4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        quaternion_wxyz_to_matrix(decoded_actions[..., 4:8]),
        quaternion_wxyz_to_matrix(actions[..., 4:8]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(decoded_actions[..., 9:], actions[..., 9:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(encoded_actions[..., 9:], expected_actions[..., 9:])


def test_jc_limits_normalization_matches_exact_affine_and_constant_dimension_behavior():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    minimum = torch.arange(25, dtype=torch.float32)
    maximum = minimum + 2.0
    maximum[-1] = minimum[-1]
    policy.config = SimpleNamespace(
        relative_pose_normalization_mode="limits",
        relative_pose_state_min=minimum.tolist(),
        relative_pose_state_max=maximum.tolist(),
        relative_pose_action_min=minimum.tolist(),
        relative_pose_action_max=maximum.tolist(),
        relative_pose_normalization_range_eps=1e-4,
    )

    midpoint = (minimum + maximum) / 2.0
    normalized_minimum = policy._normalize_relative_pose(minimum, "state")
    normalized_maximum = policy._normalize_relative_pose(maximum, "state")
    normalized_midpoint = policy._normalize_relative_pose(midpoint, "state")

    torch.testing.assert_close(normalized_minimum[:-1], torch.full((24,), -1.0))
    torch.testing.assert_close(normalized_maximum[:-1], torch.full((24,), 1.0))
    torch.testing.assert_close(normalized_midpoint, torch.zeros(25))
    torch.testing.assert_close(
        policy._unnormalize_relative_pose(normalized_midpoint, "state"),
        midpoint,
    )


def test_training_encoding_strictly_validates_compact_flags_and_quaternion_norm():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        relative_pose_state_feature_names=list(SOURCE_NAMES),
        relative_pose_action_feature_names=list(SOURCE_NAMES),
        relative_pose_normalization_mode="limits",
        relative_pose_state_min=[-1.0] * 25,
        relative_pose_state_max=[1.0] * 25,
        relative_pose_action_min=[-1.0] * 25,
        relative_pose_action_max=[1.0] * 25,
        relative_pose_normalization_range_eps=1e-4,
        relative_pose_validate_training_values=True,
        relative_pose_quaternion_norm_tolerance=1e-5,
        relative_pose_validity_threshold=0.5,
        relative_pose_eps=1e-8,
    )
    policy.train()
    state = _make_compact(
        torch.zeros(3),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.zeros(16),
    ).unsqueeze(0)
    action = state.unsqueeze(1).clone()

    invalid_state = state.clone()
    invalid_state[..., SOURCE_NAMES.index("ee_valid")] = 0
    with pytest.raises(ValueError, match="invalid end-effector"):
        policy._encode_relative_pose_state(invalid_state)

    non_unit_action = action.clone()
    non_unit_action[..., SOURCE_NAMES.index("ee_qw")] = 0.9
    with pytest.raises(ValueError, match="non-unit quaternion"):
        policy._encode_relative_pose_actions(non_unit_action, state)


class _FixedSampleModel:
    def __init__(self, normalized_actions: torch.Tensor):
        self.normalized_actions = normalized_actions
        self.received_state: torch.Tensor | None = None

    def sample_actions(self, _images, _img_masks, _tokens, _masks, state, **_kwargs):
        self.received_state = state.detach().clone()
        return self.normalized_actions.clone()


def test_inference_policy_boundary_exactly_matches_jc_canonical_rotation6d_io():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    horizon = 4
    limits_min = [-2.0] * 25
    limits_max = [2.0] * 25
    policy.config = SimpleNamespace(
        relative_pose_rotation6d_enabled=True,
        relative_pose_inference_io_mode=JC_CANONICAL_ROTATION6D_IO,
        relative_pose_normalization_mode="limits",
        relative_pose_state_min=limits_min,
        relative_pose_state_max=limits_max,
        relative_pose_action_min=limits_min,
        relative_pose_action_max=limits_max,
        relative_pose_normalization_range_eps=1e-4,
        relative_pose_eps=1e-8,
        max_state_dim=25,
        max_action_dim=25,
        action_feature=SimpleNamespace(shape=(25,)),
        relative_xyz_enabled=False,
        adapt_to_pi_aloha=False,
    )
    anchor = torch.zeros(1, 25)
    anchor[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    target = anchor[:, None, :].expand(-1, horizon, -1).clone()
    target[..., :3] = torch.tensor([0.1, -0.2, 0.3])
    target[..., 3:9] = quaternion_wxyz_to_rotation6d(torch.tensor([0.9238795, 0.0, 0.0, 0.3826834]))
    target[..., 9:] = 0.25
    relative_action = compute_world_pose_delta(anchor, target)
    normalized_action = policy._normalize_relative_pose(relative_action, "action")
    fixed_model = _FixedSampleModel(normalized_action)
    policy.model = fixed_model
    policy.prepare_images = lambda _batch: ([], [])
    policy._queues = {}
    policy.eval()

    batch = {
        OBS_STATE: anchor,
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }
    result = policy._get_action_chunk(batch)

    assert result.shape == (1, horizon, 25)
    torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        fixed_model.received_state,
        policy._normalize_relative_pose(anchor, "state"),
    )


class _FixedLossModel:
    def __init__(self, losses: torch.Tensor):
        self.losses = losses

    def forward(self, *_args, **_kwargs):
        return self.losses.clone()


@pytest.mark.parametrize(
    ("mask_padded_actions", "expected_loss"),
    ((True, 1.0), (False, 5.0)),
)
def test_padding_loss_switch_reproduces_jc_supervised_tail_repetition(
    mask_padded_actions,
    expected_loss,
):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    losses = torch.tensor([[[1.0, 1.0], [9.0, 9.0]]])
    policy.model = _FixedLossModel(losses)
    policy.prepare_images = lambda _batch: ([], [])
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        relative_pose_rotation6d_enabled=False,
        relative_xyz_enabled=False,
        max_state_dim=2,
        max_action_dim=2,
        action_feature=SimpleNamespace(shape=(2,)),
        quaternion_geodesic_weight=0.0,
        relative_xyz_anchor_weight=0.0,
        action_loss_groups={"all": [0, 1]},
        action_loss_group_weights={"all": 1.0},
        action_loss_validity_indices=None,
        action_loss_horizon_weights=None,
        mask_padded_actions=mask_padded_actions,
    )
    batch = {
        OBS_STATE: torch.zeros(1, 2),
        ACTION: torch.zeros(1, 2, 2),
        "action_is_pad": torch.tensor([[False, True]]),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }

    loss, _ = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(expected_loss))


def test_delta_rotation_path_has_finite_gradients():
    generator = torch.Generator().manual_seed(31)
    anchor_quaternion = _random_quaternion((3,), generator=generator)
    target_quaternion = _random_quaternion((3,), generator=generator)
    anchor_rotation6d = quaternion_wxyz_to_rotation6d(anchor_quaternion).requires_grad_()
    target_rotation6d = quaternion_wxyz_to_rotation6d(target_quaternion).requires_grad_()
    anchor = torch.cat(
        (torch.zeros(3, 3, dtype=torch.float64), anchor_rotation6d, torch.zeros(3, 16, dtype=torch.float64)),
        dim=-1,
    )
    target = torch.cat(
        (torch.zeros(3, 3, dtype=torch.float64), target_rotation6d, torch.zeros(3, 16, dtype=torch.float64)),
        dim=-1,
    )

    loss = (
        matrix_to_rotation6d(rotation6d_to_matrix(compute_world_pose_delta(anchor, target)[:, 3:9]))
        .square()
        .sum()
    )
    loss.backward()

    assert anchor_rotation6d.grad is not None and torch.isfinite(anchor_rotation6d.grad).all()
    assert target_rotation6d.grad is not None and torch.isfinite(target_rotation6d.grad).all()


def test_global_mean_helper_reduces_to_exact_local_mean_without_ddp():
    values = torch.tensor([1.0, 3.0], requires_grad=True)
    loss = _ddp_global_mean_from_local_sum(values.sum(), torch.tensor(2.0))

    torch.testing.assert_close(loss, torch.tensor(2.0))
    loss.backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 0.5))
