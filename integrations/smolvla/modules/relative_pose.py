#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nero compact-state and world-relative rotation6D transforms for SmolVLA.

The training-dataset boundary uses the named compact 25-D layout::

    ee_valid + xyz3 + quaternion_wxyz4 + hand_valid + hand16

The policy operates internally on JC's 25-D canonical layout::

    xyz3 + rotation6D6 + hand16

JC-compatible online inference exposes that canonical layout directly for
both observations and absolute actions.  A separately configured legacy mode
retains the compact-quaternion online boundary for older SmolVLA callers.

Actions use the same width, but their first nine values are relative to the
current observation in the world frame.  Every point in an action chunk uses
the same current-observation anchor (there is no autoregressive integration).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

COMPACT_DIM = 25
INTERNAL_DIM = 25
POSE_DIM = 9
RELATIVE_POSE_CONVENTION = "jc_world_fixed_anchor_rotation6d_v1"
COMPACT_QUATERNION_IO = "compact_quaternion"
JC_CANONICAL_ROTATION6D_IO = "jc_canonical_rotation6d"
RELATIVE_POSE_INFERENCE_IO_MODES = frozenset({COMPACT_QUATERNION_IO, JC_CANONICAL_ROTATION6D_IO})
CANONICAL_ROTATION6D_ORTHONORMALITY_TOLERANCE = 1e-4

XYZ_NAMES = ("ee_x", "ee_y", "ee_z")
QUATERNION_WXYZ_NAMES = ("ee_qw", "ee_qx", "ee_qy", "ee_qz")

# Canonical per-finger order used by JC's DexGraspVLA model and controller.
CANONICAL_HAND_SOURCE_NAMES = (
    "hand_thumb_base",
    "hand_thumb_abd",
    "hand_thumb_horiz_abd",
    "hand_thumb_tip",
    "hand_index_abd",
    "hand_index_base",
    "hand_index_tip",
    "hand_middle_abd",
    "hand_middle_base",
    "hand_middle_tip",
    "hand_ring_abd",
    "hand_ring_base",
    "hand_ring_tip",
    "hand_pinky_abd",
    "hand_pinky_base",
    "hand_pinky_tip",
)

REQUIRED_COMPACT_NAMES = {
    "ee_valid",
    *XYZ_NAMES,
    *QUATERNION_WXYZ_NAMES,
    "hand_valid",
    *CANONICAL_HAND_SOURCE_NAMES,
}


def validate_compact_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze a named compact Nero feature schema."""
    names = tuple(feature_names)
    if len(names) != COMPACT_DIM:
        raise ValueError(f"Expected {COMPACT_DIM} compact feature names, got {len(names)}.")
    if len(set(names)) != len(names):
        raise ValueError("Compact feature names must be unique.")
    missing = REQUIRED_COMPACT_NAMES - set(names)
    extra = set(names) - REQUIRED_COMPACT_NAMES
    if missing or extra:
        raise ValueError(
            f"Compact feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return names


def _indices(feature_names: Sequence[str], selected_names: Sequence[str], device: torch.device) -> Tensor:
    positions = {name: index for index, name in enumerate(feature_names)}
    return torch.tensor([positions[name] for name in selected_names], dtype=torch.long, device=device)


def quaternion_wxyz_to_matrix(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` matrices."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {tuple(quaternion.shape)}.")

    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(eps)
    w, x, y, z = quaternion.unbind(dim=-1)
    two_s = 2.0
    return torch.stack(
        (
            1.0 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1.0 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1.0 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def matrix_to_rotation6d(matrix: Tensor) -> Tensor:
    """Return the first two matrix columns, concatenated column by column."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix shape (..., 3, 3), got {tuple(matrix.shape)}.")
    return matrix[..., :, :2].transpose(-2, -1).flatten(start_dim=-2)


def quaternion_wxyz_to_rotation6d(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    return matrix_to_rotation6d(quaternion_wxyz_to_matrix(quaternion, eps=eps))


def rotation6d_to_matrix(rotation6d: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert rotation6D to a matrix with guarded Gram-Schmidt projection.

    Finite degenerate predictions are mapped to a valid rotation by choosing a
    deterministic orthogonal fallback. Non-finite predictions stay non-finite
    and are rejected by the inference decoder before an action is returned.
    """
    if rotation6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation6D shape (..., 6), got {tuple(rotation6d.shape)}.")

    a1 = rotation6d[..., :3]
    a2 = rotation6d[..., 3:6]
    a1_norm = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    default_b1 = torch.zeros_like(a1)
    default_b1[..., 0] = 1.0
    b1 = torch.where(a1_norm > eps, a1 / a1_norm.clamp_min(eps), default_b1)

    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    a2_norm = torch.linalg.vector_norm(a2_orthogonal, dim=-1, keepdim=True)

    # Select the Cartesian axis least aligned with b1, then project it into
    # b1's orthogonal plane. This handles parallel/zero rotation6D columns.
    fallback_axis_index = b1.abs().argmin(dim=-1)
    fallback_axis = torch.nn.functional.one_hot(fallback_axis_index, num_classes=3).to(dtype=rotation6d.dtype)
    fallback_b2 = fallback_axis - (fallback_axis * b1).sum(dim=-1, keepdim=True) * b1
    fallback_b2 = fallback_b2 / torch.linalg.vector_norm(fallback_b2, dim=-1, keepdim=True).clamp_min(eps)
    b2 = torch.where(a2_norm > eps, a2_orthogonal / a2_norm.clamp_min(eps), fallback_b2)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _sqrt_positive_part(value: Tensor) -> Tensor:
    return torch.sqrt(torch.clamp(value, min=0.0))


def matrix_to_quaternion_wxyz(matrix: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert rotation matrices to normalized wxyz quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix shape (..., 3, 3), got {tuple(matrix.shape)}.")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(eps))
    selection = torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4).bool()
    quaternion = candidates[selection].reshape(*matrix.shape[:-2], 4)
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(eps)


def rotation6d_to_quaternion_wxyz(rotation6d: Tensor, eps: float = 1e-8) -> Tensor:
    return matrix_to_quaternion_wxyz(rotation6d_to_matrix(rotation6d, eps=eps), eps=eps)


def compact_to_absolute_internal(
    compact: Tensor,
    feature_names: Sequence[str],
    *,
    validate_values: bool = False,
    validity_threshold: float = 0.5,
    eps: float = 1e-8,
    quaternion_norm_tolerance: float | None = None,
) -> Tensor:
    """Convert named compact values to canonical absolute xyz+rot6d+hand."""
    names = validate_compact_feature_names(feature_names)
    if compact.shape[-1] != len(names):
        raise ValueError(f"Compact tensor dimension {compact.shape[-1]} does not match {len(names)} names.")
    if validate_values:
        ee_valid = compact[..., names.index("ee_valid")]
        hand_valid = compact[..., names.index("hand_valid")]
        if not bool(torch.isfinite(compact).all()):
            raise ValueError("Compact tensor contains non-finite values.")
        if not bool((ee_valid > validity_threshold).all()):
            raise ValueError("Compact tensor contains invalid end-effector poses.")
        if not bool((hand_valid > validity_threshold).all()):
            raise ValueError("Compact tensor contains invalid hand values.")
        quaternion = torch.index_select(compact, -1, _indices(names, QUATERNION_WXYZ_NAMES, compact.device))
        quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
        if not bool((quaternion_norm > eps).all()):
            raise ValueError(f"Compact tensor contains a near-zero quaternion with norm <= {eps}.")
        if quaternion_norm_tolerance is not None and not bool(
            ((quaternion_norm - 1.0).abs() <= quaternion_norm_tolerance).all()
        ):
            raise ValueError(
                "Compact tensor contains a non-unit quaternion outside tolerance "
                f"{quaternion_norm_tolerance}."
            )

    xyz = torch.index_select(compact, -1, _indices(names, XYZ_NAMES, compact.device))
    quaternion = torch.index_select(compact, -1, _indices(names, QUATERNION_WXYZ_NAMES, compact.device))
    rotation6d = quaternion_wxyz_to_rotation6d(quaternion, eps=eps)
    hand = torch.index_select(compact, -1, _indices(names, CANONICAL_HAND_SOURCE_NAMES, compact.device))
    return torch.cat((xyz, rotation6d, hand), dim=-1)


def validate_canonical_internal(
    values: Tensor,
    *,
    eps: float = 1e-8,
    orthonormality_tolerance: float = CANONICAL_ROTATION6D_ORTHONORMALITY_TOLERANCE,
    label: str = "Canonical rotation6D tensor",
) -> None:
    """Fail closed on malformed JC online-policy state/action values.

    JC's online boundary is ``xyz3 + rotation6D6 + hand16``. Merely checking
    its width is unsafe because the compact-quaternion parquet layout is also
    25-D. Canonical absolute states/actions come from rotation matrices, so
    their two stored columns must be unit length and mutually orthogonal. This
    validation rejects malformed values before they can be silently projected
    to an unrelated rotation. (No value-only check can distinguish every
    mathematically coincident 25-D layout; the configured I/O mode remains the
    authoritative interface tag.)
    """
    if values.shape[-1] != INTERNAL_DIM:
        raise ValueError(f"{label} must have final dimension {INTERNAL_DIM}, got {values.shape[-1]}.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{label} contains non-finite values.")

    first_column = values[..., 3:6]
    second_column = values[..., 6:9]
    first_norm = torch.linalg.vector_norm(first_column, dim=-1, keepdim=True)
    second_norm = torch.linalg.vector_norm(second_column, dim=-1)
    first_unit = first_column / first_norm.clamp_min(eps)
    second_orthogonal = second_column - (first_unit * second_column).sum(dim=-1, keepdim=True) * first_unit
    second_orthogonal_norm = torch.linalg.vector_norm(second_orthogonal, dim=-1)
    if not bool(((first_norm.squeeze(-1) > eps) & (second_orthogonal_norm > eps)).all()):
        raise ValueError(f"{label} contains a degenerate rotation6D value.")

    first_norm = first_norm.squeeze(-1)
    column_dot = (first_column * second_column).sum(dim=-1)
    orthonormal = (
        ((first_norm - 1.0).abs() <= orthonormality_tolerance)
        & ((second_norm - 1.0).abs() <= orthonormality_tolerance)
        & (column_dot.abs() <= orthonormality_tolerance)
    )
    if not bool(orthonormal.all()):
        raise ValueError(
            f"{label} contains a non-orthonormal rotation6D value outside tolerance "
            f"{orthonormality_tolerance}."
        )


def compact_quaternion_wxyz(compact: Tensor, feature_names: Sequence[str]) -> Tensor:
    names = validate_compact_feature_names(feature_names)
    return torch.index_select(compact, -1, _indices(names, QUATERNION_WXYZ_NAMES, compact.device))


def absolute_internal_to_compact(
    absolute_internal: Tensor,
    feature_names: Sequence[str],
    *,
    reference_quaternion_wxyz: Tensor | None = None,
    validity_value: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Pack canonical absolute xyz+rot6d+hand into a named compact tensor."""
    names = validate_compact_feature_names(feature_names)
    if absolute_internal.shape[-1] != INTERNAL_DIM:
        raise ValueError(f"Expected internal dimension {INTERNAL_DIM}, got {absolute_internal.shape[-1]}.")

    quaternion = rotation6d_to_quaternion_wxyz(absolute_internal[..., 3:9], eps=eps)
    if reference_quaternion_wxyz is not None:
        reference = reference_quaternion_wxyz
        while reference.ndim < quaternion.ndim:
            reference = reference.unsqueeze(-2)
        sign = torch.where(
            (quaternion * reference).sum(dim=-1, keepdim=True) < 0,
            -torch.ones((), dtype=quaternion.dtype, device=quaternion.device),
            torch.ones((), dtype=quaternion.dtype, device=quaternion.device),
        )
        quaternion = quaternion * sign

    compact = torch.zeros(
        *absolute_internal.shape[:-1],
        len(names),
        dtype=absolute_internal.dtype,
        device=absolute_internal.device,
    )
    compact[..., names.index("ee_valid")] = validity_value
    compact[..., names.index("hand_valid")] = validity_value
    compact[..., _indices(names, XYZ_NAMES, compact.device)] = absolute_internal[..., :3]
    compact[..., _indices(names, QUATERNION_WXYZ_NAMES, compact.device)] = quaternion
    compact[..., _indices(names, CANONICAL_HAND_SOURCE_NAMES, compact.device)] = absolute_internal[..., 9:]
    return compact


def compute_world_pose_delta(anchor_absolute: Tensor, target_absolute: Tensor) -> Tensor:
    """Encode targets relative to one anchor using JC's world-frame convention."""
    if anchor_absolute.shape[-1] != INTERNAL_DIM or target_absolute.shape[-1] != INTERNAL_DIM:
        raise ValueError("Anchor and target must both use the canonical 25-D internal layout.")
    anchor = anchor_absolute
    while anchor.ndim < target_absolute.ndim:
        anchor = anchor.unsqueeze(-2)

    xyz_delta = target_absolute[..., :3] - anchor[..., :3]
    anchor_matrix = rotation6d_to_matrix(anchor[..., 3:9])
    target_matrix = rotation6d_to_matrix(target_absolute[..., 3:9])
    delta_matrix = target_matrix @ anchor_matrix.transpose(-1, -2)
    delta_rotation6d = matrix_to_rotation6d(delta_matrix)
    return torch.cat((xyz_delta, delta_rotation6d, target_absolute[..., 9:]), dim=-1)


def reconstruct_absolute_action(anchor_absolute: Tensor, relative_action: Tensor) -> Tensor:
    """Invert :func:`compute_world_pose_delta`; hand values stay absolute."""
    if anchor_absolute.shape[-1] != INTERNAL_DIM or relative_action.shape[-1] != INTERNAL_DIM:
        raise ValueError("Anchor and action must both use the canonical 25-D internal layout.")
    anchor = anchor_absolute
    while anchor.ndim < relative_action.ndim:
        anchor = anchor.unsqueeze(-2)

    target_xyz = anchor[..., :3] + relative_action[..., :3]
    anchor_matrix = rotation6d_to_matrix(anchor[..., 3:9])
    delta_matrix = rotation6d_to_matrix(relative_action[..., 3:9])
    target_matrix = delta_matrix @ anchor_matrix
    target_rotation6d = matrix_to_rotation6d(target_matrix)
    return torch.cat((target_xyz, target_rotation6d, relative_action[..., 9:]), dim=-1)
