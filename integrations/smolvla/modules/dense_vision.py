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

"""Dense DINOv2 workspace features for JC-parity SmolVLA.

This branch intentionally receives the same RGB frame as the existing SmolVLM
branch.  It crops a fixed, audited workspace ROI and exposes all 37x37 DINOv2
patch tokens to the action expert through a gated cross-attention residual.

The DINO backbone is frozen, but the projection adapter and cross-attention are
trainable.  A zero-initialized residual gate makes a warm-started dense policy
exactly equivalent to the base policy before the first optimization update.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as T
from torch import Tensor, nn


DINOV2_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(root: Path) -> str:
    """Hash the executable DINO source tree independent of cache files."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"DINOv2 source tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256_file(path)}  {relative}\n".encode())
    return digest.hexdigest()


class DenseDinoVisionEncoder(nn.Module):
    """Frozen DINOv2 backbone plus a trainable patch-token projection.

    Inputs are RGB tensors in ``[0, 1]`` with shape ``(B, 3, H, W)``.  The
    configured ROI uses source-image ``(x1, y1, x2, y2)`` pixel coordinates.
    """

    def __init__(
        self,
        *,
        repository: str,
        repository_sha256: str,
        weights_path: str,
        weights_sha256: str,
        model_type: str,
        roi_xyxy: tuple[int, int, int, int],
        image_size: int,
        patch_size: int,
        output_dim: int,
        adapter_dropout: float,
        freeze_backbone: bool,
        backbone_dtype: str,
        color_jitter: bool,
    ) -> None:
        super().__init__()
        repository_path = Path(repository).expanduser().resolve(strict=True)
        weights = Path(weights_path).expanduser().resolve(strict=True)
        if not repository_path.is_dir() or not (repository_path / "hubconf.py").is_file():
            raise FileNotFoundError(f"invalid local DINOv2 repository: {repository_path}")
        actual_repository_hash = sha256_source_tree(repository_path)
        if actual_repository_hash != repository_sha256.lower():
            raise ValueError(
                "DINOv2 source tree SHA256 mismatch: "
                f"expected {repository_sha256}, got {actual_repository_hash}"
            )
        actual_hash = sha256_file(weights)
        if actual_hash != weights_sha256.lower():
            raise ValueError(
                f"DINOv2 weights SHA256 mismatch: expected {weights_sha256}, got {actual_hash}"
            )
        if model_type not in DINOV2_FEATURE_DIMS:
            raise ValueError(f"unsupported DINOv2 model type: {model_type!r}")

        self.dino = torch.hub.load(
            str(repository_path),
            model_type,
            pretrained=False,
            force_reload=False,
            trust_repo=True,
            source="local",
        )
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        self.dino.load_state_dict(state_dict, strict=True)

        if backbone_dtype != "bfloat16":
            raise ValueError(f"unsupported dense backbone dtype: {backbone_dtype!r}")
        self.backbone_dtype = torch.bfloat16
        self.dino.to(dtype=self.backbone_dtype)

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for parameter in self.dino.parameters():
                parameter.requires_grad = False
        self.dino.eval()

        feature_dim = DINOV2_FEATURE_DIMS[model_type]
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.roi_xyxy = tuple(int(value) for value in roi_xyxy)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.expected_tokens = (self.image_size // self.patch_size) ** 2
        self.color_jitter = (
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            if color_jitter
            else None
        )
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.dino.eval()
        return self

    def _preprocess(self, image_rgb: Tensor, *, training: bool) -> Tensor:
        if image_rgb.ndim != 4 or image_rgb.shape[1] != 3:
            raise ValueError(
                f"dense vision expects BCHW RGB input with 3 channels, got {tuple(image_rgb.shape)}"
            )
        if not torch.is_floating_point(image_rgb):
            raise TypeError("dense vision expects floating-point RGB values in [0,1]")
        if not bool(torch.isfinite(image_rgb).all()):
            raise ValueError("dense vision input contains NaN or Inf")
        # Allow tiny decoder/interpolation overshoot, but fail closed on a wrong
        # normalization or uint8-like input.
        if bool((image_rgb.min() < -1e-4) | (image_rgb.max() > 1.0001)):
            raise ValueError("dense vision RGB input is outside [0,1]")

        _, _, height, width = image_rgb.shape
        x1, y1, x2, y2 = self.roi_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"dense ROI {self.roi_xyxy} is outside input image {(width, height)}"
            )
        workspace = image_rgb[:, :, y1:y2, x1:x2]
        workspace = F.interpolate(
            workspace,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        if training and self.color_jitter is not None:
            workspace = self.color_jitter(workspace)
        return (workspace - self.image_mean) / self.image_std

    def forward(self, image_rgb: Tensor, *, training: bool) -> Tensor:
        workspace = self._preprocess(image_rgb, training=training)
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            dense_tokens = self.dino.get_intermediate_layers(
                workspace.to(dtype=self.backbone_dtype), n=1
            )[0]
        if dense_tokens.ndim != 3 or dense_tokens.shape[1] != self.expected_tokens:
            raise RuntimeError(
                "DINOv2 dense token shape mismatch: "
                f"expected (*,{self.expected_tokens},*), got {tuple(dense_tokens.shape)}"
            )
        adapter_dtype = self.adapter[0].weight.dtype
        return self.adapter(dense_tokens.to(dtype=adapter_dtype))


class GatedDenseCrossAttention(nn.Module):
    """Fuse dense patch tokens into action tokens without changing I/O shape."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("dense cross-attention hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        # A per-channel gate is more expressive than one scalar while retaining
        # exact zero-residual warm-start behavior.
        self.gate = nn.Parameter(torch.full((hidden_dim,), float(gate_init)))

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def prepare_context(self, dense_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Project dense K/V once so ten flow steps can share the result."""
        if dense_tokens.ndim != 3 or dense_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"invalid dense token shape: {tuple(dense_tokens.shape)}")
        context = self.context_norm(dense_tokens)
        return (
            self._split_heads(self.key_proj(context)),
            self._split_heads(self.value_proj(context)),
        )

    def forward(
        self,
        action_tokens: Tensor,
        dense_tokens: Tensor | None = None,
        projected_context: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"invalid action token shape: {tuple(action_tokens.shape)}")
        if projected_context is None:
            if dense_tokens is None:
                raise ValueError("dense tokens or a projected dense context are required")
            dense_tokens = dense_tokens.to(dtype=action_tokens.dtype)
            projected_context = self.prepare_context(dense_tokens)
        key, value = projected_context
        if key.shape[0] != action_tokens.shape[0] or value.shape != key.shape:
            raise ValueError("projected dense context has an invalid batch or shape")
        query = self._split_heads(self.query_proj(self.query_norm(action_tokens)))
        correction = F.scaled_dot_product_attention(
            query,
            key.to(dtype=query.dtype),
            value.to(dtype=query.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        correction = correction.transpose(1, 2).reshape(
            action_tokens.shape[0], action_tokens.shape[1], self.hidden_dim
        )
        correction = self.output_proj(correction)
        return action_tokens + torch.tanh(self.gate).view(1, 1, -1) * correction
