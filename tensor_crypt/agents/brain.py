"""Bloodline-aware policy/value networks for Tensor Crypt.

This module owns the canonical policy/value architecture surface. Active UIDs
reach a live brain through slot bindings in the registry, but family topology
and observation-shape contracts are checkpoint-visible invariants that should
not drift silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ..config_bridge import cfg


_CANONICAL_RAY_KEYS = ("canonical_rays", "canonical_self", "canonical_context")
_EXPERIMENTAL_KEYS = ("experimental_rays", "experimental_self", "experimental_context")
_LEGACY_KEYS = ("rays", "state", "genome", "position", "context")


def get_bloodline_families() -> Tuple[str, ...]:
    return tuple(cfg.BRAIN.FAMILY_ORDER)


def validate_bloodline_family(family_id: str) -> str:
    if family_id not in cfg.BRAIN.FAMILY_ORDER:
        raise ValueError(f"Unknown bloodline family: {family_id}")
    return family_id


def get_bloodline_color(family_id: str) -> tuple[int, int, int]:
    validate_bloodline_family(family_id)
    if cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET and family_id == str(cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY):
        return tuple(int(channel) for channel in cfg.BRAIN.EXPERIMENTAL_BRANCH_COLOR)
    return tuple(int(channel) for channel in cfg.BRAIN.FAMILY_COLORS[family_id])


def _build_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _empty_canonical_batch(device: torch.device, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rays = torch.zeros(
        batch_size,
        cfg.PERCEPT.NUM_RAYS,
        cfg.PERCEPT.CANONICAL_RAY_FEATURES,
        device=device,
        dtype=dtype,
    )
    self_features = torch.zeros(
        batch_size,
        cfg.PERCEPT.CANONICAL_SELF_FEATURES,
        device=device,
        dtype=dtype,
    )
    context = torch.zeros(
        batch_size,
        cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES,
        device=device,
        dtype=dtype,
    )
    return rays, self_features, context


def _validate_canonical_observation_tensors(
    canonical_rays: torch.Tensor,
    canonical_self: torch.Tensor,
    canonical_context: torch.Tensor,
) -> None:
    expected_rays_shape = (cfg.PERCEPT.NUM_RAYS, cfg.PERCEPT.CANONICAL_RAY_FEATURES)
    if canonical_rays.dim() != 3 or tuple(canonical_rays.shape[1:]) != expected_rays_shape:
        raise ValueError(
            f"canonical_rays shape mismatch: expected (batch, {expected_rays_shape[0]}, {expected_rays_shape[1]}), got {tuple(canonical_rays.shape)}"
        )
    if canonical_self.dim() != 2 or canonical_self.shape[1] != cfg.PERCEPT.CANONICAL_SELF_FEATURES:
        raise ValueError(
            f"canonical_self shape mismatch: expected (batch, {cfg.PERCEPT.CANONICAL_SELF_FEATURES}), got {tuple(canonical_self.shape)}"
        )
    if canonical_context.dim() != 2 or canonical_context.shape[1] != cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES:
        raise ValueError(
            f"canonical_context shape mismatch: expected (batch, {cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES}), got {tuple(canonical_context.shape)}"
        )

    batch_size = canonical_rays.shape[0]
    if canonical_self.shape[0] != batch_size or canonical_context.shape[0] != batch_size:
        raise ValueError(
            "Canonical observation batch mismatch: canonical_rays, canonical_self, and canonical_context must share the same batch dimension"
        )


def _validate_legacy_observation_tensors(
    rays: torch.Tensor,
    state: torch.Tensor,
    genome: torch.Tensor,
    position: torch.Tensor,
    context: torch.Tensor,
) -> None:
    if rays.dim() != 3:
        raise ValueError(f"Legacy observation rays must be rank 3, got shape {tuple(rays.shape)}")
    for name, tensor in (("state", state), ("genome", genome), ("position", position), ("context", context)):
        if tensor.dim() != 2:
            raise ValueError(f"Legacy observation {name} must be rank 2, got shape {tuple(tensor.shape)}")

    batch_size = rays.shape[0]
    if any(tensor.shape[0] != batch_size for tensor in (state, genome, position, context)):
        raise ValueError("Legacy observation batch mismatch across rays/state/genome/position/context")


def _adapt_legacy_observation_to_canonical(obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    missing = [key for key in _LEGACY_KEYS if key not in obs]
    if missing:
        raise KeyError(f"Observation is missing canonical keys and legacy fallback keys: {missing}")

    rays = obs["rays"]
    state = obs["state"]
    genome = obs["genome"]
    position = obs["position"]
    context = obs["context"]

    _validate_legacy_observation_tensors(rays, state, genome, position, context)

    batch_size = rays.shape[0]
    device = rays.device
    dtype = rays.dtype
    canonical_rays, canonical_self, canonical_context = _empty_canonical_batch(device, batch_size, dtype)

    if rays.shape[-1] != cfg.PERCEPT.LEGACY_RAY_FEATURES:
        raise ValueError(
            f"Legacy observation rays shape mismatch: expected last dim {cfg.PERCEPT.LEGACY_RAY_FEATURES}, got {rays.shape[-1]}"
        )
    if state.shape[-1] != cfg.PERCEPT.LEGACY_STATE_FEATURES:
        raise ValueError(
            f"Legacy observation state shape mismatch: expected last dim {cfg.PERCEPT.LEGACY_STATE_FEATURES}, got {state.shape[-1]}"
        )
    if genome.shape[-1] != cfg.PERCEPT.LEGACY_GENOME_FEATURES:
        raise ValueError(
            f"Legacy observation genome shape mismatch: expected last dim {cfg.PERCEPT.LEGACY_GENOME_FEATURES}, got {genome.shape[-1]}"
        )
    if position.shape[-1] != cfg.PERCEPT.LEGACY_POSITION_FEATURES:
        raise ValueError(
            f"Legacy observation position shape mismatch: expected last dim {cfg.PERCEPT.LEGACY_POSITION_FEATURES}, got {position.shape[-1]}"
        )
    if context.shape[-1] != cfg.PERCEPT.LEGACY_CONTEXT_FEATURES:
        raise ValueError(
            f"Legacy observation context shape mismatch: expected last dim {cfg.PERCEPT.LEGACY_CONTEXT_FEATURES}, got {context.shape[-1]}"
        )

    canonical_rays[..., 1] = rays[..., 0]
    canonical_rays[..., 2] = rays[..., 1]
    canonical_rays[..., 3] = rays[..., 2]
    canonical_rays[..., 4] = rays[..., 4]
    canonical_rays[..., 6] = rays[..., 3]
    canonical_rays[..., 0] = torch.clamp(1.0 - canonical_rays[..., 1] - canonical_rays[..., 2], 0.0, 1.0)

    canonical_self[:, 0] = state[:, 0]
    canonical_self[:, 2:6] = genome
    canonical_self[:, 6:8] = position
    canonical_self[:, 8] = context[:, 0]
    canonical_self[:, 9] = context[:, 1]
    canonical_self[:, 10] = state[:, 1]
    canonical_self[:, 1] = torch.clamp(1.0 - canonical_self[:, 0], 0.0, 1.0)

    canonical_context[:, 0] = context[:, 2]
    return canonical_rays, canonical_self, canonical_context


def extract_canonical_observation(obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if all(key in obs for key in _CANONICAL_RAY_KEYS):
        canonical = (obs["canonical_rays"], obs["canonical_self"], obs["canonical_context"])
    else:
        if not cfg.BRAIN.ALLOW_LEGACY_OBS_FALLBACK:
            raise KeyError("Bloodline MLP brain requires canonical observations")
        canonical = _adapt_legacy_observation_to_canonical(obs)

    _validate_canonical_observation_tensors(*canonical)
    return canonical


def extract_experimental_observation(obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    missing = [key for key in _EXPERIMENTAL_KEYS if key not in obs]
    if missing:
        raise KeyError(f"Experimental bloodline MLP brain requires experimental observations: missing {missing}")

    experimental = (obs["experimental_rays"], obs["experimental_self"], obs["experimental_context"])
    experimental_rays, experimental_self, experimental_context = experimental
    expected_rays_shape = (cfg.PERCEPT.NUM_RAYS, cfg.PERCEPT.EXPERIMENTAL_RAY_FEATURES)
    if experimental_rays.dim() != 3 or tuple(experimental_rays.shape[1:]) != expected_rays_shape:
        raise ValueError(
            f"experimental_rays shape mismatch: expected (batch, {expected_rays_shape[0]}, {expected_rays_shape[1]}), got {tuple(experimental_rays.shape)}"
        )
    if experimental_self.dim() != 2 or experimental_self.shape[1] != cfg.PERCEPT.EXPERIMENTAL_SELF_FEATURES:
        raise ValueError(
            f"experimental_self shape mismatch: expected (batch, {cfg.PERCEPT.EXPERIMENTAL_SELF_FEATURES}), got {tuple(experimental_self.shape)}"
        )
    if experimental_context.dim() != 2 or experimental_context.shape[1] != cfg.PERCEPT.EXPERIMENTAL_CONTEXT_FEATURES:
        raise ValueError(
            f"experimental_context shape mismatch: expected (batch, {cfg.PERCEPT.EXPERIMENTAL_CONTEXT_FEATURES}), got {tuple(experimental_context.shape)}"
        )
    batch_size = experimental_rays.shape[0]
    if experimental_self.shape[0] != batch_size or experimental_context.shape[0] != batch_size:
        raise ValueError(
            "Experimental observation batch mismatch: experimental_rays, experimental_self, and experimental_context must share the same batch dimension"
        )
    return experimental


def extract_observation_for_contract(obs: dict, observation_contract: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contract = str(observation_contract)
    if contract == "canonical_v2":
        return extract_canonical_observation(obs)
    if contract == "experimental_selfcentric_v1":
        return extract_experimental_observation(obs)
    raise ValueError(f"Unsupported observation contract: {observation_contract}")


@dataclass(frozen=True)
class _FamilySpec:
    family_id: str
    hidden_widths: tuple[int, ...]
    activation: str
    normalization: str
    residual: bool
    gated: bool
    split_inputs: bool
    split_ray_width: int
    split_scalar_width: int
    dropout: float
    observation_contract: str


def get_family_spec(family_id: str) -> _FamilySpec:
    family_id = validate_bloodline_family(family_id)
    if cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET and family_id == str(cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY):
        raw = cfg.BRAIN.EXPERIMENTAL_BRANCH_SPEC
    else:
        raw = cfg.BRAIN.FAMILY_SPECS[family_id]
    return _FamilySpec(
        family_id=family_id,
        hidden_widths=tuple(int(v) for v in raw.hidden_widths),
        activation=str(raw.activation),
        normalization=str(raw.normalization),
        residual=bool(raw.residual),
        gated=bool(raw.gated),
        split_inputs=bool(raw.split_inputs),
        split_ray_width=int(raw.split_ray_width),
        split_scalar_width=int(raw.split_scalar_width),
        dropout=float(raw.dropout),
        observation_contract=str(getattr(raw, "observation_contract", "canonical_v2")),
    )


class _ResidualMLPBlock(nn.Module):
    """Width-preserving residual block with optional gating and norm placement."""

    def __init__(self, width: int, *, activation: str, normalization: str, gated: bool, dropout: float):
        super().__init__()
        self.normalization = normalization
        self.norm_in = nn.LayerNorm(width) if normalization == "pre" else nn.Identity()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate = nn.Linear(width, width) if gated else None
        self.norm_out = nn.LayerNorm(width) if normalization == "post" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm_in(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.gate is not None:
            x = x * torch.sigmoid(self.gate(x))
        x = x + residual
        x = self.norm_out(x)
        return x


class _TransitionBlock(nn.Module):
    """Dimension-changing block used at bloodline stage boundaries."""

    def __init__(self, in_width: int, out_width: int, *, activation: str, normalization: str, dropout: float):
        super().__init__()
        self.normalization = normalization
        self.norm_in = nn.LayerNorm(in_width) if normalization == "pre" else nn.Identity()
        self.fc = nn.Linear(in_width, out_width)
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm_out = nn.LayerNorm(out_width) if normalization == "post" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(x)
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm_out(x)
        return x


class Brain(nn.Module):
    """
    Canonical bloodline-aware MLP brain.

    Invariants:
    - every instance belongs to exactly one bloodline family
    - within a family, parameter topology is fully shape-identical
    - forward always returns `(logits, value)` on the canonical observation contract
    """

    def __init__(self, family_id: str | None = None):
        super().__init__()
        self.family_id = validate_bloodline_family(family_id or cfg.BRAIN.DEFAULT_FAMILY)
        self.spec = get_family_spec(self.family_id)

        if self.spec.observation_contract == "canonical_v2":
            ray_features = cfg.PERCEPT.CANONICAL_RAY_FEATURES
            self_features = cfg.PERCEPT.CANONICAL_SELF_FEATURES
            context_features = cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES
        elif self.spec.observation_contract == "experimental_selfcentric_v1":
            ray_features = cfg.PERCEPT.EXPERIMENTAL_RAY_FEATURES
            self_features = cfg.PERCEPT.EXPERIMENTAL_SELF_FEATURES
            context_features = cfg.PERCEPT.EXPERIMENTAL_CONTEXT_FEATURES
        else:
            raise ValueError(f"Unsupported observation contract for family {self.family_id}: {self.spec.observation_contract}")

        ray_dim = cfg.PERCEPT.NUM_RAYS * ray_features
        scalar_dim = self_features + context_features
        self.input_dim = ray_dim + scalar_dim

        if self.spec.split_inputs:
            if self.spec.split_ray_width <= 0 or self.spec.split_scalar_width <= 0:
                raise ValueError(f"{self.family_id} requires positive split input widths")
            self.ray_proj = nn.Linear(ray_dim, self.spec.split_ray_width)
            self.scalar_proj = nn.Linear(scalar_dim, self.spec.split_scalar_width)
            mix_width = self.spec.split_ray_width + self.spec.split_scalar_width
            self.input_gate = nn.Linear(mix_width, mix_width) if self.spec.gated else None
            self.input_proj = nn.Linear(mix_width, self.spec.hidden_widths[0])
        else:
            self.ray_proj = None
            self.scalar_proj = None
            self.input_gate = None
            self.input_proj = nn.Linear(self.input_dim, self.spec.hidden_widths[0])

        blocks: list[nn.Module] = []
        width_iter = list(self.spec.hidden_widths)
        for in_width, out_width in zip(width_iter, width_iter[1:]):
            if self.spec.residual and in_width == out_width:
                blocks.append(
                    _ResidualMLPBlock(
                        out_width,
                        activation=self.spec.activation,
                        normalization=self.spec.normalization,
                        gated=self.spec.gated,
                        dropout=self.spec.dropout,
                    )
                )
            else:
                blocks.append(
                    _TransitionBlock(
                        in_width,
                        out_width,
                        activation=self.spec.activation,
                        normalization=self.spec.normalization,
                        dropout=self.spec.dropout,
                    )
                )

        self.trunk = nn.ModuleList(blocks)
        final_width = self.spec.hidden_widths[-1]
        self.head_norm = nn.LayerNorm(final_width)
        self.actor = nn.Sequential(
            nn.Linear(final_width, final_width),
            _build_activation(self.spec.activation),
            nn.Linear(final_width, cfg.BRAIN.ACTION_DIM),
        )
        self.critic = nn.Sequential(
            nn.Linear(final_width, final_width),
            _build_activation(self.spec.activation),
            nn.Linear(final_width, cfg.BRAIN.VALUE_DIM),
        )

    def _encode_inputs(self, canonical_rays: torch.Tensor, canonical_self: torch.Tensor, canonical_context: torch.Tensor) -> torch.Tensor:
        batch_size = canonical_rays.shape[0]
        if batch_size == 0:
            return torch.empty(0, self.spec.hidden_widths[0], device=canonical_rays.device, dtype=canonical_rays.dtype)

        ray_flat = canonical_rays.reshape(batch_size, -1)
        scalar_flat = torch.cat([canonical_self, canonical_context], dim=1)

        if self.spec.split_inputs:
            mixed = torch.cat([self.ray_proj(ray_flat), self.scalar_proj(scalar_flat)], dim=1)
            if self.input_gate is not None:
                mixed = mixed * torch.sigmoid(self.input_gate(mixed))
            return self.input_proj(mixed)

        flat = torch.cat([ray_flat, scalar_flat], dim=1)
        return self.input_proj(flat)

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_rays, input_self, input_context = extract_observation_for_contract(obs, self.spec.observation_contract)
        batch_size = input_rays.shape[0]
        if batch_size == 0:
            device = input_rays.device
            return (
                torch.empty(0, cfg.BRAIN.ACTION_DIM, device=device),
                torch.empty(0, cfg.BRAIN.VALUE_DIM, device=device),
            )

        x = self._encode_inputs(input_rays, input_self, input_context)
        for block in self.trunk:
            x = block(x)
        x = self.head_norm(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    @torch.no_grad()
    def get_param_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @torch.no_grad()
    def get_topology_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((name, tuple(param.shape)) for name, param in self.named_parameters())

    @torch.no_grad()
    def describe_family(self) -> dict:
        return {
            "family_id": self.family_id,
            "hidden_widths": list(self.spec.hidden_widths),
            "activation": self.spec.activation,
            "normalization": self.spec.normalization,
            "residual": self.spec.residual,
            "gated": self.spec.gated,
            "split_inputs": self.spec.split_inputs,
            "observation_contract": self.spec.observation_contract,
        }


def create_brain(family_id: str | None = None) -> Brain:
    return Brain(family_id=family_id)
