import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_bridge import cfg


class RotaryPositionalEncoding(nn.Module):
    """Implements Rotary Positional Encoding (RoPE) for the vision stream."""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Brain(nn.Module):
    """
    The main agent brain model.

    Responsibility boundary:
    - owns the policy/value network architecture
    - does not own training state, rollout buffering, or agent lifecycle
    """

    def __init__(self):
        super().__init__()

        d_model = cfg.BRAIN.D_MODEL
        n_heads = cfg.BRAIN.N_HEADS
        n_layers = cfg.BRAIN.FUSION_LAYERS
        k_queries = cfg.BRAIN.K_QUERIES

        self.use_gru = cfg.BRAIN.USE_GRU
        self.k_features = k_queries * d_model

        decision_input_features = self.k_features
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=self.k_features,
                hidden_size=cfg.BRAIN.GRU_HIDDEN,
                num_layers=1,
                batch_first=False,
            )
            decision_input_features = cfg.BRAIN.GRU_HIDDEN

        self.vision_proj = nn.Linear(5, d_model)
        self.vision_norm = nn.LayerNorm(d_model)
        self.vision_pe = RotaryPositionalEncoding(d_model, max_len=cfg.PERCEPT.NUM_RAYS)

        self.state_proj = nn.Linear(2, d_model)
        self.genome_proj = nn.Linear(4, d_model)
        self.position_proj = nn.Linear(2, d_model)
        self.context_proj = nn.Linear(3, d_model)
        self.context_norm = nn.LayerNorm(d_model)

        self.fusion_layers = nn.ModuleList([FusionLayer(d_model, n_heads) for _ in range(n_layers)])

        self.k_queries = nn.Parameter(torch.randn(k_queries, d_model) * 0.02)

        self.actor = nn.Sequential(
            nn.Linear(decision_input_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 9),
        )
        self.critic = nn.Sequential(
            nn.Linear(decision_input_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: dict):
        batch_size = obs["rays"].shape[0]

        if batch_size == 0:
            device = obs["rays"].device
            return torch.empty(0, 9, device=device), torch.empty(0, 1, device=device)

        rays = self.vision_proj(obs["rays"])
        rays = self.vision_norm(rays)
        rays = self.vision_pe(rays)

        state = self.state_proj(obs["state"]).unsqueeze(1)
        genome = self.genome_proj(obs["genome"]).unsqueeze(1)
        position = self.position_proj(obs["position"]).unsqueeze(1)
        context = self.context_proj(obs["context"]).unsqueeze(1)

        context_tokens = torch.cat([state, genome, position, context], dim=1)
        context_tokens = self.context_norm(context_tokens)

        for layer in self.fusion_layers:
            rays, context_tokens = layer(rays, context_tokens)

        k_queries = self.k_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attn_scores = torch.bmm(k_queries, rays.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, rays)
        pooled = pooled.reshape(batch_size, -1)

        if self.use_gru:
            pooled = pooled.unsqueeze(0)
            pooled, _ = self.gru(pooled)
            pooled = pooled.squeeze(0)

        logits = self.actor(pooled)
        value = self.critic(pooled)
        return logits, value

    @torch.no_grad()
    def get_param_count(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class FusionLayer(nn.Module):
    """A single transformer-style fusion layer."""

    def __init__(self, d_model, n_heads):
        super().__init__()

        self.vision_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.context_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.vision_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.context_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.vision_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.context_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.vision_norm1 = nn.LayerNorm(d_model)
        self.vision_norm2 = nn.LayerNorm(d_model)
        self.vision_norm3 = nn.LayerNorm(d_model)
        self.context_norm1 = nn.LayerNorm(d_model)
        self.context_norm2 = nn.LayerNorm(d_model)
        self.context_norm3 = nn.LayerNorm(d_model)

    def forward(self, rays, context):
        rays = rays + self.vision_self_attn(rays, rays, rays)[0]
        rays = self.vision_norm1(rays)

        context = context + self.context_self_attn(context, context, context)[0]
        context = self.context_norm1(context)

        rays_cross = rays + self.vision_cross_attn(rays, context, context)[0]
        rays = self.vision_norm2(rays_cross)

        context_cross = context + self.context_cross_attn(context, rays, rays)[0]
        context = self.context_norm2(context_cross)

        rays = rays + self.vision_ffn(rays)
        rays = self.vision_norm3(rays)

        context = context + self.context_ffn(context)
        context = self.context_norm3(context)

        return rays, context
