# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Prompt-conditioned fusion encoder for TP-CLAP."""

import torch.nn as nn


class TPCLAPFusionLayer(nn.Module):
    """Pre-norm block: audio self-attention, audio->text cross-attention, FFN."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.norm_self = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

        self.norm_cross = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, query_seq, context_seq, context_mask=None):
        # query_seq   = audio  (B, T_a, D)
        # context_seq = text   (B, T_t, D)
        # context_mask: True = PAD (text only)

        # 1. Self-attention over audio (no mask)
        q_norm = self.norm_self(query_seq)
        attn_self, _ = self.self_attn(
            query=q_norm,
            key=q_norm,
            value=q_norm
        )
        x = query_seq + attn_self

        # 2. Cross-attention (audio -> text), masking padded text positions
        q_norm = self.norm_cross(x)
        attn_cross, _ = self.cross_attn(
            query=q_norm,              # audio
            key=context_seq,           # text
            value=context_seq,
            key_padding_mask=context_mask
        )
        x = x + attn_cross

        # 3. Feed-forward
        x = x + self.ffn(self.norm_ffn(x))

        return x


class TPCLAPFusionEncoder(nn.Module):
    """Stack of fusion layers producing a single prompt-conditioned audio vector."""

    def __init__(self, hidden_size, num_layers=2, num_heads=12):
        super().__init__()

        self.layers = nn.ModuleList([
            TPCLAPFusionLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        audio_features,         # (B, T_a, D)
        prompt_features,        # (B, T_t, D)
        prompt_attention_mask,  # (B, T_t) - 1 for valid, 0 for pad
    ):
        # nn.MultiheadAttention expects True for positions to be masked out
        prompt_attention_mask = (prompt_attention_mask == 0) if prompt_attention_mask is not None else None

        fused_features = audio_features

        for layer in self.layers:
            fused_features = layer(fused_features, prompt_features, prompt_attention_mask)

        fused_features = self.final_norm(fused_features)
        fused_embeds = fused_features.mean(1)

        return fused_embeds


__all__ = ["TPCLAPFusionLayer", "TPCLAPFusionEncoder"]
