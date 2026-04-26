"""
model.py — PatchTST Channel-Independent classifier (étape 6 v5.0).

Implementation from-scratch de PatchTST (Yu et al., ICLR 2023) avec head de
classification binaire. Self-contained, dépend uniquement de PyTorch.

Architecture:
  Input (batch, seq_len, n_channels)
    → RevIN per-channel (reversible instance normalization)
    → Patching (n_patches, patch_len) per channel
    → Linear projection patch → d_model
    → Positional embedding learnable
    → Transformer Encoder (shared weights across channels = "channel-independent")
    → Mean pooling over patches per channel
    → Concat across channels
    → MLP head → logit binaire

Référence: "A Time Series is Worth 64 Words" (PatchTST, ICLR 2023).

Voir STATUS_v5.0.md et experiments/patchtst_v5/README.md.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RevIN — Reversible Instance Normalization
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Per-channel instance normalization (Kim et al., ICLR 2022).

    Pour la classification on n'a pas besoin de la dénormalisation (pas de
    reconstruction de la série), donc seul le forward est implémenté.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, seq_len) → normalize per (batch, channel)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


# ---------------------------------------------------------------------------
# PatchTST classifier
# ---------------------------------------------------------------------------

class PatchTSTClassifier(nn.Module):
    """Channel-Independent PatchTST with binary classification head.

    Args:
        n_channels: number of input feature channels (e.g., 24)
        seq_len:    sequence length in bars (default 96)
        patch_len:  patch size in bars (default 12 → 8 patches)
        d_model:    transformer hidden dim (default 128)
        n_heads:    transformer attention heads (default 4)
        n_layers:   transformer encoder layers (default 3)
        dropout:    dropout in transformer + head (default 0.2)
        head_hidden: MLP head hidden dim (default 128)
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int = 96,
        patch_len: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        head_hidden: int = 128,
        use_revin: bool = True,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(n_channels)

        # Patch projection (shared across channels)
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Positional embedding (learnable, shared across channels)
        self.pos_emb = nn.Parameter(torch.empty(self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder (channel-independent: same weights for every channel)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head: flatten (n_channels × d_model) → MLP → logit
        self.head = nn.Sequential(
            nn.LayerNorm(n_channels * d_model),
            nn.Linear(n_channels * d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward jusqu'à l'embedding (avant classification head).

        Returns:
            embedding: (batch, n_channels * d_model) — utilisable pour Triplet Loss / contrastive.
        """
        b, t, c = x.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {t}")
        if c != self.n_channels:
            raise ValueError(f"Expected n_channels={self.n_channels}, got {c}")

        # (batch, n_channels, seq_len)
        x = x.transpose(1, 2)
        if self.use_revin:
            x = self.revin(x)

        # Patch via reshape (no overlap): (batch, n_channels, n_patches, patch_len)
        x = x.reshape(b, c, self.n_patches, self.patch_len)

        # Project: (batch, n_channels, n_patches, d_model)
        x = self.patch_proj(x)

        # Add positional embedding (broadcast across batch and channels)
        x = x + self.pos_emb.unsqueeze(0).unsqueeze(0)

        # Channel-independent transformer: process each channel as a separate sequence
        # Reshape: (batch * n_channels, n_patches, d_model)
        x = x.reshape(b * c, self.n_patches, self.d_model)
        x = self.encoder(x)

        # Mean-pool over patches: (batch * n_channels, d_model)
        x = x.mean(dim=1)

        # Reshape back to per-batch concat over channels: (batch, n_channels * d_model)
        return x.reshape(b, c * self.d_model)

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply classification head on a precomputed embedding."""
        return self.head(embedding).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_channels) float32
        Returns:
            logits: (batch,) — apply sigmoid for probability
        """
        return self.classify(self.encode(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_default_model(n_channels: int, seq_len: int = 96) -> PatchTSTClassifier:
    """Build the default PatchTST CI classifier with sensible defaults for v5.0."""
    return PatchTSTClassifier(
        n_channels=n_channels,
        seq_len=seq_len,
        patch_len=12,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dropout=0.2,
        head_hidden=128,
        use_revin=True,
    )
