"""
Spatial and Temporal Pooling for 2D Slice-Encoded CT Tokens

Reduces the massive token count from slice-by-slice DINOv3 encoding:
    Raw: num_slices × N_patches_per_slice × 768
    After spatial pool: num_slices × K × 768  (K << N_patches)
    After temporal pool: T × K × 768  (T << num_slices)

Supports:
    Spatial:  mean, max, stride, attention
    Temporal: mean, max, uniform sample, attention-weighted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialPooler(nn.Module):
    """Compress per-slice patch tokens spatially."""

    def __init__(self, dim: int = 768, method: str = "attention",
                 output_tokens: int = 64, num_heads: int = 8):
        super().__init__()
        self.method = method
        self.output_tokens = output_tokens

        if method == "attention":
            self.queries = nn.Parameter(torch.randn(output_tokens, dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, batch_first=True
            )
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_patches, D) per-slice patch tokens
        Returns:
            pooled: (B, output_tokens, D)
        """
        if self.method == "mean":
            # Reshape to grid and pool
            B, N, D = x.shape
            side = int(N ** 0.5)
            stride = max(1, side // int(self.output_tokens ** 0.5))
            x_grid = x.view(B, side, side, D)
            x_pool = F.adaptive_avg_pool2d(
                x_grid.permute(0, 3, 1, 2),
                int(self.output_tokens ** 0.5)
            )
            return x_pool.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :self.output_tokens]

        elif self.method == "max":
            B, N, D = x.shape
            side = int(N ** 0.5)
            x_grid = x.view(B, side, side, D)
            x_pool = F.adaptive_max_pool2d(
                x_grid.permute(0, 3, 1, 2),
                int(self.output_tokens ** 0.5)
            )
            return x_pool.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :self.output_tokens]

        elif self.method == "attention":
            B = x.shape[0]
            q = self.norm_q(self.queries.unsqueeze(0).expand(B, -1, -1))
            kv = self.norm_kv(x)
            out, _ = self.cross_attn(q, kv, kv)
            return out

        elif self.method == "stride":
            B, N, D = x.shape
            stride = max(1, N // self.output_tokens)
            return x[:, ::stride, :][:, :self.output_tokens]


class TemporalPooler(nn.Module):
    """Compress across slices (depth dimension)."""

    def __init__(self, dim: int = 768, method: str = "attention",
                 output_slices: int = 8, num_heads: int = 8):
        super().__init__()
        self.method = method
        self.output_slices = output_slices

        if method == "attention":
            self.queries = nn.Parameter(torch.randn(output_slices, dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, batch_first=True
            )
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
            self.pos_embed = nn.Parameter(torch.randn(1, 512, dim) * 0.02)

    def forward(self, x: torch.Tensor, num_slices: int) -> torch.Tensor:
        """
        Args:
            x: (B, num_slices * tokens_per_slice, D)
            num_slices: number of slices
        Returns:
            pooled: (B, output_slices * tokens_per_slice, D) if attention
                    or (B, output_slices * tokens_per_slice, D) if sampling
        """
        B, total_tokens, D = x.shape
        tokens_per_slice = total_tokens // num_slices

        if self.method == "mean":
            x_slices = x.view(B, num_slices, tokens_per_slice, D)
            # Group slices into output_slices bins
            bin_size = max(1, num_slices // self.output_slices)
            pooled = []
            for i in range(self.output_slices):
                start = i * bin_size
                end = min(start + bin_size, num_slices)
                if start >= num_slices:
                    break
                pooled.append(x_slices[:, start:end].mean(dim=1))
            return torch.cat(pooled, dim=1)

        elif self.method == "uniform":
            x_slices = x.view(B, num_slices, tokens_per_slice, D)
            indices = torch.linspace(0, num_slices - 1, self.output_slices).long()
            selected = x_slices[:, indices]
            return selected.reshape(B, self.output_slices * tokens_per_slice, D)

        elif self.method == "attention":
            # Add positional embeddings for depth ordering
            pos = self.pos_embed[:, :total_tokens, :]
            x_pos = x + pos

            B_q = x.shape[0]
            q = self.norm_q(self.queries.unsqueeze(0).expand(B_q, -1, -1))
            kv = self.norm_kv(x_pos)
            out, _ = self.cross_attn(q, kv, kv)
            return out


class HybridPooler(nn.Module):
    """
    Two-stage hybrid pooling: spatial first, then temporal.

    Raw tokens: (num_slices × N_patches, D)
    After spatial: (num_slices × K_spatial, D)
    After temporal: (T × K_spatial, D) or (num_temporal_queries, D)
    """

    def __init__(self, dim: int = 768, spatial_method: str = "attention",
                 temporal_method: str = "attention",
                 spatial_output: int = 64, temporal_output: int = 8,
                 num_heads: int = 8):
        super().__init__()
        self.spatial = SpatialPooler(dim, spatial_method, spatial_output, num_heads)
        self.temporal = TemporalPooler(dim, temporal_method, temporal_output, num_heads)
        self.spatial_output = spatial_output
        self.temporal_output = temporal_output

    def forward(self, all_slice_tokens: list) -> torch.Tensor:
        """
        Args:
            all_slice_tokens: list of (1, N_patches, D) tensors, one per slice
        Returns:
            pooled: (1, final_tokens, D)
        """
        num_slices = len(all_slice_tokens)

        # Stage 1: Spatial pooling per slice
        spatial_pooled = []
        for tokens in all_slice_tokens:
            pooled = self.spatial(tokens)  # (1, K_spatial, D)
            spatial_pooled.append(pooled)

        # Concatenate across slices: (1, num_slices * K_spatial, D)
        concat = torch.cat(spatial_pooled, dim=1)

        # Stage 2: Temporal pooling across slices
        output = self.temporal(concat, num_slices=num_slices)

        return output
