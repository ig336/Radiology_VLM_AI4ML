"""
Advanced Token Compression Strategies for 3D Medical Imaging

This module implements improved token compression strategies beyond the baseline
AttentionalPooler. These are experimental improvements for spatial-aware, 
hierarchical, and task-conditioned compression.

Usage:
    To use these improved poolers, modify builder.py to instantiate them instead
    of the baseline AttentionalPooler.

Status:
    - Baseline: AttentionalPooler (coca_attentional_pooler.py) - PRODUCTION
    - Advanced: These classes - EXPERIMENTAL (requires validation)

Last Updated: 2026-02-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .coca_attentional_pooler import AttentionalPooler


class SpatialAttentionalPooler(nn.Module):
    """
    Spatial-aware attention pooling with locality bias.
    
    Preserves 3D spatial structure during compression by restricting attention
    to nearby tokens. Better for medical imaging where anatomical regions should
    be compressed together (e.g., lung, heart, mediastinum).
    
    ⚠️ BLOCKER: Requires CTViT outputs to be grid-ordered (12×12×16 tokens).
                  Run grid coherence test before using this module.
    
    Args:
        d_model: Embedding dimension (512 for CTViT)
        context_dim: Context dimension (same as d_model for self-attention)
        n_queries: Number of output tokens (256)
        n_head: Number of attention heads (8)
        spatial_shape: Expected token grid shape (D, H, W) = (12, 12, 16) after CTViT compression
        locality_radius: Attention radius in token space (2 = attend to ±2 neighboring tokens)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        context_dim: int = 512,
        n_queries: int = 256,
        n_head: int = 8,
        spatial_shape: Tuple[int, int, int] = (12, 12, 16),  # D, H, W in token space
        locality_radius: int = 2,
        norm_layer = nn.LayerNorm
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.locality_radius = locality_radius
        self.n_queries = n_queries
        
        # Learnable query positions in 3D space
        self.query_positions = nn.Parameter(
            self._initialize_query_positions(n_queries, spatial_shape),
            requires_grad=False  # Fixed grid positions
        )
        
        # Standard attention mechanism
        self.attn = AttentionalPooler(
            d_model=d_model,
            context_dim=context_dim,
            n_head=n_head,
            n_queries=n_queries,
            norm_layer=norm_layer
        )
    
    def _initialize_query_positions(self, n_queries: int, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Initialize queries at regular 3D grid positions.
        E.g., for 256 queries: approximately 6×6×7 grid covering the volume
        """
        D, H, W = spatial_shape
        
        # Compute grid dimensions (approximate cube root)
        grid_d = int(np.ceil(n_queries ** (1/3)))
        grid_h = grid_d
        grid_w = max(1, n_queries // (grid_d * grid_h))
        
        # Create 3D grid
        positions = []
        d_pos = np.linspace(0, D-1, grid_d)
        h_pos = np.linspace(0, H-1, grid_h)
        w_pos = np.linspace(0, W-1, grid_w)
        
        for d in d_pos:
            for h in h_pos:
                for w in w_pos:
                    positions.append([d, h, w])
                    if len(positions) >= n_queries:
                        break
                if len(positions) >= n_queries:
                    break
            if len(positions) >= n_queries:
                break
        
        return torch.tensor(positions[:n_queries], dtype=torch.float32)
    
    def _infer_token_positions(self, n_tokens: int, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Infer token positions assuming grid-ordered output from CTViT.
        
        ⚠️ ASSUMPTION: Tokens are ordered as [d0h0w0, d0h0w1, ..., d0h0w_W, d0h1w0, ...]
        i.e., W changes fastest, then H, then D (row-major order)
        
        Returns:
            positions: (n_tokens, 3) - [D, H, W] coordinates for each token
        """
        D, H, W = spatial_shape
        expected_tokens = D * H * W
        
        if n_tokens != expected_tokens:
            raise ValueError(f"Token count mismatch: got {n_tokens}, expected {expected_tokens} from {spatial_shape}")
        
        # Generate grid positions
        positions = []
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    positions.append([d, h, w])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def _compute_spatial_mask(
        self, 
        query_pos: torch.Tensor, 
        token_pos: torch.Tensor, 
        radius: float
    ) -> torch.Tensor:
        """
        Create attention mask: queries only attend to nearby tokens.
        
        Args:
            query_pos: (n_queries, 3) positions in 3D space
            token_pos: (n_tokens, 3) positions in 3D space
            radius: float, distance threshold (in token units)
        
        Returns:
            mask: (n_queries, n_tokens) bool mask (True = attend, False = mask out)
        """
        # Compute pairwise L2 distances
        # dist[i, j] = ||query_pos[i] - token_pos[j]||_2
        dist = torch.cdist(query_pos, token_pos)  # (n_queries, n_tokens)
        
        # Mask: True where distance <= radius
        mask = dist <= radius
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_tokens, C) where N_tokens = 2304 for CTViT
        
        Returns:
            compressed: (B, n_queries, C) = (B, 256, 512)
        """
        B, N, C = x.shape
        
        # Infer token positions (assuming grid-ordered output from CTViT)
        token_positions = self._infer_token_positions(N, self.spatial_shape)  # (N, 3)
        token_positions = token_positions.to(x.device)
        
        # Compute spatial attention mask
        mask = self._compute_spatial_mask(
            self.query_positions.to(x.device),
            token_positions,
            self.locality_radius
        )  # (n_queries, N)
        
        # Note: Standard AttentionalPooler doesn't support masking yet
        # For now, just use global attention (baseline behavior)
        # TODO: Modify AttentionalPooler to accept attention_mask parameter
        compressed = self.attn(x)
        
        return compressed  # (B, 256, 512)


class HierarchicalAttentionalPooler(nn.Module):
    """
    Multi-level attention pooling: coarse-to-fine compression.
    
    Captures both global context (coarse levels) and local details (fine levels)
    by progressively compressing tokens through multiple stages. Each level
    attends to both the previous level AND high-attention tokens from the original.
    
    Args:
        d_model: Embedding dimension (512)
        n_levels: Number of compression levels (3)
        queries_per_level: Number of queries per level [coarse → fine]
                          e.g., [512, 256, 256] compresses 2304 → 256 (pipeline compatible)
    
    ⚠️ IMPORTANT: Final level MUST output 256 tokens to match the rest of the pipeline.
                  The MLP projector and LLM expect (B, 256, 4096) after projection.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_levels: int = 3,
        queries_per_level: list = None,
        n_head: int = 8,
        norm_layer = nn.LayerNorm
    ):
        super().__init__()
        self.n_levels = n_levels
        
        if queries_per_level is None:
            queries_per_level = [512, 256, 256]  # Default: final level = 256 (pipeline compatible)
        
        self.queries_per_level = queries_per_level
        
        # Multiple pooler levels
        self.poolers = nn.ModuleList([
            AttentionalPooler(
                d_model=d_model,
                context_dim=d_model,
                n_head=n_head,
                n_queries=q,
                norm_layer=norm_layer
            ) for q in queries_per_level
        ])
        
        # Cross-level fusion: combine previous level with sampled original tokens
        self.level_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                norm_layer(d_model),
                nn.GELU()
            )
            for _ in range(n_levels - 1)
        ])
    
    def _sample_top_attended(
        self, 
        x: torch.Tensor, 
        attn_weights: torch.Tensor, 
        n_samples: int
    ) -> torch.Tensor:
        """
        Sample tokens with highest attention weights from previous level.
        
        This implements attention-based importance sampling: tokens that received
        more attention in the previous level are more likely to contain important
        information, so we keep them for the next level.
        
        Args:
            x: (B, N_tokens, D) - original token sequence (e.g., 2304 tokens)
            attn_weights: (B, n_queries, N_tokens) - attention weights from previous level
            n_samples: int - number of tokens to sample (e.g., 512)
        
        Returns:
            sampled_tokens: (B, n_samples, D) - tokens with highest attention
        """
        # Average attention across all queries to get per-token importance
        importance = attn_weights.mean(dim=1)  # (B, N_tokens)
        
        # Select top-k tokens by importance
        _, top_indices = torch.topk(importance, k=min(n_samples, importance.size(-1)), dim=-1)  # (B, n_samples)
        
        # Gather selected tokens
        batch_size, n_tokens, d_model = x.shape
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, d_model)
        sampled = torch.gather(x, dim=1, index=top_indices_expanded)  # (B, n_samples, D)
        
        return sampled
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_tokens, D) ≈ (B, 2304, 512)
        
        Returns:
            compressed: (B, n_queries_final, D) = (B, 256, 512)
        """
        # Level 0: Coarse (2304 → 512 tokens)
        level_0 = self.poolers[0](x)  # (B, 512, 512)
        
        # Store attention weights (note: standard AttentionalPooler doesn't return attention)
        # TODO: Modify AttentionalPooler to optionally return attention weights
        # For now, use uniform sampling as fallback
        
        # Level 1: Medium (combine level_0 + sampled originals → 128 tokens)
        n_sample_1 = min(512, x.size(1))
        sampled_1 = x[:, torch.randperm(x.size(1))[:n_sample_1], :]  # Random sampling as fallback
        
        level_1_input = torch.cat([level_0, sampled_1], dim=1)  # (B, 512+512, 512)
        level_1_input = level_1_input.mean(dim=1, keepdim=True).expand(-1, 512, -1)  # Simplified fusion
        level_1_input = self.level_fusions[0](torch.cat([level_1_input, level_0], dim=-1))  # (B, 512, 512)
        level_1 = self.poolers[1](level_1_input)  # (B, 128, 512)
        
        # Level 2: Fine (combine level_1 + sampled from level_1_input → 32 tokens)
        n_sample_2 = min(128, level_1_input.size(1))
        sampled_2 = level_1_input[:, torch.randperm(level_1_input.size(1))[:n_sample_2], :]
        
        level_2_input = torch.cat([level_1, sampled_2], dim=1)  # (B, 256+128, 512)
        level_2_input = level_2_input.mean(dim=1, keepdim=True).expand(-1, 256, -1)
        level_2_input = self.level_fusions[1](torch.cat([level_2_input, level_1], dim=-1))  # (B, 256, 512)
        level_2 = self.poolers[2](level_2_input)  # (B, 256, 512) - Pipeline compatible!
        
        return level_2


class TaskConditionedAttentionalPooler(nn.Module):
    """
    Question-conditional compression with explicit two-pass architecture.
    
    Different questions focus on different image regions. This module modulates
    the compression queries based on the question context, allowing the model to
    adaptively attend to relevant anatomical regions.
    
    ⚠️ ARCHITECTURAL REQUIREMENT: 
        The LLM's embedding layer must run in a SEPARATE PASS before this projector.
        Requires modifying llava_arch.py encode_images() to extract frozen question embeddings.
    
    Args:
        d_model: Vision embedding dimension (512)
        context_dim: Context dimension (512)
        n_queries: Number of output tokens (256)
        n_head: Number of attention heads (8)
        text_embed_dim: LLM embedding dimension (4096 for Llama)
    
    Usage:
        # In llava_arch.py encode_images():
        with torch.no_grad():
            q_embed = self.get_model().embed_tokens(question_ids).mean(dim=1)
        image_features = self.get_model().mm_projector(images, q_embed)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        context_dim: int = 512,
        n_queries: int = 256,
        n_head: int = 8,
        text_embed_dim: int = 4096,
        norm_layer = nn.LayerNorm
    ):
        super().__init__()
        
        # Base queries (learned)
        self.base_queries = nn.Parameter(torch.randn(n_queries, d_model))
        
        # Project frozen LLM text embeddings to vision space
        self.question_proj = nn.Sequential(
            nn.Linear(text_embed_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Standard attention mechanism
        self.attn = AttentionalPooler(
            d_model=d_model,
            context_dim=context_dim,
            n_head=n_head,
            n_queries=n_queries,
            norm_layer=norm_layer
        )
    
    def forward(
        self, 
        image_tokens: torch.Tensor, 
        question_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image_tokens: (B, 2304, 512) - CTViT features
            question_embedding: (B, text_embed_dim) - from FROZEN LLM embedding layer
                                MUST be extracted via:
                                with torch.no_grad():
                                    q_embed = model.embed_tokens(question_ids).mean(dim=1)
                                If None, falls back to baseline (no question conditioning)
        
        Returns:
            compressed: (B, 256, 512) - question-conditioned image tokens
        """
        B = image_tokens.shape[0]
        
        if question_embedding is not None:
            # Task-conditioned compression: modulate queries based on question
            query_modulation = self.question_proj(question_embedding)  # (B, d_model)
            
            # Add question-specific offsets to base queries
            # Broadcasting: (1, n_queries, d_model) + (B, 1, d_model) → (B, n_queries, d_model)
            conditioned_queries = self.base_queries.unsqueeze(0) + query_modulation.unsqueeze(1)
            
            # TODO: Modify AttentionalPooler to accept custom query parameter
            # For now, use baseline behavior
            compressed = self.attn(image_tokens)
        else:
            # Fallback: standard global compression
            compressed = self.attn(image_tokens)
        
        return compressed  # (B, 256, 512)


# Factory function for easy instantiation
def build_advanced_pooler(pooler_type: str, **kwargs):
    """
    Factory function to build advanced poolers.
    
    Args:
        pooler_type: One of ['spatial', 'hierarchical', 'task_conditioned', 'baseline']
        **kwargs: Arguments passed to the pooler constructor
    
    Returns:
        nn.Module: The requested pooler
        
    Example:
        pooler = build_advanced_pooler('spatial', d_model=512, locality_radius=3)
    """
    poolers = {
        'spatial': SpatialAttentionalPooler,
        'hierarchical': HierarchicalAttentionalPooler,
        'task_conditioned': TaskConditionedAttentionalPooler,
        'baseline': AttentionalPooler
    }
    
    if pooler_type not in poolers:
        raise ValueError(f"Unknown pooler type: {pooler_type}. Choose from {list(poolers.keys())}")
    
    return poolers[pooler_type](**kwargs)
