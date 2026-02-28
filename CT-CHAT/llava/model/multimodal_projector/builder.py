import pdb
import torch
import torch.nn as nn
import re
from .coca_attentional_pooler import AttentionalPoolProjector
from .advanced_poolers import (
    SpatialAttentionalPooler,
    HierarchicalAttentionalPooler,
    TaskConditionedAttentionalPooler,
    build_advanced_pooler
)


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type.startswith('attn_pool'):
        # Parse pattern: attn_pool+[pooler_type]+[mlp_projector]
        # Examples:
        #   - attn_pool+mlp2x_gelu → baseline pooler + mlp
        #   - attn_pool+hierarchical+mlp2x_gelu → hierarchical pooler + mlp
        #   - attn_pool+spatial+mlp2x_gelu → spatial pooler + mlp
        parts = projector_type.split('+')
        
        # Determine pooler type and MLP projector
        if len(parts) == 2:
            # attn_pool+mlp2x_gelu → baseline pooler
            pooler_type = 'baseline'
            mlp_projector = parts[1]
        elif len(parts) == 3:
            # attn_pool+hierarchical+mlp2x_gelu → advanced pooler
            pooler_type = parts[1]
            mlp_projector = parts[2]
        else:
            raise ValueError(f"Invalid attn_pool format: {projector_type}")
        
        # Build MLP projector
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mlp_projector)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            projector = nn.Sequential(*modules)
        else:
            projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        
        # Build pooler based on type
        if pooler_type == 'baseline':
            # Original baseline pooler
            mm_projector = AttentionalPoolProjector(
                embed_dim=config.mm_hidden_size,
                context_dim=config.mm_context_size,
                projector=projector
            )
        elif pooler_type in ['spatial', 'hierarchical', 'task_conditioned']:
            # Advanced poolers - use factory function
            pooler = build_advanced_pooler(
                pooler_type=pooler_type,
                d_model=config.mm_hidden_size
            )
            # Wrap with projector (pooler + MLP)
            mm_projector = nn.Sequential(pooler, projector)
        else:
            raise ValueError(f"Unknown pooler type: {pooler_type}. Choose from: baseline, spatial, hierarchical, task_conditioned")
        
        return mm_projector

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')