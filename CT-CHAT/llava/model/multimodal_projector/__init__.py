"""
Multimodal Projector Module

This module provides various pooling and projection strategies for compressing
vision encoder outputs before feeding to the LLM.
"""

from .coca_attentional_pooler import AttentionalPooler, AttentionalPoolProjector
from .advanced_poolers import (
    SpatialAttentionalPooler,
    HierarchicalAttentionalPooler,
    TaskConditionedAttentionalPooler,
    build_advanced_pooler
)
from .builder import build_vision_projector

__all__ = [
    # Base poolers
    'AttentionalPooler',
    'AttentionalPoolProjector',
    
    # Advanced poolers
    'SpatialAttentionalPooler',
    'HierarchicalAttentionalPooler',
    'TaskConditionedAttentionalPooler',
    
    # Factory functions
    'build_advanced_pooler',
    'build_vision_projector',
]
