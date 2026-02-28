"""
Vision Tower Builder for CT-CHAT

NOTE: In production, vision encoding is done offline via encode_script.py using CTViT.
This builder is kept for compatibility but not actively used during training.
The vision tower is bypassed - we load pre-encoded features directly.
"""

import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Build vision tower (legacy support - not used in production pipeline).
    
    Production pipeline uses pre-encoded CTViT features loaded from .npz files.
    """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower) if vision_tower else False
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    if is_absolute_path_exists or (vision_tower and (vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower)):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # Fallback - return basic CLIP tower
    if vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')
