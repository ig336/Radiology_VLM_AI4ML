"""
HyperCT_UPDT Configuration

Central config for all pipeline stages. All paths and hyperparameters
are overridable via CLI argparse in each script.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


RADIOLOGICAL_TASKS = [
    "opacity", "nodule", "consolidation", "atelectasis",
    "pleural_effusion", "cardiomegaly", "emphysema", "fibrosis",
    "bronchiectasis", "lymphadenopathy", "mass", "pneumothorax",
    "pericardial_effusion", "calcification", "medical_material",
    "mosaic_attenuation", "peribronchial_thickening", "hiatal_hernia",
]


@dataclass
class VisionConfig:
    encoder_name: str = "facebook/dinov2-with-registers-base"
    encoder_dim: int = 768
    num_slices: int = 32
    slice_size: Tuple[int, int] = (518, 518)
    spatial_pool: str = "mean"
    temporal_pool: str = "attention"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["qkv", "proj"])


@dataclass
class HyperNetConfig:
    task_embed_dim: int = 128
    hidden_dim: int = 256
    num_tasks: int = len(RADIOLOGICAL_TASKS)
    lora_rank: int = 16


@dataclass
class QFormerConfig:
    num_queries: int = 64
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    num_tasks: int = len(RADIOLOGICAL_TASKS)
    tokens_per_task: int = 64
    total_output_tokens: int = 256


@dataclass
class VLMConfig:
    llm_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_hidden_size: int = 4096
    vision_dim: int = 768
    vision_tokens: int = 256
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
