"""
HyperCT_UPDT Configuration

Central config for all pipeline stages. All paths and hyperparameters
are overridable via CLI argparse in each script.
"""

from dataclasses import dataclass
from typing import Tuple


RADIOLOGICAL_TASKS = [
    # Aligned with the official CT-RATE multi-abnormality labels
    # (`*_predicted_labels.csv`). 18 categories produced by the RadBERT
    # text classifier shipped with the dataset. Order is fixed for
    # checkpoint compatibility — do NOT reorder without retraining.
    "medical_material",
    "arterial_wall_calcification",
    "cardiomegaly",
    "pericardial_effusion",
    "coronary_artery_wall_calcification",
    "hiatal_hernia",
    "lymphadenopathy",
    "emphysema",
    "atelectasis",
    "lung_nodule",
    "lung_opacity",
    "pulmonary_fibrotic_sequela",
    "pleural_effusion",
    "mosaic_attenuation_pattern",
    "peribronchial_thickening",
    "consolidation",
    "bronchiectasis",
    "interlobular_septal_thickening",
]

# Canonical column-name → task-id map for the official CT-RATE label CSVs.
# Used by the dataset loader to read `*_predicted_labels.csv` directly.
CT_RATE_CSV_COLUMNS = {
    "Medical material": "medical_material",
    "Arterial wall calcification": "arterial_wall_calcification",
    "Cardiomegaly": "cardiomegaly",
    "Pericardial effusion": "pericardial_effusion",
    "Coronary artery wall calcification": "coronary_artery_wall_calcification",
    "Hiatal hernia": "hiatal_hernia",
    "Lymphadenopathy": "lymphadenopathy",
    "Emphysema": "emphysema",
    "Atelectasis": "atelectasis",
    "Lung nodule": "lung_nodule",
    "Lung opacity": "lung_opacity",
    "Pulmonary fibrotic sequela": "pulmonary_fibrotic_sequela",
    "Pleural effusion": "pleural_effusion",
    "Mosaic attenuation pattern": "mosaic_attenuation_pattern",
    "Peribronchial thickening": "peribronchial_thickening",
    "Consolidation": "consolidation",
    "Bronchiectasis": "bronchiectasis",
    "Interlobular septal thickening": "interlobular_septal_thickening",
}


@dataclass
class VisionConfig:
    encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    encoder_dim: int = 768
    num_slices: int = 90  # divisible by 3
    # DINOv3 ViT-B native resolution (patch_size=16)
    slice_size: Tuple[int, int] = (224, 224)
    cube_pool_levels: int = 2  # 2x2x2 cube merging levels
    lora_rank: int = 16
    lora_scaling: float = 1.0  # LoRA output scaling factor (reference default)
    lora_dropout: float = 0.05


@dataclass
class HyperNetConfig:
    num_tasks: int = len(RADIOLOGICAL_TASKS)
    lora_rank: int = 16
    latent_size: int = 128
    head_in_size: int = 768  # matches DINOv3 feature dim (reference default)


@dataclass
class QFormerConfig:
    num_queries: int = 64
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1


@dataclass
class VLMConfig:
    llm_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_hidden_size: int = 4096
    vision_dim: int = 768
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
