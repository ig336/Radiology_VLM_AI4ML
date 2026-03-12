"""
Precompute Vision Tokens for HyperCT_UPDT Pipeline

Processes 3D CT volumes (.nii.gz) through DINOv3 + task-specific LoRA
and saves pooled vision tokens as .npz files for downstream training.

Pipeline per volume:
    1. Load .nii.gz → HU windowing → resample → slice into 2D
    2. For each task (18 radiological labels):
        a. Encode each slice: DINOv3 + task LoRA → (N_patches, 768)
        b. Spatial pool: (N_patches, 768) → (K, 768)
        c. Temporal pool across slices: → (T, 768)
    3. Concatenate task tokens: (18 * T, 768)
    4. Save as .npz

Usage:
    python precompute_tokens.py \
        --data_dir /path/to/nifti_files \
        --output_dir ./precomputed_tokens \
        --checkpoint /path/to/encoder_checkpoint.pt \
        --num_slices 32 \
        --spatial_pool attention \
        --temporal_pool attention
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import VisionConfig, HyperNetConfig, RADIOLOGICAL_TASKS
from models.encoder import DINOv3LoRAEncoder
from models.pooling import HybridPooler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_nifti_slices(path: str, num_slices: int, slice_size: tuple,
                      hu_min: float = -1000, hu_max: float = 1000) -> torch.Tensor:
    """
    Load .nii.gz, apply HU windowing, resample to uniform slices.

    Returns:
        slices: (num_slices, H, W) float32 tensor normalized to [0, 1]
    """
    nii = nib.load(str(path))
    volume = nii.get_fdata().astype(np.float32)

    # HU windowing
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)  # [0, 1]

    # Transpose to (D, H, W) if needed
    if volume.shape[2] < volume.shape[0]:
        volume = np.transpose(volume, (2, 0, 1))

    D, H, W = volume.shape
    volume_t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Resample depth
    volume_t = F.interpolate(volume_t, size=(num_slices, H, W), mode="trilinear", align_corners=False)

    # Resample spatial
    volume_t = F.interpolate(
        volume_t.squeeze(0),  # (1, num_slices, H, W)
        size=(slice_size[0], slice_size[1]),
        mode="bilinear", align_corners=False,
    )  # (1, num_slices, sH, sW)

    return volume_t.squeeze(0)  # (num_slices, sH, sW)


def slice_to_rgb(gray_slice: torch.Tensor) -> torch.Tensor:
    """Convert single-channel slice to 3-channel for DINOv3. (H, W) → (3, H, W)."""
    return gray_slice.unsqueeze(0).expand(3, -1, -1)


def precompute_single_volume(
    volume_path: str,
    encoder: DINOv3LoRAEncoder,
    pooler: HybridPooler,
    processor,
    num_slices: int,
    slice_size: tuple,
    device: torch.device,
) -> dict:
    """
    Precompute vision tokens for one CT volume, all tasks.

    Returns:
        dict with keys: 'tokens' (num_tasks, T_out, D), 'tasks' list
    """
    slices = load_nifti_slices(volume_path, num_slices, slice_size)

    all_task_tokens = []

    for task_idx in range(len(RADIOLOGICAL_TASKS)):
        task_id = torch.tensor([task_idx], device=device)
        slice_tokens = []

        for s in range(num_slices):
            rgb = slice_to_rgb(slices[s])  # (3, H, W)
            # DINOv3 processor expects PIL or numpy; we use pixel_values directly
            pixel_values = rgb.unsqueeze(0).to(device)  # (1, 3, H, W)
            tokens = encoder.encode_slice(pixel_values, task_id)  # (1, N_patches, 768)
            slice_tokens.append(tokens)

        # Hybrid pool: spatial + temporal
        pooled = pooler(slice_tokens)  # (1, T_out, 768)
        all_task_tokens.append(pooled.cpu())

    # Stack: (num_tasks, T_out, 768)
    stacked = torch.cat(all_task_tokens, dim=0)
    return {"tokens": stacked.numpy(), "tasks": RADIOLOGICAL_TASKS}


def main():
    parser = argparse.ArgumentParser(description="Precompute HyperCT vision tokens")
    parser.add_argument("--data_dir", type=str, required=True, help="Dir with .nii.gz files")
    parser.add_argument("--output_dir", type=str, default="./precomputed_tokens")
    parser.add_argument("--checkpoint", type=str, default=None, help="Encoder+HyperNet checkpoint")
    parser.add_argument("--num_slices", type=int, default=32)
    parser.add_argument("--slice_height", type=int, default=518)
    parser.add_argument("--slice_width", type=int, default=518)
    parser.add_argument("--spatial_pool", type=str, default="attention")
    parser.add_argument("--temporal_pool", type=str, default="attention")
    parser.add_argument("--spatial_output", type=int, default=64)
    parser.add_argument("--temporal_output", type=int, default=8)
    parser.add_argument("--encoder_name", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Initializing DINOv3 encoder with HyperNetwork...")
    encoder = DINOv3LoRAEncoder(
        encoder_name=args.encoder_name,
        num_tasks=len(RADIOLOGICAL_TASKS),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(state["encoder"], strict=False)
        log.info(f"Loaded checkpoint: {args.checkpoint}")

    encoder.eval()

    pooler = HybridPooler(
        dim=768,
        spatial_method=args.spatial_pool,
        temporal_method=args.temporal_pool,
        spatial_output=args.spatial_output,
        temporal_output=args.temporal_output,
    ).to(device)

    if args.checkpoint and "pooler" in state:
        pooler.load_state_dict(state["pooler"])

    pooler.eval()

    nifti_files = sorted(Path(args.data_dir).glob("*.nii.gz"))
    log.info(f"Found {len(nifti_files)} volumes in {args.data_dir}")

    slice_size = (args.slice_height, args.slice_width)

    for vol_path in tqdm(nifti_files, desc="Encoding volumes"):
        vol_name = vol_path.stem.replace(".nii", "")
        out_path = os.path.join(args.output_dir, f"{vol_name}.npz")

        if os.path.exists(out_path):
            continue

        with torch.no_grad():
            result = precompute_single_volume(
                str(vol_path), encoder, pooler,
                encoder.processor, args.num_slices, slice_size, device,
            )

        np.savez_compressed(out_path, **result)
        log.info(f"Saved {out_path} — shape {result['tokens'].shape}")

    log.info("Precomputation complete.")


if __name__ == "__main__":
    main()
