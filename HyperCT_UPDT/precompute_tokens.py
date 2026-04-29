"""Precompute Vision Tokens for HyperCT_UPDT Pipeline

Processes preprocessed CT tensors (.pt) or raw 3D CT volumes (.nii.gz)
through DINOv3 ViT-B + task-specific LoRA and saves pooled vision tokens
as .npz files for downstream training.

Uses facebook/dinov3-vitb16-pretrain-lvd1689m (arXiv 2508.10104).
Follows HyperCT reference architecture (github.com/lfb-1/HyperCT).

Pipeline per volume (following HyperCT reference architecture):
    1. Prefer train_hypernet/preprocess_volumes .pt tensors; optionally fall
       back to .nii.gz -> HU windowing -> ensure_length(divisible by 3) -> slice
    2. Group 3 consecutive slices into RGB images (R=slice_i, G=slice_{i+1},
       B=slice_{i+2}) — provides inter-slice anatomical context
    3. For each task (18 radiological labels):
        a. Encode RGB images: DINOv3 + task LoRA -> patch tokens
        b. 2x2x2 cube pooling -> compressed tokens
    4. Run classifier on pooled features per task
    5. Save tokens + predictions as .npz

Usage:
    python precompute_tokens.py \\
        --data_dir /path/to/nifti_files \\
        --output_dir ./precomputed_tokens \\
        --num_slices 90 \\
        --cube_pool_levels 2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.pooling import CubePooler, ensure_length, pad_volume_slices
from models.encoder import DINOv3LoRAEncoder
from config import RADIOLOGICAL_TASKS
import os
import argparse
import logging

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_nifti_slices(path: str, num_slices: int, slice_size: tuple,
                      hu_min: float = -1000, hu_max: float = 1000) -> torch.Tensor:
    """
    Load .nii.gz, apply HU windowing, ensure divisible by 3, resample.

    Returns:
        slices: (num_slices, H, W) float32 tensor normalized to [0, 1]
    """
    nii = nib.load(str(path))
    volume = nii.get_fdata().astype(np.float32)

    # HU windowing
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)  # [0, 1]

    # NIfTI CT volumes are typically stored as (H, W, D) in RAS convention.
    # We need (D, H, W) for axial slice extraction.
    # Identify depth as the smallest spatial dimension.
    dims = volume.shape
    depth_axis = int(np.argmin(dims))
    if depth_axis != 0:
        # Move depth axis to front: (D, H, W)
        volume = np.moveaxis(volume, depth_axis, 0)

    D, H, W = volume.shape
    volume_t = torch.from_numpy(volume).unsqueeze(
        0).unsqueeze(0)  # (1, 1, D, H, W)

    # Resample depth
    volume_t = F.interpolate(volume_t, size=(
        num_slices, H, W), mode="trilinear", align_corners=False)

    # Resample spatial
    volume_t = F.interpolate(
        volume_t.squeeze(0),  # (1, num_slices, H, W)
        size=(slice_size[0], slice_size[1]),
        mode="bilinear", align_corners=False,
    )  # (1, num_slices, sH, sW)

    slices = volume_t.squeeze(0)  # (num_slices, sH, sW)

    # Ensure divisible by 3
    target = ensure_length(num_slices, divisor=3)
    slices = pad_volume_slices(slices, target)

    return slices


def load_preprocessed_slices(path: str, num_slices: int,
                             slice_size: tuple) -> torch.Tensor:
    """
    Load slices from preprocess_volumes.py output.

    The training dataset consumes these same tensors, so using them here keeps
    Stage 1 training and Stage 2 token generation on the same deterministic
    volume representation while avoiding repeated NIfTI decompression.
    """
    try:
        slices = torch.load(path, weights_only=True)
    except TypeError:
        slices = torch.load(path)
    if isinstance(slices, np.ndarray):
        slices = torch.from_numpy(slices)
    if not torch.is_tensor(slices):
        raise TypeError(f"Expected tensor/ndarray in {path}, got {type(slices)}")
    slices = slices.float()
    if slices.ndim == 4 and slices.shape[0] == 1:
        slices = slices.squeeze(0)
    if slices.ndim != 3:
        raise ValueError(f"Expected preprocessed slices with shape (D,H,W), "
                         f"got {tuple(slices.shape)} from {path}")

    target = ensure_length(num_slices, divisor=3)
    slices = pad_volume_slices(slices, target)
    if tuple(slices.shape[-2:]) != tuple(slice_size):
        log.warning(f"Preprocessed tensor {path} has spatial shape "
                    f"{tuple(slices.shape[-2:])}; resizing to {slice_size}")
        slices = F.interpolate(
            slices.unsqueeze(0),
            size=(slice_size[0], slice_size[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return slices


def resolve_preprocessed_path(preprocess_dir: str, volume_path: str) -> str:
    """Map a raw NIfTI path to the flat .pt name produced by preprocess_volumes.py."""
    if not preprocess_dir:
        return None
    vol_name = Path(volume_path).name
    if vol_name.endswith(".nii.gz"):
        pt_name = vol_name[:-7] + ".pt"
    else:
        pt_name = Path(vol_name).stem + ".pt"
    pt_path = os.path.join(preprocess_dir, pt_name)
    return pt_path if os.path.exists(pt_path) else None


def slices_to_rgb(volume_slices: torch.Tensor, group_idx: int) -> torch.Tensor:
    """
    Stack 3 consecutive CT slices as RGB channels (HyperCT reference).

    Each group of 3 slices forms one RGB image providing inter-slice
    anatomical context: R=slice_{3i}, G=slice_{3i+1}, B=slice_{3i+2}.

    Args:
        volume_slices: (num_slices, H, W) full volume slices
        group_idx: which group of 3 (0-indexed)
    Returns:
        rgb: (3, H, W) 3-channel tensor
    """
    start = group_idx * 3
    return volume_slices[start:start + 3]  # (3, H, W)


def precompute_single_volume(
    volume_path: str,
    encoder: DINOv3LoRAEncoder,
    pooler: CubePooler,
    num_slices: int,
    slice_size: tuple,
    device: torch.device,
    preprocess_path: str = None,
) -> dict:
    """
    Precompute vision tokens for one CT volume, all tasks.
    Uses forward_with_lora for manual LoRA injection per task.
    Also runs classifier per task.

    Returns:
        dict with keys:
            'tokens': (num_tasks, N_tokens, D) pooled tokens
            'predictions': (num_tasks, num_tasks) classifier logits
            'tasks': list of task names
    """
    if preprocess_path is not None:
        slices = load_preprocessed_slices(preprocess_path, num_slices, slice_size)
    else:
        slices = load_nifti_slices(volume_path, num_slices, slice_size)
    actual_slices = slices.shape[0]
    num_rgb_images = actual_slices // 3  # HyperCT: 3 slices per RGB image

    all_task_tokens = []
    all_predictions = []

    for task_idx in range(len(RADIOLOGICAL_TASKS)):
        task_id = torch.tensor([task_idx], device=device)

        # Generate LoRA weights ONCE per task (same across all slices)
        lora_weights = encoder.hypernet.generate_full_model_lora(task_id)

        # Batch all RGB groups into a single forward pass (11x fewer DINOv3
        # forward passes per task). LoRA weights are shared across the batch.
        all_rgb = torch.stack(
            [slices_to_rgb(slices, g) for g in range(num_rgb_images)]
        ).to(device)  # (num_rgb, 3, H, W)
        all_tokens = encoder.forward_with_lora(
            all_rgb, lora_weights)  # (num_rgb, N_patches, 768)
        slice_tokens = [all_tokens[g:g+1]
                        for g in range(num_rgb_images)]  # list of (1, N, D)

        # 2×2×2 cube pooling
        pooled = pooler(slice_tokens)  # (1, final_tokens, D)

        # Classifier prediction from each task's LoRA-adapted features
        pred = encoder.classify(pooled)  # (1, num_tasks)

        all_task_tokens.append(pooled.cpu())
        all_predictions.append(pred.cpu())

    # (num_tasks, N_tokens, D)
    stacked_tokens = torch.cat(all_task_tokens, dim=0)
    # (num_tasks, num_tasks)
    stacked_preds = torch.cat(all_predictions, dim=0)

    return {
        "tokens": stacked_tokens.numpy(),
        "predictions": stacked_preds.numpy(),
        "tasks": RADIOLOGICAL_TASKS,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Precompute HyperCT vision tokens")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dir with .nii.gz files")
    parser.add_argument("--output_dir", type=str,
                        default="./precomputed_tokens")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Encoder+HyperNet checkpoint")
    parser.add_argument("--preprocess_dir", type=str, default=None,
                        help="Optional directory with preprocess_volumes.py .pt tensors.")
    parser.add_argument("--allow_raw_fallback", action="store_true",
                        help="Allow missing .pt tensors to fall back to raw .nii.gz. "
                             "Default is strict 100%% .pt coverage when --preprocess_dir is set.")
    parser.add_argument("--num_slices", type=int, default=90)
    parser.add_argument("--slice_height", type=int, default=224)
    parser.add_argument("--slice_width", type=int, default=224)
    parser.add_argument("--cube_pool_levels", type=int, default=2)
    parser.add_argument("--encoder_name", type=str,
                        default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_scaling", type=float, default=1.0)
    args = parser.parse_args()

    # Ensure num_slices is divisible by 3
    args.num_slices = ensure_length(args.num_slices, divisor=3)
    log.info(f"Using {args.num_slices} slices (ensured divisible by 3)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Initializing DINOv3 encoder with HyperNetwork...")
    encoder = DINOv3LoRAEncoder(
        encoder_name=args.encoder_name,
        num_tasks=len(RADIOLOGICAL_TASKS),
        lora_rank=args.lora_rank,
        lora_scaling=args.lora_scaling,
    ).to(device)

    checkpoint_state = None
    if args.checkpoint:
        checkpoint_state = torch.load(
            args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(checkpoint_state["encoder"], strict=False)
        log.info(f"Loaded checkpoint: {args.checkpoint}")

    encoder.eval()

    pooler = CubePooler(
        dim=768,
        num_levels=args.cube_pool_levels,
    ).to(device)

    if checkpoint_state is not None and "pooler" in checkpoint_state:
        pooler.load_state_dict(checkpoint_state["pooler"])

    pooler.eval()

    nifti_files = sorted(Path(args.data_dir).rglob("*.nii.gz"))
    log.info(f"Found {len(nifti_files)} volumes in {args.data_dir}")
    preprocessed_paths = {}
    if args.preprocess_dir:
        if not os.path.isdir(args.preprocess_dir):
            raise FileNotFoundError(
                f"--preprocess_dir does not exist: {args.preprocess_dir}")
        preprocessed_paths = {
            str(vol_path): resolve_preprocessed_path(args.preprocess_dir, str(vol_path))
            for vol_path in nifti_files
        }
        cache_hits = sum(path is not None for path in preprocessed_paths.values())
        coverage = 100.0 * cache_hits / max(len(nifti_files), 1)
        log.info(f"Preprocessed cache coverage: {cache_hits}/{len(nifti_files)} "
                 f"({coverage:.1f}%) in {args.preprocess_dir}")
        if cache_hits != len(nifti_files) and not args.allow_raw_fallback:
            missing = [
                str(vol_path) for vol_path in nifti_files
                if preprocessed_paths[str(vol_path)] is None
            ]
            preview = "\n".join(missing[:10])
            raise RuntimeError(
                f"Strict preprocessed mode requires 100% .pt coverage, but found "
                f"{cache_hits}/{len(nifti_files)}. First missing raw volumes:\n"
                f"{preview}"
            )

    slice_size = (args.slice_height, args.slice_width)

    for vol_path in tqdm(nifti_files, desc="Encoding volumes"):
        vol_name = vol_path.stem.replace(".nii", "")
        out_path = os.path.join(args.output_dir, f"{vol_name}.npz")

        if os.path.exists(out_path):
            continue

        with torch.no_grad():
            result = precompute_single_volume(
                str(vol_path), encoder, pooler,
                args.num_slices, slice_size, device,
                preprocess_path=preprocessed_paths.get(str(vol_path)),
            )

        np.savez_compressed(out_path, **result)
        log.info(f"Saved {out_path} — tokens shape {result['tokens'].shape}")

    log.info("Precomputation complete.")


if __name__ == "__main__":
    main()
