#!/bin/bash
#SBATCH --job-name=hyperct_precompute
#SBATCH -p sablab-gpu-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hyperct_precompute_%j.out
#SBATCH --error=hyperct_precompute_%j.err

set -euo pipefail
module purge
module load anaconda3
conda init bash
conda activate test

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/AI4ML-initiative---Medical-VLM-Model
cd "$PROJECT_DIR"

python HyperCT_UPDT/precompute_tokens.py \
    --data_dir /path/to/nifti_files \
    --output_dir ./precomputed_tokens \
    --num_slices 32 \
    --slice_height 518 \
    --slice_width 518 \
    --spatial_pool attention \
    --temporal_pool attention \
    --spatial_output 64 \
    --temporal_output 8 \
    --encoder_name facebook/dinov2-with-registers-base \
    --lora_rank 16 \
    --lora_alpha 32
