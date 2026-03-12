#!/bin/bash
#SBATCH --job-name=hyperct_train_vlm
#SBATCH -p sablab-gpu-low
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=hyperct_train_vlm_%j.out
#SBATCH --error=hyperct_train_vlm_%j.err

set -euo pipefail
module purge
module load anaconda3
conda init bash
conda activate test

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/AI4ML-initiative---Medical-VLM-Model
cd "$PROJECT_DIR"

torchrun --nproc_per_node=4 HyperCT_UPDT/train_vlm.py \
    --tokens_dir ./precomputed_tokens \
    --data_json /path/to/train_data.json \
    --output_dir ./checkpoints/hyperct_vlm \
    --llm_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_hidden_size 4096 \
    --vision_dim 768 \
    --num_queries 64 \
    --qformer_layers 6 \
    --qformer_heads 12 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lr 2e-5 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --max_length 2048 \
    --bf16
