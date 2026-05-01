#!/bin/bash
#SBATCH --job-name=hyperct_train_vlm
#SBATCH -p sablab-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=hyperct_train_vlm_%j.out
#SBATCH --error=hyperct_train_vlm_%j.err

set -euo pipefail
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate test

# Install dependencies (confirmed working setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip uninstall -y deepspeed
pip install einops open_clip_torch timm ninja
pip install "transformers>=4.56.0" nibabel tqdm
pip install "numpy<2"
pip install --upgrade peft
pip install --upgrade pip wheel
pip install --force-reinstall --no-deps markupsafe==3.0.3

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/Radiology_VLM_AI4ML/HyperCT_UPDT
cd "$PROJECT_DIR"

if [ ! -d ./precompute_tokens_ff ]; then
    echo "Missing ./precompute_tokens_ff. Run scripts/precompute.sh before train_vlm.sh."
    exit 1
fi

if [ ! -d ./precompute_tokens_valid_ff ]; then
    echo "Missing ./precompute_tokens_valid_ff. Run scripts/precompute_valid.sh before train_vlm.sh."
    exit 1
fi

torchrun --nproc_per_node=4 train_vlm.py \
    --tokens_dir ./precompute_tokens_ff \
    --data_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/train_vqa.json \
    --val_data_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/valid_vqa.json \
    --val_tokens_dir ./precompute_tokens_valid_ff \
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
    --batch_size 4 \
    --eval_batch_size 4 \
    --grad_accum 2 \
    --max_length 2048 \
    --num_task_tokens 3 \
    --eval_strategy epoch \
    --generation_eval_samples 512 \
    --generation_max_new_tokens 128 \
    --llm_score_samples 64 \
    --green_score_samples 64 \
    --judge_max_new_tokens 160 \
    --official_green_samples 0 \
    --bf16 \
    --attn_implementation sdpa
