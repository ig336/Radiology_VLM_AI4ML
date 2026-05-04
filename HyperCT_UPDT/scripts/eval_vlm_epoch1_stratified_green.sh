#!/bin/bash
# Evaluate the completed epoch-1 VLM checkpoint with CT-CHAT-style
# question-type stratification, constrained generation, and official GREEN.
# This does not retrain and does not overwrite the original epoch-1 checkpoint.
#SBATCH --job-name=hyperct_eval_vlm_strat_green
#SBATCH -p sablab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=hyperct_eval_vlm_strat_green_%j.out
#SBATCH --error=hyperct_eval_vlm_strat_green_%j.err

set -euo pipefail
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate test

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip uninstall -y deepspeed || true
pip install einops open_clip_torch timm ninja
pip install "transformers>=4.56.0" nibabel tqdm
pip install "numpy<2"
pip install --upgrade peft
pip install --upgrade pip wheel
pip install --force-reinstall --no-deps markupsafe==3.0.3
pip install green_score || pip install green-score || true

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/Radiology_VLM_AI4ML/HyperCT_UPDT
cd "$PROJECT_DIR"

python train_vlm.py \
    --tokens_dir ./precompute_tokens_ff \
    --data_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/train_vqa.json \
    --val_data_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/valid_vqa.json \
    --val_tokens_dir ./precompute_tokens_valid_ff \
    --output_dir ./checkpoints/hyperct_vlm_epoch1_stratified_green_eval \
    --qformer_checkpoint ./checkpoints/hyperct_vlm_epoch1/qformer_final.pt \
    --llm_lora_checkpoint ./checkpoints/hyperct_vlm_epoch1/llm_lora \
    --eval_only \
    --llm_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_hidden_size 4096 \
    --vision_dim 768 \
    --num_queries 64 \
    --qformer_layers 6 \
    --qformer_heads 12 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --grad_accum 2 \
    --max_length 2048 \
    --num_task_tokens 3 \
    --eval_strategy epoch \
    --generation_eval_samples 512 \
    --generation_max_new_tokens 128 \
    --generation_num_beams 1 \
    --type_aware_prompts \
    --constrained_generation \
    --llm_score_samples 64 \
    --green_score_samples 64 \
    --official_green_model StanfordAIMI/GREEN-radllama2-7b \
    --official_green_samples 64 \
    --judge_max_new_tokens 160 \
    --bf16 \
    --attn_implementation sdpa
