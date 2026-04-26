#!/bin/bash
#SBATCH --job-name=hyperct_train_hypernet
#SBATCH -p sablab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=hyperct_train_hypernet_%j.out
#SBATCH --error=hyperct_train_hypernet_%j.err



module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate test

# Install dependencies (confirmed working setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops open_clip_torch timm deepspeed ninja
pip install flash-attn --no-build-isolation
pip install "transformers>=4.56.0" nibabel tqdm
pip install "numpy<2"
pip install --upgrade peft
pip install --upgrade pip wheel
pip install --force-reinstall --no-deps markupsafe==3.0.3

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/Radiology_VLM_AI4ML/HyperCT_UPDT
cd "$PROJECT_DIR"

python train_hypernet.py \
    --data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/train_fixed \
    --labels_csv /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/multi_abnormality_labels/train_predicted_labels.csv \
    --val_labels_csv /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/multi_abnormality_labels/valid_predicted_labels.csv \
    --val_data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/valid_fixed \
    --preprocess_dir /midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/Radiology_VLM_AI4ML/HyperCT_UPDT/PreProcessed_train1 \
    --val_preprocess_dir /midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/Radiology_VLM_AI4ML/HyperCT_UPDT/PreProcessed_valid1 \
    --output_dir ./checkpoint_ff \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --lora_rank 16 \
    --lora_scaling 1.0 \
    --num_slices 90 \
    --slice_height 224 \
    --slice_width 224 \
    --batch_size 1 \
    --cube_pool_levels 2 \
    --lr 5e-5 \
    --weight_decay 1e-2 \
    --amp_dtype bf16 \
    --grad_accum_steps 4 \
    --max_grad_norm 1.0 \
    --pos_weight_power 0.5 \
    --pos_weight_cap 20.0 \
    --epochs 8 \
    --num_workers 4 \
    --max_batches_per_epoch 10000 \
    --early_stop_patience 4 \
    --seed 42
