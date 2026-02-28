"""
CT Volume Pre-Encoding Script for CT-CHAT

This script processes 3D CT volumes (.nii.gz files) and encodes them using CTViT
(CT Vision Transformer) from CT-CLIP. The encoded features are saved as .npz files
for efficient training.

Pipeline:
    1. Load .nii.gz with nibabel
    2. Apply DICOM rescaling (slope/intercept)
    3. HU windowing [-1000, 1000]
    4. Resample to target spacing (0.75mm xy, 1.5mm z)
    5. Crop/pad to fixed size (480, 480, 240)
    6. Normalize (/1000)
    7. Encode with CTViT → (N_tokens, 512)
    8. Save as .npz

Why Pre-encoding?
    - Training Speed: 10x faster (no repeated encoding per epoch)
    - Memory: 40x less RAM (512-dim embeddings vs full 3D volumes)
    - Reproducibility: Same features across experiments
    - Storage: ~4.7MB per .npz vs ~200MB per .nii.gz

Usage:
    python encode_script.py \
        --path /path/to/volume.nii.gz \
        --slope 1.0 \
        --intercept 0.0 \
        --xy_spacing 0.7 \
        --z_spacing 1.5

Output:
    ./embeddings/volume_name.npz containing {'arr': numpy array of shape (1, N_tokens, 512)}

Requirements:
    - transformer_maskgit library (for CTViT)
    - CTViT weights: ./CT_CLIP_encoder/clip_visual_encoder.pth
"""

import torch
from transformer_maskgit import CTViT
import numpy as np
import nibabel as nib
import argparse
import torch.nn.functional as F


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array



def nii_img_to_tensor(path, slope, intercept, xy_spacing, z_spacing):
    """
    Load and preprocess a NIfTI CT volume for CTViT encoding.
    
    Pipeline:
        1. Load .nii.gz with nibabel
        2. Apply DICOM rescaling: HU = slope * pixel_value + intercept
        3. HU windowing: clip to [-1000, 1000] (soft tissue + lung)
        4. Resample to target spacing (0.75mm x 0.75mm x 1.5mm)
        5. Crop/pad to fixed size: (H, W, D) = (480, 480, 240)
        6. Normalize: divide by 1000 to get [-1, 1] range
        7. Permute to (D, H, W) format: (240, 480, 480)
        8. Add single batch dimension: (1, D, H, W) = (1, 240, 480, 480)
    
    Args:
        path (str): Path to .nii.gz file
        slope (float): DICOM RescaleSlope (typically 1.0)
        intercept (float): DICOM RescaleIntercept (typically 0.0 or -1024)
        xy_spacing (float): Original pixel spacing in mm (x and y)
        z_spacing (float): Original slice spacing in mm (z)
    
    Returns:
        torch.Tensor: Shape (1, D, H, W) = (1, 240, 480, 480) on CUDA
                     Note: This is NOT a channel dimension - it's a placeholder.
                     In main(), this gets unsqueezed to (B, C, D, H, W) = (1, 1, 240, 480, 480)
                     where B=batch_size, C=1 (grayscale CT), D=depth, H=height, W=width
    """
    # Load NIfTI file
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    # Define the target spacing values (standard for CT-CLIP)
    target_x_spacing = 0.75  # mm
    target_y_spacing = 0.75  # mm
    target_z_spacing = 1.5   # mm

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (((img_data) / 1000)).astype(np.float32)
    slices = []

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    return tensor.cuda()

def main():
    parser = argparse.ArgumentParser(description='Process NIfTI image and encode it using a transformer model.')

    parser.add_argument('--path', type=str, required=True, help='Path to the NIfTI image file.')
    parser.add_argument('--slope', type=float, default=1, help='Slope for rescaling the image.')
    parser.add_argument('--intercept', type=float, default=0, help='Intercept for rescaling the image.')
    parser.add_argument('--xy_spacing', type=float, default=1, help='XY spacing of the image.')
    parser.add_argument('--z_spacing', type=float, default=1, help='Z spacing of the image.')
    
    # Data augmentation options
    parser.add_argument('--augment_flip', action='store_true', 
                        help='Encode with left-right flip augmentation (doubles storage, anatomically valid for chest CT)')
    parser.add_argument('--augment_flip_all', action='store_true',
                        help='Encode all flip orientations: LR, UD, Both (4x storage, requires --augment_flip)')

    args = parser.parse_args()

    # Initialize CTViT encoder (CT Vision Transformer from CT-CLIP)
    # Architecture designed specifically for 3D medical imaging with temporal awareness
    image_encoder = CTViT(
        dim=512,                    # Hidden dimension for transformer
        codebook_size=8192,         # VQ-VAE codebook size (for CT-CLIP pre-training)
        image_size=480,             # Spatial dimensions (H, W)
        patch_size=20,              # Spatial patch size (20x20 pixels)
        temporal_patch_size=10,     # Temporal patch size (along depth/slices)
        spatial_depth=4,            # Number of spatial transformer blocks
        temporal_depth=4,           # Number of temporal transformer blocks (handles 3D structure)
        dim_head=32,                # Dimension per attention head
        heads=8                     # Number of attention heads (8 * 32 = 256)
    ).cuda().eval()

    # Load pre-trained weights from CT-CLIP
    image_encoder.load("./CT_CLIP_encoder/clip_visual_encoder.pth")

    # Process the CT volume
    image = nii_img_to_tensor(
        path=args.path, 
        slope=args.slope, 
        intercept=args.intercept, 
        xy_spacing=args.xy_spacing, 
        z_spacing=args.z_spacing
    )

    # Encode: Add batch dimension first to get (B, C, D, H, W) format
    # image shape: (1, D, H, W) = (1, 240, 480, 480)
    # After unsqueeze(0): (B, C, D, H, W) = (1, 1, 240, 480, 480)
    # 
    # CTViT processes:
    # - Spatial patches: (480/20) x (480/20) = 24 x 24 = 576 patches per slice
    # - Temporal patches: (240/10) = 24 slices
    # - Total patches: 24 x 24 x 24 = 13,824 patches
    # - After spatial + temporal transformers: compressed to ~2304 tokens
    # 
    # Output: (B, N_tokens, 512) where N_tokens ≈ 2304
    
    image_name = args.path.split("/")[-1].split(".")[0]
    
    # Option 1: Encode original only (default)
    image_encoded = image_encoder(image.unsqueeze(0), return_encoded_tokens=True)
    output_path = f'./embeddings/{image_name}.npz'
    np.savez(output_path, arr=image_encoded.cpu().detach().numpy())
    
    print(f"✅ Encoded {args.path}")
    print(f"   Shape: {image_encoded.shape}")
    print(f"   Saved to: {output_path}")
    
    # Option 2: Encode with left-right flip augmentation (if --augment_flip flag is set)
    # This doubles storage but provides anatomically-valid augmentation for chest CT
    if getattr(args, 'augment_flip', False):
        # Left-right flip (along width dimension)
        image_flipped_lr = torch.flip(image, dims=[3])  # Flip W dimension
        encoded_flipped_lr = image_encoder(image_flipped_lr.unsqueeze(0), return_encoded_tokens=True)
        output_path_lr = f'./embeddings/{image_name}_flip_lr.npz'
        np.savez(output_path_lr, arr=encoded_flipped_lr.cpu().detach().numpy())
        print(f"   + Flip LR: {output_path_lr}")
        
        # Optional: Up-down flip (if --augment_flip_all flag is set)
        if getattr(args, 'augment_flip_all', False):
            # Up-down flip (along height dimension)
            image_flipped_ud = torch.flip(image, dims=[2])  # Flip H dimension
            encoded_flipped_ud = image_encoder(image_flipped_ud.unsqueeze(0), return_encoded_tokens=True)
            output_path_ud = f'./embeddings/{image_name}_flip_ud.npz'
            np.savez(output_path_ud, arr=encoded_flipped_ud.cpu().detach().numpy())
            print(f"   + Flip UD: {output_path_ud}")
            
            # Both flips
            image_flipped_both = torch.flip(image, dims=[2, 3])
            encoded_flipped_both = image_encoder(image_flipped_both.unsqueeze(0), return_encoded_tokens=True)
            output_path_both = f'./embeddings/{image_name}_flip_both.npz'
            np.savez(output_path_both, arr=encoded_flipped_both.cpu().detach().numpy())
            print(f"   + Flip Both: {output_path_both}")

if __name__ == '__main__':
    main()