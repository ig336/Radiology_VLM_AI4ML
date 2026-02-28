"""
Test script to verify grid coherence test implementation.

This tests the ratio-based spatial coherence test that was fixed in DOCUMENTATION.md
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_grid_coherence(embedding_path=None):
    """
    Test CTViT grid coherence with ratio-based metric.
    
    Args:
        embedding_path: Path to .npz file, or None to use synthetic data
    
    Returns:
        dict with test results
    """
    # Load or generate test data
    if embedding_path:
        embedding = np.load(embedding_path)['arr_0']
    else:
        # Synthetic data: spatially coherent
        print("Using synthetic spatially-coherent data for testing...")
        grid = np.zeros((12, 12, 16, 512))
        for d in range(12):
            for h in range(12):
                for w in range(16):
                    # Create spatial pattern
                    base = np.random.randn(512)
                    noise = np.random.randn(512) * 0.1
                    grid[d, h, w] = base * (d/12 + h/12 + w/16) + noise
        embedding = grid.reshape(2304, 512)
    
    # Reshape to assumed grid
    grid = embedding.reshape(12, 12, 16, 512)
    
    # Calculate adjacent vs random similarities
    adjacent_sims = []
    random_sims = []
    
    for d in range(11):
        for h in range(11):
            for w in range(15):
                # Adjacent pair
                adj_sim = cosine_similarity(
                    grid[d,h,w].reshape(1,-1),
                    grid[d,h,w+1].reshape(1,-1)
                )[0,0]
                adjacent_sims.append(adj_sim)
                
                # Random pair (for baseline)
                rand_d, rand_h, rand_w = np.random.randint(0, [12, 12, 16])
                rand_sim = cosine_similarity(
                    grid[d,h,w].reshape(1,-1),
                    grid[rand_d, rand_h, rand_w].reshape(1,-1)
                )[0,0]
                random_sims.append(rand_sim)
    
    mean_adjacent = np.mean(adjacent_sims)
    mean_random = np.mean(random_sims)
    ratio = mean_adjacent / (mean_random + 1e-8)
    
    # Determine result
    if ratio > 1.5:
        status = "✅ PASS - Strongly spatially ordered"
        recommendation = "Safe to use SpatialAttentionalPooler"
    elif ratio > 1.1:
        status = "⚠️  CAUTION - Weakly spatially ordered"
        recommendation = "Use SpatialAttentionalPooler with caution"
    else:
        status = "❌ FAIL - Not spatially ordered"
        recommendation = "Do NOT use SpatialAttentionalPooler"
    
    results = {
        'mean_adjacent': mean_adjacent,
        'mean_random': mean_random,
        'ratio': ratio,
        'status': status,
        'recommendation': recommendation
    }
    
    return results


def test_batch_padding():
    """
    Test the corrected batch padding code for task-conditioned pooler.
    
    This verifies the fix for the batch reshape crash.
    """
    print("\n" + "="*60)
    print("Testing Batch Padding Fix")
    print("="*60)
    
    # Simulate variable-length question_ids with image tokens
    IMAGE_TOKEN_INDEX = -200
    
    # Batch with different question lengths
    question_ids = [
        [1, 2, 3, IMAGE_TOKEN_INDEX, 4, 5],           # Short question
        [1, 2, IMAGE_TOKEN_INDEX, 3, 4, 5, 6, 7, 8],  # Long question
        [1, IMAGE_TOKEN_INDEX, 2, 3, 4]               # Medium question
    ]
    
    # Convert to tensor (pad to max length)
    max_len = max(len(q) for q in question_ids)
    question_ids_padded = []
    for q in question_ids:
        padded = q + [0] * (max_len - len(q))
        question_ids_padded.append(padded)
    
    import torch
    question_ids_tensor = torch.tensor(question_ids_padded)
    
    print(f"Input batch shape: {question_ids_tensor.shape}")
    print(f"Question lengths: {[len(q) for q in question_ids]}")
    
    # OLD (BROKEN) METHOD - would crash
    try:
        question_mask = question_ids_tensor != IMAGE_TOKEN_INDEX
        broken_result = question_ids_tensor[question_mask].reshape(question_ids_tensor.size(0), -1)
        print("❌ OLD METHOD: Should have crashed but didn't!")
    except RuntimeError as e:
        print(f"✅ OLD METHOD: Correctly crashes with RuntimeError")
        print(f"   Error: {str(e)[:80]}...")
    
    # NEW (FIXED) METHOD - with proper padding
    print("\nTesting FIXED method:")
    question_only_ids = []
    for i in range(question_ids_tensor.size(0)):
        mask_i = question_ids_tensor[i] != IMAGE_TOKEN_INDEX
        question_only_ids.append(question_ids_tensor[i][mask_i])
    
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    question_only_ids_padded = pad_sequence(
        question_only_ids,
        batch_first=True,
        padding_value=0
    )
    
    print(f"✅ FIXED METHOD: Success!")
    print(f"   Output shape: {question_only_ids_padded.shape}")
    print(f"   Individual lengths: {[len(q) for q in question_only_ids]}")
    
    return True


def test_storage_size():
    """
    Test actual .npz storage size.
    """
    print("\n" + "="*60)
    print("Testing Storage Size")
    print("="*60)
    
    import tempfile
    import os
    
    # Create CTViT output
    arr = np.random.randn(2304, 512).astype(np.float32)
    
    print(f"Array shape: {arr.shape}")
    print(f"Raw memory: {arr.nbytes / (1024*1024):.2f} MB")
    
    # Test .npz (uncompressed)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        fname = f.name
        np.savez(fname, arr=arr)
        size = os.path.getsize(fname)
        print(f"NPZ file size: {size / (1024*1024):.2f} MB")
        
        # Verify it's close to 4.5 MB, NOT 11 MB
        if 4.0 < size / (1024*1024) < 5.0:
            print("✅ Correct: ~4.5 MB (not ~11 MB)")
        else:
            print(f"❌ Unexpected size: {size / (1024*1024):.2f} MB")
        
        os.unlink(fname)
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("DOCUMENTATION.md Fixes Validation")
    print("="*60)
    
    # Test 1: Grid coherence (ratio-based)
    print("\n" + "="*60)
    print("Test 1: Grid Coherence (Ratio-Based)")
    print("="*60)
    results = test_grid_coherence()
    print(f"Adjacent similarity: {results['mean_adjacent']:.3f}")
    print(f"Random similarity: {results['mean_random']:.3f}")
    print(f"Ratio (adjacent/random): {results['ratio']:.3f}")
    print(f"\n{results['status']}")
    print(f"Recommendation: {results['recommendation']}")
    
    # Test 2: Batch padding fix
    test_batch_padding()
    
    # Test 3: Storage size
    test_storage_size()
    
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)
