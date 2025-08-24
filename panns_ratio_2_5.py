#!/usr/bin/env python
"""
Test PANNs with 2.5:1 cost ratio
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Import functions from the simplified PANNs implementation
from panns_simple_gpu import (
    SimplePANNs, 
    extract_mel_spectrogram,
    prepare_batch_gpu,
    train_with_cost_sensitivity,
    evaluate_model,
    augment_audio_simple
)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


def main():
    print("="*70)
    print("PANNs WITH 2.5:1 COST RATIO")
    print("="*70)
    
    # Check GPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'mps':
        print("✓ MPS GPU acceleration enabled")
    
    # Load data
    print("\nLoading audio data...")
    DATA_DIR = str(Path.cwd() / 'raw-audio')
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=8000)
    file_dict = loader.get_file_list()
    
    # Load samples
    X_all, y_all = [], []
    
    for file_path in file_dict['poi']:
        audio, sr = loader.load_audio_file(file_path)
        if audio is not None:
            audio = audio[:20*sr]
            if len(audio) < 20*sr:
                audio = np.pad(audio, (0, 20*sr - len(audio)), 'constant')
            X_all.append(audio)
            y_all.append(1)
    
    for file_path in file_dict['nopoi']:
        audio, sr = loader.load_audio_file(file_path)
        if audio is not None:
            audio = audio[:20*sr]
            if len(audio) < 20*sr:
                audio = np.pad(audio, (0, 20*sr - len(audio)), 'constant')
            X_all.append(audio)
            y_all.append(0)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"Total: {len(X_all)} samples ({np.sum(y_all==1)} POI, {np.sum(y_all==0)} Non-POI)")
    
    # Split data (same as before for consistency)
    poi_indices = np.where(y_all == 1)[0]
    nopoi_indices = np.where(y_all == 0)[0]
    
    np.random.seed(42)
    poi_indices = np.random.permutation(poi_indices)
    nopoi_indices = np.random.permutation(nopoi_indices)
    
    test_indices = np.concatenate([poi_indices[:5], nopoi_indices[:20]])
    val_indices = np.concatenate([poi_indices[5:10], nopoi_indices[20:40]])
    train_indices = np.concatenate([poi_indices[10:], nopoi_indices[40:]])
    
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Augment training data
    print("\nAugmenting training data...")
    X_train_aug, y_train_aug = [], []
    for x, y in zip(X_train, y_train):
        if y == 1:  # POI
            augmented = augment_audio_simple(x, n_aug=8)
            X_train_aug.extend(augmented)
            y_train_aug.extend([1] * len(augmented))
        else:
            X_train_aug.append(x)
            y_train_aug.append(0)
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    print(f"After augmentation: {len(X_train_aug)} samples")
    print(f"POI: {np.sum(y_train_aug==1)}, Non-POI: {np.sum(y_train_aug==0)}")
    
    # Test 2.5:1 ratio
    print(f"\n{'='*70}")
    print(f"TRAINING WITH 2.5:1 COST RATIO")
    print('='*70)
    
    # Create model
    model = SimplePANNs(n_mels=64, n_classes=2)
    
    # Train with 2.5:1 ratio
    model, threshold = train_with_cost_sensitivity(
        model, X_train_aug, y_train_aug, X_val, y_val,
        fn_cost=2.5, fp_cost=1.0,
        epochs=30, batch_size=32, device=device
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test, threshold, 
                            fn_cost=2.5, fp_cost=1.0, device=device)
    
    print(f"\nResults for 2.5:1 Ratio:")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
    print(f"  NPV: {metrics['npv']:.1%}")
    print(f"  PPV: {metrics['ppv']:.1%}")
    print(f"  Total Cost: {metrics['total_cost']:.1f}")
    
    # Confusion matrix analysis
    cm = np.array(metrics['confusion_matrix'])
    if cm.shape == (2, 2):
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 No POI    POI")
        print(f"Actual No POI      {cm[0,0]:3d}     {cm[0,1]:3d}")
        print(f"Actual POI         {cm[1,0]:3d}     {cm[1,1]:3d}")
        
        print("\nBreakdown:")
        print(f"• True Negatives: {cm[0,0]} out of 20 non-POI correctly identified")
        print(f"• False Positives: {cm[0,1]} non-POI incorrectly flagged as POI")
        print(f"• False Negatives: {cm[1,0]} POI cases missed")
        print(f"• True Positives: {cm[1,1]} out of 5 POI correctly detected")
    
    # Save results
    result = {
        'name': 'Ratio 2.5:1',
        'fn_cost': 2.5,
        'fp_cost': 1.0,
        'metrics': metrics,
        'threshold': threshold
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/panns_ratio_2_5_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n✓ Results saved to results/panns_ratio_2_5_results.json")
    
    # Visualize confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    # Create labels
    labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                       for j in range(cm.shape[1])] 
                      for i in range(cm.shape[0])])
    
    # Plot
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
               cbar=False, ax=ax,
               xticklabels=['Pred: No POI', 'Pred: POI'],
               yticklabels=['True: No POI', 'True: POI'])
    
    ax.set_title(f"PANNs Ratio 2.5:1\n" + 
                f"Sens: {metrics['sensitivity']:.0%}, " +
                f"Spec: {metrics['specificity']:.0%}, " +
                f"NPV: {metrics['npv']:.1%}",
                fontsize=12, fontweight='bold')
    
    ax.text(0.5, -0.15, 
           f"Threshold: {threshold:.2f}, Total Cost: {metrics['total_cost']:.0f}",
           transform=ax.transAxes, ha='center', fontsize=9)
    
    # Highlight if perfect detection
    if metrics['sensitivity'] == 1.0:
        rect = plt.Rectangle((1, 1), 0.98, 0.98, fill=False, 
                            edgecolor='green', linewidth=3)
        ax.add_patch(rect)
        ax.text(0.5, -0.25, '⭐ Perfect Sensitivity!', 
               transform=ax.transAxes, ha='center', 
               fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/panns_ratio_2_5_confusion.png', dpi=150, bbox_inches='tight')
    print("✓ Saved confusion matrix to results/panns_ratio_2_5_confusion.png")
    
    # Compare with other ratios
    print("\n" + "="*70)
    print("COMPARISON WITH OTHER RATIOS")
    print("="*70)
    
    # Load previous results
    with open('results/panns_gpu_results.json', 'r') as f:
        prev_results = json.load(f)
    
    print(f"\n{'Ratio':<10} {'Sens':<8} {'Spec':<8} {'NPV':<8} {'PPV':<8} {'Cost':<8}")
    print("-" * 60)
    
    # Previous results
    for exp in prev_results['experiments']:
        ratio = exp['fn_cost'] / exp['fp_cost']
        m = exp['metrics']
        print(f"{ratio:<10.1f} {m['sensitivity']:<8.1%} {m['specificity']:<8.1%} "
              f"{m['npv']:<8.1%} {m['ppv']:<8.1%} {m['total_cost']:<8.0f}")
    
    # New 2.5:1 result
    print(f"{2.5:<10.1f} {metrics['sensitivity']:<8.1%} {metrics['specificity']:<8.1%} "
          f"{metrics['npv']:<8.1%} {metrics['ppv']:<8.1%} {metrics['total_cost']:<8.0f}")
    
    print("\n" + "="*70)
    print("CLINICAL INTERPRETATION")
    print("="*70)
    
    if metrics['sensitivity'] >= 0.8:
        print("✓ HIGH SENSITIVITY: Good POI detection capability")
    if metrics['npv'] >= 0.9:
        print("✓ HIGH NPV: High confidence when clearing patients")
    if metrics['specificity'] >= 0.7:
        print("✓ GOOD SPECIFICITY: Reasonable false positive rate")
    
    print(f"\nRecommendation based on 2.5:1 ratio performance:")
    if metrics['sensitivity'] == 1.0 and metrics['npv'] == 1.0:
        print("EXCELLENT for screening - perfect detection with no missed cases!")
    elif metrics['sensitivity'] >= 0.8:
        print("Good for screening - catches most POI cases")
    else:
        print("May miss too many POI cases for primary screening")


if __name__ == '__main__':
    main()