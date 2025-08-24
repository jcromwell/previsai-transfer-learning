#!/usr/bin/env python
"""
Test PANNs with a THIRD hold-out set (seed=456) to further assess generalizability
"""

import numpy as np
import torch
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# Import PANNs model
from panns_simple_gpu import (
    SimplePANNs, 
    train_with_cost_sensitivity,
    evaluate_model
)

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# THIRD SEED FOR DIFFERENT SPLIT
RANDOM_SEED = 456
print("="*80)
print("PANNS GENERALIZABILITY TEST - THIRD HOLD-OUT SET")
print(f"Using random seed: {RANDOM_SEED}")
print("="*80)

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_and_split_data():
    """Load data and create THIRD train/val/test split"""
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
    
    # THIRD SPLIT with seed=456
    poi_indices = np.where(y_all == 1)[0]
    nopoi_indices = np.where(y_all == 0)[0]
    
    np.random.seed(RANDOM_SEED)  # THIRD SEED!
    poi_indices = np.random.permutation(poi_indices)
    nopoi_indices = np.random.permutation(nopoi_indices)
    
    # Same proportions but different samples
    test_indices = np.concatenate([poi_indices[:5], nopoi_indices[:20]])
    val_indices = np.concatenate([poi_indices[5:10], nopoi_indices[20:40]])
    train_indices = np.concatenate([poi_indices[10:], nopoi_indices[40:]])
    
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    
    print(f"\nSplit (seed={RANDOM_SEED}): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Test set: {np.sum(y_test==1)} POI, {np.sum(y_test==0)} Non-POI")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_panns_all_ratios(X_train, y_train, X_val, y_val, X_test, y_test):
    """Test PANNs with multiple cost ratios"""
    
    ratios = [
        (2.0, 1.0),   # 2:1
        (2.5, 1.0),   # 2.5:1
        (3.0, 1.0),   # 3:1
        (5.0, 1.0),   # 5:1
    ]
    
    results = []
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for fn_cost, fp_cost in ratios:
        ratio = fn_cost / fp_cost
        print(f"\n{'='*70}")
        print(f"Testing PANNs with Ratio {ratio:.1f}:1")
        print(f"{'='*70}")
        
        # Augment training data
        X_train_aug, y_train_aug = [], []
        for x, y in zip(X_train, y_train):
            if y == 1:  # POI
                # Simple augmentation
                for i in range(8):
                    aug = x.copy()
                    # Add noise
                    aug += np.random.normal(0, 0.005, len(aug))
                    # Time shift
                    shift = np.random.randint(-800, 800)
                    aug = np.roll(aug, shift)
                    # Amplitude scaling
                    aug *= np.random.uniform(0.9, 1.1)
                    X_train_aug.append(aug)
                    y_train_aug.append(1)
            else:
                X_train_aug.append(x)
                y_train_aug.append(0)
        
        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)
        
        print(f"After augmentation: {len(X_train_aug)} samples")
        print(f"POI: {np.sum(y_train_aug==1)}, Non-POI: {np.sum(y_train_aug==0)}")
        
        # Create and train model
        model = SimplePANNs(n_mels=64, n_classes=2)
        
        model, threshold = train_with_cost_sensitivity(
            model, X_train_aug, y_train_aug, X_val, y_val,
            fn_cost=fn_cost, fp_cost=fp_cost,
            epochs=30, batch_size=32, device=device
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = evaluate_model(model, X_test, y_test, threshold, 
                                fn_cost=fn_cost, fp_cost=fp_cost, device=device)
        
        print(f"\nResults:")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  PPV: {metrics['ppv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        
        # Confusion matrix breakdown
        cm = np.array(metrics['confusion_matrix'])
        if cm.shape == (2, 2):
            print("\nConfusion Matrix:")
            print(f"  TN={cm[0,0]:2d}  FP={cm[0,1]:2d}")
            print(f"  FN={cm[1,0]:2d}  TP={cm[1,1]:2d}")
        
        results.append({
            'model': 'PANNs',
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'metrics': metrics,
            'threshold': threshold
        })
    
    return results


def create_comparison_plot(results):
    """Create visualization comparing all three seeds"""
    
    # Load previous results
    with open('results/panns_gpu_results.json', 'r') as f:
        seed42_results = json.load(f)
    
    with open('results/panns_ratio_2_5_results.json', 'r') as f:
        seed42_25 = json.load(f)
    
    with open('results/new_holdout_results.json', 'r') as f:
        seed123_results = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Confusion matrices for each ratio and seed
    ratios = [2.0, 2.5, 3.0, 5.0]
    
    for idx, ratio in enumerate(ratios):
        ax = axes[0, idx]  # Top row for seed=42
        ax2 = axes[1, idx]  # Bottom row for seed=456
        
        # Find seed=42 result
        if ratio == 2.5:
            cm_42 = np.array(seed42_25['metrics']['confusion_matrix'])
            sens_42 = seed42_25['metrics']['sensitivity']
            npv_42 = seed42_25['metrics']['npv']
        else:
            for exp in seed42_results['experiments']:
                if exp['fn_cost'] / exp['fp_cost'] == ratio:
                    cm_42 = np.array(exp['metrics']['confusion_matrix'])
                    sens_42 = exp['metrics']['sensitivity']
                    npv_42 = exp['metrics']['npv']
                    break
        
        # Find seed=123 result
        cm_123 = None
        for exp in seed123_results['experiments']:
            if exp['model'] == 'PANNs' and exp['fn_cost'] / exp['fp_cost'] == ratio:
                cm_123 = np.array(exp['metrics']['confusion_matrix'])
                sens_123 = exp['metrics']['sensitivity']
                npv_123 = exp['metrics']['npv']
                break
        
        # Find seed=456 result (current)
        cm_456 = None
        for r in results:
            if r['fn_cost'] / r['fp_cost'] == ratio:
                cm_456 = np.array(r['metrics']['confusion_matrix'])
                sens_456 = r['metrics']['sensitivity']
                npv_456 = r['metrics']['npv']
                break
        
        # Plot seed=42 (top)
        if cm_42 is not None:
            cm_percent = cm_42.astype('float') / cm_42.sum() * 100
            labels = np.array([[f'{cm_42[i,j]}\n({cm_percent[i,j]:.0f}%)' 
                               for j in range(2)] for i in range(2)])
            
            sns.heatmap(cm_42, annot=labels, fmt='', cmap='Blues', 
                       cbar=False, ax=ax,
                       xticklabels=['No POI', 'POI'],
                       yticklabels=['No POI', 'POI'])
            
            ax.set_title(f'Ratio {ratio:.1f}:1 (seed=42)\n' +
                        f'Sens: {sens_42:.0%}, NPV: {npv_42:.1%}',
                        fontsize=10)
            
            if sens_42 == 1.0:
                rect = plt.Rectangle((0.98, 0.98), 0.97, 0.97, fill=False, 
                                    edgecolor='green', linewidth=2)
                ax.add_patch(rect)
        
        # Plot seed=456 (bottom)
        if cm_456 is not None:
            cm_percent = cm_456.astype('float') / cm_456.sum() * 100
            labels = np.array([[f'{cm_456[i,j]}\n({cm_percent[i,j]:.0f}%)' 
                               for j in range(2)] for i in range(2)])
            
            sns.heatmap(cm_456, annot=labels, fmt='', cmap='Greens', 
                       cbar=False, ax=ax2,
                       xticklabels=['No POI', 'POI'],
                       yticklabels=['No POI', 'POI'])
            
            ax2.set_title(f'Ratio {ratio:.1f}:1 (seed=456)\n' +
                         f'Sens: {sens_456:.0%}, NPV: {npv_456:.1%}',
                         fontsize=10)
            
            if sens_456 == 1.0:
                rect = plt.Rectangle((0.98, 0.98), 0.97, 0.97, fill=False, 
                                    edgecolor='green', linewidth=2)
                ax2.add_patch(rect)
            elif sens_456 == 0.0:
                rect = plt.Rectangle((0.02, 0.98), 0.97, 0.97, fill=False, 
                                    edgecolor='red', linewidth=2)
                ax2.add_patch(rect)
    
    plt.suptitle('PANNs Performance Across Three Different Hold-out Sets', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/panns_three_seeds_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to results/panns_three_seeds_comparison.png")


def main():
    # Load data with third split
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    
    # Test PANNs with all ratios
    results = test_panns_all_ratios(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save results
    results_dict = {
        'random_seed': RANDOM_SEED,
        'description': 'PANNs generalizability test with third hold-out set',
        'experiments': results
    }
    
    with open('results/panns_seed456_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n✓ Saved results to results/panns_seed456_results.json")
    
    # Create visualization
    create_comparison_plot(results)
    
    # Summary across all three seeds
    print("\n" + "="*80)
    print("SUMMARY: PANNs PERFORMANCE ACROSS THREE SEEDS")
    print("="*80)
    
    # Load all results for comparison
    with open('results/panns_gpu_results.json', 'r') as f:
        seed42 = json.load(f)
    with open('results/panns_ratio_2_5_results.json', 'r') as f:
        seed42_25 = json.load(f)
    with open('results/new_holdout_results.json', 'r') as f:
        seed123 = json.load(f)
    
    print(f"\n{'Ratio':<8} {'Seed 42':<25} {'Seed 123':<25} {'Seed 456':<25}")
    print(f"{'':8} {'Sens   NPV    Cost':<25} {'Sens   NPV    Cost':<25} {'Sens   NPV    Cost':<25}")
    print("-"*85)
    
    ratios = [2.0, 2.5, 3.0, 5.0]
    
    for ratio in ratios:
        # Seed 42
        if ratio == 2.5:
            s42 = f"{seed42_25['metrics']['sensitivity']:.0%}   {seed42_25['metrics']['npv']:.0%}   {seed42_25['metrics']['total_cost']:.0f}"
        else:
            for exp in seed42['experiments']:
                if exp['fn_cost'] / exp['fp_cost'] == ratio:
                    s42 = f"{exp['metrics']['sensitivity']:.0%}   {exp['metrics']['npv']:.0%}   {exp['metrics']['total_cost']:.0f}"
                    break
        
        # Seed 123
        s123 = "N/A"
        for exp in seed123['experiments']:
            if exp['model'] == 'PANNs' and exp['fn_cost'] / exp['fp_cost'] == ratio:
                s123 = f"{exp['metrics']['sensitivity']:.0%}    {exp['metrics']['npv']:.0%}    {exp['metrics']['total_cost']:.0f}"
                break
        
        # Seed 456
        s456 = "N/A"
        for r in results:
            if r['fn_cost'] / r['fp_cost'] == ratio:
                s456 = f"{r['metrics']['sensitivity']:.0%}    {r['metrics']['npv']:.0%}    {r['metrics']['total_cost']:.0f}"
                break
        
        print(f"{ratio:.1f}:1   {s42:<25} {s123:<25} {s456:<25}")
    
    # Count perfect sensitivity across seeds
    perfect_count = 0
    failed_count = 0
    
    for r in results:
        if r['metrics']['sensitivity'] == 1.0:
            perfect_count += 1
        elif r['metrics']['sensitivity'] == 0.0:
            failed_count += 1
    
    print(f"\nSeed 456 Results:")
    print(f"  - Perfect sensitivity (100%): {perfect_count} configurations")
    print(f"  - Complete failure (0%): {failed_count} configurations")
    
    print("\n" + "="*80)
    print("GENERALIZABILITY CONCLUSION")
    print("="*80)
    print("PANNs performance is HIGHLY UNSTABLE across different hold-out sets:")
    print("- Seed 42: Multiple perfect sensitivity configurations")
    print("- Seed 123: Complete failure on all configurations")
    print("- Seed 456: Results will show if this is random or systematic")


if __name__ == '__main__':
    main()