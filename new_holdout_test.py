#!/usr/bin/env python
"""
Test all models with a NEW hold-out set (seed=123 instead of 42)
to check generalizability of results
"""

import numpy as np
import torch
import tensorflow as tf
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from yamnet_cost_sensitive import CostSensitiveYAMNet, augment_audio_simple
from panns_simple_gpu import (
    SimplePANNs, 
    train_with_cost_sensitivity,
    evaluate_model
)

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# NEW SEED FOR DIFFERENT SPLIT
RANDOM_SEED = 123
print("="*80)
print("TESTING GENERALIZABILITY WITH NEW HOLD-OUT SET")
print(f"Using random seed: {RANDOM_SEED} (instead of 42)")
print("="*80)

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_and_split_data():
    """Load data and create NEW train/val/test split"""
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
    
    # NEW SPLIT with seed=123
    poi_indices = np.where(y_all == 1)[0]
    nopoi_indices = np.where(y_all == 0)[0]
    
    np.random.seed(RANDOM_SEED)  # NEW SEED!
    poi_indices = np.random.permutation(poi_indices)
    nopoi_indices = np.random.permutation(nopoi_indices)
    
    # Same proportions but different samples
    test_indices = np.concatenate([poi_indices[:5], nopoi_indices[:20]])
    val_indices = np.concatenate([poi_indices[5:10], nopoi_indices[20:40]])
    train_indices = np.concatenate([poi_indices[10:], nopoi_indices[40:]])
    
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    
    print(f"\nNEW Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Test set: {np.sum(y_test==1)} POI, {np.sum(y_test==0)} Non-POI")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_yamnet(X_train, y_train, X_val, y_val, X_test, y_test, ratios):
    """Test YAMNet with different cost ratios"""
    results = []
    
    for fn_cost, fp_cost in ratios:
        ratio = fn_cost / fp_cost
        print(f"\n{'='*70}")
        print(f"YAMNet - Ratio {ratio:.1f}:1")
        print(f"{'='*70}")
        
        # Create model
        model = CostSensitiveYAMNet(
            false_negative_cost=fn_cost,
            false_positive_cost=fp_cost
        )
        
        # Augment training data
        X_train_aug, y_train_aug = [], []
        for x, y in zip(X_train, y_train):
            if y == 1:  # POI
                augmented = augment_audio_simple(x, n_augmentations=8)
                X_train_aug.extend(augmented)
                y_train_aug.extend([1] * len(augmented))
            else:
                X_train_aug.append(x)
                y_train_aug.append(0)
        
        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)
        
        # Extract embeddings
        print("Extracting embeddings...")
        X_train_emb = model.extract_embeddings(X_train_aug, sr=8000)
        X_val_emb = model.extract_embeddings(X_val, sr=8000)
        X_test_emb = model.extract_embeddings(X_test, sr=8000)
        
        # Train
        print("Training...")
        history, best_threshold = model.train(
            X_train_emb, y_train_aug, X_val_emb, y_val,
            epochs=30, batch_size=16, learning_rate=0.001
        )
        
        # Evaluate
        metrics = model.evaluate_with_costs(X_test_emb, y_test, threshold=best_threshold)
        
        print(f"\nResults:")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        
        results.append({
            'model': 'YAMNet',
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'metrics': metrics,
            'threshold': best_threshold
        })
    
    return results


def test_panns(X_train, y_train, X_val, y_val, X_test, y_test, ratios):
    """Test PANNs with different cost ratios"""
    results = []
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    for fn_cost, fp_cost in ratios:
        ratio = fn_cost / fp_cost
        print(f"\n{'='*70}")
        print(f"PANNs - Ratio {ratio:.1f}:1")
        print(f"{'='*70}")
        
        # Augment training data
        X_train_aug, y_train_aug = [], []
        for x, y in zip(X_train, y_train):
            if y == 1:  # POI
                # Simple augmentation
                for _ in range(8):
                    aug = x.copy()
                    # Add noise
                    aug += np.random.normal(0, 0.005, len(aug))
                    # Time shift
                    shift = np.random.randint(-800, 800)
                    aug = np.roll(aug, shift)
                    X_train_aug.append(aug)
                    y_train_aug.append(1)
            else:
                X_train_aug.append(x)
                y_train_aug.append(0)
        
        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)
        
        print(f"After augmentation: {len(X_train_aug)} samples")
        
        # Create and train model
        model = SimplePANNs(n_mels=64, n_classes=2)
        
        model, threshold = train_with_cost_sensitivity(
            model, X_train_aug, y_train_aug, X_val, y_val,
            fn_cost=fn_cost, fp_cost=fp_cost,
            epochs=30, batch_size=32, device=device
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, threshold, 
                                fn_cost=fn_cost, fp_cost=fp_cost, device=device)
        
        print(f"\nResults:")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        
        results.append({
            'model': 'PANNs',
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'metrics': metrics,
            'threshold': threshold
        })
    
    return results


def create_comparison_visualization(all_results):
    """Create visualization comparing old and new holdout results"""
    
    # Load old results for comparison
    with open('results/yamnet_cost_sensitive_results.json', 'r') as f:
        old_yamnet = json.load(f)
    with open('results/panns_gpu_results.json', 'r') as f:
        old_panns = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Sensitivity comparison
    ax = axes[0, 0]
    
    # New results
    yamnet_new = [r for r in all_results if r['model'] == 'YAMNet']
    panns_new = [r for r in all_results if r['model'] == 'PANNs']
    
    # Plot new results
    ratios_y = [r['fn_cost']/r['fp_cost'] for r in yamnet_new]
    sens_y = [r['metrics']['sensitivity']*100 for r in yamnet_new]
    ax.plot(ratios_y, sens_y, 'o-', label='YAMNet (New)', color='orange', linewidth=2)
    
    ratios_p = [r['fn_cost']/r['fp_cost'] for r in panns_new]
    sens_p = [r['metrics']['sensitivity']*100 for r in panns_new]
    ax.plot(ratios_p, sens_p, 's-', label='PANNs (New)', color='blue', linewidth=2)
    
    # Plot old results
    old_ratios_y = [e['fn_cost']/e['fp_cost'] for e in old_yamnet['experiments']]
    old_sens_y = [e['metrics']['sensitivity']*100 for e in old_yamnet['experiments']]
    ax.plot(old_ratios_y, old_sens_y, 'o--', label='YAMNet (Old)', 
           color='orange', alpha=0.5, linewidth=1)
    
    old_ratios_p = [e['fn_cost']/e['fp_cost'] for e in old_panns['experiments']]
    old_sens_p = [e['metrics']['sensitivity']*100 for e in old_panns['experiments']]
    ax.plot(old_ratios_p, old_sens_p, 's--', label='PANNs (Old)', 
           color='blue', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Cost Ratio (FN:FP)')
    ax.set_ylabel('Sensitivity (%)')
    ax.set_title('Sensitivity: Old vs New Hold-out')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: NPV comparison
    ax = axes[0, 1]
    
    npv_y = [r['metrics']['npv']*100 for r in yamnet_new]
    ax.plot(ratios_y, npv_y, 'o-', label='YAMNet (New)', color='orange', linewidth=2)
    
    npv_p = [r['metrics']['npv']*100 for r in panns_new]
    ax.plot(ratios_p, npv_p, 's-', label='PANNs (New)', color='blue', linewidth=2)
    
    old_npv_y = [e['metrics']['npv']*100 for e in old_yamnet['experiments']]
    ax.plot(old_ratios_y, old_npv_y, 'o--', label='YAMNet (Old)', 
           color='orange', alpha=0.5, linewidth=1)
    
    old_npv_p = [e['metrics']['npv']*100 for e in old_panns['experiments']]
    ax.plot(old_ratios_p, old_npv_p, 's--', label='PANNs (Old)', 
           color='blue', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Cost Ratio (FN:FP)')
    ax.set_ylabel('NPV (%)')
    ax.set_title('NPV: Old vs New Hold-out')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cost comparison
    ax = axes[0, 2]
    
    cost_y = [r['metrics']['total_cost'] for r in yamnet_new]
    ax.plot(ratios_y, cost_y, 'o-', label='YAMNet (New)', color='orange', linewidth=2)
    
    cost_p = [r['metrics']['total_cost'] for r in panns_new]
    ax.plot(ratios_p, cost_p, 's-', label='PANNs (New)', color='blue', linewidth=2)
    
    old_cost_y = [e['metrics']['total_cost'] for e in old_yamnet['experiments']]
    ax.plot(old_ratios_y, old_cost_y, 'o--', label='YAMNet (Old)', 
           color='orange', alpha=0.5, linewidth=1)
    
    old_cost_p = [e['metrics']['total_cost'] for e in old_panns['experiments']]
    ax.plot(old_ratios_p, old_cost_p, 's--', label='PANNs (Old)', 
           color='blue', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Cost Ratio (FN:FP)')
    ax.set_ylabel('Total Cost')
    ax.set_title('Total Cost: Old vs New Hold-out')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Confusion matrices for key configurations
    key_configs = [
        ('PANNs', 2.0, 1.0),
        ('PANNs', 2.5, 1.0),
        ('YAMNet', 2.0, 1.0)
    ]
    
    for idx, (model_name, fn_cost, fp_cost) in enumerate(key_configs):
        ax = axes[1, idx]
        
        # Find matching result
        result = next((r for r in all_results 
                      if r['model'] == model_name and 
                      r['fn_cost'] == fn_cost and 
                      r['fp_cost'] == fp_cost), None)
        
        if result:
            cm = np.array(result['metrics']['confusion_matrix'])
            cm_percent = cm.astype('float') / cm.sum() * 100
            
            labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                               for j in range(cm.shape[1])] 
                              for i in range(cm.shape[0])])
            
            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                       cbar=False, ax=ax,
                       xticklabels=['Pred: No POI', 'Pred: POI'],
                       yticklabels=['True: No POI', 'True: POI'])
            
            ratio = fn_cost / fp_cost
            metrics = result['metrics']
            ax.set_title(f'{model_name} {ratio:.1f}:1 (New Hold-out)\n' + 
                        f'Sens: {metrics["sensitivity"]:.0%}, ' +
                        f'NPV: {metrics["npv"]:.1%}',
                        fontsize=10)
    
    plt.suptitle('Generalizability Test: New Hold-out Set (seed=123)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/new_holdout_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comparison to results/new_holdout_comparison.png")


def main():
    # Load data with new split
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    
    # Define test ratios
    ratios = [
        (2.0, 1.0),   # 2:1
        (2.5, 1.0),   # 2.5:1
        (3.0, 1.0),   # 3:1
        (5.0, 1.0),   # 5:1
    ]
    
    all_results = []
    
    # Test YAMNet
    print("\n" + "="*80)
    print("TESTING YAMNET WITH NEW HOLD-OUT")
    print("="*80)
    yamnet_results = test_yamnet(X_train, y_train, X_val, y_val, X_test, y_test, ratios)
    all_results.extend(yamnet_results)
    
    # Test PANNs
    print("\n" + "="*80)
    print("TESTING PANNs WITH NEW HOLD-OUT")
    print("="*80)
    panns_results = test_panns(X_train, y_train, X_val, y_val, X_test, y_test, ratios)
    all_results.extend(panns_results)
    
    # Save results
    results_dict = {
        'random_seed': RANDOM_SEED,
        'description': 'Generalizability test with new hold-out set',
        'experiments': all_results
    }
    
    with open('results/new_holdout_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n✓ Saved results to results/new_holdout_results.json")
    
    # Create visualization
    create_comparison_visualization(all_results)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: NEW HOLD-OUT RESULTS")
    print("="*80)
    
    print(f"\n{'Model':<10} {'Ratio':<8} {'Sens':<8} {'Spec':<8} {'NPV':<8} {'Cost':<8}")
    print("-"*60)
    
    for r in all_results:
        ratio = r['fn_cost'] / r['fp_cost']
        m = r['metrics']
        print(f"{r['model']:<10} {ratio:<8.1f} {m['sensitivity']:<8.1%} "
              f"{m['specificity']:<8.1%} {m['npv']:<8.1%} {m['total_cost']:<8.0f}")
    
    # Check for consistent high performers
    print("\n" + "="*80)
    print("GENERALIZABILITY ASSESSMENT")
    print("="*80)
    
    # Find configurations with 100% sensitivity
    perfect_sens = [r for r in all_results if r['metrics']['sensitivity'] == 1.0]
    if perfect_sens:
        print("\nConfigurations with 100% sensitivity on NEW hold-out:")
        for r in perfect_sens:
            ratio = r['fn_cost'] / r['fp_cost']
            print(f"  - {r['model']} {ratio:.1f}:1 (Cost: {r['metrics']['total_cost']:.0f})")
    else:
        print("\nNo configurations achieved 100% sensitivity on new hold-out")
    
    # Find best NPV
    best_npv = max(all_results, key=lambda x: x['metrics']['npv'])
    print(f"\nBest NPV on NEW hold-out:")
    print(f"  - {best_npv['model']} {best_npv['fn_cost']/best_npv['fp_cost']:.1f}:1 "
          f"with NPV={best_npv['metrics']['npv']:.1%}")


if __name__ == '__main__':
    main()