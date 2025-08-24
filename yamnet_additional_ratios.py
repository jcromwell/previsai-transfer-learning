#!/usr/bin/env python
"""
Test additional cost ratios (2.5:1 and 3:1) for YAMNet
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader
from yamnet_cost_sensitive import CostSensitiveYAMNet, augment_audio_simple

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_single_experiment(X_train, y_train, X_val, y_val, X_test, y_test, 
                         fn_cost, fp_cost, name):
    """Run a single cost-sensitive experiment"""
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {name}")
    print(f"FN Cost: {fn_cost}, FP Cost: {fp_cost}")
    print("="*70)
    
    # Create model with specific costs
    model = CostSensitiveYAMNet(
        false_negative_cost=fn_cost,
        false_positive_cost=fp_cost
    )
    
    # Augment training data
    print("\nAugmenting POI samples...")
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
    
    print(f"After augmentation: {len(X_train_aug)} samples")
    print(f"POI: {np.sum(y_train_aug==1)}, Non-POI: {np.sum(y_train_aug==0)}")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    X_train_emb = model.extract_embeddings(X_train_aug, sr=8000)
    X_val_emb = model.extract_embeddings(X_val, sr=8000)
    X_test_emb = model.extract_embeddings(X_test, sr=8000)
    
    # Train
    print("\nTraining classifier...")
    history, best_threshold = model.train(
        X_train_emb, y_train_aug, X_val_emb, y_val,
        epochs=30, batch_size=16, learning_rate=0.001
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model.evaluate_with_costs(X_test_emb, y_test, threshold=best_threshold)
    
    # Print results
    print(f"\nResults for {name}:")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
    print(f"  NPV: {metrics['npv']:.1%}")
    print(f"  PPV: {metrics['ppv']:.1%}")
    print(f"  Total Cost: {metrics['total_cost']:.1f}")
    
    return {
        'name': name,
        'fn_cost': fn_cost,
        'fp_cost': fp_cost,
        'metrics': metrics,
        'threshold': best_threshold
    }


def visualize_all_ratios(all_results):
    """Create comprehensive visualization of all cost ratios"""
    
    # Sort by FN/FP ratio
    all_results.sort(key=lambda x: x['fn_cost'] / x['fp_cost'])
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fig.suptitle('YAMNet Performance Across Different Cost Ratios', fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(all_results[:6]):  # Show up to 6 configurations
        ax = axes[idx]
        
        # Get confusion matrix
        cm = np.array(result['metrics']['confusion_matrix'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # Create labels with counts and percentages
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['Pred: No POI', 'Pred: POI'],
                   yticklabels=['True: No POI', 'True: POI'])
        
        # Add title
        ratio = result['fn_cost'] / result['fp_cost']
        metrics = result['metrics']
        ax.set_title(f'Ratio {ratio:.1f}:1 ({result["name"]})\n' + 
                    f'Sens: {metrics["sensitivity"]:.0%}, ' +
                    f'Spec: {metrics["specificity"]:.0%}, ' +
                    f'NPV: {metrics["npv"]:.1%}',
                    fontsize=11)
        
        # Add threshold info
        ax.text(0.5, -0.12, 
               f'Threshold: {result["threshold"]:.2f}, Cost: {metrics["total_cost"]:.0f}',
               transform=ax.transAxes, ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/yamnet_all_ratios_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to results/yamnet_all_ratios_comparison.png")
    
    # Create performance trend plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    ratios = [r['fn_cost'] / r['fp_cost'] for r in all_results]
    sensitivities = [r['metrics']['sensitivity'] for r in all_results]
    specificities = [r['metrics']['specificity'] for r in all_results]
    npvs = [r['metrics']['npv'] for r in all_results]
    costs = [r['metrics']['total_cost'] for r in all_results]
    
    # Sensitivity vs Ratio
    axes2[0,0].plot(ratios, sensitivities, 'o-', color='red', linewidth=2, markersize=8)
    axes2[0,0].set_xlabel('Cost Ratio (FN:FP)')
    axes2[0,0].set_ylabel('Sensitivity')
    axes2[0,0].set_title('Sensitivity vs Cost Ratio')
    axes2[0,0].grid(True, alpha=0.3)
    axes2[0,0].axhline(y=0.5, color='gray', linestyle='--', label='SMOLK baseline')
    
    # Specificity vs Ratio
    axes2[0,1].plot(ratios, specificities, 's-', color='blue', linewidth=2, markersize=8)
    axes2[0,1].set_xlabel('Cost Ratio (FN:FP)')
    axes2[0,1].set_ylabel('Specificity')
    axes2[0,1].set_title('Specificity vs Cost Ratio')
    axes2[0,1].grid(True, alpha=0.3)
    
    # NPV vs Ratio
    axes2[1,0].plot(ratios, npvs, '^-', color='green', linewidth=2, markersize=8)
    axes2[1,0].set_xlabel('Cost Ratio (FN:FP)')
    axes2[1,0].set_ylabel('NPV')
    axes2[1,0].set_title('NPV vs Cost Ratio')
    axes2[1,0].grid(True, alpha=0.3)
    axes2[1,0].axhline(y=0.846, color='gray', linestyle='--', label='SMOLK baseline')
    
    # Total Cost vs Ratio
    axes2[1,1].plot(ratios, costs, 'd-', color='purple', linewidth=2, markersize=8)
    axes2[1,1].set_xlabel('Cost Ratio (FN:FP)')
    axes2[1,1].set_ylabel('Total Cost')
    axes2[1,1].set_title('Total Cost vs Cost Ratio')
    axes2[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Performance Metrics vs Cost Ratio Trends', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/yamnet_ratio_trends.png', dpi=150, bbox_inches='tight')
    print("✓ Saved trend analysis to results/yamnet_ratio_trends.png")


def main():
    print("="*70)
    print("ADDITIONAL COST RATIO EXPERIMENTS FOR YAMNET")
    print("Testing 2.5:1 and 3:1 ratios")
    print("="*70)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
    
    # Load data
    print("\nLoading audio data...")
    DATA_DIR = str(Path.cwd() / 'raw-audio')
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=8000)
    file_dict = loader.get_file_list()
    
    # Load samples
    X_all, y_all = [], []
    
    # Load POI
    for file_path in file_dict['poi']:
        audio, sr = loader.load_audio_file(file_path)
        if audio is not None:
            audio = audio[:20*sr]
            if len(audio) < 20*sr:
                audio = np.pad(audio, (0, 20*sr - len(audio)), 'constant')
            X_all.append(audio)
            y_all.append(1)
    
    # Load Non-POI
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
    
    print(f"\nTotal: {len(X_all)} samples ({np.sum(y_all==1)} POI, {np.sum(y_all==0)} Non-POI)")
    
    # Split data (same as before)
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
    
    # Run new experiments
    new_results = []
    
    # Test 2.5:1 ratio
    result_2_5 = run_single_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        fn_cost=2.5, fp_cost=1.0, name="Ratio 2.5:1"
    )
    new_results.append(result_2_5)
    
    # Test 3:1 ratio (already done as "Moderate POI priority")
    result_3 = run_single_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        fn_cost=3.0, fp_cost=1.0, name="Ratio 3:1"
    )
    new_results.append(result_3)
    
    # Load previous results for comparison
    with open('results/yamnet_cost_sensitive_results.json', 'r') as f:
        prev_results = json.load(f)
    
    # Combine all results
    all_results = []
    
    # Add previous key results
    for exp in prev_results['experiments']:
        all_results.append({
            'name': exp['name'],
            'fn_cost': exp['fn_cost'],
            'fp_cost': exp['fp_cost'],
            'metrics': exp['metrics'],
            'threshold': exp['metrics']['threshold']
        })
    
    # Add new results
    all_results.extend(new_results)
    
    # Save new results
    with open('results/yamnet_additional_ratios.json', 'w') as f:
        json.dump({
            'new_experiments': [
                {
                    'name': r['name'],
                    'fn_cost': r['fn_cost'],
                    'fp_cost': r['fp_cost'],
                    'metrics': r['metrics'],
                    'threshold': r['threshold']
                }
                for r in new_results
            ]
        }, f, indent=2)
    
    print("\n✓ Saved results to results/yamnet_additional_ratios.json")
    
    # Create visualizations
    visualize_all_ratios(all_results)
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON OF ALL COST RATIOS")
    print("="*70)
    
    print(f"\n{'Ratio':<10} {'Name':<25} {'Sens':<8} {'Spec':<8} {'NPV':<8} {'PPV':<8} {'Cost':<8}")
    print("-" * 85)
    
    # Sort by ratio for display
    all_results.sort(key=lambda x: x['fn_cost'] / x['fp_cost'])
    
    for result in all_results:
        ratio = result['fn_cost'] / result['fp_cost']
        metrics = result['metrics']
        print(f"{ratio:<10.1f} {result['name']:<25} "
              f"{metrics['sensitivity']:<8.1%} {metrics['specificity']:<8.1%} "
              f"{metrics['npv']:<8.1%} {metrics['ppv']:<8.1%} {metrics['total_cost']:<8.0f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find optimal ratio for balanced performance
    best_balanced = None
    best_score = 0
    
    for result in all_results:
        metrics = result['metrics']
        # Balance score: weighted combination of sensitivity and NPV
        score = metrics['sensitivity'] * 0.5 + metrics['npv'] * 0.5
        if score > best_score:
            best_score = score
            best_balanced = result
    
    if best_balanced:
        print(f"\nBest Balanced Performance:")
        print(f"  Ratio: {best_balanced['fn_cost']}/{best_balanced['fp_cost']}")
        print(f"  Sensitivity: {best_balanced['metrics']['sensitivity']:.1%}")
        print(f"  NPV: {best_balanced['metrics']['npv']:.1%}")
        print(f"  Specificity: {best_balanced['metrics']['specificity']:.1%}")


if __name__ == '__main__':
    main()