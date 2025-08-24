#!/usr/bin/env python
"""
Visualize all confusion matrices comparing old vs new holdout results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
with open('results/new_holdout_results.json', 'r') as f:
    new_results = json.load(f)

with open('results/yamnet_cost_sensitive_results.json', 'r') as f:
    old_yamnet = json.load(f)

with open('results/panns_gpu_results.json', 'r') as f:
    old_panns = json.load(f)

with open('results/panns_ratio_2_5_results.json', 'r') as f:
    old_panns_25 = json.load(f)

# Create comprehensive confusion matrix visualization
fig = plt.figure(figsize=(20, 12))

# Define layout: 4 rows x 5 columns
# Row 1: PANNs 2:1 (old vs new)
# Row 2: PANNs 2.5:1 (old vs new)  
# Row 3: PANNs 3:1 (old vs new)
# Row 4: YAMNet 2:1, 3:1, 5:1 (old vs new)

ratios_to_show = [
    ('PANNs', 2.0, 1.0),
    ('PANNs', 2.5, 1.0),
    ('PANNs', 3.0, 1.0),
    ('PANNs', 5.0, 1.0),
    ('YAMNet', 2.0, 1.0),
    ('YAMNet', 3.0, 1.0),
    ('YAMNet', 5.0, 1.0),
]

plot_idx = 1

for model_name, fn_cost, fp_cost in ratios_to_show:
    ratio = fn_cost / fp_cost
    
    # Find old result
    old_result = None
    if model_name == 'PANNs':
        if ratio == 2.5:
            old_result = {
                'metrics': old_panns_25['metrics'],
                'threshold': old_panns_25['threshold']
            }
        else:
            for exp in old_panns['experiments']:
                if exp['fn_cost'] == fn_cost and exp['fp_cost'] == fp_cost:
                    old_result = exp
                    break
    else:  # YAMNet
        for exp in old_yamnet['experiments']:
            if exp['fn_cost'] == fn_cost and exp['fp_cost'] == fp_cost:
                old_result = exp
                break
    
    # Find new result
    new_result = None
    for exp in new_results['experiments']:
        if (exp['model'] == model_name and 
            exp['fn_cost'] == fn_cost and 
            exp['fp_cost'] == fp_cost):
            new_result = exp
            break
    
    # Plot OLD confusion matrix
    if old_result:
        ax = plt.subplot(7, 4, plot_idx)
        plot_idx += 1
        
        cm = np.array(old_result['metrics']['confusion_matrix'])
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['No POI', 'POI'],
                   yticklabels=['No POI', 'POI'])
        
        metrics = old_result['metrics']
        thresh_val = old_result.get('threshold', 0.5)
        ax.set_title(f'{model_name} {ratio:.1f}:1 (OLD seed=42)\n' + 
                    f'Sens: {metrics["sensitivity"]:.0%}, ' +
                    f'Spec: {metrics["specificity"]:.0%}, ' +
                    f'NPV: {metrics["npv"]:.1%}\n' +
                    f'Cost: {metrics["total_cost"]:.0f}, ' +
                    f'Thresh: {thresh_val:.2f}',
                    fontsize=9)
        
        # Highlight perfect sensitivity
        if metrics['sensitivity'] == 1.0:
            rect = plt.Rectangle((0.98, 0.98), 0.97, 0.97, fill=False, 
                                edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    
    # Plot NEW confusion matrix
    if new_result:
        ax = plt.subplot(7, 4, plot_idx)
        plot_idx += 1
        
        cm = np.array(new_result['metrics']['confusion_matrix'])
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', 
                   cbar=False, ax=ax,
                   xticklabels=['No POI', 'POI'],
                   yticklabels=['No POI', 'POI'])
        
        metrics = new_result['metrics']
        ax.set_title(f'{model_name} {ratio:.1f}:1 (NEW seed=123)\n' + 
                    f'Sens: {metrics["sensitivity"]:.0%}, ' +
                    f'Spec: {metrics["specificity"]:.0%}, ' +
                    f'NPV: {metrics["npv"]:.1%}\n' +
                    f'Cost: {metrics["total_cost"]:.0f}, ' +
                    f'Thresh: {new_result["threshold"]:.2f}',
                    fontsize=9)
        
        # Highlight failures (0% sensitivity)
        if metrics['sensitivity'] == 0.0:
            rect = plt.Rectangle((0.02, 0.98), 0.97, 0.97, fill=False, 
                                edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        # Highlight perfect sensitivity
        elif metrics['sensitivity'] == 1.0:
            rect = plt.Rectangle((0.98, 0.98), 0.97, 0.97, fill=False, 
                                edgecolor='green', linewidth=2)
            ax.add_patch(rect)

plt.suptitle('Confusion Matrix Comparison: Old (seed=42) vs New (seed=123) Hold-out Sets', 
            fontsize=14, fontweight='bold')

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='#3182bd', alpha=0.3, label='Old Hold-out (seed=42)'),
    plt.Rectangle((0, 0), 1, 1, fc='#fd8d3c', alpha=0.3, label='New Hold-out (seed=123)'),
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='green', linewidth=2, label='Perfect Sensitivity (100%)'),
    plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=2, label='Failed Detection (0%)')
]
plt.figlegend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('results/all_confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/all_confusion_matrices_comparison.png")

# Create summary statistics table
print("\n" + "="*80)
print("CONFUSION MATRIX SUMMARY")
print("="*80)

print("\n" + "-"*80)
print(f"{'Model':<12} {'Ratio':<8} {'Holdout':<10} {'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6} {'Sens':<8} {'Spec':<8} {'NPV':<8}")
print("-"*80)

for model_name, fn_cost, fp_cost in ratios_to_show:
    ratio = fn_cost / fp_cost
    
    # Old result
    old_result = None
    if model_name == 'PANNs':
        if ratio == 2.5:
            cm = np.array(old_panns_25['metrics']['confusion_matrix'])
            old_result = old_panns_25['metrics']
        else:
            for exp in old_panns['experiments']:
                if exp['fn_cost'] == fn_cost and exp['fp_cost'] == fp_cost:
                    cm = np.array(exp['metrics']['confusion_matrix'])
                    old_result = exp['metrics']
                    break
    else:
        for exp in old_yamnet['experiments']:
            if exp['fn_cost'] == fn_cost and exp['fp_cost'] == fp_cost:
                cm = np.array(exp['metrics']['confusion_matrix'])
                old_result = exp['metrics']
                break
    
    if old_result and cm.shape == (2, 2):
        print(f"{model_name:<12} {ratio:<8.1f} {'OLD':<10} {cm[0,0]:<6} {cm[0,1]:<6} "
              f"{cm[1,0]:<6} {cm[1,1]:<6} {old_result['sensitivity']:<8.1%} "
              f"{old_result['specificity']:<8.1%} {old_result['npv']:<8.1%}")
    
    # New result
    for exp in new_results['experiments']:
        if (exp['model'] == model_name and 
            exp['fn_cost'] == fn_cost and 
            exp['fp_cost'] == fp_cost):
            cm = np.array(exp['metrics']['confusion_matrix'])
            if cm.shape == (2, 2):
                print(f"{model_name:<12} {ratio:<8.1f} {'NEW':<10} {cm[0,0]:<6} {cm[0,1]:<6} "
                      f"{cm[1,0]:<6} {cm[1,1]:<6} {exp['metrics']['sensitivity']:<8.1%} "
                      f"{exp['metrics']['specificity']:<8.1%} {exp['metrics']['npv']:<8.1%}")
            break

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

# Count perfect sensitivity cases
old_perfect = 0
new_perfect = 0
old_failed = 0
new_failed = 0

for model_name, fn_cost, fp_cost in ratios_to_show:
    # Check old results
    if model_name == 'PANNs':
        if fn_cost == 2.5:
            if old_panns_25['metrics']['sensitivity'] == 1.0:
                old_perfect += 1
            elif old_panns_25['metrics']['sensitivity'] == 0.0:
                old_failed += 1
        else:
            for exp in old_panns['experiments']:
                if exp['fn_cost'] == fn_cost and exp['fp_cost'] == fp_cost:
                    if exp['metrics']['sensitivity'] == 1.0:
                        old_perfect += 1
                    elif exp['metrics']['sensitivity'] == 0.0:
                        old_failed += 1
                    break
    else:
        for exp in old_yamnet['experiments']:
            if exp['fn_cost'] == fn_cost and exp['fp_cost'] == fp_cost:
                if exp['metrics']['sensitivity'] == 1.0:
                    old_perfect += 1
                elif exp['metrics']['sensitivity'] == 0.0:
                    old_failed += 1
                break
    
    # Check new results
    for exp in new_results['experiments']:
        if (exp['model'] == model_name and 
            exp['fn_cost'] == fn_cost and 
            exp['fp_cost'] == fp_cost):
            if exp['metrics']['sensitivity'] == 1.0:
                new_perfect += 1
            elif exp['metrics']['sensitivity'] == 0.0:
                new_failed += 1
            break

print(f"\n1. Perfect Sensitivity (100% POI detection):")
print(f"   - Old holdout: {old_perfect} configurations")
print(f"   - New holdout: {new_perfect} configurations")

print(f"\n2. Complete Detection Failure (0% sensitivity):")
print(f"   - Old holdout: {old_failed} configurations")
print(f"   - New holdout: {new_failed} configurations")

print(f"\n3. Generalizability Issue:")
print(f"   - PANNs lost ALL perfect sensitivity when tested on new data")
print(f"   - Most configurations show dramatically different performance")
print(f"   - Results are NOT stable across different test sets")

print("\nLegend:")
print("  TN = True Negatives (correctly identified non-POI)")
print("  FP = False Positives (incorrectly flagged as POI)")
print("  FN = False Negatives (missed POI cases)")
print("  TP = True Positives (correctly identified POI)")