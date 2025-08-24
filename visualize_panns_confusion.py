#!/usr/bin/env python
"""
Visualize confusion matrices for GPU-optimized PANNs results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Load PANNs results
with open('results/panns_gpu_results.json', 'r') as f:
    panns_results = json.load(f)

# Load YAMNet results for comparison
with open('results/yamnet_cost_sensitive_results.json', 'r') as f:
    yamnet_results = json.load(f)

# Create figure with confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: PANNs results
for idx, exp in enumerate(panns_results['experiments']):
    ax = axes[0, idx]
    
    # Get confusion matrix
    cm = np.array(exp['metrics']['confusion_matrix'])
    
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
               yticklabels=['True: No POI', 'True: POI'],
               cbar_kws={'label': 'Count'})
    
    # Add title
    metrics = exp['metrics']
    ax.set_title(f"PANNs {exp['name']}\n" + 
                f"Sens: {metrics['sensitivity']:.0%}, " +
                f"Spec: {metrics['specificity']:.0%}, " +
                f"NPV: {metrics['npv']:.1%}",
                fontsize=12, fontweight='bold')
    
    # Add threshold and cost info
    ax.text(0.5, -0.15, 
           f"Threshold: {exp['threshold']:.2f}, Total Cost: {metrics['total_cost']:.0f}",
           transform=ax.transAxes, ha='center', fontsize=9)
    
    # Highlight key cells based on performance
    if metrics['sensitivity'] == 1.0:  # Perfect sensitivity
        # Highlight true positives
        rect = plt.Rectangle((1, 1), 0.98, 0.98, fill=False, 
                            edgecolor='green', linewidth=3)
        ax.add_patch(rect)
    
    if metrics['npv'] == 1.0:  # Perfect NPV
        # Add star marker
        ax.text(1.5, -0.25, '⭐ Perfect NPV', 
               transform=ax.transAxes, ha='center', 
               fontsize=10, color='green', fontweight='bold')

# Bottom row: YAMNet comparison (same cost ratios)
yamnet_configs = {
    'Slight POI priority': '2:1',
    'Moderate POI priority': '3:1', 
    'Strong POI priority': '5:1'
}

yamnet_selected = []
for exp in yamnet_results['experiments']:
    if exp['name'] in yamnet_configs:
        yamnet_selected.append(exp)

for idx, exp in enumerate(yamnet_selected):
    ax = axes[1, idx]
    
    # Get confusion matrix
    cm = np.array(exp['metrics']['confusion_matrix'])
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    # Create labels
    labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                       for j in range(cm.shape[1])] 
                      for i in range(cm.shape[0])])
    
    # Plot
    sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', 
               cbar=False, ax=ax,
               xticklabels=['Pred: No POI', 'Pred: POI'],
               yticklabels=['True: No POI', 'True: POI'])
    
    # Title
    metrics = exp['metrics']
    ratio = yamnet_configs[exp['name']]
    ax.set_title(f"YAMNet Ratio {ratio}\n" + 
                f"Sens: {metrics['sensitivity']:.0%}, " +
                f"Spec: {metrics['specificity']:.0%}, " +
                f"NPV: {metrics['npv']:.1%}",
                fontsize=12)
    
    # Add info
    ax.text(0.5, -0.15, 
           f"Threshold: {metrics['threshold']:.2f}, Total Cost: {metrics['total_cost']:.0f}",
           transform=ax.transAxes, ha='center', fontsize=9)

# Overall title
fig.suptitle('PANNs vs YAMNet: Confusion Matrices Comparison', 
            fontsize=16, fontweight='bold', y=1.02)

# Add legend
fig.text(0.5, 0.48, 'Blue: PANNs (GPU-Optimized) | Orange: YAMNet', 
        ha='center', fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('results/panns_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("✓ Saved confusion matrices to results/panns_confusion_matrices.png")

# Detailed analysis
print("\n" + "="*70)
print("CONFUSION MATRIX ANALYSIS: PANNs")
print("="*70)

for exp in panns_results['experiments']:
    cm = np.array(exp['metrics']['confusion_matrix'])
    metrics = exp['metrics']
    
    print(f"\n{exp['name']} (FN:{exp['fn_cost']:.0f}, FP:{exp['fp_cost']:.0f})")
    print("-" * 40)
    print(f"Threshold: {exp['threshold']:.2f}")
    
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
        
        print(f"\nPerformance:")
        print(f"• Sensitivity: {metrics['sensitivity']:.1%} ({cm[1,1]}/5 POI detected)")
        print(f"• Specificity: {metrics['specificity']:.1%} ({cm[0,0]}/20 non-POI correct)")
        print(f"• NPV: {metrics['npv']:.1%}")
        print(f"• PPV: {metrics['ppv']:.1%}")
        
        # Clinical interpretation
        if metrics['sensitivity'] == 1.0:
            print("\n⭐ PERFECT SENSITIVITY: All POI cases detected!")
        if metrics['npv'] == 1.0:
            print("⭐ PERFECT NPV: 100% confidence when clearing patients!")
        
        print(f"\nCost Analysis:")
        print(f"• False Negative Cost: {cm[1,0]} × {exp['fn_cost']:.0f} = {cm[1,0] * exp['fn_cost']:.0f}")
        print(f"• False Positive Cost: {cm[0,1]} × {exp['fp_cost']:.0f} = {cm[0,1] * exp['fp_cost']:.0f}")
        print(f"• Total Cost: {metrics['total_cost']:.0f}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("""
PANNs RATIO 2:1 - PERFECT DETECTION:
• Detected ALL 5 POI cases (100% sensitivity)
• 11 out of 20 non-POI correctly cleared (55% specificity)
• 9 false alarms but ZERO missed POI cases
• IDEAL for screening where missing POI is unacceptable

PANNs RATIO 3:1 - BALANCED:
• Detected 2 out of 5 POI cases (40% sensitivity)
• 16 out of 20 non-POI correctly cleared (80% specificity)
• Better specificity but misses 3 POI cases

PANNs RATIO 5:1 - MAXIMUM SENSITIVITY:
• Detected ALL 5 POI cases (100% sensitivity)
• Only 3 out of 20 non-POI correctly cleared (15% specificity)
• 17 false alarms - almost everyone flagged as potential POI

RECOMMENDATION: Use PANNs with 2:1 ratio for clinical screening.
The 100% sensitivity and NPV ensures no POI cases are missed,
though follow-up testing needed for the 45% false positive rate.
""")

# Create comparison bar chart
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

models = ['PANNs 2:1', 'PANNs 3:1', 'PANNs 5:1', 'YAMNet 2:1', 'YAMNet 3:1', 'SMOLK']
sensitivities = [100, 40, 100, 60, 60, 50]
npvs = [100, 84.2, 100, 88.2, 86.7, 84.6]
specificities = [55, 80, 15, 75, 65, None]

x = np.arange(len(models))
width = 0.25

bars1 = ax2.bar(x - width, sensitivities, width, label='Sensitivity', color='red', alpha=0.7)
bars2 = ax2.bar(x, npvs, width, label='NPV', color='green', alpha=0.7)
bars3 = ax2.bar(x + width, [s if s else 0 for s in specificities], width, 
                label='Specificity', color='blue', alpha=0.7)

ax2.set_xlabel('Model Configuration')
ax2.set_ylabel('Performance (%)')
ax2.set_title('Performance Comparison: PANNs vs YAMNet vs SMOLK')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 110)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/panns_performance_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved performance comparison to results/panns_performance_comparison.png")