#!/usr/bin/env python
"""
Visualize confusion matrices for cost-sensitive YAMNet configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Load results
with open('results/yamnet_cost_sensitive_results.json', 'r') as f:
    results = json.load(f)

# Find the three key configurations
configs_to_plot = {
    'Best NPV (2:1 ratio)': None,
    'Best Sensitivity (5:1 ratio)': None, 
    'Minimize False Alarms (1:5 ratio)': None
}

for exp in results['experiments']:
    if exp['name'] == 'Slight POI priority' and exp['fn_cost'] == 2.0:
        configs_to_plot['Best NPV (2:1 ratio)'] = exp
    elif exp['name'] == 'Strong POI priority' and exp['fn_cost'] == 5.0:
        configs_to_plot['Best Sensitivity (5:1 ratio)'] = exp
    elif exp['name'] == 'Minimize false alarms' and exp['fp_cost'] == 5.0:
        configs_to_plot['Minimize False Alarms (1:5 ratio)'] = exp

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Cost-Sensitive YAMNet: Confusion Matrices for Different Cost Ratios', fontsize=14, fontweight='bold')

for idx, (title, config) in enumerate(configs_to_plot.items()):
    ax = axes[idx]
    
    if config and 'metrics' in config:
        # Get confusion matrix
        cm = np.array(config['metrics']['confusion_matrix'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # Create labels with both counts and percentages
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   cbar=False, ax=ax, 
                   xticklabels=['Pred: No POI', 'Pred: POI'],
                   yticklabels=['True: No POI', 'True: POI'])
        
        # Add title and metrics
        metrics = config['metrics']
        ax.set_title(f'{title}\n' + 
                    f'Sens: {metrics["sensitivity"]:.0%}, ' +
                    f'Spec: {metrics["specificity"]:.0%}, ' +
                    f'NPV: {metrics["npv"]:.1%}',
                    fontsize=11)
        
        # Add cost information
        ax.text(0.5, -0.15, 
               f'FN Cost: {config["fn_cost"]}, FP Cost: {config["fp_cost"]}\n' +
               f'Threshold: {metrics["threshold"]:.2f}',
               transform=ax.transAxes, ha='center', fontsize=9)
        
        # Highlight key cells based on configuration
        if 'Best Sensitivity' in title:
            # Highlight true positives (high sensitivity)
            rect = plt.Rectangle((1, 1), 0.98, 0.98, fill=False, 
                                edgecolor='green', linewidth=3)
            ax.add_patch(rect)
        elif 'Minimize False' in title:
            # Highlight true negatives (high specificity)
            rect = plt.Rectangle((0, 0), 0.98, 0.98, fill=False, 
                                edgecolor='blue', linewidth=3)
            ax.add_patch(rect)

plt.tight_layout()
plt.savefig('results/yamnet_confusion_matrices.png', dpi=150, bbox_inches='tight')
# plt.show()  # Commented out for non-interactive mode

print("\n" + "="*70)
print("CONFUSION MATRIX ANALYSIS")
print("="*70)

for title, config in configs_to_plot.items():
    if config and 'metrics' in config:
        cm = np.array(config['metrics']['confusion_matrix'])
        metrics = config['metrics']
        
        print(f"\n{title}")
        print("-" * 40)
        print(f"Cost Ratio - FN:{config['fn_cost']:.0f}, FP:{config['fp_cost']:.0f}")
        print(f"Decision Threshold: {metrics['threshold']:.2f}")
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 No POI    POI")
        if cm.shape == (2, 2):
            print(f"Actual No POI      {cm[0,0]:3d}     {cm[0,1]:3d}")
            print(f"Actual POI         {cm[1,0]:3d}     {cm[1,1]:3d}")
            
            # Analysis
            print("\nAnalysis:")
            print(f"• True Negatives: {cm[0,0]} (Correctly identified non-POI)")
            print(f"• False Positives: {cm[0,1]} (False alarms)")
            print(f"• False Negatives: {cm[1,0]} (Missed POI cases)")
            print(f"• True Positives: {cm[1,1]} (Correctly identified POI)")
            
            print(f"\nPerformance:")
            print(f"• Sensitivity: {metrics['sensitivity']:.1%} ({cm[1,1]}/{cm[1,0]+cm[1,1]} POI detected)")
            print(f"• Specificity: {metrics['specificity']:.1%} ({cm[0,0]}/{cm[0,0]+cm[0,1]} non-POI correct)")
            print(f"• NPV: {metrics['npv']:.1%} (confidence when predicting no POI)")
            print(f"• PPV: {metrics['ppv']:.1%} (confidence when predicting POI)")
            
            # Cost analysis
            total_cost = cm[1,0] * config['fn_cost'] + cm[0,1] * config['fp_cost']
            print(f"\nCost Analysis:")
            print(f"• False Negative Cost: {cm[1,0]} × {config['fn_cost']:.0f} = {cm[1,0] * config['fn_cost']:.0f}")
            print(f"• False Positive Cost: {cm[0,1]} × {config['fp_cost']:.0f} = {cm[0,1] * config['fp_cost']:.0f}")
            print(f"• Total Cost: {total_cost:.0f}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("""
1. BEST NPV (2:1 ratio) - Balanced Approach:
   - Misses 2 out of 5 POI cases (40% missed)
   - 5 false alarms out of 20 non-POI (25% false positive rate)
   - Good for initial screening with follow-up tests

2. BEST SENSITIVITY (5:1 ratio) - Aggressive POI Detection:
   - Catches 4 out of 5 POI cases (only 1 missed!)
   - But 17 false alarms out of 20 non-POI (85% false positive rate)
   - Use when missing POI is catastrophic

3. MINIMIZE FALSE ALARMS (1:5 ratio) - Conservative Approach:
   - Perfect specificity: zero false alarms
   - But misses 3 out of 5 POI cases (60% missed)
   - Use when false alarms are very costly

RECOMMENDATION: For medical screening, the 2:1 or 5:1 ratio is preferred
since missing a POI case has serious clinical consequences.
""")

print("\n✓ Confusion matrices saved to results/yamnet_confusion_matrices.png")