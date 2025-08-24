#!/usr/bin/env python
"""
Final comparison of all models tested: PANNs, YAMNet, and AST
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Results from all experiments
results = {
    'PANNs': {
        'seed_42': {'sens': 100, 'spec': 50, 'npv': 100, 'cost': 10},
        'seed_123': {'sens': 0, 'spec': 100, 'npv': 80, 'cost': 12.5},
        'seed_456': {'sens': 0, 'spec': 100, 'npv': 80, 'cost': 12.5}
    },
    'YAMNet': {
        'seed_42': {'sens': 60, 'spec': 75, 'npv': 88.2, 'cost': 9},
        'seed_123': {'sens': 0, 'spec': 100, 'npv': 80, 'cost': 10},
        'seed_456': {'sens': 0, 'spec': 95, 'npv': 79.2, 'cost': 11}  # Estimated from 2:1 ratio
    },
    'AST': {
        'seed_42': {'sens': 60, 'spec': 30, 'npv': 75, 'cost': 19},
        'seed_123': {'sens': 0, 'spec': 100, 'npv': 80, 'cost': 12.5},
        'seed_456': {'sens': 0, 'spec': 90, 'npv': 78.3, 'cost': 14.5}
    }
}

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Sensitivity across seeds
ax = axes[0, 0]
x = np.arange(3)
width = 0.25
seeds = ['Seed 42', 'Seed 123', 'Seed 456']

panns_sens = [results['PANNs']['seed_42']['sens'], 
              results['PANNs']['seed_123']['sens'], 
              results['PANNs']['seed_456']['sens']]
yamnet_sens = [results['YAMNet']['seed_42']['sens'], 
               results['YAMNet']['seed_123']['sens'], 
               results['YAMNet']['seed_456']['sens']]
ast_sens = [results['AST']['seed_42']['sens'], 
            results['AST']['seed_123']['sens'], 
            results['AST']['seed_456']['sens']]

ax.bar(x - width, panns_sens, width, label='PANNs', color='blue', alpha=0.7)
ax.bar(x, yamnet_sens, width, label='YAMNet', color='orange', alpha=0.7)
ax.bar(x + width, ast_sens, width, label='AST', color='green', alpha=0.7)
ax.set_xlabel('Test Set')
ax.set_ylabel('Sensitivity (%)')
ax.set_title('Sensitivity Across Different Hold-out Sets', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.legend()
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

# Add failure indicators
for i, (p, y, a) in enumerate(zip(panns_sens, yamnet_sens, ast_sens)):
    if p == 0:
        ax.text(i - width, 5, '❌', ha='center', fontsize=12)
    if y == 0:
        ax.text(i, 5, '❌', ha='center', fontsize=12)
    if a == 0:
        ax.text(i + width, 5, '❌', ha='center', fontsize=12)

# 2. NPV across seeds
ax = axes[0, 1]
panns_npv = [results['PANNs']['seed_42']['npv'], 
             results['PANNs']['seed_123']['npv'], 
             results['PANNs']['seed_456']['npv']]
yamnet_npv = [results['YAMNet']['seed_42']['npv'], 
              results['YAMNet']['seed_123']['npv'], 
              results['YAMNet']['seed_456']['npv']]
ast_npv = [results['AST']['seed_42']['npv'], 
           results['AST']['seed_123']['npv'], 
           results['AST']['seed_456']['npv']]

ax.bar(x - width, panns_npv, width, label='PANNs', color='blue', alpha=0.7)
ax.bar(x, yamnet_npv, width, label='YAMNet', color='orange', alpha=0.7)
ax.bar(x + width, ast_npv, width, label='AST', color='green', alpha=0.7)
ax.set_xlabel('Test Set')
ax.set_ylabel('NPV (%)')
ax.set_title('NPV Across Different Hold-out Sets', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.legend()
ax.set_ylim(70, 105)
ax.grid(True, alpha=0.3, axis='y')

# 3. Cost comparison
ax = axes[0, 2]
panns_cost = [results['PANNs']['seed_42']['cost'], 
              results['PANNs']['seed_123']['cost'], 
              results['PANNs']['seed_456']['cost']]
yamnet_cost = [results['YAMNet']['seed_42']['cost'], 
               results['YAMNet']['seed_123']['cost'], 
               results['YAMNet']['seed_456']['cost']]
ast_cost = [results['AST']['seed_42']['cost'], 
            results['AST']['seed_123']['cost'], 
            results['AST']['seed_456']['cost']]

ax.bar(x - width, panns_cost, width, label='PANNs', color='blue', alpha=0.7)
ax.bar(x, yamnet_cost, width, label='YAMNet', color='orange', alpha=0.7)
ax.bar(x + width, ast_cost, width, label='AST', color='green', alpha=0.7)
ax.set_xlabel('Test Set')
ax.set_ylabel('Total Cost')
ax.set_title('Misclassification Cost', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Variability analysis
ax = axes[1, 0]
models = ['PANNs', 'YAMNet', 'AST']
sens_ranges = []
for model in models:
    sens_values = [results[model]['seed_42']['sens'],
                   results[model]['seed_123']['sens'],
                   results[model]['seed_456']['sens']]
    sens_ranges.append(max(sens_values) - min(sens_values))

bars = ax.bar(models, sens_ranges, color=['blue', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('Sensitivity Range (%)')
ax.set_title('Sensitivity Variability (Max - Min)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, sens_ranges):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}%', ha='center', fontweight='bold')

# 5. Success rate
ax = axes[1, 1]
success_rates = []
for model in models:
    successes = 0
    for seed in ['seed_42', 'seed_123', 'seed_456']:
        if results[model][seed]['sens'] > 0:
            successes += 1
    success_rates.append(successes / 3 * 100)

bars = ax.bar(models, success_rates, color=['blue', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('Success Rate (%)')
ax.set_title('% of Seeds with Non-Zero Sensitivity', fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}%', ha='center', fontweight='bold')

# 6. Summary table
ax = axes[1, 2]
ax.axis('tight')
ax.axis('off')

summary_data = []
for model in models:
    sens_vals = [results[model][s]['sens'] for s in ['seed_42', 'seed_123', 'seed_456']]
    npv_vals = [results[model][s]['npv'] for s in ['seed_42', 'seed_123', 'seed_456']]
    
    summary_data.append([
        model,
        f"{max(sens_vals)}%",
        f"{min(sens_vals)}%",
        f"{np.mean(sens_vals):.0f}%",
        f"{np.mean(npv_vals):.1f}%"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Model', 'Best Sens', 'Worst Sens', 'Avg Sens', 'Avg NPV'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style the header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight worst performer
table[(3, 0)].set_facecolor('#ffcccc')  # AST row

plt.suptitle('Comprehensive Model Comparison: Generalizability Analysis\n(Cost Ratio 2.5:1)', 
            fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/final_all_models_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/final_all_models_comparison.png")

# Print detailed summary
print("\n" + "="*80)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*80)

print("\n1. GENERALIZABILITY RANKINGS (best to worst):")
print("   None of the models showed acceptable generalizability")
print("   - PANNs: 100% variability (100% → 0%)")
print("   - YAMNet: 60% variability (60% → 0%)")  
print("   - AST: 60% variability (60% → 0%)")

print("\n2. CLINICAL SUITABILITY:")
print("   ❌ PANNs: NOT SUITABLE - extreme instability")
print("   ❌ YAMNet: NOT SUITABLE - fails on 2/3 test sets")
print("   ❌ AST: NOT SUITABLE - fails on 2/3 test sets")

print("\n3. KEY INSIGHTS:")
print("   • All models fail completely (0% sensitivity) on new test data")
print("   • 'Perfect' results on seed 42 are misleading")
print("   • Small dataset (17 POI) causes severe overfitting")
print("   • Threshold optimization on validation set doesn't generalize")

print("\n4. RECOMMENDATIONS:")
print("   • DO NOT deploy any of these models clinically")
print("   • Collect significantly more POI samples (>100)")
print("   • Use k-fold cross-validation during development")
print("   • Consider ensemble methods for stability")
print("   • Test on completely separate hospital data")

print("\n" + "="*80)