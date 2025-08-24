#!/usr/bin/env python
"""
Audio Spectrogram Transformer (AST) with cost-sensitive training for POI detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Check if transformers is available
try:
    from transformers import ASTModel, ASTFeatureExtractor
    print("✓ Transformers library available")
    AST_AVAILABLE = True
except ImportError:
    print("✗ Transformers library not available. Installing...")
    AST_AVAILABLE = False

if not AST_AVAILABLE:
    # Install transformers
    import subprocess
    subprocess.check_call(['pip', 'install', 'transformers'])
    from transformers import ASTModel, ASTFeatureExtractor
    print("✓ Transformers installed successfully")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class CostSensitiveAST(nn.Module):
    """Audio Spectrogram Transformer with cost-sensitive classification head"""
    
    def __init__(self, fn_cost=2.0, fp_cost=1.0, n_classes=2):
        super().__init__()
        
        # Load pre-trained AST
        print("Loading pre-trained AST model...")
        self.ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Freeze base model initially
        for param in self.ast_model.parameters():
            param.requires_grad = False
        
        # Add trainable classification head
        hidden_size = self.ast_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
        
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        
    def forward(self, audio_values):
        # Get AST embeddings
        outputs = self.ast_model(audio_values)
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits
    
    def compute_cost_sensitive_loss(self, logits, labels):
        """Compute weighted cross-entropy loss based on misclassification costs"""
        probs = F.softmax(logits, dim=-1)
        
        # Create cost matrix
        batch_size = labels.shape[0]
        costs = torch.zeros_like(probs)
        
        # For true negative (label=0): cost of FP
        costs[labels == 0, 1] = self.fp_cost
        
        # For true positive (label=1): cost of FN  
        costs[labels == 1, 0] = self.fn_cost
        
        # Compute weighted loss
        loss = -torch.log(probs[range(batch_size), labels] + 1e-7)
        
        # Apply cost weights
        weights = torch.where(labels == 1, self.fn_cost, self.fp_cost)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


def extract_features(audio, feature_extractor, target_length=160000):
    """Extract AST features from audio"""
    # AST expects 16kHz audio, but we have 8kHz
    # Resample by repeating samples (simple upsampling)
    audio_16k = np.repeat(audio, 2)
    
    # Pad or truncate to target length
    if len(audio_16k) > target_length:
        audio_16k = audio_16k[:target_length]
    else:
        audio_16k = np.pad(audio_16k, (0, target_length - len(audio_16k)), 'constant')
    
    # Extract features
    inputs = feature_extractor(
        audio_16k, 
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=1024
    )
    
    return inputs.input_values


def augment_audio_ast(audio, n_aug=4):
    """Simple augmentation for AST training"""
    augmented = []
    for i in range(n_aug):
        aug = audio.copy()
        
        # Add noise
        if np.random.random() > 0.5:
            aug += np.random.normal(0, 0.005, len(aug))
        
        # Time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-1600, 1600)
            aug = np.roll(aug, shift)
        
        # Amplitude scaling
        if np.random.random() > 0.5:
            aug *= np.random.uniform(0.8, 1.2)
        
        augmented.append(aug)
    
    return augmented


def train_ast_model(model, X_train, y_train, X_val, y_val, 
                   epochs=20, batch_size=8, lr=1e-4, device='cpu'):
    """Train AST with cost-sensitive loss"""
    
    model = model.to(device)
    optimizer = AdamW(model.classifier.parameters(), lr=lr)
    
    # Prepare data
    print("Extracting features...")
    train_features = []
    val_features = []
    
    for x in X_train:
        feat = extract_features(x, model.feature_extractor)
        train_features.append(feat)
    
    for x in X_val:
        feat = extract_features(x, model.feature_extractor)
        val_features.append(feat)
    
    train_features = torch.cat(train_features, dim=0)
    val_features = torch.cat(val_features, dim=0)
    
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    best_val_loss = float('inf')
    best_threshold = 0.5
    
    for epoch in range(epochs):
        # Training
        model.train()
        indices = torch.randperm(len(train_features))
        train_loss = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = train_features[batch_idx].to(device)
            batch_y = y_train_t[batch_idx].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = model.compute_cost_sensitive_loss(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_features.to(device))
            val_loss = model.compute_cost_sensitive_loss(val_logits, y_val_t.to(device))
            
            # Find optimal threshold
            val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            
            best_f1 = 0
            for thresh in np.arange(0.1, 0.9, 0.05):
                preds = (val_probs >= thresh).astype(int)
                tp = np.sum((preds == 1) & (y_val == 1))
                fp = np.sum((preds == 1) & (y_val == 0))
                fn = np.sum((preds == 0) & (y_val == 1))
                
                if tp > 0:
                    precision = tp / (tp + fp + 1e-7)
                    recall = tp / (tp + fn + 1e-7)
                    f1 = 2 * precision * recall / (precision + recall + 1e-7)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}, Threshold={best_threshold:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model, best_threshold


def evaluate_ast(model, X_test, y_test, threshold, fn_cost, fp_cost, device='cpu'):
    """Evaluate AST model with cost metrics"""
    
    model.eval()
    model = model.to(device)
    
    # Extract features
    test_features = []
    for x in X_test:
        feat = extract_features(x, model.feature_extractor)
        test_features.append(feat)
    
    test_features = torch.cat(test_features, dim=0)
    
    with torch.no_grad():
        logits = model(test_features.to(device))
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    
    # Predictions
    predictions = (probs >= threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    # Cost
    total_cost = fn * fn_cost + fp * fp_cost
    
    return {
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'npv': npv,
        'ppv': ppv,
        'accuracy': accuracy,
        'total_cost': total_cost
    }


def main():
    print("="*80)
    print("AUDIO SPECTROGRAM TRANSFORMER (AST) WITH COST-SENSITIVE TRAINING")
    print("="*80)
    
    # Check device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Note: AST may not work well with MPS, fallback to CPU if needed
    if device == 'mps':
        print("Note: AST may have issues with MPS, falling back to CPU")
        device = 'cpu'
    
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
    
    # Split data
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
            augmented = augment_audio_ast(x, n_aug=6)
            X_train_aug.extend(augmented)
            y_train_aug.extend([1] * len(augmented))
        else:
            X_train_aug.append(x)
            y_train_aug.append(0)
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    print(f"After augmentation: {len(X_train_aug)} samples")
    print(f"POI: {np.sum(y_train_aug==1)}, Non-POI: {np.sum(y_train_aug==0)}")
    
    # Test different cost ratios
    ratios = [(2.0, 1.0), (2.5, 1.0), (3.0, 1.0)]
    results = []
    
    for fn_cost, fp_cost in ratios:
        ratio = fn_cost / fp_cost
        print(f"\n{'='*70}")
        print(f"Training AST with Ratio {ratio:.1f}:1")
        print(f"{'='*70}")
        
        # Create model
        model = CostSensitiveAST(fn_cost=fn_cost, fp_cost=fp_cost)
        
        # Train
        model, threshold = train_ast_model(
            model, X_train_aug, y_train_aug, X_val, y_val,
            epochs=15, batch_size=8, lr=1e-4, device=device
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = evaluate_ast(model, X_test, y_test, threshold, 
                              fn_cost, fp_cost, device=device)
        
        print(f"\nResults for {ratio:.1f}:1 Ratio:")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  PPV: {metrics['ppv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        
        results.append({
            'model': 'AST',
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'metrics': metrics,
            'threshold': threshold
        })
    
    # Save results
    with open('results/ast_cost_sensitive_results.json', 'w') as f:
        json.dump({'experiments': results}, f, indent=2)
    
    print("\n✓ Saved results to results/ast_cost_sensitive_results.json")
    
    # Visualize confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        cm = np.array(result['metrics']['confusion_matrix'])
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(2)] for i in range(2)])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['Pred: No POI', 'Pred: POI'],
                   yticklabels=['True: No POI', 'True: POI'])
        
        ratio = result['fn_cost'] / result['fp_cost']
        metrics = result['metrics']
        ax.set_title(f'AST Ratio {ratio:.1f}:1\n' + 
                    f'Sens: {metrics["sensitivity"]:.0%}, ' +
                    f'NPV: {metrics["npv"]:.1%}',
                    fontsize=11)
        
        if metrics['sensitivity'] == 1.0:
            rect = plt.Rectangle((0.98, 0.98), 0.97, 0.97, fill=False, 
                                edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    
    plt.suptitle('Audio Spectrogram Transformer (AST) Results', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/ast_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("✓ Saved confusion matrices to results/ast_confusion_matrices.png")


if __name__ == '__main__':
    main()