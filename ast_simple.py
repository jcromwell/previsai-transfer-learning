#!/usr/bin/env python
"""
Simplified Audio Spectrogram Transformer (AST) implementation without torchaudio
Using Vision Transformer (ViT) on spectrograms for audio classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import librosa
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Try to use transformers ViT as a base
try:
    from transformers import ViTModel, ViTConfig
    print("✓ Using Vision Transformer from transformers")
    VIT_AVAILABLE = True
except ImportError:
    print("✗ ViT not available, using custom implementation")
    VIT_AVAILABLE = False

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class SimpleASTModel(nn.Module):
    """Simplified AST using spectrograms and vision transformer architecture"""
    
    def __init__(self, fn_cost=2.0, fp_cost=1.0, n_mels=128, patch_size=16, 
                 hidden_dim=768, n_heads=12, n_layers=6, n_classes=2):
        super().__init__()
        
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.n_mels = n_mels
        self.patch_size = patch_size
        
        if VIT_AVAILABLE:
            # Use pre-trained ViT
            config = ViTConfig(
                image_size=224,  # We'll resize spectrograms to this
                patch_size=patch_size,
                num_channels=1,  # Grayscale spectrogram
                hidden_size=hidden_dim,
                num_hidden_layers=n_layers,
                num_attention_heads=n_heads,
                intermediate_size=hidden_dim * 4,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            self.vit = ViTModel(config)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes)
            )
        else:
            # Simple CNN as fallback
            self.features = nn.Sequential(
                # Conv blocks
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes)
            )
    
    def forward(self, spectrograms):
        if VIT_AVAILABLE:
            # Use ViT
            outputs = self.vit(spectrograms)
            pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
            logits = self.classifier(pooled)
        else:
            # Use CNN
            features = self.features(spectrograms)
            features = features.view(features.size(0), -1)
            logits = self.classifier(features)
        
        return logits
    
    def compute_cost_sensitive_loss(self, logits, labels):
        """Weighted cross-entropy based on misclassification costs"""
        # Standard cross-entropy with class weights
        weights = torch.tensor([self.fp_cost, self.fn_cost], device=logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        return loss_fn(logits, labels)


def audio_to_spectrogram(audio, sr=8000, n_mels=128, n_fft=1024, hop_length=256):
    """Convert audio to mel-spectrogram"""
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, 
        n_fft=n_fft, hop_length=hop_length
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    
    return log_mel


def prepare_spectrogram_batch(audios, sr=8000, n_mels=128, target_size=(224, 224)):
    """Prepare batch of spectrograms for model input"""
    spectrograms = []
    
    for audio in audios:
        # Convert to spectrogram
        spec = audio_to_spectrogram(audio, sr, n_mels)
        
        # Resize to target size (simple interpolation)
        from scipy.ndimage import zoom
        h, w = spec.shape
        zoom_h = target_size[0] / h
        zoom_w = target_size[1] / w
        spec_resized = zoom(spec, (zoom_h, zoom_w), order=1)
        
        # Add channel dimension
        spec_resized = spec_resized[np.newaxis, :, :]
        spectrograms.append(spec_resized)
    
    return np.array(spectrograms, dtype=np.float32)


def augment_audio_simple(audio, n_aug=4):
    """Simple audio augmentation"""
    augmented = []
    for i in range(n_aug):
        aug = audio.copy()
        
        # Add noise
        if np.random.random() > 0.3:
            aug += np.random.normal(0, 0.005, len(aug))
        
        # Time shift
        if np.random.random() > 0.3:
            shift = np.random.randint(-800, 800)
            aug = np.roll(aug, shift)
        
        # Amplitude scaling
        if np.random.random() > 0.3:
            aug *= np.random.uniform(0.8, 1.2)
        
        augmented.append(aug)
    
    return augmented


def train_ast_simple(model, X_train, y_train, X_val, y_val, 
                     epochs=20, batch_size=16, lr=1e-4, device='cpu'):
    """Train simplified AST model"""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Convert audio to spectrograms
    print("Converting audio to spectrograms...")
    X_train_spec = prepare_spectrogram_batch(X_train)
    X_val_spec = prepare_spectrogram_batch(X_val)
    
    X_train_t = torch.tensor(X_train_spec, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_spec, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    best_npv = 0
    best_threshold = 0.5
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        indices = torch.randperm(len(X_train_t))
        train_loss = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_train_t[batch_idx].to(device)
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
            val_logits = model(X_val_t.to(device))
            val_loss = model.compute_cost_sensitive_loss(val_logits, y_val_t.to(device))
            
            # Find optimal threshold for NPV
            val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            
            best_val_npv = 0
            for thresh in np.arange(0.1, 0.9, 0.05):
                preds = (val_probs >= thresh).astype(int)
                tn = np.sum((preds == 0) & (y_val == 0))
                fn = np.sum((preds == 0) & (y_val == 1))
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                if npv > best_val_npv:
                    best_val_npv = npv
                    if npv > best_npv:
                        best_npv = npv
                        best_threshold = thresh
                        best_state = model.state_dict()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss={train_loss:.3f}, Val NPV={best_val_npv:.1%}, Thresh={best_threshold:.2f}")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_threshold


def evaluate_ast_simple(model, X_test, y_test, threshold, fn_cost, fp_cost, device='cpu'):
    """Evaluate simplified AST model"""
    
    model.eval()
    model = model.to(device)
    
    # Convert to spectrograms
    X_test_spec = prepare_spectrogram_batch(X_test)
    X_test_t = torch.tensor(X_test_spec, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X_test_t.to(device))
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
    print("SIMPLIFIED AUDIO SPECTROGRAM TRANSFORMER (AST)")
    print("="*80)
    
    # Check device - use MPS if available
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using MPS GPU acceleration")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"Using CPU (slower)")
    
    print(f"Device: {device}")
    
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
    
    # Test with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Testing with seed={seed}")
        print(f"{'='*70}")
        
        # Split data
        np.random.seed(seed)
        poi_indices = np.where(y_all == 1)[0]
        nopoi_indices = np.where(y_all == 0)[0]
        
        poi_indices = np.random.permutation(poi_indices)
        nopoi_indices = np.random.permutation(nopoi_indices)
        
        test_indices = np.concatenate([poi_indices[:5], nopoi_indices[:20]])
        val_indices = np.concatenate([poi_indices[5:10], nopoi_indices[20:40]])
        train_indices = np.concatenate([poi_indices[10:], nopoi_indices[40:]])
        
        X_test, y_test = X_all[test_indices], y_all[test_indices]
        X_val, y_val = X_all[val_indices], y_all[val_indices]
        X_train, y_train = X_all[train_indices], y_all[train_indices]
        
        # Augment training data
        X_train_aug, y_train_aug = [], []
        for x, y in zip(X_train, y_train):
            if y == 1:  # POI
                augmented = augment_audio_simple(x, n_aug=6)
                X_train_aug.extend(augmented)
                y_train_aug.extend([1] * len(augmented))
            else:
                X_train_aug.append(x)
                y_train_aug.append(0)
        
        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)
        
        print(f"Training samples: {len(X_train_aug)} (POI: {np.sum(y_train_aug==1)})")
        
        # Test 2.5:1 ratio
        fn_cost, fp_cost = 2.5, 1.0
        
        print(f"\nTraining AST with {fn_cost}:{fp_cost} ratio...")
        model = SimpleASTModel(fn_cost=fn_cost, fp_cost=fp_cost)
        
        model, threshold = train_ast_simple(
            model, X_train_aug, y_train_aug, X_val, y_val,
            epochs=10, batch_size=16, lr=1e-4, device=device
        )
        
        print("\nEvaluating...")
        metrics = evaluate_ast_simple(model, X_test, y_test, threshold, 
                                     fn_cost, fp_cost, device=device)
        
        print(f"\nResults (seed={seed}):")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        
        all_results.append({
            'seed': seed,
            'metrics': metrics,
            'threshold': threshold
        })
    
    # Save results
    with open('results/ast_simple_results.json', 'w') as f:
        json.dump({'experiments': all_results}, f, indent=2)
    
    print("\n✓ Saved results to results/ast_simple_results.json")
    
    # Visualize confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        cm = np.array(result['metrics']['confusion_matrix'])
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(2)] for i in range(2)])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['No POI', 'POI'],
                   yticklabels=['No POI', 'POI'])
        
        metrics = result['metrics']
        ax.set_title(f'AST (seed={result["seed"]})\n' + 
                    f'Sens: {metrics["sensitivity"]:.0%}, ' +
                    f'NPV: {metrics["npv"]:.1%}',
                    fontsize=11)
        
        if metrics['sensitivity'] == 1.0:
            rect = plt.Rectangle((0.98, 0.98), 0.97, 0.97, fill=False, 
                                edgecolor='green', linewidth=2)
            ax.add_patch(rect)
        elif metrics['sensitivity'] == 0.0:
            rect = plt.Rectangle((0.02, 0.98), 0.97, 0.97, fill=False, 
                                edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    plt.suptitle('Audio Spectrogram Transformer - Generalizability Test', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/ast_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("✓ Saved confusion matrices to results/ast_confusion_matrices.png")
    
    # Summary
    print("\n" + "="*80)
    print("GENERALIZABILITY SUMMARY")
    print("="*80)
    
    sensitivities = [r['metrics']['sensitivity'] for r in all_results]
    npvs = [r['metrics']['npv'] for r in all_results]
    
    print(f"Sensitivity range: {min(sensitivities):.0%} - {max(sensitivities):.0%}")
    print(f"NPV range: {min(npvs):.1%} - {max(npvs):.1%}")
    
    if max(sensitivities) - min(sensitivities) > 0.5:
        print("\n⚠️ HIGH VARIABILITY detected - model is not stable across different test sets")
    else:
        print("\n✓ Relatively stable performance across seeds")


if __name__ == '__main__':
    main()