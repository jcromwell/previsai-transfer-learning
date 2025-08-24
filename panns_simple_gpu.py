#!/usr/bin/env python
"""
Simplified PANNs-style CNN with cost sensitivity, optimized for GPU
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

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Force MPS if available
if torch.backends.mps.is_available():
    torch.backends.mps.force_fallback = False


class SimplePANNs(nn.Module):
    """Simplified PANNs-like architecture optimized for MPS GPU"""
    
    def __init__(self, n_mels=64, n_classes=2):
        super().__init__()
        
        # Conv blocks - keeping operations MPS-compatible
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def extract_mel_spectrogram(audio, sr=8000, n_mels=64, n_fft=512, hop_length=256):
    """Extract mel-spectrogram for CNN input"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, 
        n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    
    return log_mel


def prepare_batch_gpu(audio_batch, sr=8000, target_length=256):
    """Prepare audio batch for GPU processing"""
    spectrograms = []
    
    for audio in audio_batch:
        mel_spec = extract_mel_spectrogram(audio, sr)
        
        # Pad or trim to fixed length
        if mel_spec.shape[1] < target_length:
            pad = target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad)), 'constant')
        else:
            mel_spec = mel_spec[:, :target_length]
        
        spectrograms.append(mel_spec)
    
    return np.array(spectrograms)


def train_with_cost_sensitivity(model, X_train, y_train, X_val, y_val,
                                fn_cost=2.0, fp_cost=1.0, 
                                epochs=30, batch_size=32, device='mps'):
    """Train with cost-sensitive loss on GPU"""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Calculate class weights
    n_samples = len(y_train)
    class_counts = np.bincount(y_train)
    weight_0 = fp_cost * (n_samples / (2 * class_counts[0]))
    weight_1 = fn_cost * (n_samples / (2 * class_counts[1]))
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32, device=device)
    
    print(f"Class weights: [Non-POI: {weight_0:.2f}, POI: {weight_1:.2f}]")
    
    # Prepare data
    X_train_spec = prepare_batch_gpu(X_train)
    X_val_spec = prepare_batch_gpu(X_val)
    
    # Convert to tensors - keep on GPU throughout
    X_train_t = torch.FloatTensor(X_train_spec).unsqueeze(1).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_spec).unsqueeze(1).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    best_val_npv = 0
    best_threshold = 0.5
    best_state = None
    
    print("\nTraining on GPU...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        indices = torch.randperm(len(X_train_t), device=device)
        total_loss = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            
            # Cost-sensitive loss
            loss = F.cross_entropy(outputs, batch_y, weight=class_weights)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_probs = F.softmax(val_outputs, dim=1)[:, 1]
                
                # Find best threshold
                best_epoch_npv = 0
                best_epoch_thresh = 0.5
                
                for thresh in torch.arange(0.3, 0.7, 0.05):
                    val_preds = (val_probs > thresh).long()
                    
                    cm = confusion_matrix(y_val_t.cpu(), val_preds.cpu())
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                        
                        if npv > best_epoch_npv:
                            best_epoch_npv = npv
                            best_epoch_thresh = thresh.item()
                
                # Save best model
                if best_epoch_npv > best_val_npv:
                    best_val_npv = best_epoch_npv
                    best_threshold = best_epoch_thresh
                    best_state = model.state_dict().copy()
                    
                    # Calculate metrics for display
                    val_preds = (val_probs > best_threshold).long()
                    cm = confusion_matrix(y_val_t.cpu(), val_preds.cpu())
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        elapsed = time.time() - start_time
                        print(f"  Epoch {epoch+1}: NPV={best_epoch_npv:.1%}, "
                              f"Sens={sens:.1%}, Spec={spec:.1%} "
                              f"(Time: {elapsed:.1f}s)")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"Training complete. Best NPV: {best_val_npv:.1%}, Threshold: {best_threshold:.2f}")
    
    return model, best_threshold


def evaluate_model(model, X_test, y_test, threshold, fn_cost, fp_cost, device='mps'):
    """Evaluate model on test set"""
    model.eval()
    
    X_test_spec = prepare_batch_gpu(X_test)
    X_test_t = torch.FloatTensor(X_test_spec).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_t)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    
    preds = (probs > threshold).astype(int)
    
    cm = confusion_matrix(y_test, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    else:
        tn = fp = fn = tp = 0
    
    return {
        'confusion_matrix': cm.tolist(),
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'total_cost': fn * fn_cost + fp * fp_cost
    }


def augment_audio_simple(audio, n_aug=5):
    """Simple augmentation"""
    augmented = [audio]
    for i in range(n_aug):
        aug = audio + np.random.normal(0, 0.005, len(audio))
        augmented.append(aug)
    return augmented


def main():
    print("="*70)
    print("SIMPLIFIED PANNs WITH GPU OPTIMIZATION")
    print("="*70)
    
    # Check GPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'mps':
        print("✓ MPS GPU acceleration enabled")
        # Test GPU
        test = torch.randn(10, 10, device=device)
        print(f"✓ GPU test successful: {test.shape}")
    
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
    
    # Test different cost ratios
    cost_configs = [
        {'fn': 2.0, 'fp': 1.0, 'name': 'Ratio 2:1'},
        {'fn': 3.0, 'fp': 1.0, 'name': 'Ratio 3:1'},
        {'fn': 5.0, 'fp': 1.0, 'name': 'Ratio 5:1'},
    ]
    
    results = []
    
    for config in cost_configs:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {config['name']} (FN:{config['fn']}, FP:{config['fp']})")
        print('='*70)
        
        # Create model
        model = SimplePANNs(n_mels=64, n_classes=2)
        
        # Train
        model, threshold = train_with_cost_sensitivity(
            model, X_train_aug, y_train_aug, X_val, y_val,
            fn_cost=config['fn'], fp_cost=config['fp'],
            epochs=30, batch_size=32, device=device
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = evaluate_model(model, X_test, y_test, threshold, 
                                config['fn'], config['fp'], device)
        
        results.append({
            'config': config,
            'metrics': metrics,
            'threshold': threshold
        })
        
        print(f"\nResults:")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        
        # Clear GPU cache
        if device == 'mps':
            torch.mps.empty_cache()
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/panns_gpu_results.json', 'w') as f:
        json.dump({
            'experiments': [
                {
                    'name': r['config']['name'],
                    'fn_cost': r['config']['fn'],
                    'fp_cost': r['config']['fp'],
                    'metrics': r['metrics'],
                    'threshold': r['threshold']
                }
                for r in results
            ]
        }, f, indent=2)
    
    print("\n✓ Results saved to results/panns_gpu_results.json")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: PANNs-style CNN Performance")
    print("="*70)
    
    print(f"\n{'Config':<15} {'Sens':<8} {'Spec':<8} {'NPV':<8} {'Cost':<8}")
    print("-" * 55)
    
    for result in results:
        config_name = result['config']['name']
        metrics = result['metrics']
        print(f"{config_name:<15} {metrics['sensitivity']:<8.1%} "
              f"{metrics['specificity']:<8.1%} {metrics['npv']:<8.1%} "
              f"{metrics['total_cost']:<8.0f}")
    
    print("\nComparison with YAMNet best (2:1 ratio):")
    print("  YAMNet: Sens=60%, Spec=75%, NPV=88.2%")
    
    best_panns = max(results, key=lambda x: x['metrics']['npv'])
    print(f"  PANNs best: Sens={best_panns['metrics']['sensitivity']:.1%}, "
          f"Spec={best_panns['metrics']['specificity']:.1%}, "
          f"NPV={best_panns['metrics']['npv']:.1%}")


if __name__ == '__main__':
    main()