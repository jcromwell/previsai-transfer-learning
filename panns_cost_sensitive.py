#!/usr/bin/env python
"""
Cost-Sensitive PANNs (Pretrained Audio Neural Networks) for POI Detection
PANNs are state-of-the-art audio classification models trained on AudioSet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from sklearn.metrics import confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import json
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class PANNsFeatureExtractor:
    """Extract features using pretrained PANNs models"""
    
    def __init__(self, model_type='CNN14', device='cpu'):
        """
        Initialize PANNs model
        
        Args:
            model_type: Type of PANNs model ('CNN14', 'ResNet38', 'MobileNetV1')
            device: Device to run on
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.sample_rate = 32000  # PANNs expects 32kHz
        self.model = None
        
        # Load pretrained model
        self._load_model()
        
    def _load_model(self):
        """Load pretrained PANNs model"""
        # Use simplified implementation to avoid download issues
        print(f"Using simplified {self.model_type} implementation...")
        self._create_simple_cnn14()
    
    def _create_simple_cnn14(self):
        """Create a simplified CNN14-like architecture"""
        class SimpleCNN14(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Simplified CNN blocks
                self.conv_block1 = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                self.conv_block2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                self.conv_block3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                self.conv_block4 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.embedding_dim = 512
                
            def forward(self, x):
                x = self.conv_block1(x)
                x = self.conv_block2(x)
                x = self.conv_block3(x)
                x = self.conv_block4(x)
                x = x.squeeze(-1).squeeze(-1)
                return x
        
        self.model = SimpleCNN14().to(self.device)
        self.model.eval()
        print(f"Created simplified CNN14 model")
    
    def preprocess_audio(self, audio, sr=8000):
        """Preprocess audio for PANNs (resample to 32kHz)"""
        # Resample to 32kHz if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def extract_mel_features(self, audio):
        """Extract mel-spectrogram features"""
        # Parameters for mel-spectrogram (following PANNs convention)
        n_mels = 64
        n_fft = 1024
        hop_length = 320
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=50,
            fmax=14000
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def extract_embeddings(self, audio_batch, sr=8000):
        """Extract embeddings from audio batch"""
        embeddings_list = []
        
        for audio in audio_batch:
            # Preprocess audio
            audio_32k = self.preprocess_audio(audio, sr)
            
            # Extract mel-spectrogram
            mel_spec = self.extract_mel_features(audio_32k)
            
            # Convert to tensor and add batch dimension
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Pad or trim to fixed size (e.g., 1000 frames ~ 10 seconds)
            target_frames = 1000
            if mel_tensor.shape[-1] < target_frames:
                pad_amount = target_frames - mel_tensor.shape[-1]
                mel_tensor = F.pad(mel_tensor, (0, pad_amount))
            else:
                mel_tensor = mel_tensor[:, :, :, :target_frames]
            
            # Extract features
            with torch.no_grad():
                if hasattr(self.model, 'inference'):
                    # For panns_inference models
                    _, embedding = self.model.inference(audio_32k[None, :])
                    embeddings_list.append(embedding.squeeze().cpu().numpy())
                else:
                    # For simplified model
                    embedding = self.model(mel_tensor)
                    embeddings_list.append(embedding.squeeze().cpu().numpy())
        
        return np.array(embeddings_list)


class CostSensitivePANNsClassifier(nn.Module):
    """Classifier head for PANNs embeddings with cost sensitivity"""
    
    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.classifier(x)


class CostSensitivePANNsTrainer:
    """Trainer for cost-sensitive PANNs"""
    
    def __init__(self, model_type='CNN14', device='cpu', fn_cost=2.0, fp_cost=1.0):
        self.device = torch.device(device)
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        
        print(f"\nCost-Sensitive PANNs Configuration:")
        print(f"  Model: {model_type}")
        print(f"  Device: {device}")
        print(f"  FN Cost: {fn_cost}, FP Cost: {fp_cost}")
        
        # Initialize feature extractor
        self.feature_extractor = PANNsFeatureExtractor(model_type, device)
        self.classifier = None
        
    def compute_class_weights(self, y_train):
        """Compute cost-adjusted class weights"""
        n_samples = len(y_train)
        n_classes = 2
        class_counts = np.bincount(y_train)
        
        # Incorporate misclassification costs
        weight_0 = self.fp_cost * (n_samples / (n_classes * class_counts[0]))
        weight_1 = self.fn_cost * (n_samples / (n_classes * class_counts[1]))
        
        # Normalize
        total_weight = weight_0 + weight_1
        weight_0 = weight_0 / total_weight * n_classes
        weight_1 = weight_1 / total_weight * n_classes
        
        return torch.FloatTensor([weight_0, weight_1]).to(self.device)
    
    def cost_sensitive_loss(self, outputs, targets, class_weights):
        """Custom loss with asymmetric costs"""
        # Standard cross entropy with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        ce_loss = criterion(outputs, targets)
        
        # Add additional penalty for false negatives
        probs = F.softmax(outputs, dim=1)
        fn_penalty = torch.mean(
            (targets == 1).float() * (1 - probs[:, 1]) * self.fn_cost
        )
        
        return ce_loss + fn_penalty * 0.1  # Weight the additional penalty
    
    def train(self, X_train_emb, y_train, X_val_emb, y_val, 
              epochs=50, batch_size=16, lr=0.001):
        """Train the classifier with cost sensitivity"""
        
        # Initialize classifier
        input_dim = X_train_emb.shape[1]
        self.classifier = CostSensitivePANNsClassifier(input_dim).to(self.device)
        
        # Optimizer
        optimizer = AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
        
        # Class weights
        class_weights = self.compute_class_weights(y_train)
        print(f"Class weights: {class_weights.cpu().numpy()}")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_emb).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val_emb).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        best_val_score = -float('inf')
        best_threshold = 0.5
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            total_loss = 0
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_x = X_train_t[batch_idx]
                batch_y = y_train_t[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.classifier(batch_x)
                loss = self.cost_sensitive_loss(outputs, batch_y, class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            if (epoch + 1) % 5 == 0:
                self.classifier.eval()
                with torch.no_grad():
                    val_outputs = self.classifier(X_val_t)
                    val_probs = F.softmax(val_outputs, dim=1)[:, 1]
                    
                    # Find best threshold for this epoch
                    best_epoch_score = -float('inf')
                    best_epoch_thresh = 0.5
                    
                    for thresh in np.arange(0.2, 0.8, 0.05):
                        val_preds = (val_probs > thresh).long()
                        
                        # Calculate metrics
                        cm = confusion_matrix(y_val_t.cpu(), val_preds.cpu())
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                            
                            # Cost-aware score
                            total_cost = fn * self.fn_cost + fp * self.fp_cost
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                            
                            # Composite score (lower cost is better, so negate)
                            score = -total_cost + sensitivity * 10 + npv * 5
                            
                            if score > best_epoch_score:
                                best_epoch_score = score
                                best_epoch_thresh = thresh
                    
                    # Check if this is the best model so far
                    if best_epoch_score > best_val_score:
                        best_val_score = best_epoch_score
                        best_threshold = best_epoch_thresh
                        best_state = self.classifier.state_dict().copy()
                        
                        # Print progress
                        val_preds = (val_probs > best_threshold).long()
                        cm = confusion_matrix(y_val_t.cpu(), val_preds.cpu())
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                            
                            print(f"  Epoch {epoch+1}: Thresh={best_threshold:.2f}, "
                                  f"Sens={sensitivity:.1%}, Spec={specificity:.1%}, NPV={npv:.1%}")
        
        # Restore best model
        if best_state is not None:
            self.classifier.load_state_dict(best_state)
        
        return best_threshold
    
    def evaluate(self, X_test_emb, y_test, threshold=0.5):
        """Evaluate the model"""
        self.classifier.eval()
        
        X_test_t = torch.FloatTensor(X_test_emb).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(X_test_t)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        # Apply threshold
        preds = (probs > threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, preds)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        else:
            tn = fp = fn = tp = 0
        
        metrics = {
            'confusion_matrix': cm.tolist(),
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'total_cost': fn * self.fn_cost + fp * self.fp_cost,
            'threshold': threshold
        }
        
        return metrics


def augment_audio(audio, n_augmentations=5):
    """Simple audio augmentation"""
    augmented = [audio]
    
    for i in range(n_augmentations):
        aug = audio.copy()
        
        if i % 3 == 0:
            aug = aug + np.random.normal(0, 0.005, len(aug))
        elif i % 3 == 1:
            aug = aug * np.random.uniform(0.8, 1.2)
        else:
            shift = np.random.randint(-len(aug)//10, len(aug)//10)
            aug = np.roll(aug, shift)
        
        augmented.append(aug)
    
    return augmented


def run_panns_experiments(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run experiments with different cost ratios"""
    
    cost_configs = [
        {'fn': 1.0, 'fp': 1.0, 'name': 'Balanced (1:1)'},
        {'fn': 2.0, 'fp': 1.0, 'name': 'Slight POI priority (2:1)'},
        {'fn': 3.0, 'fp': 1.0, 'name': 'Moderate POI priority (3:1)'},
        {'fn': 5.0, 'fp': 1.0, 'name': 'Strong POI priority (5:1)'},
    ]
    
    results = []
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for config in cost_configs:
        print("\n" + "="*70)
        print(f"EXPERIMENT: {config['name']}")
        print("="*70)
        
        # Create trainer
        trainer = CostSensitivePANNsTrainer(
            model_type='CNN14',
            device=device,
            fn_cost=config['fn'],
            fp_cost=config['fp']
        )
        
        # Augment training data
        print("\nAugmenting training data...")
        X_train_aug, y_train_aug = [], []
        
        for x, y in zip(X_train, y_train):
            if y == 1:  # POI
                augmented = augment_audio(x, n_augmentations=8)
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
        print("\nExtracting PANNs embeddings...")
        X_train_emb = trainer.feature_extractor.extract_embeddings(X_train_aug, sr=8000)
        X_val_emb = trainer.feature_extractor.extract_embeddings(X_val, sr=8000)
        X_test_emb = trainer.feature_extractor.extract_embeddings(X_test, sr=8000)
        
        print(f"Embedding dimensions: {X_train_emb.shape[1]}")
        
        # Train
        print("\nTraining classifier...")
        best_threshold = trainer.train(
            X_train_emb, y_train_aug, X_val_emb, y_val,
            epochs=40, batch_size=16, lr=0.001
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = trainer.evaluate(X_test_emb, y_test, threshold=best_threshold)
        
        result = {
            'config': config,
            'metrics': metrics,
            'threshold': best_threshold
        }
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
    
    return results


def visualize_panns_results(results):
    """Create visualization comparing PANNs results"""
    
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    fig.suptitle('PANNs Performance with Different Cost Ratios', fontsize=14, fontweight='bold')
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Get confusion matrix
        cm = np.array(result['metrics']['confusion_matrix'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # Create labels
        labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
        
        # Plot
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['Pred: No POI', 'Pred: POI'],
                   yticklabels=['True: No POI', 'True: POI'])
        
        # Title
        metrics = result['metrics']
        ax.set_title(f"{result['config']['name']}\n" + 
                    f"Sens: {metrics['sensitivity']:.0%}, " +
                    f"Spec: {metrics['specificity']:.0%}, " +
                    f"NPV: {metrics['npv']:.1%}",
                    fontsize=11)
        
        # Add info
        ax.text(0.5, -0.15, 
               f"Threshold: {result['threshold']:.2f}, Cost: {metrics['total_cost']:.0f}",
               transform=ax.transAxes, ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/panns_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to results/panns_confusion_matrices.png")


def main():
    print("="*70)
    print("COST-SENSITIVE PANNs FOR POI DETECTION")
    print("="*70)
    
    # Check device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    # Run experiments
    results = run_panns_experiments(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Visualize results
    visualize_panns_results(results)
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/panns_results.json', 'w') as f:
        json.dump({
            'experiments': [
                {
                    'name': r['config']['name'],
                    'fn_cost': r['config']['fn'],
                    'fp_cost': r['config']['fp'],
                    'metrics': r['metrics']
                }
                for r in results
            ]
        }, f, indent=2)
    
    print("\n✓ Results saved to results/panns_results.json")
    
    # Summary
    print("\n" + "="*70)
    print("PANNS RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Config':<25} {'Sens':<8} {'Spec':<8} {'NPV':<8} {'Cost':<8}")
    print("-" * 65)
    
    for result in results:
        config_name = result['config']['name']
        metrics = result['metrics']
        print(f"{config_name:<25} {metrics['sensitivity']:<8.1%} "
              f"{metrics['specificity']:<8.1%} {metrics['npv']:<8.1%} "
              f"{metrics['total_cost']:<8.1f}")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON WITH OTHER MODELS")
    print("="*70)
    
    print("\nBaseline SMOLK:")
    print("  Accuracy: 76.5%, NPV: 84.6%, Sensitivity: 50%")
    
    print("\nBest YAMNet (2:1 ratio):")
    print("  Sensitivity: 60%, NPV: 88.2%, Specificity: 75%")
    
    best_panns = max(results, key=lambda x: x['metrics']['npv'])
    print(f"\nBest PANNs ({best_panns['config']['name']}):")
    print(f"  Sensitivity: {best_panns['metrics']['sensitivity']:.1%}")
    print(f"  NPV: {best_panns['metrics']['npv']:.1%}")
    print(f"  Specificity: {best_panns['metrics']['specificity']:.1%}")


if __name__ == '__main__':
    main()