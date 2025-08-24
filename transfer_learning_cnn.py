#!/usr/bin/env python
"""
Transfer Learning with Pre-trained CNNs for POI Detection
Uses spectrograms as input to leverage ImageNet pre-trained models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import time
import json
from PIL import Image
import io

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class SpectrogramGenerator:
    """Convert audio to spectrograms for CNN input"""
    
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def audio_to_spectrogram(self, audio, spec_type='mel'):
        """Convert audio to spectrogram image"""
        
        if spec_type == 'mel':
            # Mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            
        elif spec_type == 'stft':
            # Regular spectrogram
            D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
        elif spec_type == 'mfcc':
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=40,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            S_dB = mfccs
            
        else:
            raise ValueError(f"Unknown spectrogram type: {spec_type}")
        
        # Normalize to 0-255 range for image
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
        S_img = (S_norm * 255).astype(np.uint8)
        
        # Convert to 3-channel RGB for pre-trained models
        # Stack the same spectrogram 3 times
        S_rgb = np.stack([S_img, S_img, S_img], axis=0)
        
        return S_rgb
    
    def batch_audio_to_spectrograms(self, audio_batch, spec_type='mel'):
        """Convert batch of audio to spectrograms"""
        spectrograms = []
        for audio in audio_batch:
            spec = self.audio_to_spectrogram(audio, spec_type)
            spectrograms.append(spec)
        return np.array(spectrograms)


class AudioSpectrogramDataset(Dataset):
    """Dataset that converts audio to spectrograms on the fly"""
    
    def __init__(self, X, y, spec_generator, spec_type='mel', augment=False):
        self.X = X
        self.y = y
        self.spec_generator = spec_generator
        self.spec_type = spec_type
        self.augment = augment
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=5),
        ])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        audio = self.X[idx]
        label = self.y[idx]
        
        # Generate spectrogram
        spec = self.spec_generator.audio_to_spectrogram(audio, self.spec_type)
        
        # Convert to tensor
        spec_tensor = torch.FloatTensor(spec) / 255.0
        
        # Apply augmentation if training
        if self.augment and np.random.random() < 0.5:
            spec_tensor = self.augment_transform(spec_tensor)
        
        # Apply ImageNet normalization
        spec_tensor = self.normalize(spec_tensor)
        
        return spec_tensor, label


class TransferLearningModel(nn.Module):
    """Transfer learning model with customizable backbone"""
    
    def __init__(self, backbone='mobilenet_v2', num_classes=2, dropout=0.5):
        super().__init__()
        
        if backbone == 'mobilenet_v2':
            # Load MobileNetV2
            self.base_model = models.mobilenet_v2(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            
            # Replace classifier
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            
        elif backbone == 'efficientnet_b0':
            # Load EfficientNet-B0
            self.base_model = models.efficientnet_b0(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            
            # Replace classifier
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            
        elif backbone == 'resnet18':
            # Load ResNet18 (lighter than ResNet50)
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
            
            # Replace final layer
            self.base_model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone = backbone
        
    def forward(self, x):
        return self.base_model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier"""
        if self.backbone == 'mobilenet_v2':
            for param in self.base_model.features.parameters():
                param.requires_grad = False
                
        elif self.backbone == 'efficientnet_b0':
            for param in self.base_model.features.parameters():
                param.requires_grad = False
                
        elif self.backbone == 'resnet18':
            # Freeze all layers except fc
            for name, param in self.base_model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
    
    def unfreeze_last_layers(self, n_layers=2):
        """Unfreeze last n layers for fine-tuning"""
        if self.backbone == 'mobilenet_v2':
            # Unfreeze last n conv blocks
            for i in range(len(self.base_model.features) - n_layers, len(self.base_model.features)):
                for param in self.base_model.features[i].parameters():
                    param.requires_grad = True
                    
        elif self.backbone == 'efficientnet_b0':
            # Unfreeze last n blocks
            for i in range(len(self.base_model.features) - n_layers, len(self.base_model.features)):
                for param in self.base_model.features[i].parameters():
                    param.requires_grad = True
                    
        elif self.backbone == 'resnet18':
            # Unfreeze layer4
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True


class TransferLearningTrainer:
    """Trainer for transfer learning models"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Trainer using device: {self.device}")
        
    def train_model(self, model, train_loader, val_loader, epochs=30, lr=0.001, 
                   class_weights=None, patience=10):
        """Train the model with early stopping"""
        
        model = model.to(self.device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=lr, weight_decay=1e-4)
        
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        best_val_npv = 0
        best_model_state = None
        patience_counter = 0
        
        train_history = {'loss': [], 'val_npv': [], 'val_sens': [], 'val_spec': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_history['loss'].append(avg_loss)
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Calculate metrics
            cm = confusion_matrix(all_labels, all_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                train_history['val_npv'].append(npv)
                train_history['val_sens'].append(sensitivity)
                train_history['val_spec'].append(specificity)
                
                # Save best model based on NPV
                if npv > best_val_npv:
                    best_val_npv = npv
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                          f"NPV={npv:.1%}, Sens={sensitivity:.1%}, Spec={specificity:.1%}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with NPV={best_val_npv:.1%}")
        
        return model, train_history
    
    def fine_tune(self, model, train_loader, val_loader, epochs=10, lr=0.0001, 
                  n_unfreeze=2, class_weights=None):
        """Fine-tune the model by unfreezing last layers"""
        
        print(f"\nFine-tuning: Unfreezing last {n_unfreeze} layers")
        model.unfreeze_last_layers(n_unfreeze)
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})")
        
        # Train with lower learning rate
        model, history = self.train_model(
            model, train_loader, val_loader, 
            epochs=epochs, lr=lr, class_weights=class_weights
        )
        
        return model, history


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    cm = confusion_matrix(y_true, y_pred)
    
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
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }


def main():
    print("="*70)
    print("TRANSFER LEARNING WITH PRE-TRAINED CNNs FOR POI DETECTION")
    print("="*70)
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading audio data...")
    DATA_DIR = str(Path.cwd() / 'raw-audio')
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=8000)
    file_dict = loader.get_file_list()
    
    print(f"Found: {len(file_dict['poi'])} POI, {len(file_dict['nopoi'])} Non-POI")
    
    # Load all samples
    X_all, y_all = [], []
    
    print("\nLoading POI samples...")
    for file_path in file_dict['poi']:
        audio, sr = loader.load_audio_file(file_path)
        if audio is not None:
            # Use 20-second segments
            audio = audio[:20*sr]
            if len(audio) < 20*sr:
                audio = np.pad(audio, (0, 20*sr - len(audio)), 'constant')
            X_all.append(audio)
            y_all.append(1)
    
    print(f"Loaded {len(X_all)} POI samples")
    
    print("\nLoading Non-POI samples...")
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
    
    # Split data (same as train_enhanced_mps.py for fair comparison)
    poi_indices = np.where(y_all == 1)[0]
    nopoi_indices = np.where(y_all == 0)[0]
    
    np.random.seed(42)
    poi_indices = np.random.permutation(poi_indices)
    nopoi_indices = np.random.permutation(nopoi_indices)
    
    # Test: 5 POI, 20 Non-POI
    test_indices = np.concatenate([poi_indices[:5], nopoi_indices[:20]])
    # Val: 5 POI, 20 Non-POI  
    val_indices = np.concatenate([poi_indices[5:10], nopoi_indices[20:40]])
    # Train: 7 POI, 33 Non-POI
    train_indices = np.concatenate([poi_indices[10:], nopoi_indices[40:]])
    
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Train: {np.sum(y_train==1)} POI, {np.sum(y_train==0)} Non-POI")
    print(f"Val: {np.sum(y_val==1)} POI, {np.sum(y_val==0)} Non-POI")
    print(f"Test: {np.sum(y_test==1)} POI, {np.sum(y_test==0)} Non-POI")
    
    # Data augmentation for POI samples
    print("\n" + "="*70)
    print("DATA AUGMENTATION")
    print("="*70)
    
    X_train_aug, y_train_aug = [], []
    for x, y in zip(X_train, y_train):
        X_train_aug.append(x)
        y_train_aug.append(y)
        
        if y == 1:  # POI - augment heavily
            for i in range(8):  # 8 augmented versions
                aug = x.copy()
                
                # Simple augmentations
                if i % 4 == 0:
                    aug = aug + np.random.normal(0, 0.005, len(aug))
                elif i % 4 == 1:
                    aug = aug * np.random.uniform(0.8, 1.2)
                elif i % 4 == 2:
                    shift = np.random.randint(-len(aug)//20, len(aug)//20)
                    aug = np.roll(aug, shift)
                else:
                    # Combine augmentations
                    aug = aug * np.random.uniform(0.9, 1.1)
                    aug = aug + np.random.normal(0, 0.003, len(aug))
                
                X_train_aug.append(aug)
                y_train_aug.append(1)
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    print(f"Augmented training set: {len(X_train_aug)} samples")
    print(f"POI: {np.sum(y_train_aug==1)}, Non-POI: {np.sum(y_train_aug==0)}")
    
    # Create spectrogram generator
    spec_gen = SpectrogramGenerator(sample_rate=8000)
    
    # Create datasets
    train_dataset = AudioSpectrogramDataset(X_train_aug, y_train_aug, spec_gen, 
                                           spec_type='mel', augment=True)
    val_dataset = AudioSpectrogramDataset(X_val, y_val, spec_gen, 
                                         spec_type='mel', augment=False)
    test_dataset = AudioSpectrogramDataset(X_test, y_test, spec_gen, 
                                          spec_type='mel', augment=False)
    
    # Create data loaders
    batch_size = 16 if device == 'mps' else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights
    class_counts = np.bincount(y_train_aug)
    class_weights = len(y_train_aug) / (2 * class_counts)
    print(f"\nClass weights: POI={class_weights[1]:.2f}, Non-POI={class_weights[0]:.2f}")
    
    # Test different backbones
    backbones = ['mobilenet_v2', 'efficientnet_b0', 'resnet18']
    results = {}
    
    for backbone in backbones:
        print("\n" + "="*70)
        print(f"TRAINING {backbone.upper()}")
        print("="*70)
        
        # Create model
        model = TransferLearningModel(backbone=backbone, num_classes=2, dropout=0.5)
        
        # Freeze backbone initially
        model.freeze_backbone()
        
        # Count parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total:,} total, {trainable:,} trainable ({trainable/total:.1%})")
        
        # Create trainer
        trainer = TransferLearningTrainer(device=device)
        
        # Stage 1: Train classifier only
        print("\nStage 1: Training classifier layers only...")
        model, history1 = trainer.train_model(
            model, train_loader, val_loader,
            epochs=30, lr=0.001, class_weights=class_weights
        )
        
        # Stage 2: Fine-tune
        print("\nStage 2: Fine-tuning with unfrozen layers...")
        model, history2 = trainer.fine_tune(
            model, train_loader, val_loader,
            epochs=15, lr=0.0001, n_unfreeze=2, class_weights=class_weights
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        results[backbone] = metrics
        
        print(f"\n{backbone} Test Results:")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  PPV: {metrics['ppv']:.1%}")
        
        # Save model
        Path('models').mkdir(exist_ok=True)
        torch.save(model.state_dict(), f'models/transfer_{backbone}.pth')
        print(f"✓ Model saved to models/transfer_{backbone}.pth")
        
        # Clear GPU memory
        if device == 'mps':
            torch.mps.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nComparison of all models:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'NPV':<10}")
    print("-" * 64)
    
    for backbone, metrics in results.items():
        print(f"{backbone:<20} {metrics['accuracy']:<10.1%} {metrics['sensitivity']:<12.1%} "
              f"{metrics['specificity']:<12.1%} {metrics['npv']:<10.1%}")
    
    # Find best model
    best_npv = 0
    best_model = None
    for backbone, metrics in results.items():
        if metrics['npv'] > best_npv:
            best_npv = metrics['npv']
            best_model = backbone
    
    print(f"\n✓ Best model: {best_model} with NPV={best_npv:.1%}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/transfer_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/transfer_learning_results.json")
    
    # Compare with SMOLK baseline
    print("\n" + "="*70)
    print("COMPARISON WITH SMOLK")
    print("="*70)
    
    print("\nSMOLK Baseline (from CLAUDE.md):")
    print("  Accuracy: 76.5%, NPV: 84.6%, Sensitivity: 50%")
    
    best_metrics = results[best_model]
    print(f"\nBest Transfer Learning ({best_model}):")
    print(f"  Accuracy: {best_metrics['accuracy']:.1%} ({(best_metrics['accuracy'] - 0.765)*100:+.1f}%)")
    print(f"  NPV: {best_metrics['npv']:.1%} ({(best_metrics['npv'] - 0.846)*100:+.1f}%)")
    print(f"  Sensitivity: {best_metrics['sensitivity']:.1%} ({(best_metrics['sensitivity'] - 0.50)*100:+.1f}%)")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()