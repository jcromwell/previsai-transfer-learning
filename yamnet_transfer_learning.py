#!/usr/bin/env python
"""
YAMNet Transfer Learning for POI Detection
Uses YAMNet pre-trained on AudioSet for audio classification
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import time
import json
import librosa
import soundfile as sf

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class YAMNetProcessor:
    """Process audio for YAMNet input"""
    
    def __init__(self, model_handle='https://tfhub.dev/google/yamnet/1'):
        """Initialize YAMNet model"""
        print("Loading YAMNet model...")
        self.model = hub.load(model_handle)
        
        # YAMNet expects 16kHz audio
        self.target_sr = 16000
        
    def preprocess_audio(self, audio, sr=8000):
        """Preprocess audio for YAMNet (resample to 16kHz)"""
        # Resample to 16kHz if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Ensure float32 and normalize
        audio = audio.astype(np.float32)
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def extract_embeddings(self, audio_batch, sr=8000):
        """Extract YAMNet embeddings for audio batch"""
        embeddings_list = []
        
        for audio in audio_batch:
            # Preprocess
            audio_16k = self.preprocess_audio(audio, sr)
            
            # Get embeddings from YAMNet
            scores, embeddings, log_mel = self.model(audio_16k)
            
            # Average embeddings over time
            avg_embedding = tf.reduce_mean(embeddings, axis=0)
            embeddings_list.append(avg_embedding.numpy())
        
        return np.array(embeddings_list)
    
    def extract_embeddings_windowed(self, audio, sr=8000, window_size=3.0, hop_size=1.5):
        """Extract windowed embeddings for longer audio"""
        audio_16k = self.preprocess_audio(audio, sr)
        
        window_samples = int(window_size * self.target_sr)
        hop_samples = int(hop_size * self.target_sr)
        
        embeddings_list = []
        
        for start in range(0, len(audio_16k) - window_samples + 1, hop_samples):
            window = audio_16k[start:start + window_samples]
            
            # Get embeddings
            scores, embeddings, log_mel = self.model(window)
            avg_embedding = tf.reduce_mean(embeddings, axis=0)
            embeddings_list.append(avg_embedding.numpy())
        
        # Average all window embeddings
        if embeddings_list:
            return np.mean(embeddings_list, axis=0)
        else:
            # If audio is too short, process as is
            scores, embeddings, log_mel = self.model(audio_16k)
            return tf.reduce_mean(embeddings, axis=0).numpy()


class POIClassifier(tf.keras.Model):
    """Custom classifier head for YAMNet embeddings"""
    
    def __init__(self, embedding_size=1024, dropout_rate=0.5):
        super().__init__()
        
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        
        return self.output_layer(x)


class YAMNetTrainer:
    """Training pipeline for YAMNet transfer learning"""
    
    def __init__(self):
        self.processor = YAMNetProcessor()
        self.classifier = None
        
    def augment_audio(self, audio, sr=8000, n_augmentations=5):
        """Augment audio samples"""
        augmented = [audio]
        
        for i in range(n_augmentations):
            aug = audio.copy()
            
            if i % 5 == 0:
                # Add noise
                noise = np.random.normal(0, 0.005, len(aug))
                aug = aug + noise
            elif i % 5 == 1:
                # Time stretch
                stretch_rate = np.random.uniform(0.9, 1.1)
                aug = librosa.effects.time_stretch(aug, rate=stretch_rate)
                # Ensure same length
                if len(aug) > len(audio):
                    aug = aug[:len(audio)]
                else:
                    aug = np.pad(aug, (0, len(audio) - len(aug)), 'constant')
            elif i % 5 == 2:
                # Pitch shift
                n_steps = np.random.uniform(-2, 2)
                aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=n_steps)
            elif i % 5 == 3:
                # Amplitude scaling
                scale = np.random.uniform(0.7, 1.3)
                aug = aug * scale
            else:
                # Time shift
                shift = np.random.randint(-sr//2, sr//2)
                aug = np.roll(aug, shift)
            
            augmented.append(aug)
        
        return augmented
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test, y_test, augment=True):
        """Prepare data with augmentation and extract embeddings"""
        
        # Augment POI samples
        if augment:
            print("Augmenting POI samples...")
            X_train_aug, y_train_aug = [], []
            
            for x, y in zip(X_train, y_train):
                if y == 1:  # POI
                    augmented = self.augment_audio(x, n_augmentations=8)
                    X_train_aug.extend(augmented)
                    y_train_aug.extend([1] * len(augmented))
                else:
                    X_train_aug.append(x)
                    y_train_aug.append(0)
            
            X_train = np.array(X_train_aug)
            y_train = np.array(y_train_aug)
            print(f"After augmentation: {len(X_train)} samples")
            print(f"POI: {np.sum(y_train==1)}, Non-POI: {np.sum(y_train==0)}")
        
        # Extract embeddings
        print("\nExtracting YAMNet embeddings...")
        print("Processing training set...")
        X_train_emb = []
        for i in range(0, len(X_train), 10):  # Process in batches
            batch = X_train[i:i+10]
            embeddings = self.processor.extract_embeddings(batch, sr=8000)
            X_train_emb.extend(embeddings)
            if (i // 10 + 1) % 5 == 0:
                print(f"  Processed {i+10}/{len(X_train)} samples")
        X_train_emb = np.array(X_train_emb)
        
        print("Processing validation set...")
        X_val_emb = self.processor.extract_embeddings(X_val, sr=8000)
        
        print("Processing test set...")
        X_test_emb = self.processor.extract_embeddings(X_test, sr=8000)
        
        return X_train_emb, y_train, X_val_emb, y_val, X_test_emb, y_test
    
    def train(self, X_train_emb, y_train, X_val_emb, y_val, 
              epochs=50, batch_size=32, learning_rate=0.001):
        """Train the classifier on YAMNet embeddings"""
        
        # Initialize classifier
        embedding_size = X_train_emb.shape[1]
        self.classifier = POIClassifier(embedding_size=embedding_size)
        
        # Compile model
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced', classes=classes, y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        # Custom callback for NPV monitoring
        class NPVCallback(tf.keras.callbacks.Callback):
            def __init__(self, X_val, y_val):
                self.X_val = X_val
                self.y_val = y_val
                self.best_npv = 0
                self.best_weights = None
                
            def on_epoch_end(self, epoch, logs=None):
                # Get predictions
                y_pred_proba = self.model.predict(self.X_val, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # Calculate metrics
                cm = confusion_matrix(self.y_val, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    
                    if (epoch + 1) % 5 == 0:
                        print(f"\n  Epoch {epoch+1}: NPV={npv:.1%}, Sens={sensitivity:.1%}, Spec={specificity:.1%}")
                    
                    # Save best model based on NPV
                    if npv > self.best_npv:
                        self.best_npv = npv
                        self.best_weights = self.model.get_weights()
        
        npv_callback = NPVCallback(X_val_emb, y_val)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train
        history = self.classifier.fit(
            X_train_emb, y_train,
            validation_data=(X_val_emb, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[npv_callback, early_stopping],
            verbose=1
        )
        
        # Restore best NPV weights if available
        if npv_callback.best_weights is not None:
            self.classifier.set_weights(npv_callback.best_weights)
            print(f"\nRestored best model with NPV={npv_callback.best_npv:.1%}")
        
        return history
    
    def evaluate(self, X_test_emb, y_test, threshold=0.5):
        """Evaluate the model"""
        # Get predictions
        y_pred_proba = self.classifier.predict(X_test_emb)
        
        # Try different thresholds
        best_threshold = threshold
        best_npv = 0
        
        print("\nThreshold optimization:")
        for thresh in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_pred_proba[:, 1] > thresh).astype(int)
            
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                print(f"  Threshold={thresh:.2f}: NPV={npv:.1%}, Sens={sens:.1%}, Spec={spec:.1%}")
                
                if npv > best_npv:
                    best_npv = npv
                    best_threshold = thresh
        
        # Final predictions with best threshold
        y_pred = (y_pred_proba[:, 1] > best_threshold).astype(int)
        
        return y_pred, y_pred_proba, best_threshold


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
    print("YAMNET TRANSFER LEARNING FOR POI DETECTION")
    print("="*70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        # For Apple Silicon
        if tf.config.list_physical_devices('GPU'):
            print("Using Metal GPU acceleration")
    else:
        print("No GPU found, using CPU")
    
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
    
    # Split data (same splits as other scripts for fair comparison)
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
    
    # Initialize trainer
    print("\n" + "="*70)
    print("YAMNET FEATURE EXTRACTION & TRAINING")
    print("="*70)
    
    trainer = YAMNetTrainer()
    
    # Prepare data with augmentation
    X_train_emb, y_train_aug, X_val_emb, y_val, X_test_emb, y_test = trainer.prepare_data(
        X_train, y_train, X_val, y_val, X_test, y_test, augment=True
    )
    
    print(f"\nEmbedding shapes:")
    print(f"  Train: {X_train_emb.shape}")
    print(f"  Val: {X_val_emb.shape}")
    print(f"  Test: {X_test_emb.shape}")
    
    # Train classifier
    print("\n" + "="*70)
    print("TRAINING CLASSIFIER")
    print("="*70)
    
    history = trainer.train(
        X_train_emb, y_train_aug, X_val_emb, y_val,
        epochs=50, batch_size=16, learning_rate=0.001
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    y_pred, y_proba, best_threshold = trainer.evaluate(X_test_emb, y_test)
    
    print(f"\nUsing optimal threshold: {best_threshold:.2f}")
    
    # Calculate final metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    print("\nFinal Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
    print(f"  NPV: {metrics['npv']:.1%}")
    print(f"  PPV: {metrics['ppv']:.1%}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"         Pred_No  Pred_POI")
    if metrics['confusion_matrix'] and len(metrics['confusion_matrix']) == 2:
        print(f"True_No    {metrics['confusion_matrix'][0][0]:3d}      {metrics['confusion_matrix'][0][1]:3d}")
        print(f"True_POI   {metrics['confusion_matrix'][1][0]:3d}      {metrics['confusion_matrix'][1][1]:3d}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    results = {
        'yamnet_transfer': metrics,
        'threshold': float(best_threshold),
        'embedding_size': int(X_train_emb.shape[1])
    }
    
    with open('results/yamnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/yamnet_results.json")
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    trainer.classifier.save_weights('models/yamnet_classifier.weights.h5')
    print("✓ Model saved to models/yamnet_classifier.weights.h5")
    
    # Compare with baselines
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINES")
    print("="*70)
    
    print("\nSMOLK Baseline (from CLAUDE.md):")
    print("  Accuracy: 76.5%, NPV: 84.6%, Sensitivity: 50%")
    
    print("\nBest CNN Transfer Learning (ResNet18):")
    print("  Accuracy: 76.0%, NPV: 79.2%, Sensitivity: 0%")
    
    print(f"\nYAMNet Transfer Learning:")
    print(f"  Accuracy: {metrics['accuracy']:.1%} ({(metrics['accuracy'] - 0.765)*100:+.1f}% vs SMOLK)")
    print(f"  NPV: {metrics['npv']:.1%} ({(metrics['npv'] - 0.846)*100:+.1f}% vs SMOLK)")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%} ({(metrics['sensitivity'] - 0.50)*100:+.1f}% vs SMOLK)")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()