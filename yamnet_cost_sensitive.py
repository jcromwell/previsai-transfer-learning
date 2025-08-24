#!/usr/bin/env python
"""
Cost-Sensitive YAMNet Transfer Learning for POI Detection
Allows control over misclassification costs to optimize for clinical priorities
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import json
import librosa

import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_loader import AudioDataLoader

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CostSensitiveYAMNet:
    """YAMNet with adjustable misclassification costs"""
    
    def __init__(self, false_negative_cost=5.0, false_positive_cost=1.0):
        """
        Initialize with misclassification costs
        
        Args:
            false_negative_cost: Cost of missing a POI case (default 5.0)
            false_positive_cost: Cost of false alarm (default 1.0)
        """
        self.fn_cost = false_negative_cost
        self.fp_cost = false_positive_cost
        
        print(f"Misclassification costs:")
        print(f"  False Negative (missing POI): {self.fn_cost}")
        print(f"  False Positive (false alarm): {self.fp_cost}")
        
        # Load YAMNet
        print("\nLoading YAMNet model...")
        self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        self.target_sr = 16000
        
        self.classifier = None
        
    def preprocess_audio(self, audio, sr=8000):
        """Resample audio to 16kHz for YAMNet"""
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def extract_embeddings(self, audio_batch, sr=8000):
        """Extract YAMNet embeddings"""
        embeddings_list = []
        
        for audio in audio_batch:
            audio_16k = self.preprocess_audio(audio, sr)
            scores, embeddings, log_mel = self.yamnet(audio_16k)
            avg_embedding = tf.reduce_mean(embeddings, axis=0)
            embeddings_list.append(avg_embedding.numpy())
        
        return np.array(embeddings_list)
    
    def create_classifier(self, input_dim=1024, dropout=0.5):
        """Create cost-aware classifier"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        return model
    
    def compute_class_weights(self, y_train):
        """Compute class weights based on misclassification costs"""
        # Base class weights for imbalance
        unique_classes = np.unique(y_train)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        class_counts = np.bincount(y_train)
        
        # Incorporate misclassification costs
        # Weight for class 0 (non-POI): influenced by false positive cost
        # Weight for class 1 (POI): influenced by false negative cost
        weight_0 = self.fp_cost * (n_samples / (n_classes * class_counts[0]))
        weight_1 = self.fn_cost * (n_samples / (n_classes * class_counts[1]))
        
        # Normalize weights
        total_weight = weight_0 + weight_1
        weight_0 = weight_0 / total_weight * n_classes
        weight_1 = weight_1 / total_weight * n_classes
        
        return {0: weight_0, 1: weight_1}
    
    def custom_loss(self, y_true, y_pred):
        """Custom loss function with asymmetric costs"""
        # Convert to float
        y_true_float = tf.cast(y_true, tf.float32)
        
        # Calculate per-sample loss
        # For POI (class 1): penalize false negatives more
        # For non-POI (class 0): penalize false positives less
        
        # Cross entropy base
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Separate losses for each class
        loss_0 = -tf.where(
            tf.equal(y_true, 0),
            tf.math.log(y_pred[:, 0]) * self.fp_cost,
            0.0
        )
        
        loss_1 = -tf.where(
            tf.equal(y_true, 1),
            tf.math.log(y_pred[:, 1]) * self.fn_cost,
            0.0
        )
        
        return tf.reduce_mean(loss_0 + loss_1)
    
    def train(self, X_train_emb, y_train, X_val_emb, y_val, 
              epochs=50, batch_size=16, learning_rate=0.001):
        """Train with cost-sensitive learning"""
        
        # Create classifier
        self.classifier = self.create_classifier(input_dim=X_train_emb.shape[1])
        
        # Compute cost-aware class weights
        class_weights = self.compute_class_weights(y_train)
        print(f"\nCost-adjusted class weights: {class_weights}")
        
        # Compile with custom loss
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=self.custom_loss,
            metrics=['accuracy']
        )
        
        # Custom callback for monitoring
        class MetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, X_val, y_val, fn_cost, fp_cost):
                self.X_val = X_val
                self.y_val = y_val
                self.fn_cost = fn_cost
                self.fp_cost = fp_cost
                self.best_score = float('inf')
                self.best_weights = None
                
            def on_epoch_end(self, epoch, logs=None):
                y_pred_proba = self.model.predict(self.X_val, verbose=0)
                
                # Find optimal threshold based on costs
                best_threshold = 0.5
                best_cost = float('inf')
                
                for thresh in np.arange(0.2, 0.8, 0.05):
                    y_pred = (y_pred_proba[:, 1] > thresh).astype(int)
                    cm = confusion_matrix(self.y_val, y_pred)
                    
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                        
                        # Calculate total cost
                        total_cost = fn * self.fn_cost + fp * self.fp_cost
                        
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_threshold = thresh
                
                # Evaluate at best threshold
                y_pred = (y_pred_proba[:, 1] > best_threshold).astype(int)
                cm = confusion_matrix(self.y_val, y_pred)
                
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    total_cost = fn * self.fn_cost + fp * self.fp_cost
                    
                    if (epoch + 1) % 5 == 0:
                        print(f"\n  Epoch {epoch+1}: Cost={total_cost:.1f}, "
                              f"Thresh={best_threshold:.2f}, "
                              f"Sens={sensitivity:.1%}, Spec={specificity:.1%}, NPV={npv:.1%}")
                    
                    # Save best model based on cost
                    if total_cost < self.best_score:
                        self.best_score = total_cost
                        self.best_weights = self.model.get_weights()
                        self.best_threshold = best_threshold
        
        metrics_callback = MetricsCallback(X_val_emb, y_val, self.fn_cost, self.fp_cost)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=False
        )
        
        # Train
        history = self.classifier.fit(
            X_train_emb, y_train,
            validation_data=(X_val_emb, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[metrics_callback, early_stopping],
            verbose=1
        )
        
        # Restore best weights
        if metrics_callback.best_weights is not None:
            self.classifier.set_weights(metrics_callback.best_weights)
            print(f"\nRestored best model with cost={metrics_callback.best_score:.1f}")
            print(f"Optimal threshold: {metrics_callback.best_threshold:.2f}")
        
        return history, metrics_callback.best_threshold
    
    def evaluate_with_costs(self, X_test_emb, y_test, threshold=0.5):
        """Evaluate model with cost analysis"""
        y_pred_proba = self.classifier.predict(X_test_emb)
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        else:
            tn = fp = fn = tp = 0
        
        # Calculate costs
        fn_total_cost = fn * self.fn_cost
        fp_total_cost = fp * self.fp_cost
        total_cost = fn_total_cost + fp_total_cost
        
        # Calculate metrics
        metrics = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'false_negative_cost': fn_total_cost,
            'false_positive_cost': fp_total_cost,
            'total_cost': total_cost,
            'threshold': threshold
        }
        
        return metrics


def augment_audio_simple(audio, n_augmentations=5):
    """Simple audio augmentation"""
    augmented = [audio]
    
    for i in range(n_augmentations):
        aug = audio.copy()
        
        if i % 3 == 0:
            # Add noise
            aug = aug + np.random.normal(0, 0.005, len(aug))
        elif i % 3 == 1:
            # Amplitude scaling
            aug = aug * np.random.uniform(0.8, 1.2)
        else:
            # Time shift
            shift = np.random.randint(-len(aug)//10, len(aug)//10)
            aug = np.roll(aug, shift)
        
        augmented.append(aug)
    
    return augmented


def experiment_with_costs(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run experiments with different cost ratios"""
    
    # Different cost scenarios
    cost_configs = [
        {'fn': 1.0, 'fp': 1.0, 'name': 'Balanced'},
        {'fn': 2.0, 'fp': 1.0, 'name': 'Slight POI priority'},
        {'fn': 5.0, 'fp': 1.0, 'name': 'Strong POI priority'},
        {'fn': 10.0, 'fp': 1.0, 'name': 'Very strong POI priority'},
        {'fn': 3.0, 'fp': 1.0, 'name': 'Moderate POI priority'},
        {'fn': 1.0, 'fp': 5.0, 'name': 'Minimize false alarms'},
    ]
    
    results = []
    
    for config in cost_configs:
        print("\n" + "="*70)
        print(f"EXPERIMENT: {config['name']}")
        print(f"FN Cost: {config['fn']}, FP Cost: {config['fp']}")
        print("="*70)
        
        # Create model with specific costs
        model = CostSensitiveYAMNet(
            false_negative_cost=config['fn'],
            false_positive_cost=config['fp']
        )
        
        # Augment training data
        print("\nAugmenting POI samples...")
        X_train_aug, y_train_aug = [], []
        
        for x, y in zip(X_train, y_train):
            if y == 1:  # POI
                augmented = augment_audio_simple(x, n_augmentations=8)
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
        print("\nExtracting embeddings...")
        X_train_emb = model.extract_embeddings(X_train_aug, sr=8000)
        X_val_emb = model.extract_embeddings(X_val, sr=8000)
        X_test_emb = model.extract_embeddings(X_test, sr=8000)
        
        # Train
        print("\nTraining classifier...")
        history, best_threshold = model.train(
            X_train_emb, y_train_aug, X_val_emb, y_val,
            epochs=30, batch_size=16, learning_rate=0.001
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = model.evaluate_with_costs(X_test_emb, y_test, threshold=best_threshold)
        
        # Store results
        result = {
            'config': config,
            'metrics': metrics,
            'threshold': best_threshold
        }
        results.append(result)
        
        # Print results
        print(f"\nResults for {config['name']}:")
        print(f"  Threshold: {best_threshold:.2f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
        print(f"  Specificity: {metrics['specificity']:.1%}")
        print(f"  NPV: {metrics['npv']:.1%}")
        print(f"  Total Cost: {metrics['total_cost']:.1f}")
        print(f"    FN Cost: {metrics['false_negative_cost']:.1f}")
        print(f"    FP Cost: {metrics['false_positive_cost']:.1f}")
    
    return results


def main():
    print("="*70)
    print("COST-SENSITIVE YAMNET FOR POI DETECTION")
    print("="*70)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
    
    # Load data
    print("\nLoading audio data...")
    DATA_DIR = str(Path.cwd() / 'raw-audio')
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=8000)
    file_dict = loader.get_file_list()
    
    print(f"Found: {len(file_dict['poi'])} POI, {len(file_dict['nopoi'])} Non-POI")
    
    # Load samples
    X_all, y_all = [], []
    
    # Load POI
    for file_path in file_dict['poi']:
        audio, sr = loader.load_audio_file(file_path)
        if audio is not None:
            audio = audio[:20*sr]
            if len(audio) < 20*sr:
                audio = np.pad(audio, (0, 20*sr - len(audio)), 'constant')
            X_all.append(audio)
            y_all.append(1)
    
    # Load Non-POI
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
    
    # Split data
    poi_indices = np.where(y_all == 1)[0]
    nopoi_indices = np.where(y_all == 0)[0]
    
    np.random.seed(42)
    poi_indices = np.random.permutation(poi_indices)
    nopoi_indices = np.random.permutation(nopoi_indices)
    
    # Same splits as before
    test_indices = np.concatenate([poi_indices[:5], nopoi_indices[:20]])
    val_indices = np.concatenate([poi_indices[5:10], nopoi_indices[20:40]])
    train_indices = np.concatenate([poi_indices[10:], nopoi_indices[40:]])
    
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Run experiments
    results = experiment_with_costs(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*70)
    
    print(f"\n{'Config':<25} {'Sens':<8} {'Spec':<8} {'NPV':<8} {'Cost':<10} {'Thresh':<8}")
    print("-" * 75)
    
    for result in results:
        config_name = result['config']['name']
        metrics = result['metrics']
        print(f"{config_name:<25} {metrics['sensitivity']:<8.1%} {metrics['specificity']:<8.1%} "
              f"{metrics['npv']:<8.1%} {metrics['total_cost']:<10.1f} {result['threshold']:<8.2f}")
    
    # Find best configurations
    best_npv = max(results, key=lambda x: x['metrics']['npv'])
    best_sens = max(results, key=lambda x: x['metrics']['sensitivity'])
    best_cost = min(results, key=lambda x: x['metrics']['total_cost'])
    
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    
    print(f"\nBest NPV: {best_npv['config']['name']}")
    print(f"  NPV: {best_npv['metrics']['npv']:.1%}")
    print(f"  Sensitivity: {best_npv['metrics']['sensitivity']:.1%}")
    
    print(f"\nBest Sensitivity: {best_sens['config']['name']}")
    print(f"  Sensitivity: {best_sens['metrics']['sensitivity']:.1%}")
    print(f"  NPV: {best_sens['metrics']['npv']:.1%}")
    
    print(f"\nLowest Cost: {best_cost['config']['name']}")
    print(f"  Total Cost: {best_cost['metrics']['total_cost']:.1f}")
    print(f"  Sensitivity: {best_cost['metrics']['sensitivity']:.1%}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    
    save_data = {
        'experiments': [
            {
                'name': r['config']['name'],
                'fn_cost': r['config']['fn'],
                'fp_cost': r['config']['fp'],
                'metrics': r['metrics']
            }
            for r in results
        ],
        'best_npv': best_npv['config']['name'],
        'best_sensitivity': best_sens['config']['name'],
        'best_cost': best_cost['config']['name']
    }
    
    with open('results/yamnet_cost_sensitive_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print("\n✓ Results saved to results/yamnet_cost_sensitive_results.json")
    
    # Comparison with SMOLK
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    
    print("\nSMOLK Baseline:")
    print("  Accuracy: 76.5%, NPV: 84.6%, Sensitivity: 50%")
    
    print(f"\nBest Cost-Sensitive YAMNet ({best_sens['config']['name']}):")
    print(f"  Sensitivity: {best_sens['metrics']['sensitivity']:.1%} "
          f"({(best_sens['metrics']['sensitivity'] - 0.50)*100:+.1f}% vs SMOLK)")
    print(f"  NPV: {best_sens['metrics']['npv']:.1%} "
          f"({(best_sens['metrics']['npv'] - 0.846)*100:+.1f}% vs SMOLK)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    print("\nFor clinical use where missing POI is costly:")
    print(f"  Use FN:FP cost ratio of {best_sens['config']['fn']}:{best_sens['config']['fp']}")
    print(f"  This achieves {best_sens['metrics']['sensitivity']:.1%} sensitivity")
    print(f"  With {best_sens['metrics']['npv']:.1%} NPV")


if __name__ == '__main__':
    main()