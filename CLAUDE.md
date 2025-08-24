# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Transfer Learning** project for medical audio classification, specifically focused on detecting postoperative ileus (POI) from intestinal audio recordings. The project implements and compares multiple state-of-the-art transfer learning approaches:

- **SMOLK** (Sparse Mixture of Learned Kernels) - Custom architecture with learnable kernels
- **PANNs** (Pre-trained Audio Neural Networks) - CNN14 architecture
- **YAMNet** (Yet Another Mobile Network) - MobileNet-based audio classification
- **AST** (Audio Spectrogram Transformer) - Vision transformer adapted for audio

The goal is to leverage pre-trained models and transfer learning techniques to build an efficient, accurate classification pipeline suitable for medical audio analysis with limited training data.

## Project Structure

```
previsai-transfer-learning/
├── raw-audio/
│   ├── poi/                  # Subjects who developed postoperative ileus
│   │   └── foreground/       # Intestinal recordings (17 samples)
│   └── nopoi/                # Subjects who did NOT develop POI
│       └── foreground/       # Intestinal recordings (73 samples)
├── results/                  # Training results and metrics
├── *.py                      # Implementation files for each model
└── requirements.txt          # Python dependencies
```

## Audio Data Conventions

- **Sample Rate**: 8 kHz (8000 Hz) - Critical for all models
- **Duration**: 4-minute WAV recordings per file
- **Processing Segments**: Model-specific (20-30 seconds for most)
- **Dataset Size**: 17 POI samples, 73 Non-POI samples (severe class imbalance)
- Focus on `foreground/` directories for classification tasks

## Transfer Learning Models

### 1. SMOLK (Sparse Mixture of Learned Kernels)
- **Files**: `smolk_implementation.py`, `smolk_improved.py`
- **Architecture**: 12 mixed-size kernels (0.5s, 1.0s, 2.0s at 8kHz)
- **Performance**: 76.5% accuracy, 84.6% NPV
- **Parameters**: 112K (can be pruned to 84K)
- **Best for**: Edge deployment, interpretability

### 2. PANNs (Pre-trained Audio Neural Networks)
- **Files**: `panns_simple_gpu.py`, `panns_cost_sensitive.py`, `panns_ratio_2_5.py`, `panns_third_seed_test.py`
- **Architecture**: CNN14 with 14 convolutional layers
- **Pre-trained on**: AudioSet (2M+ audio clips)
- **Performance**: Varies by configuration (see results/)
- **Best for**: General audio classification tasks

### 3. YAMNet (Yet Another Mobile Network)
- **Files**: `yamnet_transfer_learning.py`, `yamnet_cost_sensitive.py`, `yamnet_additional_ratios.py`
- **Architecture**: MobileNetV1 adapted for audio
- **Pre-trained on**: AudioSet
- **Performance**: Strong with cost-sensitive learning
- **Best for**: Mobile/edge deployment

### 4. AST (Audio Spectrogram Transformer)
- **Files**: `ast_simple.py`, `ast_cost_sensitive.py`
- **Architecture**: Vision Transformer (ViT) for audio spectrograms
- **Pre-trained on**: AudioSet/ImageNet
- **Performance**: State-of-the-art on many benchmarks
- **Best for**: High accuracy when computational resources available

## Key Implementation Details

### Data Loading
All models use similar data loading patterns:
- Load 4-minute WAV files from raw-audio/
- Segment into appropriate lengths for each model
- Apply model-specific preprocessing (mel-spectrograms, etc.)

### Class Imbalance Handling
- Cost-sensitive learning with class weights
- Various pos_weight ratios tested (1.5, 2.0, 2.5, 3.0)
- Stratified train/validation splits

### GPU Acceleration
- MPS support for Apple Silicon (device='mps')
- CUDA support for NVIDIA GPUs
- Automatic device detection in most scripts

## Running the Models

```bash
# SMOLK
python smolk_implementation.py

# PANNs
python panns_simple_gpu.py
python panns_cost_sensitive.py

# YAMNet
python yamnet_transfer_learning.py
python yamnet_cost_sensitive.py

# AST
python ast_simple.py
python ast_cost_sensitive.py

# Visualization
python final_comparison_visualization.py
python visualize_all_confusion_matrices.py
```

## Performance Comparison

Results are stored in `results/` directory as JSON files. Key metrics:
- Accuracy
- NPV (Negative Predictive Value) - Critical for screening
- Sensitivity/Specificity
- Training time

Use `final_model_comparison.py` to compare all models.

## Transfer Learning Benefits

1. **Limited Data**: Only 90 total samples (17 POI, 73 non-POI)
2. **Pre-trained Features**: Leverage features learned from millions of audio samples
3. **Faster Training**: Fine-tuning vs training from scratch
4. **Better Generalization**: Pre-trained models often generalize better

## Best Practices

1. **Always use 8kHz sample rate** - Audio files are natively 8kHz
2. **Monitor for overfitting** - Small dataset prone to overfitting
3. **Use stratified splits** - Maintain class distribution
4. **Apply strong regularization** - Dropout, weight decay
5. **Ensemble methods** - Combine predictions from multiple models

## Dependencies

Key libraries required:
- PyTorch (SMOLK, PANNs, AST)
- TensorFlow (YAMNet)
- transformers (AST)
- librosa (audio processing)
- scikit-learn (metrics, splitting)
- numpy, matplotlib, seaborn (data manipulation and visualization)

See `requirements.txt` for complete list.

## Clinical Integration Notes

- NPV >80% suitable for initial screening
- Always combine with clinical assessment
- Not for sole diagnostic use
- Consider ensemble of best performing models

## Hardware Requirements

- **Minimum**: 8GB RAM, any CPU
- **Recommended**: 16GB RAM, GPU (NVIDIA/Apple Silicon)
- **Training Time**: Varies from minutes (YAMNet) to hours (AST)