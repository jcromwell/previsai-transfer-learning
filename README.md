# Transfer Learning for POI Audio Classification

## Overview

This repository implements state-of-the-art transfer learning approaches for detecting postoperative ileus (POI) from intestinal audio recordings. The project addresses the challenge of limited medical audio data (90 samples total) by leveraging pre-trained models from large-scale audio datasets.

## Models Implemented

### 1. **SMOLK** (Sparse Mixture of Learned Kernels)
- Custom architecture with 12 learnable kernels
- 76.5% accuracy, 84.6% NPV
- Interpretable and efficient (84K parameters after pruning)

### 2. **PANNs** (Pre-trained Audio Neural Networks)
- CNN14 architecture pre-trained on AudioSet
- Transfer learning with custom classification head
- Multiple configurations for cost-sensitive learning

### 3. **YAMNet** (Yet Another Mobile Network)
- MobileNet-based audio classifier
- Pre-trained on 521 AudioSet classes
- Optimized for edge deployment

### 4. **AST** (Audio Spectrogram Transformer)
- Vision Transformer adapted for audio spectrograms
- State-of-the-art performance on audio benchmarks
- Requires more computational resources

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/previsai-transfer-learning.git
cd previsai-transfer-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Models

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

# Compare all models
python final_model_comparison.py
```

## Dataset

- **Total Samples**: 90 (17 POI, 73 Non-POI)
- **Audio Format**: 8 kHz WAV files, 4 minutes each
- **Location**: `raw-audio/poi/foreground/` and `raw-audio/nopoi/foreground/`

## Results

All training results are saved in the `results/` directory as JSON files. Key metrics include:
- Accuracy
- NPV (Negative Predictive Value)
- Sensitivity and Specificity
- Training time

## Key Features

- **Transfer Learning**: Leverage pre-trained models from AudioSet
- **Class Imbalance Handling**: Cost-sensitive learning with various pos_weight ratios
- **GPU Support**: Automatic detection of CUDA and Apple MPS
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, and medical metrics

## Project Structure

```
previsai-transfer-learning/
├── raw-audio/                    # Audio dataset
├── results/                      # Training results
├── *.py                         # Model implementations
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # AI assistant guidelines
├── design.md                    # Project design document
└── README.md                    # This file
```

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, GPU (NVIDIA CUDA or Apple MPS)
- **Optimal**: 32GB RAM, High-end GPU for transformer models

## Clinical Integration

These models are designed for **screening support only** and should:
- Always be combined with clinical assessment
- Not be used as sole diagnostic tools
- Maintain audit trails of all predictions

## Citation

If you use this code in your research, please cite:

```bibtex
@software{previsai_transfer_learning,
  title = {Transfer Learning for POI Audio Classification},
  year = {2025},
  url = {https://github.com/yourusername/previsai-transfer-learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PANNs: Kong et al. (2020)
- YAMNet: Google Research
- AST: Gong et al. (2021)
- SMOLK: Chen et al. (2024)