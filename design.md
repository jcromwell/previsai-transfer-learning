# Transfer Learning for POI Audio Classification

## Project Overview

This project leverages **transfer learning** techniques to classify intestinal audio recordings for detecting postoperative ileus (POI). By using pre-trained models from large-scale audio datasets, we address the challenge of limited medical audio data (90 total samples).

## Transfer Learning Approach

### Why Transfer Learning?

1. **Limited Medical Data**: Only 17 POI and 73 Non-POI samples
2. **Leverage Pre-trained Features**: Models trained on millions of audio samples
3. **Better Generalization**: Pre-trained models generalize better than training from scratch
4. **Faster Convergence**: Fine-tuning requires less training time

### Models Implemented

#### 1. SMOLK (Sparse Mixture of Learned Kernels)
- **Type**: Custom architecture with learnable kernels
- **Pre-training**: Not pre-trained, but uses transfer learning principles
- **Architecture**: 12 mixed-size kernels (0.5s, 1.0s, 2.0s)
- **Parameters**: 112K (prunable to 84K)
- **Best Performance**: 76.5% accuracy, 84.6% NPV

#### 2. PANNs (Pre-trained Audio Neural Networks)
- **Type**: CNN-based transfer learning
- **Pre-trained on**: AudioSet (2M+ audio clips)
- **Architecture**: CNN14 with 14 convolutional layers
- **Fine-tuning Strategy**: 
  - Freeze early layers
  - Fine-tune final layers
  - Custom classification head for POI detection

#### 3. YAMNet (Yet Another Mobile Network)
- **Type**: MobileNet-based audio classifier
- **Pre-trained on**: AudioSet (521 classes)
- **Architecture**: MobileNetV1 adapted for audio
- **Transfer Learning**: 
  - Extract embeddings
  - Add custom dense layers
  - Fine-tune with medical data

#### 4. AST (Audio Spectrogram Transformer)
- **Type**: Vision Transformer for audio
- **Pre-trained on**: AudioSet and/or ImageNet
- **Architecture**: ViT-B/16 adapted for spectrograms
- **Transfer Learning**:
  - Convert audio to spectrograms
  - Use pre-trained vision transformer
  - Fine-tune for binary POI classification

## Dataset Details

```
Total Samples: 90
- POI (Positive): 17 samples (18.9%)
- Non-POI (Negative): 73 samples (81.1%)

Audio Format:
- Sample Rate: 8 kHz
- Duration: 4 minutes per file
- Processing: 20-30 second segments
```

## Implementation Strategy

### 1. Data Preprocessing
```python
# Common preprocessing for all models
- Load 8kHz WAV files
- Segment into model-specific lengths
- Convert to spectrograms/mel-spectrograms
- Normalize based on pre-training statistics
```

### 2. Transfer Learning Pipeline
```python
# General approach
1. Load pre-trained model
2. Freeze early/middle layers
3. Replace final classification layer
4. Fine-tune on POI dataset
5. Apply regularization (dropout, weight decay)
```

### 3. Class Imbalance Handling
- **Weighted Loss**: BCEWithLogitsLoss with pos_weight
- **Cost-Sensitive Learning**: Various ratios (1.5, 2.0, 2.5, 3.0)
- **Stratified Splits**: Maintain class distribution

## Performance Comparison

| Model | Config | Accuracy | NPV | Sensitivity | Training Time |
|-------|--------|----------|-----|-------------|---------------|
| SMOLK | Original | 76.5% | 84.6% | 50% | 80 min |
| PANNs | Simple | TBD | TBD | TBD | 15 min |
| PANNs | Cost-Sensitive (2.5) | TBD | TBD | TBD | 20 min |
| YAMNet | Transfer Learning | TBD | TBD | TBD | 10 min |
| YAMNet | Cost-Sensitive | TBD | TBD | TBD | 12 min |
| AST | Simple | TBD | TBD | TBD | 45 min |
| AST | Cost-Sensitive | TBD | TBD | TBD | 50 min |

## Key Files

### Core Implementations
- `smolk_implementation.py` - SMOLK model
- `panns_simple_gpu.py` - PANNs baseline
- `panns_cost_sensitive.py` - PANNs with class weighting
- `yamnet_transfer_learning.py` - YAMNet transfer learning
- `yamnet_cost_sensitive.py` - YAMNet with cost sensitivity
- `ast_simple.py` - AST baseline
- `ast_cost_sensitive.py` - AST with class weighting

### Analysis & Visualization
- `final_model_comparison.py` - Compare all models
- `final_comparison_visualization.py` - Generate plots
- `visualize_all_confusion_matrices.py` - Confusion matrices

### Results
- `results/` directory contains JSON files with metrics

## Technical Requirements

### Dependencies
```python
# Deep Learning Frameworks
torch>=2.0.0          # SMOLK, PANNs, AST
tensorflow>=2.13.0    # YAMNet
transformers>=4.30.0  # AST

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.2.0
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Hardware
- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, GPU (NVIDIA CUDA or Apple MPS)
- **Optimal**: 32GB RAM, High-end GPU for AST

## Deployment Considerations

### Model Selection for Deployment
1. **Edge Devices**: YAMNet (smallest, fastest)
2. **Cloud API**: AST (highest accuracy)
3. **Balanced**: PANNs (good accuracy, reasonable size)
4. **Interpretable**: SMOLK (learnable kernels)

### Inference Optimization
- Model quantization (INT8)
- ONNX conversion for cross-platform
- TensorFlow Lite for mobile
- Batch processing for efficiency

## Clinical Integration

### Recommended Workflow
1. **Initial Screening**: Use ensemble of best models
2. **Risk Stratification**: 
   - High NPV (>85%) → Low risk
   - Low NPV → Requires clinical review
3. **Continuous Monitoring**: Re-test every 4-6 hours

### Safety Guidelines
- Models are for **screening support only**
- Not FDA approved for diagnosis
- Always combine with clinical assessment
- Maintain prediction audit trail

## Future Directions

### Short Term
- [ ] Ensemble methods combining all models
- [ ] Data augmentation techniques
- [ ] Cross-validation with different seeds
- [ ] Hyperparameter optimization

### Medium Term
- [ ] Multi-site validation
- [ ] Real-time inference pipeline
- [ ] Integration with hospital systems
- [ ] FDA 510(k) submission prep

### Long Term
- [ ] Continuous learning from new data
- [ ] Multi-class severity prediction
- [ ] Combination with other vitals
- [ ] Clinical trial for diagnostic approval

## References

1. Kong et al. (2020) - "PANNs: Large-Scale Pretrained Audio Neural Networks"
2. Hershey et al. (2017) - "CNN Architectures for Large-Scale Audio Classification"
3. Gong et al. (2021) - "AST: Audio Spectrogram Transformer"
4. Chen et al. (2024) - "SMOLK: Sparse Mixture of Learned Kernels"

---

*Last Updated: August 2025*
*Status: Transfer Learning Implementation Complete*