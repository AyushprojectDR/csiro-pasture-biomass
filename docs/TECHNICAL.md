# Technical Documentation

## Detailed Solution Approach

### Problem Analysis

The CSIRO Pasture Biomass Challenge presents several unique challenges:

1. **Small Dataset**: Only 357 training images makes overfitting a primary concern
2. **Wide Images**: 2000×1000 pixel images require careful processing to preserve spatial information
3. **Correlated Targets**: Five biomass components with inherent biological relationships
4. **Imbalanced Metric**: Weighted R² gives much higher weight to Total (0.5) and GDM (0.2)

### Key Innovations

#### 1. Separate Models Per Target

**Discovery**: Training separate models for each target eliminates gradient conflicts
- Each target (Dead, Clover, Green) gets its own complete model
- Independent backbones, fusion layers, and regression heads
- No shared parameters that could cause competing gradients

**Implementation**:
```python
# Three independent models
model_dead = BiomassModelSingle(...)
model_clover = BiomassModelSingle(...)
model_green = BiomassModelSingle(...)

# Sequential training
train_target("dead", ...)    # Train dead model first
train_target("clover", ...)  # Then clover
train_target("green", ...)   # Finally green

# At inference, derive constrained targets
gdm = pred_green + pred_clover
total = pred_green + pred_dead + pred_clover
```

**Impact**: 
- Eliminated gradient conflicts completely
- Each model can specialize for its target
- More stable training curves
- Better overall R² performance

#### 2. LocalMambaBlock Fusion

**Problem**: Simply concatenating left/right crop features loses sequential relationships

**Solution**: LocalMambaBlock with gating mechanism
```python
class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, 
                                padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Gated fusion with depthwise convolution
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        return shortcut + self.drop(self.proj(x))
```

**Impact**: Better feature fusion compared to simple concatenation

#### 3. Freeze-then-Unfreeze Strategy

**Rationale**: 
- Start with frozen backbone to prevent overfitting
- After head converges (5 epochs), unfreeze for fine-tuning
- Allows adaptation to specific biomass prediction task

**Results**:
- First 5 epochs: Head learns with frozen features
- Remaining epochs: Full model fine-tunes together
- Better than always-frozen or always-unfrozen

### Experimental Journey

#### Early Experiments (R² 0.54-0.61)

**Baseline Approaches**:
- Single EfficientNet-B0 with all 5 targets: R² ~0.54
- Log-transform targets: R² ~0.56 (didn't help)
- Box-Cox transform: R² ~0.58 (misaligned with metric)

**Key Learnings**:
- Transform-space optimization doesn't correlate with original-scale R²
- Need to train on original scale targets
- Small dataset requires aggressive regularization

#### Mid-Stage Experiments (R² 0.62-0.68)

**Improvements**:
- Switched to DINOv2 backbone: +0.04 R²
- Implemented dual-crop processing: +0.03 R²
- Froze backbone weights: +0.05 R²
- Proper cross-validation (no state-based splitting): More reliable estimates

**Failed Experiments**:
- Complex multi-head attention aggregation: No improvement, increased overfitting
- Auxiliary metadata inputs (NDVI, height): Minimal impact (<0.01 R²)
- Heavy augmentation: Hurt performance on validation

#### Final Experiments (R² 0.70+)

**Breakthrough**:
- Biological constraint enforcement: +0.06 R²
- Careful hyperparameter tuning (dropout, learning rate)
- XGBoost stacking with fold predictions: +0.02 R²
- Test-time augmentation: +0.01 R²

### Hyperparameter Tuning

**Critical Parameters**:

| Parameter | Value | Impact |
|-----------|-------|--------|
| Model | DINOv3 Huge Plus | Critical |
| Image Size | 512×512 | Medium |
| Freeze Epochs | 5 | High |
| Dropout (Head) | 0.35 | High |
| Dropout (Fusion) | 0.2 | Medium |
| LR Backbone | 5e-5 | High |
| LR Head | 2e-4 | High |
| Batch Size | 4 | Low |
| Accumulation Steps | 4 | Medium |
| Weight Decay | 5e-4 | Medium |
| Gradient Clipping | 0.5 | Medium |
| Early Stopping Patience | 10 | Medium |
| Number of Folds | 4 | Medium |
| Warmup Epochs | 3 | Low |

### Cross-Validation Strategy

**What Didn't Work**:
- State-based CV: Severe data leakage, unreliable scores
- Leave-one-state-out: Imbalanced folds, poor generalization

**What Worked**:
- 5-Fold K-Fold: Balanced, reliable
- GroupKFold by image: Good for tracking multi-target consistency

**Validation-LB Correlation**:
- Proper K-Fold: R² ~0.98 correlation with LB
- State-based: R² ~0.65 correlation (unreliable)

### Ensemble Methods

**Approaches Tested**:

1. **Simple Averaging**: Average predictions from 5 folds
   - Quick to implement
   - R² improvement: +0.01

2. **XGBoost Stacking**: Train XGBoost on fold predictions
   - Features: All fold predictions + metadata
   - R² improvement: +0.02
   - Required careful CV to avoid overfitting

3. **Weighted Averaging**: Weight folds by validation R²
   - Minimal improvement over simple average
   - Not worth the complexity

**Final Ensemble**: Simple 5-fold average + XGBoost stacking

### Computational Resources

**Hardware Used**:
- Kaggle: 2× Tesla T4 (16GB) or A100 (40GB)
- Batch size: 4-8 depending on model size
- Training time: ~2-3 hours per fold
- Total experiment time: ~150 hours

**Optimization Tricks**:
- Mixed precision training (fp16)
- Gradient accumulation for larger effective batch size
- Pin memory and multiple workers for dataloading
- Efficient caching of frozen backbone features

### What Didn't Work

**Failed Experiments**:

1. **Complex Architectures**:
   - Cross-attention between crops: Overfit quickly
   - Transformer aggregation layers: No improvement
   - Multi-scale processing: Computational cost without benefit

2. **Metadata Integration**:
   - Adding NDVI/height as extra inputs: <0.01 R² improvement
   - State embeddings: No improvement
   - Species one-hot encoding: No benefit

3. **Advanced Augmentation**:
   - CutMix/MixUp: Hurt performance on this dataset
   - Heavy color augmentation: Reduced accuracy
   - Random erasing: No benefit

4. **Alternative Backbones**:
   - Swin Transformer: Similar to DINOv2, no advantage
   - ConvNeXt: Slightly worse than DINOv2
   - CLIP: Good but not better than DINOv2

### Lessons for Small Datasets

1. **Simplicity wins**: Complex architectures overfit faster
2. **Pre-training is critical**: Frozen features > fine-tuned on small data
3. **Proper regularization**: Dropout, early stopping, frozen weights
4. **Biological constraints as inductive bias**: Domain knowledge helps
5. **Careful validation**: Poor CV strategy leads to false confidence

### Future Improvements

If I were to continue this project:

1. **Semi-supervised learning**: Use unlabeled pasture images
2. **Multi-task learning**: Predict species/state as auxiliary tasks
3. **Ensemble diversity**: Train with different architectures
4. **Better augmentation**: Domain-specific augmentations for pastures
5. **Pseudo-labeling**: Use test predictions to expand training set

### Reproducibility Notes

**Seeds Set**:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Random: `random.seed(42)`

**Deterministic Operations**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Expected Variance**:
- Different hardware: ±0.01 R² (due to floating point)
- Different PyTorch versions: ±0.02 R²
- Different random seeds: ±0.01 R²

---

## References

- **DINOv2**: [Self-Supervised Learning of Visual Features](https://arxiv.org/abs/2304.07193)
- **Vision Transformers**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Timm Library**: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
