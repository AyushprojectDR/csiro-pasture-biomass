# Setup and Usage Guide

This guide will help you set up and run the CSIRO Pasture Biomass prediction pipeline.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- 32GB+ RAM
- ~50GB disk space for data and models

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/csiro-pasture-biomass.git
cd csiro-pasture-biomass
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n pasture-biomass python=3.10
conda activate pasture-biomass
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Competition Data

1. Go to [Kaggle Competition Page](https://www.kaggle.com/competitions/csiro-pasture-biomass/data)
2. Download the data files
3. Extract to the `data/` directory following this structure:

```
data/
├── train.csv
├── test.csv
├── train_images/
│   └── *.jpg
└── test_images/
    └── *.jpg
```

## Training

### Quick Start

Train all three targets (Dead → Clover → Green) sequentially:

```bash
python src/train.py
```

### Training Specific Folds

To train only certain folds:

```bash
python src/train.py --folds 0,1,2  # Train only folds 0, 1, 2
```

### Skip Training (Validation Only)

If you already have trained models and just want to run final validation:

```bash
python src/train.py --skip_training
```

### Debug Mode

For quick testing with a small subset:

```bash
python src/train.py --debug
```

### Expected Training Time

- **Per target per fold**: ~1-2 hours on A100 (with early stopping)
- **Full training** (3 targets × 4 folds): ~12-24 hours total
- Training is done sequentially: Dead → Clover → Green

### Training Process

The script trains 3 separate models in this order:

1. **Dead model**: Trains on all 4 folds, saves best checkpoints
2. **Clover model**: Trains on all 4 folds, saves best checkpoints  
3. **Green model**: Trains on all 4 folds, saves best checkpoints
4. **Final validation**: Loads all 12 models and computes combined metrics

### Monitoring Training

Training logs will show:
- Target being trained (DEAD, CLOVER, or GREEN)
- Current fold and epoch
- Train and validation losses  
- RMSE and R² scores
- Best model saves
- Early stopping status

Example output:
```
============================================================
Training Target: DEAD (binned by Dry_Dead_g)
============================================================
--- Fold 0 ---
Backbone frozen for first 5 epochs

Epoch 1/50 | Train: 156.34 | Val: 142.56 | RMSE: 142.56 | R2: 0.4123
  >> Saved (RMSE: 142.56)
Epoch 2/50 | Train: 134.21 | Val: 128.45 | RMSE: 128.45 | R2: 0.5234
  >> Saved (RMSE: 128.45)
...
Epoch 6/50 | Train: 98.76 | Val: 102.34 | RMSE: 102.34 | R2: 0.6845
Backbone unfrozen
  >> Saved (RMSE: 102.34)
...

============================================================
Final Combined Validation
============================================================
--- Fold 0 ---
  Target   RMSE         R2          
  green    89.2345      0.7123      
  dead     112.3456     0.4567      
  clover   76.5432      0.6789      
  gdm      98.7654      0.7234      
  total    145.6789     0.7456
```

## Inference

### Generate Predictions

After training all folds:

```bash
python src/inference.py
```

This will:
1. Load all fold checkpoints
2. Make predictions with test-time augmentation
3. Ensemble predictions across folds
4. Save submission to `submissions/submission.csv`

### Submission Format

The output file will have the required format:

```csv
sample_id,target_name,target
test_001,Dry_Green_g,123.45
test_001,Dry_Dead_g,234.56
...
```

## Project Structure

```
csiro-pasture-biomass/
├── README.md              # Main documentation
├── requirements.txt       # Dependencies
├── LICENSE               # MIT License
│
├── data/                 # Competition data (download separately)
│   ├── README.md         # Data download guide
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   └── test_images/
│
├── src/                  # Source code
│   ├── dataset.py        # Dataset and transforms
│   ├── train.py          # Training script
│   ├── inference.py      # Inference script
│   └── models/
│       └── dual_crop_vit.py  # Model architecture
│
├── configs/              # Configuration files
│   └── dinov2_base.yaml
│
├── notebooks/            # Jupyter notebooks (optional)
│   └── eda.ipynb
│
├── docs/                 # Additional documentation
│   └── TECHNICAL.md      # Detailed technical docs
│
├── submissions/          # Prediction outputs
│   └── submission.csv
│
└── checkpoints/          # Saved models (created during training)
    ├── fold_0_best.pth
    ├── fold_1_best.pth
    └── ...
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter GPU OOM errors:

1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 2  # Reduce from 4
   ```

2. Use gradient accumulation (add to train.py):
   ```python
   accumulation_steps = 2
   ```

3. Reduce image size:
   ```yaml
   data:
     image_size: 512  # Reduce from 1000
   ```

### Poor Validation Scores

If validation R² is low (<0.60):

1. Check data loading:
   ```bash
   python -c "from src.dataset import *; test_dataset_loading()"
   ```

2. Verify biological constraints are enforced
3. Ensure backbone is frozen
4. Check for data leakage in cross-validation

### Slow Training

To speed up training:

1. Enable mixed precision:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

2. Increase num_workers:
   ```yaml
   training:
     num_workers: 8  # Increase from 4
   ```

3. Use smaller backbone:
   ```yaml
   model:
     backbone_name: 'vit_small_patch14_dinov2.lvd142m'
   ```

## Advanced Usage

### Hyperparameter Tuning

Use the provided config as a starting point and experiment with:

- **Dropout**: Test 0.2, 0.3, 0.4
- **Learning Rate**: Try 5e-5, 1e-4, 2e-4
- **Backbone**: Try different DINOv2 or DINOv3 variants

### Ensemble Methods

To add XGBoost stacking:

```python
from xgboost import XGBRegressor

# Train meta-model on fold predictions
meta_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)
meta_model.fit(fold_predictions, targets)
```

### Custom Backbones

To use a different backbone:

```yaml
model:
  backbone_name: 'vit_large_patch14_dinov2.lvd142m'
  # Or try: 'efficientnet_b3', 'convnext_base', etc.
```

## Performance Benchmarks

Expected performance on different hardware:

| Hardware | Batch Size | Training Speed | Memory Usage |
|----------|-----------|----------------|--------------|
| Tesla T4 | 4 | ~15 min/epoch | 14GB |
| A100 40GB | 8 | ~8 min/epoch | 28GB |
| A100 80GB | 16 | ~5 min/epoch | 45GB |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this code, please cite:

```bibtex
@software{csiro_pasture_biomass_2025,
  author = {Your Name},
  title = {CSIRO Pasture Biomass Prediction - Silver Medal Solution},
  year = {2025},
  url = {https://github.com/yourusername/csiro-pasture-biomass}
}
```

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

---

Good luck with your experiments! 🚀
