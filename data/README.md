# Data Directory

## Download Instructions

1. Visit the [CSIRO Pasture Biomass Competition](https://www.kaggle.com/competitions/csiro-pasture-biomass/data) on Kaggle

2. Download the following files:
   - `train.csv` - Training labels and metadata
   - `test.csv` - Test set metadata
   - `train_images/` - Training images (2000×1000 pixels)
   - `test_images/` - Test images (2000×1000 pixels)

3. Place files in this directory with the following structure:

```
data/
├── train.csv
├── test.csv
├── train_images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── test_images/
    ├── test_001.jpg
    ├── test_002.jpg
    └── ...
```

## Dataset Overview

- **Training Images**: 357 unique images
- **Image Size**: 2000×1000 pixels (RGB)
- **Geographic Coverage**: NSW, Tasmania, Victoria, Western Australia
- **Target Variables**: 5 biomass components per image
  - Dry_Green_g
  - Dry_Dead_g
  - Dry_Clover_g
  - GDM_g (Green Dry Matter)
  - Dry_Total_g

## Metadata Features

- **Sampling_Date**: Date of image capture
- **State**: Australian state location
- **Species**: Dominant pasture species
- **Pre_GSHH_NDVI**: NDVI reading before grazing
- **Height_Ave_cm**: Average pasture height

## Important Notes

- Only 3 targets are mathematically independent (Green, Dead, Clover)
- GDM = Green + Clover
- Total = Green + Dead + Clover
- These constraints are enforced in our model architecture
