"""
Custom dataset for CSIRO Pasture Biomass Challenge
Implements dual-crop processing for 2000×1000 images with synchronized augmentation
"""

import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BiomassDataset(Dataset):
    """
    Single-target dataset for training individual models (Dead, Clover, or Green).
    
    Args:
        df: DataFrame with image paths and targets
        transform: Albumentations transforms
        img_dir: Directory containing images
        target_col: Target column name (e.g., 'Dry_Dead_g')
        use_log_transform: Whether to log-transform targets
        is_training: Whether in training mode (affects augmentation synchronization)
    """
    
    def __init__(
        self,
        df,
        transform,
        img_dir,
        target_col,
        use_log_transform=False,
        is_training=True
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = Path(img_dir)
        self.target_col = target_col
        self.use_log_transform = use_log_transform
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img = cv2.imread(str(self.img_dir / row["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Dual-crop: left and right halves (using height as crop size)
        left = img[:, :h]
        right = img[:, w - h:]
        
        # Apply transforms with synchronization for training
        if self.is_training:
            # Use same random seed for both crops to ensure synchronized augmentation
            seed = random.randint(0, 2**31)
            random.seed(seed)
            np.random.seed(seed)
            left = self.transform(image=left)['image']
            random.seed(seed)
            np.random.seed(seed)
            right = self.transform(image=right)['image']
        else:
            left = self.transform(image=left)['image']
            right = self.transform(image=right)['image']
        
        # Get target value
        target = row[self.target_col]
        if self.use_log_transform:
            target = np.log1p(target + 1e-6)
        
        return (left, right), torch.tensor(target, dtype=torch.float32)


class BiomassDatasetMulti(Dataset):
    """
    Multi-target dataset for final validation (returns all 3 targets: Green, Dead, Clover).
    Used for computing derived targets (GDM, Total) and final metrics.
    
    Args:
        df: DataFrame with image paths and all targets
        transform: Albumentations transforms  
        img_dir: Directory containing images
        use_log_transform: Whether to log-transform targets
    """
    
    def __init__(self, df, transform, img_dir, use_log_transform=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = Path(img_dir)
        self.use_log_transform = use_log_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img = cv2.imread(str(self.img_dir / row["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Dual-crop
        left = img[:, :h]
        right = img[:, w - h:]
        
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        
        # Get all 3 independent targets
        green = row["Dry_Green_g"]
        dead = row["Dry_Dead_g"]
        clover = row["Dry_Clover_g"]
        
        if self.use_log_transform:
            eps = 1e-6
            green = np.log1p(green + eps)
            dead = np.log1p(dead + eps)
            clover = np.log1p(clover + eps)
        
        targets = torch.tensor([green, dead, clover], dtype=torch.float32)
        return (left, right), targets


def collate_fn(batch):
    """
    Custom collate function for dual-crop batches.
    
    Args:
        batch: List of (crops, target) tuples
        
    Returns:
        (imgs1, imgs2): Stacked left and right crops
        targets: Stacked targets
    """
    imgs1 = torch.stack([b[0][0] for b in batch])
    imgs2 = torch.stack([b[0][1] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return (imgs1, imgs2), targets


def get_train_transforms(img_size):
    """Training augmentations with synchronized flips and rotations"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(-10, 10), p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        # ColorJitter disabled - found to hurt performance
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size):
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

