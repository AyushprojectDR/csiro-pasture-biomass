"""
CSIRO Biomass Prediction - Training Pipeline V2
Separate models for: Dead, Clover, Green (trained sequentially)
Final validation computes: GDM = Green + Clover, Total = Green + Dead + Clover
"""

import os
import random
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import gc
import warnings
warnings.filterwarnings("ignore")


class CFG:
    DATA_DIR = Path("/home/ayush/Documents/Kaggle_comp/bio-mass/csiro-biomass")
    OUTPUT_DIR = Path("./v2_trained")

    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
    IMG_SIZE = 512
    GRAD_CHECKPOINTING = True

    N_FOLDS = 4
    TRAIN_FOLDS = [0, 1, 2, 3]
    EPOCHS = 50
    FREEZE_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 10
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4
    NUM_WORKERS = 4

    LR_BACKBONE = 5e-5
    LR_HEAD = 2e-4
    WD = 5e-4
    GRAD_CLIP = 0.5

    USE_LOG_TRANSFORM = False
    TARGETS = ["dead", "clover", "green"]  # Training order
    TARGET_COLS = {"dead": "Dry_Dead_g", "clover": "Dry_Clover_g", "green": "Dry_Green_g"}

    WARMUP_EPOCHS = 3
    USE_AMP = True
    SEED = 42
    DEBUG = False


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(-10, 10), p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        #A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class BiomassDataset(Dataset):
    """Single-target dataset for training individual models."""
    def __init__(self, df, transform, img_dir, target_col, use_log_transform=True, is_training=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir
        self.target_col = target_col
        self.use_log_transform = use_log_transform
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.img_dir / row["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        left = img[:, :h]
        right = img[:, w - h:]

        # Synchronized augmentation: same transform for both crops
        if self.is_training:
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

        target = row[self.target_col]
        if self.use_log_transform:
            target = np.log1p(target + 1e-6)

        return (left, right), torch.tensor(target, dtype=torch.float32)


class BiomassDatasetMulti(Dataset):
    """Multi-target dataset for final validation (returns all targets)."""
    def __init__(self, df, transform, img_dir, use_log_transform=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir
        self.use_log_transform = use_log_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.img_dir / row["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        left = img[:, :h]
        right = img[:, w - h:]

        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']

        green, dead, clover = row["Dry_Green_g"], row["Dry_Dead_g"], row["Dry_Clover_g"]

        if self.use_log_transform:
            eps = 1e-6
            green, dead, clover = np.log1p(green + eps), np.log1p(dead + eps), np.log1p(clover + eps)

        targets = torch.tensor([green, dead, clover], dtype=torch.float32)
        return (left, right), targets


def collate_fn(batch):
    imgs1 = torch.stack([b[0][0] for b in batch])
    imgs2 = torch.stack([b[0][1] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return (imgs1, imgs2), targets


class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        return shortcut + self.drop(self.proj(x))


class BiomassModelSingle(nn.Module):
    """Single-target model with its own backbone, fusion, and head."""
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')

        if hasattr(self.backbone, 'set_grad_checkpointing') and CFG.GRAD_CHECKPOINTING:
            self.backbone.set_grad_checkpointing(True)

        nf = self.backbone.num_features
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.2),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.2)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.35), nn.Linear(nf // 2, 1))

    def forward(self, x):
        left, right = x
        x_l, x_r = self.backbone(left), self.backbone(right)
        x_fused = self.fusion(torch.cat([x_l, x_r], dim=1))
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        return self.head(x_pool).squeeze(-1)


class RMSELoss(nn.Module):
    """Single-target RMSE loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        return torch.sqrt(mse + 1e-8)


def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def set_backbone_grad(model, requires_grad):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, cfg, target_name):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, ((imgs1, imgs2), targets) in enumerate(tqdm(loader, desc=f"Train [{target_name}]", leave=False)):
        imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
            loss = criterion(model((imgs1, imgs2)), targets) / cfg.ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * cfg.ACCUMULATION_STEPS

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, target_name):
    """Validate single-target model."""
    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    for (imgs1, imgs2), targets in tqdm(loader, desc=f"Valid [{target_name}]", leave=False):
        imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
            preds = model((imgs1, imgs2))
            loss = criterion(preds, targets)

        running_loss += loss.item()

        preds_np, targets_np = preds.cpu().numpy(), targets.cpu().numpy()
        if cfg.USE_LOG_TRANSFORM:
            preds_np = np.expm1(np.clip(preds_np, -20, 20))
            targets_np = np.expm1(np.clip(targets_np, -20, 20))

        all_preds.append(preds_np)
        all_targets.append(targets_np)

    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = r2_score(all_targets, all_preds)

    return {
        "val_loss": running_loss / len(loader),
        "rmse": rmse,
        "r2": r2
    }


def create_folds(df, target_col, n_folds, seed):
    """Create stratified folds based on target-specific binning."""
    df = df.copy()
    df["target_bin"] = pd.qcut(df[target_col], q=10, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["target_bin"])):
        df.loc[val_idx, "fold"] = fold

    df.drop("target_bin", axis=1, inplace=True)
    return df


def train_fold(fold, train_df, target_name, target_col, cfg, device):
    """Train a single-target model for one fold."""
    print(f"\n{'-'*40}\nFold {fold} [{target_name}]\n{'-'*40}")

    train_data = train_df[train_df["fold"] != fold].reset_index(drop=True)
    valid_data = train_df[train_df["fold"] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    train_loader = DataLoader(
        BiomassDataset(train_data, get_train_transforms(cfg.IMG_SIZE), cfg.DATA_DIR, target_col, cfg.USE_LOG_TRANSFORM, is_training=True),
        batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, collate_fn=collate_fn, drop_last=True
    )
    valid_loader = DataLoader(
        BiomassDataset(valid_data, get_val_transforms(cfg.IMG_SIZE), cfg.DATA_DIR, target_col, cfg.USE_LOG_TRANSFORM, is_training=False),
        batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, collate_fn=collate_fn
    )

    model = BiomassModelSingle(cfg.MODEL_NAME, pretrained=True).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    optimizer = AdamW([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": cfg.LR_HEAD}
    ], weight_decay=cfg.WD)

    scheduler = get_scheduler(optimizer, cfg.WARMUP_EPOCHS, cfg.EPOCHS)
    criterion = RMSELoss()
    scaler = torch.cuda.amp.GradScaler()

    best_rmse, best_epoch, patience_counter = float("inf"), 0, 0

    # Freeze backbone initially
    if cfg.FREEZE_EPOCHS > 0:
        set_backbone_grad(model, False)
        print(f"Backbone frozen for first {cfg.FREEZE_EPOCHS} epochs")

    for epoch in range(1, cfg.EPOCHS + 1):
        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == cfg.FREEZE_EPOCHS + 1:
            set_backbone_grad(model, True)
            print("Backbone unfrozen")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg, target_name)
        metrics = validate(model, valid_loader, criterion, device, cfg, target_name)
        scheduler.step()

        print(f"Epoch {epoch}/{cfg.EPOCHS} | Train: {train_loss:.4f} | Val: {metrics['val_loss']:.4f} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")

        if metrics["rmse"] < best_rmse:
            best_rmse, best_epoch = metrics["rmse"], epoch
            patience_counter = 0
            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f"fold{fold}_{target_name}_best.pth")
            print(f"  >> Saved (RMSE: {best_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg.EARLY_STOPPING_PATIENCE})")

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"Fold {fold} [{target_name}] Best: RMSE={best_rmse:.4f} @ Epoch {best_epoch}")

    del model, optimizer, scheduler, train_loader, valid_loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_rmse


def train_target(target_name, train_wide, cfg, device):
    """Train all folds for a single target."""
    target_col = cfg.TARGET_COLS[target_name]
    print(f"\n{'='*60}\nTraining Target: {target_name.upper()} (binned by {target_col})\n{'='*60}")

    # Create folds with target-specific binning
    train_df = create_folds(train_wide, target_col, cfg.N_FOLDS, cfg.SEED)

    scores = []
    for fold in cfg.TRAIN_FOLDS:
        score = train_fold(fold, train_df, target_name, target_col, cfg, device)
        scores.append(score)

    mean_score = np.mean(scores)
    print(f"\n{target_name.upper()} Complete! Mean RMSE: {mean_score:.4f}")
    for f, s in zip(cfg.TRAIN_FOLDS, scores):
        print(f"  Fold {f}: {s:.4f}")

    return scores


@torch.no_grad()
def final_validation(train_wide, cfg, device):
    """
    Load all trained models and compute combined metrics (GDM, Total).
    Uses green model's fold splits for final validation since green is trained last.
    """
    print(f"\n{'='*60}\nFinal Combined Validation\n{'='*60}")

    # Use green's binning for final validation (since green is trained last)
    train_df = create_folds(train_wide, cfg.TARGET_COLS["green"], cfg.N_FOLDS, cfg.SEED)

    all_fold_metrics = []

    for fold in cfg.TRAIN_FOLDS:
        print(f"\n--- Fold {fold} ---")
        valid_data = train_df[train_df["fold"] == fold].reset_index(drop=True)

        valid_loader = DataLoader(
            BiomassDatasetMulti(valid_data, get_val_transforms(cfg.IMG_SIZE), cfg.DATA_DIR, cfg.USE_LOG_TRANSFORM),
            batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, collate_fn=collate_fn
        )

        # Load models for this fold
        models = {}
        for target_name in ["green", "dead", "clover"]:
            model = BiomassModelSingle(cfg.MODEL_NAME, pretrained=False).to(device)
            model.load_state_dict(torch.load(cfg.OUTPUT_DIR / f"fold{fold}_{target_name}_best.pth", map_location=device))
            model.eval()
            models[target_name] = model

        all_preds = {"green": [], "dead": [], "clover": []}
        all_targets = {"green": [], "dead": [], "clover": []}

        for (imgs1, imgs2), targets in tqdm(valid_loader, desc=f"Final Valid [Fold {fold}]", leave=False):
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)

            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                pred_green = models["green"]((imgs1, imgs2)).cpu().numpy()
                pred_dead = models["dead"]((imgs1, imgs2)).cpu().numpy()
                pred_clover = models["clover"]((imgs1, imgs2)).cpu().numpy()

            targets_np = targets.numpy()  # [green, dead, clover]

            if cfg.USE_LOG_TRANSFORM:
                pred_green = np.expm1(np.clip(pred_green, -20, 20))
                pred_dead = np.expm1(np.clip(pred_dead, -20, 20))
                pred_clover = np.expm1(np.clip(pred_clover, -20, 20))
                targets_np = np.expm1(np.clip(targets_np, -20, 20))

            all_preds["green"].append(pred_green)
            all_preds["dead"].append(pred_dead)
            all_preds["clover"].append(pred_clover)
            all_targets["green"].append(targets_np[:, 0])
            all_targets["dead"].append(targets_np[:, 1])
            all_targets["clover"].append(targets_np[:, 2])

        # Concatenate all predictions and targets
        for k in all_preds:
            all_preds[k] = np.concatenate(all_preds[k])
            all_targets[k] = np.concatenate(all_targets[k])

        # Compute derived targets
        pred_gdm = all_preds["green"] + all_preds["clover"]
        pred_total = all_preds["green"] + all_preds["dead"] + all_preds["clover"]
        target_gdm = all_targets["green"] + all_targets["clover"]
        target_total = all_targets["green"] + all_targets["dead"] + all_targets["clover"]

        # Calculate metrics
        metrics = {}
        for name in ["green", "dead", "clover"]:
            metrics[f"rmse_{name}"] = np.sqrt(np.mean((all_preds[name] - all_targets[name]) ** 2))
            metrics[f"r2_{name}"] = r2_score(all_targets[name], all_preds[name])

        metrics["rmse_gdm"] = np.sqrt(np.mean((pred_gdm - target_gdm) ** 2))
        metrics["r2_gdm"] = r2_score(target_gdm, pred_gdm)
        metrics["rmse_total"] = np.sqrt(np.mean((pred_total - target_total) ** 2))
        metrics["r2_total"] = r2_score(target_total, pred_total)

        # Print fold results
        print(f"  {'Target':<8} {'RMSE':<12} {'R2':<12}")
        for name in ["green", "dead", "clover", "gdm", "total"]:
            print(f"  {name:<8} {metrics[f'rmse_{name}']:<12.4f} {metrics[f'r2_{name}']:<12.4f}")

        all_fold_metrics.append(metrics)

        # Clean up models
        for m in models.values():
            del m
        torch.cuda.empty_cache()

    # Compute mean metrics across folds
    print(f"\n{'='*60}\nMean Metrics Across Folds\n{'='*60}")
    print(f"  {'Target':<8} {'RMSE':<12} {'R2':<12}")
    for name in ["green", "dead", "clover", "gdm", "total"]:
        mean_rmse = np.mean([m[f"rmse_{name}"] for m in all_fold_metrics])
        mean_r2 = np.mean([m[f"r2_{name}"] for m in all_fold_metrics])
        print(f"  {name:<8} {mean_rmse:<12.4f} {mean_r2:<12.4f}")

    return all_fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--folds", type=str, default="0,1,2,3")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run final validation")
    args = parser.parse_args()

    cfg = CFG()
    cfg.DEBUG = args.debug
    cfg.TRAIN_FOLDS = [int(f) for f in args.folds.split(",")]

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    train_wide = train_df.pivot(index="image_path", columns="target_name", values="target").reset_index()
    train_wide.columns.name = None
    print(f"Images: {len(train_wide)}")

    if cfg.DEBUG:
        train_wide = train_wide.head(100)
        cfg.EPOCHS = 2

    if not args.skip_training:
        # Train each target sequentially: dead -> clover -> green
        all_scores = {}
        for target_name in cfg.TARGETS:  # ["dead", "clover", "green"]
            scores = train_target(target_name, train_wide, cfg, device)
            all_scores[target_name] = scores

        # Print summary
        print(f"\n{'='*60}\nTraining Summary\n{'='*60}")
        for target_name in cfg.TARGETS:
            mean_score = np.mean(all_scores[target_name])
            print(f"{target_name.upper()}: Mean RMSE = {mean_score:.4f}")

    # Final combined validation
    final_validation(train_wide, cfg, device)

    print(f"\n{'='*60}\nAll Done!\n{'='*60}")


if __name__ == "__main__":
    main()

