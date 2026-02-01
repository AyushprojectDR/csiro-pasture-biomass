"""
Biomass Prediction Model with LocalMambaBlock Fusion
DINOv3 Huge Plus backbone with separate models per target
"""

import torch
import torch.nn as nn
import timm


class LocalMambaBlock(nn.Module):
    """
    Local Mamba-inspired block for sequential feature fusion.
    Uses depthwise convolution with gating mechanism.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            x: [batch_size, seq_len, dim]
        """
        shortcut = x
        x = self.norm(x)
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        return shortcut + self.drop(self.proj(x))


class BiomassModelSingle(nn.Module):
    """
    Single-target biomass prediction model.
    
    Architecture:
    - DINOv3 Huge Plus backbone (per crop)
    - LocalMambaBlock fusion (2 layers)
    - Adaptive pooling + MLP head
    
    Args:
        model_name: timm model name (e.g., 'vit_huge_plus_patch16_dinov3.lvd1689m')
        pretrained: Whether to load pretrained weights
        grad_checkpointing: Enable gradient checkpointing to save memory
    """
    def __init__(
        self,
        model_name='vit_huge_plus_patch16_dinov3.lvd1689m',
        pretrained=True,
        grad_checkpointing=True
    ):
        super().__init__()
        
        # Backbone: no classification head, no global pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Enable gradient checkpointing if supported
        if hasattr(self.backbone, 'set_grad_checkpointing') and grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        
        # Feature dimension from backbone
        nf = self.backbone.num_features
        
        # Fusion layers: 2 LocalMambaBlocks
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.2),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.2)
        )
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(nf // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tuple of (left_crop, right_crop)
               Each crop: [batch_size, 3, H, W]
        
        Returns:
            predictions: [batch_size] single target values
        """
        left, right = x
        
        # Extract features from both crops
        x_l = self.backbone(left)  # [batch_size, seq_len, nf]
        x_r = self.backbone(right)  # [batch_size, seq_len, nf]
        
        # Concatenate along sequence dimension
        x_fused = torch.cat([x_l, x_r], dim=1)  # [batch_size, 2*seq_len, nf]
        
        # Fusion through LocalMambaBlocks
        x_fused = self.fusion(x_fused)  # [batch_size, 2*seq_len, nf]
        
        # Pool to fixed size
        x_pool = self.pool(x_fused.transpose(1, 2))  # [batch_size, nf, 1]
        x_pool = x_pool.flatten(1)  # [batch_size, nf]
        
        # Regression head
        out = self.head(x_pool)  # [batch_size, 1]
        
        return out.squeeze(-1)  # [batch_size]


def create_model(config):
    """
    Factory function to create model from config.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        model: BiomassModelSingle instance
    """
    model = BiomassModelSingle(
        model_name=config.get('model_name', 'vit_huge_plus_patch16_dinov3.lvd1689m'),
        pretrained=config.get('pretrained', True),
        grad_checkpointing=config.get('grad_checkpointing', True)
    )
    
    return model


def set_backbone_grad(model, requires_grad):
    """
    Freeze or unfreeze backbone parameters.
    
    Args:
        model: BiomassModelSingle instance
        requires_grad: True to unfreeze, False to freeze
    """
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


if __name__ == '__main__':
    # Test model
    print("Testing BiomassModelSingle...")
    model = BiomassModelSingle(
        model_name='vit_huge_plus_patch16_dinov3.lvd1689m',
        pretrained=False  # Don't download weights for test
    )
    
    # Create dummy input (dual crops)
    batch_size = 2
    left = torch.randn(batch_size, 3, 512, 512)
    right = torch.randn(batch_size, 3, 512, 512)
    
    # Forward pass
    out = model((left, right))
    
    print(f"Input shapes: left={left.shape}, right={right.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: ({batch_size},)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Model test passed!")

