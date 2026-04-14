"""
encoder.py
----------
3D CNN encoder for PI-SSL and SimCLR pre-training on Widar3.0 BVP volumes.

Architecture
------------
Input BVP volume from BVPDataset: (B, 1, 20, 20, 20)
  dim 0 : batch
  dim 1 : channel (always 1)
  dim 2 : Doppler velocity bins  (20)
  dim 3 : spatial / angle bins   (20)
  dim 4 : time frames            (20)

BVPEncoder  → 128-dim embedding  (used for fine-tuning)
ProjectionHead → 64-dim projection (used only during NT-Xent pre-training)

Usage — SSL pre-training:
    encoder    = BVPEncoder()
    proj_head  = ProjectionHead(in_dim=128, hidden_dim=128, out_dim=64)
    z = encoder(x)       # (B, 128)
    h = proj_head(z)     # (B, 64)  ← pass to NTXentLoss

Usage — fine-tuning:
    encoder    = BVPEncoder()
    classifier = nn.Linear(128, num_classes)
    # load pre-trained encoder weights, freeze or fine-tune
    logits = classifier(encoder(x))
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ConvBlock3d(nn.Module):
    """Conv3d → BatchNorm3d → ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class BVPEncoder(nn.Module):
    """
    3D CNN encoder for BVP volumes.

    Three convolutional blocks progressively halve spatial resolution
    via MaxPool3d(2); global average pooling collapses the remaining
    5×5×5 feature map to a 128-dim embedding.

    Spatial progression:
        (B,  1, 20, 20, 20)
      → (B, 32, 10, 10, 10)   after block 1 + pool
      → (B, 64,  5,  5,  5)   after block 2 + pool
      → (B,128,  5,  5,  5)   after block 3
      → (B,128,  1,  1,  1)   after AdaptiveAvgPool3d(1)
      → (B,128)               after flatten
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        self.layer1 = nn.Sequential(
            ConvBlock3d(1, 32),
            nn.MaxPool3d(kernel_size=2),          # 20 → 10
        )
        self.layer2 = nn.Sequential(
            ConvBlock3d(32, 64),
            nn.MaxPool3d(kernel_size=2),          # 10 → 5
        )
        self.layer3 = ConvBlock3d(64, embed_dim)  # 5 → 5 (no pool)

        self.pool = nn.AdaptiveAvgPool3d(1)       # 5×5×5 → 1×1×1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 20, 20, 20)

        Returns
        -------
        z : (B, embed_dim)
        """
        x = self.layer1(x)   # (B, 32, 10, 10, 10)
        x = self.layer2(x)   # (B, 64,  5,  5,  5)
        x = self.layer3(x)   # (B,128,  5,  5,  5)
        x = self.pool(x)     # (B,128,  1,  1,  1)
        return x.flatten(1)  # (B, 128)


# ---------------------------------------------------------------------------
# Projection head  (used only during NT-Xent pre-training)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head from SimCLR (Chen et al., 2020).

    Maps encoder embeddings to a lower-dimensional space where the
    NT-Xent contrastive loss is applied.  After pre-training, this
    head is discarded; the encoder output z is used for fine-tuning.

    Default: 128 → 128 (BN+ReLU) → 64
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128,
                 out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, in_dim)

        Returns
        -------
        h : (B, out_dim)   — L2-normalised in NTXentLoss, not here
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch

    encoder  = BVPEncoder(embed_dim=128)
    proj     = ProjectionHead(in_dim=128, hidden_dim=128, out_dim=64)

    x = torch.zeros(4, 1, 20, 20, 20)
    z = encoder(x)
    h = proj(z)

    enc_params  = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in proj.parameters()    if p.requires_grad)

    print(f'Input  shape : {x.shape}')
    print(f'Embed  shape : {z.shape}   (expect [4, 128])')
    print(f'Proj   shape : {h.shape}   (expect [4,  64])')
    print(f'Encoder params     : {enc_params:,}')
    print(f'Proj head params   : {proj_params:,}')
    print(f'Total SSL params   : {enc_params + proj_params:,}')
