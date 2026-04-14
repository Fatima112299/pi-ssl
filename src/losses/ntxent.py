"""
ntxent.py
---------
NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss for
contrastive self-supervised learning (SimCLR, Chen et al. 2020).

Given a batch of N samples, each augmented twice to produce views
(z_i, z_j), the loss maximises agreement between the two views of
the same sample while pushing apart views from different samples.

Reference: Chen et al., "A Simple Framework for Contrastive Learning
of Visual Representations", ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for a batch of paired embeddings.

    Parameters
    ----------
    temperature : float
        Scaling factor τ.  Lower values make the distribution sharper
        (harder negatives).  SimCLR uses τ=0.5; smaller batches may
        benefit from slightly higher values (0.5–1.0).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_i : (B, d)  — projection of first  augmented view
        z_j : (B, d)  — projection of second augmented view

        Both are raw (un-normalised) outputs of the projection head.
        L2 normalisation is applied here.

        Returns
        -------
        loss : scalar tensor
        """
        B = z_i.size(0)
        device = z_i.device

        # L2 normalise → unit hypersphere
        z_i = F.normalize(z_i, dim=1)   # (B, d)
        z_j = F.normalize(z_j, dim=1)   # (B, d)

        # Concatenate both views: (2B, d)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity on the diagonal
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float('-inf'))

        # Positive pair indices:
        #   for row i  in [0, B)  → positive is row i+B
        #   for row i  in [B, 2B) → positive is row i-B
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B,     device=device),
        ])

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)

    loss_fn = NTXentLoss(temperature=0.5)

    B, d = 8, 64
    # Perfect positive pairs → loss should be near 0
    z = F.normalize(torch.randn(B, d), dim=1)
    loss_perfect = loss_fn(z, z.clone())

    # Random pairs → loss should be close to log(2B-1)
    import math
    z_i = torch.randn(B, d)
    z_j = torch.randn(B, d)
    loss_random = loss_fn(z_i, z_j)
    expected_random = math.log(2 * B - 1)

    print(f'NT-Xent loss (perfect positives)  : {loss_perfect.item():.4f}  (expect ≈ 0)')
    print(f'NT-Xent loss (random)             : {loss_random.item():.4f}')
    print(f'Expected random upper bound       : {expected_random:.4f}  = log({2*B}-1)')
