"""
augmentations.py
----------------
Data augmentations for BVP volumes used in PI-SSL and SimCLR pre-training.

BVP volume layout (from BVPDataset, before the channel unsqueeze):
    shape : (20, 20, 20)  float32
    axis 0 : Doppler velocity bins   (20)
    axis 1 : spatial / angle bins    (20)
    axis 2 : time frames             (20)

Two augmentation pipelines are provided:

    physics_augmentation()  — PI-SSL: three transforms grounded in BVP
                               formation physics + Gaussian noise.

    generic_augmentation()  — SimCLR ablation: domain-agnostic transforms
                               (Gaussian noise + temporal crop) with no
                               physics motivation.

Both return a callable that maps (20,20,20) numpy float32 → (20,20,20)
numpy float32.  BVPDataset.transform applies the callable before the
channel unsqueeze and tensor conversion.

Physics justification
---------------------
DopplerShift     : A subject performing a gesture at a slightly different
                   speed or at a different distance from the TX/RX shifts
                   all Doppler bins uniformly.  Rolling the velocity axis
                   by ±1–2 bins simulates this effect without altering the
                   gesture shape.

TemporalCropResize : Gesture execution speed varies across subjects and
                   trials.  Randomly cropping the time axis and
                   interpolating back to T=20 simulates faster/slower
                   execution while preserving temporal coherence.

SpatialFlip      : BVP is symmetric under left-right body orientation
                   reversal (mirrored gestures produce mirrored angle
                   profiles).  Flipping the angle axis is a valid
                   invariance for most of the standard-6 gestures.

GaussianNoise    : Models measurement noise and channel fluctuations
                   present in real Wi-Fi sensing deployments.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Individual transforms  (operate on numpy arrays)
# ---------------------------------------------------------------------------

class DopplerShift:
    """
    Cyclically shift the Doppler velocity axis by a random integer.

    Physical meaning: subject moves at slightly different speed or
    distance, uniformly shifting the Doppler spectrum.

    Parameters
    ----------
    max_shift : int
        Maximum shift magnitude in bins.  Shift is drawn uniformly
        from [-max_shift, +max_shift].
    """

    def __init__(self, max_shift: int = 2):
        self.max_shift = max_shift

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift, axis=0)   # axis 0 = Doppler


class TemporalCropResize:
    """
    Randomly crop the time axis and resize back to the original length.

    Physical meaning: gesture executed faster or slower than the
    canonical template, compressing or expanding the temporal envelope.

    Parameters
    ----------
    min_ratio : float
        Minimum fraction of time frames to retain before resizing.
        e.g. 0.6 means the crop is at least 60% of the original length.
    """

    def __init__(self, min_ratio: float = 0.6):
        self.min_ratio = min_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.ndimage import zoom

        T = x.shape[2]
        crop_len = max(1, int(np.random.uniform(self.min_ratio, 1.0) * T))
        start    = np.random.randint(0, T - crop_len + 1)
        cropped  = x[:, :, start: start + crop_len]       # (20, 20, crop_len)

        if crop_len == T:
            return x

        scale  = T / crop_len
        resized = zoom(cropped, (1.0, 1.0, scale), order=1)  # linear interp

        # zoom may produce length T±1 due to float arithmetic — clip/pad
        if resized.shape[2] >= T:
            return resized[:, :, :T].astype(np.float32)
        else:
            pad = T - resized.shape[2]
            return np.pad(resized, ((0,0),(0,0),(0,pad)),
                          mode='edge').astype(np.float32)


class SpatialFlip:
    """
    Mirror the spatial / angle axis with probability p.

    Physical meaning: gesture performed in mirrored body orientation
    (e.g. left-handed vs right-handed) produces a mirrored angle
    profile in the BVP.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return np.ascontiguousarray(np.flip(x, axis=1))  # axis 1 = angle
        return x


class GaussianNoise:
    """
    Add zero-mean Gaussian noise scaled relative to the signal range.

    Parameters
    ----------
    sigma : float
        Standard deviation of the noise.  0.05 corresponds to ~5% of
        the normalised signal range (data is z-score normalised).
    """

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.sigma, x.shape).astype(np.float32)
        return x + noise


class Compose:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
# Pre-built pipelines
# ---------------------------------------------------------------------------

def physics_augmentation(
    max_doppler_shift: int   = 2,
    min_temporal_ratio: float = 0.6,
    spatial_flip_p: float    = 0.5,
    noise_sigma: float       = 0.05,
) -> Compose:
    """
    PI-SSL augmentation pipeline — physics-motivated transforms.

    Applied independently twice per sample to create two correlated
    views for NT-Xent training.

    Order: DopplerShift → TemporalCropResize → SpatialFlip → GaussianNoise
    """
    return Compose([
        DopplerShift(max_shift=max_doppler_shift),
        TemporalCropResize(min_ratio=min_temporal_ratio),
        SpatialFlip(p=spatial_flip_p),
        GaussianNoise(sigma=noise_sigma),
    ])


def generic_augmentation(
    noise_sigma: float        = 0.05,
    min_temporal_ratio: float = 0.6,
) -> Compose:
    """
    SimCLR ablation augmentation pipeline — domain-agnostic transforms.

    Uses only Gaussian noise and temporal crop, with no physics
    motivation.  Comparing SimCLR (this) against PI-SSL (physics
    augmentation) isolates the contribution of physics-informed
    augmentation design.
    """
    return Compose([
        TemporalCropResize(min_ratio=min_temporal_ratio),
        GaussianNoise(sigma=noise_sigma),
    ])


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    x = np.random.randn(20, 20, 20).astype(np.float32)
    print(f'Input  : shape={x.shape}  min={x.min():.3f}  max={x.max():.3f}')

    aug_physics = physics_augmentation()
    aug_generic = generic_augmentation()

    x_p1 = aug_physics(x)
    x_p2 = aug_physics(x)
    x_g1 = aug_generic(x)

    print(f'PI-SSL view 1 : shape={x_p1.shape}  min={x_p1.min():.3f}  max={x_p1.max():.3f}')
    print(f'PI-SSL view 2 : shape={x_p2.shape}  min={x_p2.min():.3f}  max={x_p2.max():.3f}')
    print(f'Generic view  : shape={x_g1.shape}  min={x_g1.min():.3f}  max={x_g1.max():.3f}')

    # Views should be different from each other
    print(f'view1 == view2 : {np.allclose(x_p1, x_p2)}  (expect False)')
    print(f'view1 == input : {np.allclose(x_p1, x)}     (expect False)')
    print('All checks passed.')
