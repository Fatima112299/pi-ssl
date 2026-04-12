"""
bvp_dataset.py
--------------
Fast PyTorch Dataset backed by preprocessed.npz instead of raw .mat files.

WHY
---
scipy.io.loadmat() parses a MATLAB binary on every __getitem__ call.
On a T4 GPU, the model forward+backward takes ~10 ms per batch; a single
.mat load takes ~1 ms — so disk I/O would eat 30-50% of training time.
Loading preprocessed.npz once at startup (1-2 s) means __getitem__ is a
pure numpy array slice: no disk, no parsing, no overhead.

USAGE
-----
    from src.data.bvp_dataset import load_npz, BVPDataset
    from src.data.splits import make_loeo_splits

    npz_data, file_list = load_npz('data/widar3/preprocessed.npz')

    pretrain, labeled, unlabeled, test = make_loeo_splits(
        bvp_root=None, fold=0, file_list=file_list
    )

    train_ds = BVPDataset(labeled, npz_data)
    bvp, label = train_ds[0]
    # bvp  : torch.FloatTensor, shape (1, 20, 20, 20)  — channel-first
    # label: int, global gesture ID (0-20)

SHAPE NOTE
----------
BVPDataset returns tensors of shape (1, 20, 20, 20):
  dim 0 : channel (always 1 — intensity of the Doppler-velocity spectrum)
  dim 1 : Doppler velocity bins   (20)
  dim 2 : spatial / angle bins    (20)
  dim 3 : time frames             (20, padded/truncated by preprocess.py)

This channel-first layout matches PyTorch Conv3d and most SSL frameworks.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# NPZ loader
# ---------------------------------------------------------------------------

def load_npz(npz_path):
    """
    Load preprocessed.npz and reconstruct a file_list compatible with
    splits.make_loeo_splits().

    Parameters
    ----------
    npz_path : str
        Path to the file produced by scripts/preprocess.py.

    Returns
    -------
    npz_data : dict
        {
          'bvp'            : float32 ndarray (N, 20, 20, 20),
          'gesture_id'     : int16   ndarray (N,),
          'gesture_name'   : str     ndarray (N,),
          'room'           : int8    ndarray (N,),
          'environment_id' : int8    ndarray (N,),
          'environment'    : str     ndarray (N,),
          'user'           : str     ndarray (N,),
          'torso'          : int8    ndarray (N,),
          'face'           : int8    ndarray (N,),
          'rep'            : int16   ndarray (N,),
        }

    file_list : list of dict
        One dict per sample. Same keys as scan_bvp_files() output plus
        'npz_idx' (integer row index into npz_data arrays). The 'path'
        key is set to None — BVPDataset uses 'npz_idx' instead.
        Compatible with make_loeo_splits(file_list=...).
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"preprocessed.npz not found: {npz_path}\n"
            "Run scripts/preprocess.py first."
        )

    raw      = np.load(npz_path, allow_pickle=False)
    npz_data = {k: raw[k] for k in raw.files}   # copy into plain dict
    raw.close()

    N = len(npz_data['bvp'])

    # Reconstruct the file_list format that splits.py expects.
    # Python scalars (int/str) rather than numpy scalars so filtering with
    # set membership (`gesture_name in STANDARD_6_GESTURES`) works correctly.
    file_list = [
        {
            'npz_idx'       : i,
            'path'          : None,
            'gesture_id'    : int(npz_data['gesture_id'][i]),
            'gesture_name'  : str(npz_data['gesture_name'][i]),
            'room'          : int(npz_data['room'][i]),
            'environment_id': int(npz_data['environment_id'][i]),
            'environment'   : str(npz_data['environment'][i]),
            'user'          : str(npz_data['user'][i]),
            'torso'         : int(npz_data['torso'][i]),
            'face'          : int(npz_data['face'][i]),
            'rep'           : int(npz_data['rep'][i]),
        }
        for i in range(N)
    ]

    return npz_data, file_list


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BVPDataset(Dataset):
    """
    PyTorch Dataset backed by a pre-loaded preprocessed.npz.

    __getitem__ performs a single numpy array slice — no disk I/O.

    Parameters
    ----------
    file_list : list of dict
        Each dict must have 'npz_idx' and 'gesture_id'.
        Build with load_npz(), then filter/split with make_loeo_splits().
    npz_data : dict
        Returned by load_npz(). Must contain the 'bvp' key.
    transform : callable, optional
        Applied to the raw (20, 20, 20) float32 numpy array before the
        tensor conversion. Use for data augmentation during SSL pre-training
        (e.g. temporal crop, Gaussian noise). Leave None for evaluation.
    """

    def __init__(self, file_list, npz_data, transform=None):
        self.file_list = file_list
        self.bvp       = npz_data['bvp']   # (N_total, 20, 20, 20) float32
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        entry = self.file_list[idx]
        bvp   = self.bvp[entry['npz_idx']]   # (20, 20, 20) float32

        if self.transform is not None:
            bvp = self.transform(bvp)

        # Add channel dimension: (20,20,20) → (1,20,20,20)
        bvp = torch.from_numpy(bvp.copy()).unsqueeze(0)

        return bvp, entry['gesture_id']


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import collections

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data.splits import make_loeo_splits

    NPZ_PATH = r'C:\Projects\pi-ssl\data\widar3\preprocessed.npz'

    print("Loading NPZ ...")
    npz_data, file_list = load_npz(NPZ_PATH)
    print(f"  Total samples : {len(file_list)}")
    print(f"  BVP array     : {npz_data['bvp'].shape}  {npz_data['bvp'].dtype}")

    print("\nBuilding Fold 0 splits (from NPZ, no disk scan) ...")
    pretrain, labeled, unlabeled, test = make_loeo_splits(
        bvp_root=None, fold=0, file_list=file_list
    )
    print(f"  SSL pre-train : {len(pretrain)}")
    print(f"  Labeled train : {len(labeled)}")
    print(f"  Unlabeled     : {len(unlabeled)}")
    print(f"  Test          : {len(test)}")

    print("\nCreating BVPDataset (labeled split) ...")
    ds = BVPDataset(labeled, npz_data)
    bvp, label = ds[0]
    print(f"  Dataset length  : {len(ds)}")
    print(f"  bvp shape       : {bvp.shape}")
    print(f"  bvp dtype       : {bvp.dtype}")
    print(f"  label           : {label}")
    print(f"  bvp min/max     : {bvp.min():.4f} / {bvp.max():.4f}")

    # Check that all 6 standard gestures appear in labeled set
    gesture_counts = collections.Counter(f['gesture_name'] for f in labeled)
    print(f"\n  Gesture balance in labeled split:")
    for g, n in sorted(gesture_counts.items()):
        print(f"    {g:>22} : {n}")

    # Spot-check: DataLoader round-trip
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch_bvp, batch_labels = next(iter(loader))
    print(f"\n  DataLoader batch shape : {batch_bvp.shape}")
    print(f"  DataLoader label shape : {batch_labels.shape}")

    print("\nAll checks passed!")
