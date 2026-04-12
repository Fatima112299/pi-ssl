"""
splits.py
---------
Builds 3-fold Leave-One-Room-Out (LORO) splits for PI-SSL training.

Protocol (matches Widar3.0 TPAMI baseline for direct comparison):
  - 3 physical rooms = 3 folds
  - Fold 0: test = Room 3 (Office),     train = Room 1 + Room 2
  - Fold 1: test = Room 1 (Classroom),  train = Room 2 + Room 3
  - Fold 2: test = Room 2 (Hall),       train = Room 1 + Room 3

For each fold, this module returns 4 file lists:

  pretrain_files   All files from training rooms (ANY gesture class).
                   Used for SSL pre-training — labels are never read.
                   Includes non-standard-6 envs (digit gestures, etc.)
                   so the SSL encoder sees as much Wi-Fi signal as possible.

  labeled_files    25% of standard-6 files from training rooms.
                   Stratified by gesture class so all 6 classes are
                   represented. Used for supervised fine-tuning.

  unlabeled_files  Remaining 75% of standard-6 training files.
                   Labels exist but are withheld — can be used as an
                   additional unlabeled pool for consistency losses.

  test_files       All standard-6 files from the held-out room.
                   Used only for final evaluation. Never seen during
                   training or fine-tuning.

Usage
-----
    from src.data.splits import make_loeo_splits

    pretrain, labeled, unlabeled, test = make_loeo_splits(BVP_ROOT, fold=0)
    # each is a list of dicts with keys: path, gesture_id, gesture_name,
    # user, environment, room, torso, face, rep, link_id, ...
"""

import os
import sys
import random
import collections

# Allow running this file directly from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.widar3_dataset import scan_bvp_files, STANDARD_6_GESTURES

# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------

FOLD_DEFINITIONS = {
    0: {
        'test_room'  : 3,
        'train_rooms': {1, 2},
        'description': 'test=Office (Room 3),    train=Classroom+Hall (Rooms 1+2)',
    },
    1: {
        'test_room'  : 1,
        'train_rooms': {2, 3},
        'description': 'test=Classroom (Room 1), train=Hall+Office (Rooms 2+3)',
    },
    2: {
        'test_room'  : 2,
        'train_rooms': {1, 3},
        'description': 'test=Hall (Room 2),      train=Classroom+Office (Rooms 1+3)',
    },
}


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def make_loeo_splits(bvp_root, fold, labeled_ratio=0.25, seed=42, *,
                     file_list=None):
    """
    Build train/test splits for one fold.

    Parameters
    ----------
    bvp_root : str
        Path to the top-level BVP directory.
        Pass None when supplying a pre-built file_list (see below).
    fold : int  (0, 1, or 2)
        Which room to hold out as the test domain.
    labeled_ratio : float, default 0.25
        Fraction of standard-6 training files to treat as labeled.
        Stratified by gesture class so every class gets labeled examples.
    seed : int, default 42
        Random seed for reproducibility.
    file_list : list of dict, optional
        Pre-built file list from bvp_dataset.load_npz().  When supplied,
        the disk scan (scan_bvp_files) is skipped — useful on Colab where
        raw .mat files are not present.  Each dict must have the same keys
        as scan_bvp_files() output (gesture_name, room, gesture_id, …).

    Returns
    -------
    pretrain_files : list of dict
        All files from training rooms (any gesture class).
        Feed to your SSL pre-training dataloader — ignore the labels.
    labeled_files : list of dict
        labeled_ratio fraction of standard-6 files from training rooms.
        Feed to your supervised fine-tuning step.
    unlabeled_files : list of dict
        Remaining (1 - labeled_ratio) standard-6 training files.
        Labels exist in each dict but should be withheld during training.
    test_files : list of dict
        All standard-6 files from the held-out room.
        Use only for evaluation — never for training.
    """
    if fold not in FOLD_DEFINITIONS:
        raise ValueError(f"fold must be 0, 1, or 2. Got: {fold}")

    fold_def    = FOLD_DEFINITIONS[fold]
    test_room   = fold_def['test_room']
    train_rooms = fold_def['train_rooms']

    # Accept a pre-built file_list (e.g. from bvp_dataset.load_npz()) so the
    # raw .mat files do not need to be present (Colab / preprocessed-only env).
    if file_list is not None:
        all_files = list(file_list)
    else:
        # Single disk scan — filter in memory to avoid scanning twice
        all_files, _ = scan_bvp_files(bvp_root)

    # Standard-6 subset (used for supervised training and evaluation)
    std6_files = [f for f in all_files if f['gesture_name'] in STANDARD_6_GESTURES]

    # Split by room
    pretrain_files = [f for f in all_files  if f['room'] in train_rooms]
    test_files     = [f for f in std6_files if f['room'] == test_room]
    train_std6     = [f for f in std6_files if f['room'] in train_rooms]

    # Labeled / unlabeled split — stratified by gesture class
    labeled_files, unlabeled_files = _stratified_split(
        train_std6, ratio=labeled_ratio, key='gesture_name', seed=seed,
    )

    return pretrain_files, labeled_files, unlabeled_files, test_files


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_all_folds(bvp_root, labeled_ratio=0.25, seed=42):
    """
    Return splits for all 3 folds at once.

    Returns
    -------
    list of 3 tuples: [(pretrain, labeled, unlabeled, test), ...]
    Index 0 = Fold 0, index 1 = Fold 1, index 2 = Fold 2.
    """
    return [
        make_loeo_splits(bvp_root, fold, labeled_ratio, seed)
        for fold in range(3)
    ]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _stratified_split(file_list, ratio, key, seed):
    """
    Split file_list into (selected, remaining) stratified by file[key].
    selected has `ratio` fraction of files from each class (min 1 per class).
    """
    rng = random.Random(seed)

    groups = collections.defaultdict(list)
    for f in file_list:
        groups[f[key]].append(f)

    selected, remaining = [], []
    for class_files in groups.values():
        shuffled = list(class_files)
        rng.shuffle(shuffled)
        n_labeled = max(1, round(len(shuffled) * ratio))
        selected.extend(shuffled[:n_labeled])
        remaining.extend(shuffled[n_labeled:])

    return selected, remaining


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    BVP_ROOT = r'C:\Projects\pi-ssl\data\widar3\BVP'

    print(f"Building 3-fold LOEO splits  (labeled_ratio=25%, seed=42)\n")

    for fold in range(3):
        desc = FOLD_DEFINITIONS[fold]['description']
        pretrain, labeled, unlabeled, test = make_loeo_splits(BVP_ROOT, fold)

        total_train = len(labeled) + len(unlabeled)
        print(f"{'='*62}")
        print(f"Fold {fold}  —  {desc}")
        print(f"{'='*62}")
        print(f"  SSL pre-train (all gestures) : {len(pretrain):>6}")
        print(f"  Labeled train  (std-6, 25%)  : {len(labeled):>6}")
        print(f"  Unlabeled train (std-6, 75%) : {len(unlabeled):>6}")
        print(f"  Test           (std-6)       : {len(test):>6}")

        # Per-class counts in labeled set
        label_counts = collections.Counter(f['gesture_name'] for f in labeled)
        test_counts  = collections.Counter(f['gesture_name'] for f in test)

        print(f"\n  Gesture balance — labeled train vs test:")
        print(f"  {'Gesture':>22}   {'Labeled':>8}   {'Test':>8}")
        print(f"  {'-'*44}")
        for g in sorted(label_counts):
            print(f"  {g:>22}   {label_counts[g]:>8}   {test_counts.get(g, 0):>8}")
        print()

    print("Done.")
