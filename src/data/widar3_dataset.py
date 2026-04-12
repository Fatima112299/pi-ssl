"""
widar3_dataset.py
-----------------
Loads BVP .mat files from the Widar3.0 dataset.

Each .mat file contains one gesture sample:
  - variable name : velocity_spectrum_ro
  - shape         : (20, 20, T) where T varies 0-38

Filename format (per official Widar3.0 README):
  {user}-{local_gesture}-{torso}-{face}-{repetition}-[algo_params]-L{link}.mat
  e.g. user1-6-3-5-14-1-1e-07-100-20-100000-L0.mat
              ^                 ^
        local gesture       repetition

  IMPORTANT: local_gesture is environment-specific. The same local number
  means different gestures in different environments. Use ENV_GESTURE_MAP
  to convert (environment, user, local_label) -> semantic gesture name.

Folder structures (two variants exist in the dataset):
  Normal  : BVP/{date}-VS/6-link/{user}/*.mat    (13 environments)
  Special : BVP/20181130-VS/{user}/*.mat          (no 6-link subfolder)
"""

import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Gesture mapping tables  (source: official Widar3.0 README)
# ---------------------------------------------------------------------------

# Maps environment -> user -> {local_label: semantic_gesture_name}
# Use '*' as user key when the mapping applies to all users in that environment.
# User-specific entries take priority over '*'.
ENV_GESTURE_MAP = {
    '20181109-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
        4: 'Slide', 5: 'Draw-Zigzag-V', 6: 'Draw-N-V',
    }},
    '20181112-VS': {'*': {
        1: 'Draw-1', 2: 'Draw-2', 3: 'Draw-3', 4: 'Draw-4', 5: 'Draw-5',
        6: 'Draw-6', 7: 'Draw-7', 8: 'Draw-8', 9: 'Draw-9', 0: 'Draw-0',
    }},
    '20181115-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
        4: 'Draw-O-V', 5: 'Draw-Zigzag-V', 6: 'Draw-N-V',
    }},
    '20181117-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
        4: 'Draw-O-V', 5: 'Draw-Zigzag-V', 6: 'Draw-N-V',
    }},
    '20181118-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
        4: 'Draw-O-V', 5: 'Draw-Zigzag-V', 6: 'Draw-N-V',
    }},
    '20181121-VS': {'*': {
        1: 'Slide', 2: 'Draw-O-H', 3: 'Draw-Zigzag-H',
        4: 'Draw-N-H', 5: 'Draw-Triangle-H', 6: 'Draw-Rectangle-H',
    }},
    '20181127-VS': {'*': {
        1: 'Slide', 2: 'Draw-O-H', 3: 'Draw-Zigzag-H',
        4: 'Draw-N-H', 5: 'Draw-Triangle-H', 6: 'Draw-Rectangle-H',
    }},
    '20181128-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
        4: 'Draw-O-H', 5: 'Draw-Zigzag-H', 6: 'Draw-N-H',
    }},
    '20181130-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide',
        5: 'Draw-O-H', 6: 'Draw-Zigzag-H', 7: 'Draw-N-H',
        8: 'Draw-Triangle-H', 9: 'Draw-Rectangle-H',
    }},
    '20181204-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide',
        5: 'Draw-O-H', 6: 'Draw-Zigzag-H', 7: 'Draw-N-H',
        8: 'Draw-Triangle-H', 9: 'Draw-Rectangle-H',
    }},
    '20181205-VS': {
        'user2': {
            1: 'Draw-O-H', 2: 'Draw-Zigzag-H', 3: 'Draw-N-H',
            4: 'Draw-Triangle-H', 5: 'Draw-Rectangle-H',
        },
        'user3': {
            1: 'Slide', 2: 'Draw-O-H', 3: 'Draw-Zigzag-H',
            4: 'Draw-N-H', 5: 'Draw-Triangle-H', 6: 'Draw-Rectangle-H',
        },
    },
    '20181208-VS': {
        'user2': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide'},
        'user3': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap'},
    },
    '20181209-VS': {
        'user2': {1: 'Push&Pull'},
        'user6': {
            1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
            4: 'Slide', 5: 'Draw-O-H', 6: 'Draw-Zigzag-H',
        },
    },
    '20181211-VS': {'*': {
        1: 'Push&Pull', 2: 'Sweep', 3: 'Clap',
        4: 'Slide', 5: 'Draw-O-H', 6: 'Draw-Zigzag-H',
    }},
}

# Maps each environment to its physical room (1=Classroom, 2=Hall, 3=Office)
ENV_ROOM_MAP = {
    '20181109-VS': 1, '20181112-VS': 1, '20181115-VS': 1,
    '20181121-VS': 1, '20181130-VS': 1,
    '20181117-VS': 2, '20181118-VS': 2, '20181127-VS': 2,
    '20181128-VS': 2, '20181204-VS': 2, '20181205-VS': 2,
    '20181208-VS': 2, '20181209-VS': 2,
    '20181211-VS': 3,
}

# Global gesture vocabulary — 22 unique semantic gesture classes (0-indexed)
# IDs 0-5 are the standard 6 benchmark gestures from the Widar3.0 TPAMI paper.
GESTURE_TO_ID = {
    'Push&Pull': 0, 'Sweep': 1, 'Clap': 2, 'Slide': 3,
    'Draw-O-H': 4, 'Draw-Zigzag-H': 5,
    'Draw-N-H': 6, 'Draw-Triangle-H': 7, 'Draw-Rectangle-H': 8,
    'Draw-O-V': 9, 'Draw-Zigzag-V': 10, 'Draw-N-V': 11,
    'Draw-1': 12, 'Draw-2': 13, 'Draw-3': 14, 'Draw-4': 15,
    'Draw-5': 16, 'Draw-6': 17, 'Draw-7': 18, 'Draw-8': 19,
    'Draw-9': 20, 'Draw-0': 21,
}

ID_TO_GESTURE = {v: k for k, v in GESTURE_TO_ID.items()}

# The 6 standard benchmark gestures from the Widar3.0 TPAMI paper
STANDARD_6_GESTURES = {'Push&Pull', 'Sweep', 'Clap', 'Slide', 'Draw-O-H', 'Draw-Zigzag-H'}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filename):
    """
    Extract metadata from a BVP filename.

    Filename format (per official Widar3.0 README):
      {user}-{local_gesture}-{torso}-{face}-{repetition}-[algo_params]-L{link}.mat
      e.g. user1-6-3-5-14-1-1e-07-100-20-100000-L0.mat

    Returns a dict with:
      user          : str, e.g. "user1"
      local_gesture : int, gesture number LOCAL to this environment
      torso         : int, torso location (1-5)
      face          : int, face orientation (1-5)
      rep           : int, repetition number
      link_id       : str, e.g. "L0" (last part of the filename)

    Returns None if the filename cannot be parsed.
    """
    name = filename.replace('.mat', '')
    parts = name.split('-')

    # Need at least: user, local_gesture, torso, face, rep
    if len(parts) < 5:
        return None

    try:
        user          = parts[0]           # e.g. "user1"
        local_gesture = int(parts[1])      # gesture label, LOCAL to this environment
        torso         = int(parts[2])      # torso location
        face          = int(parts[3])      # face orientation
        rep           = int(parts[4])      # repetition number
        link_id       = parts[-1]          # e.g. "L0"
    except ValueError:
        return None

    return {
        'user'         : user,
        'local_gesture': local_gesture,
        'torso'        : torso,
        'face'         : face,
        'rep'          : rep,
        'link_id'      : link_id,
    }


def resolve_gesture(environment, user, local_gesture):
    """
    Convert a local gesture label to a global semantic gesture name.

    Parameters
    ----------
    environment   : str, e.g. "20181109-VS"
    user          : str, e.g. "user1"
    local_gesture : int, the gesture number from the filename

    Returns
    -------
    str  : semantic gesture name, e.g. "Push&Pull"
    None : if the mapping is not found
    """
    env_map = ENV_GESTURE_MAP.get(environment)
    if env_map is None:
        return None

    # User-specific mapping takes priority over wildcard '*'
    user_map = env_map.get(user) or env_map.get('*')
    if user_map is None:
        return None

    return user_map.get(local_gesture)


# ---------------------------------------------------------------------------
# Dataset scanner
# ---------------------------------------------------------------------------

def scan_bvp_files(bvp_root, gestures=None):
    """
    Walk the BVP folder and collect all valid .mat files.

    Handles both folder structures present in Widar3.0:
      Normal  : {bvp_root}/{date}-VS/6-link/{user}/*.mat
      Special : {bvp_root}/20181130-VS/{user}/*.mat  (no 6-link subfolder)

    Parameters
    ----------
    bvp_root : str
        Path to the top-level BVP directory.
    gestures : set of str, optional
        If given, only files whose semantic gesture name is in this set are
        included. E.g. pass STANDARD_6_GESTURES to restrict to the 6
        benchmark gestures.

    Returns
    -------
    file_list : list of dict
        Each entry contains:
          path          : full path to the .mat file
          filename      : basename of the file
          user          : str, e.g. "user1"
          local_gesture : int, local gesture label from filename
          gesture_name  : str, semantic gesture name, e.g. "Push&Pull"
          gesture_id    : int, global 0-indexed ID from GESTURE_TO_ID
          torso         : int
          face          : int
          rep           : int
          link_id       : str, e.g. "L0"
          environment   : str, e.g. "20181109-VS"
          environment_id: int, 0-indexed across all date folders
          room          : int, physical room (1=Classroom, 2=Hall, 3=Office)

    skipped : dict
        Counts of files skipped, keyed by reason:
          bad_filename    : .mat file whose name could not be parsed
          unknown_gesture : parsed but (env, user, local_gesture) not in map
          filtered_out    : valid but excluded by the gestures filter
    """
    file_list = []
    skipped = {'bad_filename': 0, 'unknown_gesture': 0, 'filtered_out': 0}

    all_date_folders = sorted([
        d for d in os.listdir(bvp_root)
        if os.path.isdir(os.path.join(bvp_root, d)) and d.endswith('-VS')
    ])
    env_id_map = {d: i for i, d in enumerate(all_date_folders)}

    for date_folder in all_date_folders:
        date_path = os.path.join(bvp_root, date_folder)
        link_path = os.path.join(date_path, '6-link')
        room      = ENV_ROOM_MAP.get(date_folder, 0)
        env_id    = env_id_map[date_folder]

        # Choose scan root: use 6-link subfolder if it exists, else scan date
        # folder directly (handles 20181130-VS which has no 6-link subfolder)
        scan_root = link_path if os.path.isdir(link_path) else date_path

        for user_folder in sorted(os.listdir(scan_root)):
            user_path = os.path.join(scan_root, user_folder)
            if not os.path.isdir(user_path):
                continue
            if not user_folder.startswith('user'):
                continue   # skip any non-user entries (e.g. stray files)

            for fname in sorted(os.listdir(user_path)):
                if not fname.endswith('.mat'):
                    continue

                meta = parse_filename(fname)
                if meta is None:
                    skipped['bad_filename'] += 1
                    continue

                gesture_name = resolve_gesture(date_folder, meta['user'], meta['local_gesture'])
                if gesture_name is None:
                    skipped['unknown_gesture'] += 1
                    continue

                if gestures is not None and gesture_name not in gestures:
                    skipped['filtered_out'] += 1
                    continue

                gesture_id = GESTURE_TO_ID.get(gesture_name)
                if gesture_id is None:
                    skipped['unknown_gesture'] += 1
                    continue

                file_list.append({
                    'path'          : os.path.join(user_path, fname),
                    'filename'      : fname,
                    'user'          : meta['user'],
                    'local_gesture' : meta['local_gesture'],
                    'gesture_name'  : gesture_name,
                    'gesture_id'    : gesture_id,
                    'torso'         : meta['torso'],
                    'face'          : meta['face'],
                    'rep'           : meta['rep'],
                    'link_id'       : meta['link_id'],
                    'environment'   : date_folder,
                    'environment_id': env_id,
                    'room'          : room,
                })

    return file_list, skipped


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class Widar3Dataset(Dataset):
    """
    PyTorch Dataset for Widar3.0 BVP data.

    Parameters
    ----------
    file_list : list of dict
        Each dict must have at least 'path' and 'gesture_id'.
        Build with scan_bvp_files() or the LOEO splitter in splits.py.
    transform : callable, optional
        Applied to the numpy array before returning it.
        Use this for normalisation, padding to T=20, and augmentation.
    """

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        entry = self.file_list[idx]

        mat = scipy.io.loadmat(entry['path'])
        bvp = mat['velocity_spectrum_ro']       # shape: (20, 20, T)
        bvp = np.array(bvp, dtype=np.float32)

        if self.transform is not None:
            bvp = self.transform(bvp)

        # gesture_id is already the global 0-indexed label
        label = entry['gesture_id']

        return bvp, label


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify everything works:
#   python src/data/widar3_dataset.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    BVP_ROOT = r'C:\Projects\pi-ssl\data\widar3\BVP'

    print("Scanning BVP folder (all gestures) ...")
    files, skipped = scan_bvp_files(BVP_ROOT)
    print(f"  Files found : {len(files)}")
    print(f"  Skipped     : {skipped}")

    if not files:
        print("ERROR: No files found. Check your BVP_ROOT path.")
        sys.exit(1)

    print("\nFirst 3 entries:")
    for f in files[:3]:
        print(f"  {f['environment']} | {f['user']} | local={f['local_gesture']}"
              f" | gesture={f['gesture_name']} (id={f['gesture_id']}) | room={f['room']}")

    envs     = set(f['environment']  for f in files)
    users    = set(f['user']         for f in files)
    gestures = set(f['gesture_name'] for f in files)
    rooms    = set(f['room']         for f in files)
    print(f"\nEnvironments : {len(envs)}  → {sorted(envs)}")
    print(f"Rooms        : {sorted(rooms)}")
    print(f"Users        : {len(users)} → {sorted(users)}")
    print(f"Gestures     : {len(gestures)} → {sorted(gestures)}")

    print("\nScanning with STANDARD_6_GESTURES filter ...")
    files6, skipped6 = scan_bvp_files(BVP_ROOT, gestures=STANDARD_6_GESTURES)
    print(f"  Files with standard-6 gestures : {len(files6)}")
    print(f"  Filtered out                   : {skipped6['filtered_out']}")

    print("\nLoading one sample ...")
    dataset = Widar3Dataset(files)
    bvp, label = dataset[0]
    print(f"  BVP shape : {bvp.shape}")
    print(f"  BVP dtype : {bvp.dtype}")
    print(f"  Label     : {label}  ({ID_TO_GESTURE[label]})")
    print(f"  Min/Max   : {bvp.min():.4f} / {bvp.max():.4f}")

    print("\nAll checks passed!")
