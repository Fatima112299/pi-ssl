"""
check_shapes.py
---------------
Scans every BVP .mat file and records the shape of velocity_spectrum_ro.

WHY:
  Before we can build a data pipeline, we need to know whether all files
  share the same array shape. A mismatch would crash the batch loader.
  This script answers:
    - What shape(s) exist across all 33,534 files?
    - Which environments / users have non-standard shapes?
    - Are there any files that fail to load?

HOW TO RUN (from the project root, inside the pissl conda env):
  python scripts/check_shapes.py
"""

import os
import sys
import collections
import scipy.io
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
BVP_ROOT = r'C:\Projects\pi-ssl\data\widar3\BVP'
# ────────────────────────────────────────────────────────────────────────────


def main():
    if not os.path.isdir(BVP_ROOT):
        print(f"ERROR: BVP_ROOT not found: {BVP_ROOT}")
        sys.exit(1)

    # Counters / collectors
    shape_counter   = collections.Counter()   # shape -> count
    shape_examples  = collections.defaultdict(list)  # shape -> list of paths (first 3)
    failed_files    = []   # files that raised an exception
    total_scanned   = 0

    print(f"Scanning: {BVP_ROOT}")
    print("(This may take a minute for 33 k files ...)\n")

    date_folders = sorted([
        d for d in os.listdir(BVP_ROOT)
        if os.path.isdir(os.path.join(BVP_ROOT, d)) and d.endswith('-VS')
    ])
    print(f"Found {len(date_folders)} environment folders: {date_folders}\n")

    for date_folder in date_folders:
        link_path = os.path.join(BVP_ROOT, date_folder, '6-link')
        if not os.path.isdir(link_path):
            print(f"  WARNING: no 6-link folder in {date_folder}, skipping.")
            continue

        for user_folder in sorted(os.listdir(link_path)):
            user_path = os.path.join(link_path, user_folder)
            if not os.path.isdir(user_path):
                continue

            for fname in sorted(os.listdir(user_path)):
                if not fname.endswith('.mat'):
                    continue

                fpath = os.path.join(user_path, fname)
                total_scanned += 1

                # Progress tick every 2000 files
                if total_scanned % 2000 == 0:
                    print(f"  ... scanned {total_scanned} files so far")

                try:
                    mat  = scipy.io.loadmat(fpath)
                    arr  = mat['velocity_spectrum_ro']
                    shp  = tuple(arr.shape)
                    shape_counter[shp] += 1
                    if len(shape_examples[shp]) < 3:
                        shape_examples[shp].append(
                            os.path.join(date_folder, user_folder, fname)
                        )
                except Exception as e:
                    failed_files.append((fpath, str(e)))

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total files scanned : {total_scanned}")
    print(f"Files that failed   : {len(failed_files)}")
    print(f"Unique shapes found : {len(shape_counter)}\n")

    print("Shape breakdown (shape -> count):")
    for shp, cnt in sorted(shape_counter.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total_scanned if total_scanned else 0
        print(f"  {str(shp):30s}  {cnt:6d} files  ({pct:.1f}%)")
        print(f"    Example files:")
        for ex in shape_examples[shp]:
            print(f"      {ex}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for fpath, err in failed_files[:20]:   # show first 20 only
            print(f"  {fpath}")
            print(f"    Error: {err}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if len(shape_counter) == 1:
        only_shape = list(shape_counter.keys())[0]
        print(f"All files share the same shape: {only_shape}")
        print("→ Good: we can use a fixed-size input without any padding/resizing.")
    else:
        dominant = shape_counter.most_common(1)[0]
        print(f"Multiple shapes found! Dominant shape: {dominant[0]} ({dominant[1]} files)")
        print("→ We will need a strategy (crop / pad / resize) to unify shapes.")
        print("  Non-dominant shapes:")
        for shp, cnt in shape_counter.items():
            if shp != dominant[0]:
                print(f"    {shp} — {cnt} files")

    print()


if __name__ == '__main__':
    main()
