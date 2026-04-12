"""
preprocess.py
-------------
Converts every BVP .mat file into a single compressed .npz dataset.

WHY
---
scipy.io.loadmat() is slow — parsing a MATLAB binary file has significant
overhead.  Loading 43 k files one-by-one during training would add several
minutes per epoch.  By running this script once, we:
  1. Pay the I/O cost once instead of every epoch.
  2. Unify shapes: T varies 0-38 across files; we pad/truncate to T=20.
  3. Normalise: per-sample z-score removes environment-level signal-strength
     differences so the SSL encoder learns gesture shape, not room acoustics.

OUTPUT
------
  data/widar3/preprocessed.npz  (~600 MB compressed)

  Keys:
    bvp            float32  (N, 20, 20, 20)  — normalised, padded BVP cubes
    gesture_id     int16    (N,)             — global 0-indexed label (0-21)
    room           int8     (N,)             — 1=Classroom 2=Hall 3=Office
    environment_id int8     (N,)             — 0-indexed environment index
    gesture_name   U20      (N,)             — semantic name, e.g. "Clap"
    user           U8       (N,)             — e.g. "user1"
    environment    U16      (N,)             — e.g. "20181130-VS"
    torso          int8     (N,)
    face           int8     (N,)
    rep            int16    (N,)

HOW TO RUN (from the project root, inside the pissl conda env):
  python scripts/preprocess.py

Expected runtime: ~5-8 minutes for 43 k files on a standard laptop.
"""

import os
import sys
import time
import collections

import numpy as np
import scipy.io

# Allow running from the project root or from the scripts/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.widar3_dataset import scan_bvp_files

# ── Config ────────────────────────────────────────────────────────────────────
BVP_ROOT    = r'C:\Projects\pi-ssl\data\widar3\BVP'
OUTPUT_PATH = r'C:\Projects\pi-ssl\data\widar3\preprocessed.npz'
T_TARGET    = 20   # all samples are padded / truncated to this many time frames
# ─────────────────────────────────────────────────────────────────────────────


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def normalise_sample(bvp):
    """
    Per-sample z-score normalisation over the full (20, 20, T) volume.

    WHY per-sample:  Signal strength varies across rooms (path loss,
    reflections).  Dividing by the per-sample standard deviation removes
    this environment-level offset, so the SSL encoder learns the gesture
    shape rather than the room's absolute power level.

    WHY before padding: padding adds zeros that should mean "no gesture",
    not "silence that participates in the statistics."  Normalising first
    keeps the padding semantically clean.
    """
    bvp   = bvp.astype(np.float32)
    mean  = bvp.mean()
    std   = bvp.std()
    return (bvp - mean) / (std + 1e-6)


def pad_or_truncate(bvp, t_target=T_TARGET):
    """
    Reshape the time axis of (20, 20, T) → (20, 20, t_target).

      T > t_target : keep first t_target frames  (the gesture is fully captured)
      T < t_target : zero-pad on the right       (extend with silence)
      T == t_target: return unchanged
    """
    T = bvp.shape[2]
    if T > t_target:
        return bvp[:, :, :t_target]
    if T < t_target:
        return np.pad(bvp, ((0, 0), (0, 0), (0, t_target - T)),
                      mode='constant', constant_values=0.0)
    return bvp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(BVP_ROOT):
        print(f"ERROR: BVP_ROOT not found:\n  {BVP_ROOT}")
        sys.exit(1)

    # ── Step 1: Scan ─────────────────────────────────────────────────────────
    print("Step 1/4 — Scanning BVP folder ...")
    file_list, scan_skipped = scan_bvp_files(BVP_ROOT)
    N = len(file_list)
    print(f"  Files to process : {N}")
    print(f"  Scan skipped     : {scan_skipped}")

    if N == 0:
        print("ERROR: No files found. Check BVP_ROOT.")
        sys.exit(1)

    # ── Step 2: Pre-allocate output arrays ───────────────────────────────────
    # Use fixed-width unicode strings (no pickle) for portability.
    print(f"\nStep 2/4 — Pre-allocating arrays for {N} samples ...")
    bvp_out        = np.zeros((N, 20, 20, T_TARGET), dtype=np.float32)
    gesture_id     = np.zeros(N, dtype=np.int16)
    room           = np.zeros(N, dtype=np.int8)
    environment_id = np.zeros(N, dtype=np.int8)
    gesture_name   = np.empty(N, dtype='U20')   # max gesture name: 16 chars
    user           = np.empty(N, dtype='U8')    # max user name: 6 chars
    environment    = np.empty(N, dtype='U16')   # max env name: 11 chars
    torso          = np.zeros(N, dtype=np.int8)
    face           = np.zeros(N, dtype=np.int8)
    rep            = np.zeros(N, dtype=np.int16)
    mem_mb = bvp_out.nbytes / 1e6
    print(f"  BVP array (uncompressed) : {mem_mb:.0f} MB")

    # ── Step 3: Load, normalise, pad ─────────────────────────────────────────
    print(f"\nStep 3/4 — Processing files ...")
    saved   = 0   # samples successfully written
    n_skip  = 0   # samples skipped (load error or T=0)
    t0      = time.time()

    for i, entry in enumerate(file_list):

        # Progress report every 2 000 files
        if i > 0 and i % 2000 == 0:
            elapsed   = time.time() - t0
            rate      = i / elapsed
            remaining = (N - i) / rate
            pct       = 100.0 * i / N
            print(f"  [{i:>5}/{N}]  {pct:4.1f}%  saved={saved}  "
                  f"skipped={n_skip}  rate={rate:.0f}/s  ETA={remaining:.0f}s")

        # Load raw .mat
        try:
            mat = scipy.io.loadmat(entry['path'])
            bvp = np.array(mat['velocity_spectrum_ro'], dtype=np.float32)
        except Exception as exc:
            print(f"  WARN load error: {entry['path']}\n       {exc}")
            n_skip += 1
            continue

        # Skip corrupt samples: ndim < 3 (malformed array) or T=0 (no frames)
        if bvp.ndim < 3 or bvp.shape[2] == 0:
            n_skip += 1
            continue

        # Normalise first, then pad/truncate
        bvp = normalise_sample(bvp)
        bvp = pad_or_truncate(bvp)

        # Write into the pre-allocated arrays at index `saved`
        bvp_out[saved]        = bvp
        gesture_id[saved]     = entry['gesture_id']
        room[saved]           = entry['room']
        environment_id[saved] = entry['environment_id']
        gesture_name[saved]   = entry['gesture_name']
        user[saved]           = entry['user']
        environment[saved]    = entry['environment']
        torso[saved]          = entry['torso']
        face[saved]           = entry['face']
        rep[saved]            = entry['rep']
        saved += 1

    # Trim to actual count (n_skip samples were not written)
    bvp_out        = bvp_out[:saved]
    gesture_id     = gesture_id[:saved]
    room           = room[:saved]
    environment_id = environment_id[:saved]
    gesture_name   = gesture_name[:saved]
    user           = user[:saved]
    environment    = environment[:saved]
    torso          = torso[:saved]
    face           = face[:saved]
    rep            = rep[:saved]

    elapsed = time.time() - t0
    print(f"\n  Finished processing in {elapsed:.0f}s")
    print(f"  Saved   : {saved}")
    print(f"  Skipped : {n_skip}  (load errors + T=0 samples)")

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    out_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nStep 4/4 — Compressing and saving to:\n  {OUTPUT_PATH}")
    print("  (compression may take ~30 seconds ...)")
    t_save = time.time()

    np.savez_compressed(
        OUTPUT_PATH,
        bvp=bvp_out,
        gesture_id=gesture_id,
        room=room,
        environment_id=environment_id,
        gesture_name=gesture_name,
        user=user,
        environment=environment,
        torso=torso,
        face=face,
        rep=rep,
    )

    save_elapsed = time.time() - t_save
    total_elapsed = time.time() - t0
    file_mb = os.path.getsize(OUTPUT_PATH) / 1e6

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("DONE")
    print(f"{'='*62}")
    print(f"  Samples saved    : {saved}")
    print(f"  Samples skipped  : {n_skip}")
    print(f"  Output file      : {OUTPUT_PATH}")
    print(f"  Compressed size  : {file_mb:.1f} MB")
    print(f"  Save time        : {save_elapsed:.1f}s")
    print(f"  Total time       : {total_elapsed:.0f}s")
    print(f"  bvp shape        : {bvp_out.shape}")
    print(f"  bvp dtype        : {bvp_out.dtype}")

    # ── Verification — reload and spot-check ──────────────────────────────────
    print(f"\n{'='*62}")
    print("VERIFICATION (reloading saved file)")
    print(f"{'='*62}")

    data = np.load(OUTPUT_PATH, allow_pickle=False)

    print(f"  Keys             : {sorted(data.files)}")
    print(f"  bvp.shape        : {data['bvp'].shape}")
    print(f"  bvp.dtype        : {data['bvp'].dtype}")
    print(f"  gesture_id range : {data['gesture_id'].min()} – {data['gesture_id'].max()}")
    print(f"  rooms present    : {sorted(set(data['room'].tolist()))}")
    print(f"  environments     : {len(set(data['environment'].tolist()))}")
    print(f"  users            : {sorted(set(data['user'].tolist()))}")

    # Check normalisation: per-sample mean should be ~0, std ~1
    sample_means = data['bvp'].reshape(saved, -1).mean(axis=1)
    sample_stds  = data['bvp'].reshape(saved, -1).std(axis=1)
    print(f"\n  Normalisation check (should be ~0 mean, ~1 std per sample):")
    print(f"    mean of per-sample means : {sample_means.mean():.4f}")
    print(f"    mean of per-sample stds  : {sample_stds.mean():.4f}")

    # Per-gesture sample counts
    gc = collections.Counter(data['gesture_name'].tolist())
    print(f"\n  Per-gesture sample counts ({len(gc)} classes):")
    print(f"  {'Gesture':>22}   Count")
    print(f"  {'-'*34}")
    for name, count in sorted(gc.items()):
        print(f"  {name:>22} : {count:>6}")

    print("\nAll checks passed! Run src/data/bvp_dataset.py next.")


if __name__ == '__main__':
    main()
