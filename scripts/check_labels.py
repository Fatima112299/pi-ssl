"""
check_labels.py
---------------
Scans every BVP .mat filename and reports the true gesture distribution
using semantic gesture names (not local environment-specific numbers).

Reports:
  1. Gesture counts by semantic name
  2. User counts
  3. Environment counts with room assignments
  4. Cross-table: environments x semantic gestures
  5. Cross-table: environments x users
  6. Per-environment summary for LOEO fold design
  7. Standard-6 gesture coverage per environment and room

WHY:
  Gesture labels in filenames are LOCAL to each environment — the same
  local number means different gestures in different environments.
  This script uses the ENV_GESTURE_MAP from widar3_dataset.py to resolve
  each file to its true semantic gesture class before counting anything.
  Running this gives us the information needed to design the LOEO splitter.

HOW TO RUN (from project root, inside pissl env):
  python scripts/check_labels.py
"""

import os
import sys
import collections

# Add project root to sys.path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.widar3_dataset import (
    scan_bvp_files,
    ENV_ROOM_MAP,
    GESTURE_TO_ID,
    STANDARD_6_GESTURES,
)

BVP_ROOT = r'C:\Projects\pi-ssl\data\widar3\BVP'


def main():
    if not os.path.isdir(BVP_ROOT):
        print(f"ERROR: BVP_ROOT not found: {BVP_ROOT}")
        sys.exit(1)

    print(f"Scanning: {BVP_ROOT}")
    print("(This may take a minute for 43,000+ files ...)\n")

    files, skipped = scan_bvp_files(BVP_ROOT)
    total = len(files)

    print(f"Total files successfully parsed : {total}")
    print(f"Skipped — bad filename          : {skipped['bad_filename']}")
    print(f"Skipped — unknown gesture map   : {skipped['unknown_gesture']}")
    print()

    all_envs     = sorted(set(f['environment']  for f in files))
    all_users    = sorted(set(f['user']          for f in files))
    all_gestures = sorted(set(f['gesture_name']  for f in files),
                          key=lambda g: GESTURE_TO_ID.get(g, 99))

    # ── 1. Gesture counts (semantic) ─────────────────────────────────────────
    gesture_counts = collections.Counter(f['gesture_name'] for f in files)

    print("=" * 60)
    print("1. GESTURE COUNTS  (semantic names, global IDs)")
    print("=" * 60)
    print(f"Unique gesture classes : {len(all_gestures)}\n")
    print(f"{'ID':>4}  {'Gesture':>22}  {'Files':>8}  {'% total':>8}")
    print("-" * 48)
    for name in sorted(GESTURE_TO_ID.keys(), key=lambda g: GESTURE_TO_ID[g]):
        if name not in gesture_counts:
            continue
        cnt    = gesture_counts[name]
        gid    = GESTURE_TO_ID[name]
        marker = " *" if name in STANDARD_6_GESTURES else ""
        print(f"{gid:>4}  {name:>22}  {cnt:>8}  {100*cnt/total:>7.1f}%{marker}")
    print("  * = standard 6 benchmark gestures (Widar3.0 TPAMI paper)")

    # ── 2. User counts ────────────────────────────────────────────────────────
    user_counts = collections.Counter(f['user'] for f in files)

    print(f"\n{'='*60}")
    print("2. USER COUNTS")
    print("=" * 60)
    print(f"Unique users : {len(all_users)}\n")
    print(f"{'User':>10}  {'Files':>8}")
    print("-" * 22)
    for u in all_users:
        print(f"{u:>10}  {user_counts[u]:>8}")

    # ── 3. Environment counts with room ──────────────────────────────────────
    env_counts = collections.Counter(f['environment'] for f in files)

    print(f"\n{'='*60}")
    print("3. ENVIRONMENT COUNTS  (with room assignments)")
    print("=" * 60)
    print(f"Unique environments : {len(all_envs)}\n")
    print(f"{'Environment':>20}  {'Room':>6}  {'Files':>8}")
    print("-" * 40)
    for e in all_envs:
        room = ENV_ROOM_MAP.get(e, '?')
        print(f"{e:>20}  {room:>6}  {env_counts[e]:>8}")

    # ── 4. Cross-table: environment x semantic gesture ────────────────────────
    env_gest = collections.Counter((f['environment'], f['gesture_name']) for f in files)

    print(f"\n{'='*60}")
    print("4. CROSS-TABLE: environments x semantic gestures")
    print("   (cell = file count;  -- = gesture absent in that env)")
    print("=" * 60)

    # Short column headers to keep table width manageable
    abbrev = {
        'Push&Pull': 'PP',  'Sweep': 'Sw',       'Clap': 'Cl',
        'Slide': 'Sl',      'Draw-O-H': 'DOH',   'Draw-Zigzag-H': 'DZH',
        'Draw-N-H': 'DNH',  'Draw-Triangle-H': 'DTH', 'Draw-Rectangle-H': 'DRH',
        'Draw-O-V': 'DOV',  'Draw-Zigzag-V': 'DZV',  'Draw-N-V': 'DNV',
        'Draw-1': 'D1', 'Draw-2': 'D2', 'Draw-3': 'D3', 'Draw-4': 'D4',
        'Draw-5': 'D5', 'Draw-6': 'D6', 'Draw-7': 'D7', 'Draw-8': 'D8',
        'Draw-9': 'D9', 'Draw-0': 'D0',
    }
    header = f"{'Env':>20}  Rm" + "".join(f"{abbrev.get(g, g[:4]):>5}" for g in all_gestures)
    print(header)
    print("-" * (24 + 5 * len(all_gestures)))

    for e in all_envs:
        room = ENV_ROOM_MAP.get(e, '?')
        row = f"{e:>20}  {room:>2}"
        for g in all_gestures:
            cnt = env_gest.get((e, g), 0)
            row += f"{'--':>5}" if cnt == 0 else f"{cnt:>5}"
        print(row)

    # ── 5. Cross-table: environment x user ────────────────────────────────────
    env_user = collections.Counter((f['environment'], f['user']) for f in files)

    print(f"\n{'='*60}")
    print("5. CROSS-TABLE: environments x users")
    print("   (cell = file count;  -- = user absent in that env)")
    print("=" * 60)

    header2 = f"{'Env':>20}  Rm" + "".join(f"{u:>8}" for u in all_users)
    print(header2)
    print("-" * (24 + 8 * len(all_users)))

    for e in all_envs:
        room = ENV_ROOM_MAP.get(e, '?')
        row = f"{e:>20}  {room:>2}"
        for u in all_users:
            cnt = env_user.get((e, u), 0)
            row += f"{'--':>8}" if cnt == 0 else f"{cnt:>8}"
        print(row)

    # ── 6. Per-environment summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("6. PER-ENVIRONMENT SUMMARY  (for LOEO fold design)")
    print("=" * 60)
    print(f"{'Env':>20}  Rm  {'Files':>7}  {'Gestures':>10}  {'Users':>7}")
    print("-" * 56)

    std6_envs = []
    for e in all_envs:
        e_records  = [f for f in files if f['environment'] == e]
        e_gestures = set(f['gesture_name'] for f in e_records)
        e_users    = set(f['user']          for f in e_records)
        room       = ENV_ROOM_MAP.get(e, '?')
        has_std6   = STANDARD_6_GESTURES.issubset(e_gestures)
        if has_std6:
            std6_envs.append(e)
        marker = " *" if has_std6 else ""
        print(f"{e:>20}  {room:>2}  {len(e_records):>7}  {len(e_gestures):>10}"
              f"  {len(e_users):>7}{marker}")

    print("  * = has all 6 standard benchmark gestures")

    # ── 7. Standard-6 coverage ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("7. STANDARD-6 GESTURE COVERAGE")
    print("=" * 60)
    print(f"Standard 6 : {sorted(STANDARD_6_GESTURES)}\n")

    print(f"Environments with ALL standard-6 gestures : {len(std6_envs)}")
    for e in std6_envs:
        room = ENV_ROOM_MAP.get(e, '?')
        print(f"  {e}  (Room {room})  {env_counts[e]} files")

    print(f"\n3-fold LOEO room breakdown:")
    room_labels = {1: 'Classroom', 2: 'Hall', 3: 'Office'}
    for room_id in [1, 2, 3]:
        room_envs  = [e for e in all_envs if ENV_ROOM_MAP.get(e) == room_id]
        room_files = sum(env_counts[e] for e in room_envs)
        room_std6  = [e for e in room_envs if e in std6_envs]
        print(f"  Room {room_id} ({room_labels[room_id]}): "
              f"{len(room_envs)} envs, {room_files} files, "
              f"{len(room_std6)} env(s) have all standard-6")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY FOR LOEO SPLITTER DESIGN")
    print("=" * 60)
    print(f"  Total files       : {total}")
    print(f"  Users             : {len(all_users)} -> {all_users}")
    print(f"  Environments      : {len(all_envs)}")
    print(f"  Gesture types     : {len(all_gestures)}")
    print(f"  Physical rooms    : 3")
    print(f"  Envs with std-6   : {std6_envs}")
    print()


if __name__ == '__main__':
    main()
