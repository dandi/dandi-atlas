"""Extract electrode x,y,z coordinates from NWB files on DANDI.

Reads dandiset_assets.json to get asset list, opens each NWB via HTTP streaming
(remfile + h5py), and extracts electrode coordinates from the electrodes table.

Outputs data/dandiset_electrodes.json grouped by dandiset and asset_id.

Usage:
    python extract_electrodes.py [--workers N] [--no-cache]
"""

import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import remfile
from tqdm import tqdm

DANDI_API = "https://api.dandiarchive.org/api"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
ASSETS_FILE = DATA_DIR / "dandiset_assets.json"
OUTPUT_FILE = DATA_DIR / "dandiset_electrodes.json"
ELECTRODE_MANIFEST_FILE = DATA_DIR / "dandisets_with_electrodes.json"
CACHE_FILE = SCRIPT_DIR / "electrode_cache.jsonl"

# Allen CCF bounds (micrometers)
ALLEN_X_MAX = 13200
ALLEN_Y_MAX = 8000
ALLEN_Z_MAX = 11400


def get_download_url(dandiset_id, asset_id, version="draft"):
    return (
        f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}"
        f"/assets/{asset_id}/download/"
    )


def extract_electrode_coords(url):
    """Open an NWB file via HTTP streaming and return electrode x,y,z coords.

    Returns list of [x, y, z] or None if no electrode coordinates found.
    """
    rf = remfile.File(url)
    with h5py.File(rf, "r") as f:
        if "general/extracellular_ephys/electrodes" not in f:
            return None
        electrodes = f["general/extracellular_ephys/electrodes"]
        if not all(col in electrodes for col in ("x", "y", "z")):
            return None

        x = electrodes["x"][()]
        y = electrodes["y"][()]
        z = electrodes["z"][()]

    coords = []
    for xi, yi, zi in zip(x, y, z):
        xi, yi, zi = float(xi), float(yi), float(zi)
        # Check for NaN
        if xi != xi or yi != yi or zi != zi:
            continue
        # Log warning for out-of-bounds but still include
        if not (0 <= xi <= ALLEN_X_MAX and 0 <= yi <= ALLEN_Y_MAX and 0 <= zi <= ALLEN_Z_MAX):
            pass  # Out of bounds, included anyway
        coords.append([round(xi, 1), round(yi, 1), round(zi, 1)])

    return coords if coords else None


def load_cache():
    """Load JSONL cache; returns dict of (dandiset_id, asset_id) -> entry."""
    cache = {}
    if not CACHE_FILE.exists():
        return cache
    with open(CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[(entry["dandiset_id"], entry["asset_id"])] = entry
    return cache


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache")
    args = parser.parse_args()

    with open(ASSETS_FILE) as f:
        dandiset_assets = json.load(f)

    cache = {} if args.no_cache else load_cache()
    if cache:
        print(f"Loaded cache with {len(cache)} entries")

    cache_lock = threading.Lock()

    # Collect work items: (dandiset_id, asset)
    work_items = []
    for dandiset_id, assets in dandiset_assets.items():
        for asset in assets:
            work_items.append((dandiset_id, asset))

    print(f"{len(work_items)} assets to process ({args.workers} workers)")

    # Results: dandiset_id -> asset_id -> coords
    results = defaultdict(dict)
    errors = 0

    def process(dandiset_id, asset):
        nonlocal errors
        asset_id = asset["asset_id"]
        path = asset["path"]
        cache_key = (dandiset_id, asset_id)

        with cache_lock:
            if cache_key in cache:
                entry = cache[cache_key]
                if entry.get("coords"):
                    results[dandiset_id][asset_id] = entry["coords"]
                return

        try:
            url = get_download_url(dandiset_id, asset_id)
            coords = extract_electrode_coords(url)
        except Exception as exc:
            tqdm.write(f"ERROR {dandiset_id}/{path}: {exc}")
            coords = None
            with cache_lock:
                errors += 1

        entry = {
            "dandiset_id": dandiset_id,
            "asset_id": asset_id,
            "path": path,
            "coords": coords,
        }

        with cache_lock:
            # Write to cache
            with open(CACHE_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            cache[cache_key] = entry
            if coords:
                results[dandiset_id][asset_id] = coords

    with tqdm(total=len(work_items), desc="Assets", unit="asset") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, did, asset): (did, asset)
                for did, asset in work_items
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    did, asset = futures[future]
                    tqdm.write(f"Worker error {did}/{asset['path']}: {exc}")
                pbar.update(1)

    # Build output: only dandisets with electrode data
    output = {}
    for dandiset_id in sorted(results):
        asset_coords = results[dandiset_id]
        if asset_coords:
            output[dandiset_id] = dict(sorted(asset_coords.items()))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    # Write manifest of dandiset IDs with electrode data (for frontend)
    with open(ELECTRODE_MANIFEST_FILE, "w") as f:
        json.dump(sorted(output.keys()), f)

    total_assets = sum(len(v) for v in output.values())
    total_electrodes = sum(
        len(coords) for asset_coords in output.values() for coords in asset_coords.values()
    )
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Wrote {ELECTRODE_MANIFEST_FILE} ({len(output)} dandisets)")
    print(f"  {len(output)} dandisets, {total_assets} assets with electrodes, {total_electrodes} electrodes")
    print(f"  {errors} errors")


if __name__ == "__main__":
    main()
