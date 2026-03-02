#!/usr/bin/env python3
"""Migrate dandiset_electrodes.json to per-dandiset files in data/electrodes/.

Reads the old monolithic data/dandiset_electrodes.json and writes one JSON file
per dandiset into data/electrodes/{dandiset_id}.json, then removes the old file.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLD_FILE = DATA_DIR / "dandiset_electrodes.json"
ELECTRODES_DIR = DATA_DIR / "electrodes"
ELECTRODE_MANIFEST_FILE = DATA_DIR / "dandisets_with_electrodes.json"


def main():
    if not OLD_FILE.exists():
        print(f"Nothing to migrate: {OLD_FILE} not found.")
        return

    with open(OLD_FILE) as f:
        data = json.load(f)

    ELECTRODES_DIR.mkdir(parents=True, exist_ok=True)

    for dandiset_id, asset_coords in sorted(data.items()):
        out = ELECTRODES_DIR / f"{dandiset_id}.json"
        with open(out, "w") as f:
            json.dump(asset_coords, f, separators=(",", ":"))
        size_kb = out.stat().st_size / 1024
        print(f"  {dandiset_id}.json: {len(asset_coords)} assets, {size_kb:.0f} KB")

    # Write manifest of dandiset IDs with electrode data (for frontend)
    with open(ELECTRODE_MANIFEST_FILE, "w") as f:
        json.dump(sorted(data.keys()), f)

    OLD_FILE.unlink()
    print(f"\nMigrated {len(data)} dandisets to {ELECTRODES_DIR}/")
    print(f"Wrote {ELECTRODE_MANIFEST_FILE}")
    print(f"Removed {OLD_FILE}")


if __name__ == "__main__":
    main()
