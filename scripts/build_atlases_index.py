"""
Build data/atlases_index.json — a small landing-page summary that the
browser fetches before any atlas is loaded.

For each atlas under data/atlases/, this reads dandi_regions.json once and
writes a per-atlas record with:
  - key           : atlas directory name (e.g. "allen_ccf")
  - name          : display name shown on the card
  - species       : "Mouse" / "Macaque"
  - preview       : relative path to the preview PNG
  - dandiset_count
  - file_count
  - regions_with_data

The aggregate counts are read from the root entry of dandi_regions.json,
which already carries total_dandiset_count / total_file_count. The
"regions_with_data" count is computed by filtering for regions where
file_count > 0 (i.e. the region itself is recorded in some asset, not
just an ancestor of one).

Run after the per-atlas update scripts (update_data.py for Allen CCF,
update_macaque_data.py for the macaque atlases). Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATLASES_DIR = REPO_ROOT / "data" / "atlases"
INDEX_PATH = REPO_ROOT / "data" / "atlases_index.json"

# Display metadata not derivable from the per-atlas JSON files. Order in
# this list defines the order of cards on the landing page.
ATLAS_DISPLAY = [
    {"key": "allen_ccf", "name": "Allen CCF v3", "species": "Mouse"},
    {"key": "d99", "name": "D99 v2.0", "species": "Macaque"},
    {"key": "nmt", "name": "NMT v2.0 sym", "species": "Macaque"},
    {"key": "mebrains", "name": "MEBRAINS", "species": "Macaque"},
    {"key": "whs_sd", "name": "WHS-SD v4", "species": "Rat"},
]


def summarize_atlas(atlas_dir: Path) -> dict:
    regions_path = atlas_dir / "dandi_regions.json"
    regions = json.loads(regions_path.read_text())

    # The root entry has the largest total_file_count. We can't assume a
    # specific structure_id (Allen uses 997, macaque atlases use 9999), so
    # pick the entry with the highest total_file_count.
    root_entry = max(regions.values(), key=lambda r: r.get("total_file_count", 0))

    regions_with_data = sum(1 for r in regions.values() if r.get("file_count", 0) > 0)

    return {
        "dandiset_count": root_entry.get("total_dandiset_count", 0),
        "file_count": root_entry.get("total_file_count", 0),
        "regions_with_data": regions_with_data,
    }


def main() -> None:
    atlases = []
    for entry in ATLAS_DISPLAY:
        atlas_dir = ATLASES_DIR / entry["key"]
        if not (atlas_dir / "dandi_regions.json").exists():
            print(f"  skipping {entry['key']}: dandi_regions.json missing")
            continue

        stats = summarize_atlas(atlas_dir)
        preview_rel = f"data/atlases/{entry['key']}/atlas_card_{entry['key']}.png"
        if not (REPO_ROOT / preview_rel).exists():
            print(f"  warning: {preview_rel} missing, card will show no image")

        record = {**entry, "preview": preview_rel, **stats}
        atlases.append(record)
        print(
            f"  {entry['key']:11s} dandisets={stats['dandiset_count']:>5}  "
            f"files={stats['file_count']:>6}  regions_with_data={stats['regions_with_data']:>4}"
        )

    INDEX_PATH.write_text(json.dumps({"atlases": atlases}, indent=2) + "\n")
    print(f"\nWrote {INDEX_PATH.relative_to(REPO_ROOT)} ({len(atlases)} atlases)")


if __name__ == "__main__":
    main()
