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
  - dandisets      : sorted IDs of the dandisets counted by dandiset_count

It also writes data/dandiset_titles.json, a flat {dandiset_id: title} map
covering every dandiset any atlas lists, so the viewer can show titles
without one DANDI API request per dandiset on every atlas load. Titles come
from the DANDI API here; a dandiset whose request fails keeps the title from
the previous file, and --skip-titles leaves the file untouched for offline
runs.

The aggregate counts are read from the root entry of dandi_regions.json,
which already carries total_dandiset_count / total_file_count /
total_dandisets. The "regions_with_data" count is computed by filtering
for regions where file_count > 0 (i.e. the region itself is recorded in
some asset, not just an ancestor of one).

The "dandisets" list lets other sites (e.g. the DANDI Archive landing
page) decide whether a given dandiset has anything to show here without
downloading the much larger per-atlas data files.

Run after the per-atlas update scripts (update_data.py for Allen CCF,
update_macaque_data.py for the macaque atlases). Idempotent.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATLASES_DIR = REPO_ROOT / "data" / "atlases"
INDEX_PATH = REPO_ROOT / "data" / "atlases_index.json"
LAST_UPDATED_PATH = REPO_ROOT / "data" / "last_updated.json"
TITLES_PATH = REPO_ROOT / "data" / "dandiset_titles.json"
DANDI_API = "https://api.dandiarchive.org/api"

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
        "dandisets": sorted(root_entry.get("total_dandisets", [])),
    }


def stamp_last_updated_date() -> None:
    """Add a plain YYYY-MM-DD `date` field to last_updated.json.

    The three per-atlas update scripts each stamp the top-level `timestamp`,
    in two different formats ("...Z" from update_data.py, isoformat with
    microseconds from the macaque and rat scripts), and whichever runs last
    wins. Neither is usable in a README badge, because shields.io prints the
    raw string it reads. This derives one short value from the newest per-atlas
    run, so the badge has something stable to point at and does not depend on
    which script happened to finish last.
    """
    if not LAST_UPDATED_PATH.exists():
        return
    try:
        record = json.loads(LAST_UPDATED_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        print("  warning: last_updated.json unreadable, leaving date unset")
        return

    stamps = [v.get("timestamp") for v in (record.get("per_atlas") or {}).values()]
    stamps.append(record.get("timestamp"))

    newest = None
    for raw in filter(None, stamps):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
        if newest is None or parsed > newest:
            newest = parsed

    if newest is None:
        print("  warning: no parseable timestamps, leaving date unset")
        return

    record["date"] = newest.date().isoformat()
    LAST_UPDATED_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  last_updated.json date = {record['date']}")


def fetch_dandiset_titles(dandiset_ids: set[str], previous: dict) -> dict:
    """Return {dandiset_id: title} for the given IDs.

    One request per dandiset. The draft name is preferred, matching what the
    viewer used to fetch at runtime, with the latest published name as the
    fallback. A request that fails, or a dandiset with no name, keeps whatever
    the previous file had so a flaky night does not blank titles out.
    """
    import requests  # local import so the index can still build without it

    titles = {did: previous[did] for did in dandiset_ids if did in previous}
    failed = 0
    for did in sorted(dandiset_ids):
        url = f"{DANDI_API}/dandisets/{did}/"
        name = None
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
            except requests.RequestException:
                resp = None
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                name = (
                    (data.get("draft_version") or {}).get("name")
                    or (data.get("most_recent_published_version") or {}).get("name")
                )
                break
            if resp is not None and resp.status_code not in (429,) and resp.status_code < 500:
                break
            time.sleep(2 * (attempt + 1))
        if name:
            titles[did] = name
        else:
            failed += 1
    if failed:
        print(f"  warning: {failed} title request(s) failed; previous titles kept where available")
    return titles


def write_dandiset_titles(atlases: list) -> None:
    ids = set()
    for atlas in atlases:
        ids.update(atlas.get("dandisets", []))
    previous = {}
    if TITLES_PATH.exists():
        try:
            previous = json.loads(TITLES_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
    titles = fetch_dandiset_titles(ids, previous)
    TITLES_PATH.write_text(
        json.dumps(dict(sorted(titles.items())), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Wrote {TITLES_PATH.relative_to(REPO_ROOT)} ({len(titles)} of {len(ids)} dandisets titled)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-titles", action="store_true",
        help="Do not refresh data/dandiset_titles.json (no network needed)",
    )
    args = parser.parse_args()

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

    if not args.skip_titles:
        write_dandiset_titles(atlases)

    stamp_last_updated_date()


if __name__ == "__main__":
    main()
