"""Rescan DANDI to get one asset per subject for all subjects.

Uses the existing label_cache.jsonl as a starting point, fetches all NWB assets
for each dandiset, picks one per subject, and processes any missing ones.
Appends new results to label_cache.jsonl, then regenerates dandiset_assets.json.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add analyze-locations to path so we can import from it
ANALYZE_DIR = Path(os.path.expanduser("~/dev/sandbox/analyze-locations"))
sys.path.insert(0, str(ANALYZE_DIR))

from label_anatomy import (
    load_or_fetch_allen_mapping,
    build_lookup_dicts,
    process_asset,
    get_nwb_assets_paged,
    LABEL_CACHE_FILE,
)

LABEL_CACHE = ANALYZE_DIR / LABEL_CACHE_FILE
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "atlases" / "allen_ccf" / "dandiset_assets.json"
FILTER_IDS = {997, 8}


def extract_subject(path):
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else path.split("_")[0]


def load_label_cache():
    cache = {}
    if LABEL_CACHE.exists():
        with open(LABEL_CACHE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (entry["dandiset_id"], entry["asset_id"])
                cache[key] = entry
    return cache


def append_label_cache(entry):
    with open(LABEL_CACHE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    # Load existing cache
    label_cache = load_label_cache()
    print(f"Loaded {len(label_cache)} entries from label_cache.jsonl")

    # Get the set of dandisets we already know about
    dandiset_ids = sorted(set(k[0] for k in label_cache.keys()))
    print(f"Found {len(dandiset_ids)} dandisets to process")

    # Load Allen CCF mapping
    os.chdir(str(ANALYZE_DIR))
    structures = load_or_fetch_allen_mapping()
    lookups = build_lookup_dicts(structures)

    total_new = 0

    for ds_id in dandiset_ids:
        print(f"\n--- Dandiset {ds_id} ---")

        # Get ALL NWB assets from DANDI API
        all_assets = list(get_nwb_assets_paged(ds_id, max_assets=None))
        print(f"  {len(all_assets)} total NWB assets on DANDI")

        # Group by subject, pick first asset per subject (by path)
        subject_assets = defaultdict(list)
        for asset in all_assets:
            subj = extract_subject(asset["path"])
            subject_assets[subj].append(asset)

        print(f"  {len(subject_assets)} subjects")

        # For each subject, pick the first asset (by path) that we need
        for subj in sorted(subject_assets):
            assets = sorted(subject_assets[subj], key=lambda a: a["path"])

            # Check if we already have ANY asset for this subject in cache
            have_cached = False
            for asset in assets:
                key = (ds_id, asset["asset_id"])
                if key in label_cache:
                    have_cached = True
                    break

            if have_cached:
                continue

            # Process the first asset for this subject
            asset = assets[0]
            print(f"  Processing {subj}: {asset['path']}")
            try:
                result = process_asset(ds_id, asset, lookups, apply=False)
                append_label_cache(result)
                label_cache[(ds_id, asset["asset_id"])] = result
                total_new += 1
                status = result["status"]
                n_regions = len(result.get("matched_locations", {}))
                print(f"    -> {status}, {n_regions} regions")
            except Exception as exc:
                print(f"    -> ERROR: {exc}")

    print(f"\n{'=' * 50}")
    print(f"Added {total_new} new entries to label_cache.jsonl")

    # Now regenerate dandiset_assets.json
    print(f"\nRegenerating {OUTPUT} ...")

    dandisets = defaultdict(lambda: defaultdict(list))
    for (ds_id, _), entry in label_cache.items():
        path = entry["path"]
        subj = extract_subject(path)
        dandisets[ds_id][subj].append(entry)

    result = {}
    for did in sorted(dandisets):
        subjects = dandisets[did]
        assets = []
        for subj in sorted(subjects):
            # Pick first entry by path
            entries = sorted(subjects[subj], key=lambda e: e["path"])
            e = entries[0]
            matched = e.get("matched_locations", {})
            regions = []
            seen = set()
            for loc_key, matches in matched.items():
                for m in matches:
                    if m["id"] not in FILTER_IDS and m["id"] not in seen:
                        seen.add(m["id"])
                        regions.append({
                            "id": m["id"],
                            "acronym": m["acronym"],
                            "name": m["name"],
                        })
            assets.append({
                "path": e["path"],
                "asset_id": e["asset_id"],
                "regions": regions,
            })
        result[did] = assets

    with open(OUTPUT, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    total_assets = sum(len(v) for v in result.values())
    total_subjects = total_assets  # one per subject
    print(f"Wrote {OUTPUT}")
    print(f"  {len(result)} dandisets, {total_subjects} subjects")


if __name__ == "__main__":
    main()
