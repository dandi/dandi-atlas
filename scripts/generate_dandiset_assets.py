"""Convert label_cache.jsonl to compact dandiset_assets.json for the viewer.

Keeps all assets per subject per dandiset, sorted by path, with session and
description metadata extracted from BIDS filenames.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

LABEL_CACHE = Path(os.environ.get(
    "LABEL_CACHE",
    os.path.expanduser("~/dev/sandbox/analyze-locations/label_cache.jsonl"),
))
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "dandiset_assets.json"

FILTER_IDS = {997, 8}  # root, grey (not useful to display)


def extract_subject(path):
    """Extract subject directory from asset path."""
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else path.split("_")[0]


def extract_session(path):
    """Extract session ID from a BIDS-style NWB filename."""
    match = re.search(r"_ses-([^_/]+)", path)
    if not match:
        return None
    session = match.group(1)
    session = re.sub(r"-processed-only$", "", session)
    return session


def extract_desc(path):
    """Extract description label from a BIDS-style NWB filename."""
    match = re.search(r"_desc-([^_/]+)", path)
    return match.group(1) if match else None


def main():
    # dandiset -> subject -> list of assets
    dandisets = defaultdict(lambda: defaultdict(list))

    with open(LABEL_CACHE) as f:
        for line in f:
            entry = json.loads(line)
            did = entry["dandiset_id"]
            asset_id = entry["asset_id"]
            path = entry["path"]
            matched = entry.get("matched_locations", {})

            regions = []
            for loc_key, matches in matched.items():
                for m in matches:
                    if m["id"] not in FILTER_IDS:
                        regions.append({
                            "id": m["id"],
                            "acronym": m["acronym"],
                            "name": m["name"],
                        })

            # Deduplicate by id
            seen = set()
            unique_regions = []
            for r in regions:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    unique_regions.append(r)

            subject = extract_subject(path)
            asset_entry = {
                "path": path,
                "asset_id": asset_id,
                "regions": unique_regions,
            }

            session = extract_session(path)
            if session:
                asset_entry["session"] = session

            desc = extract_desc(path)
            if desc:
                asset_entry["desc"] = desc

            dandisets[did][subject].append(asset_entry)

    # Keep ALL assets per subject, sorted by path
    result = {}
    for did in sorted(dandisets):
        subjects = dandisets[did]
        assets = []
        for subj in sorted(subjects):
            sorted_assets = sorted(subjects[subj], key=lambda a: a["path"])
            assets.extend(sorted_assets)
        result[did] = assets

    with open(OUTPUT, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    total_assets = sum(len(v) for v in result.values())
    total_subjects = sum(
        len(set(extract_subject(a["path"]) for a in assets))
        for assets in result.values()
    )
    print(f"Wrote {OUTPUT}")
    print(f"  {len(result)} dandisets, {total_subjects} subjects, {total_assets} assets")


if __name__ == "__main__":
    main()
