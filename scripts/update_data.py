#!/usr/bin/env python3
"""Update DANDI data files for the brain atlas viewer.

Fetches NWB files from DANDI, matches brain region locations against the
Allen CCF, extracts electrode coordinates, and rebuilds all data files.

Usage:
    python scripts/update_data.py                        # incremental (default)
    python scripts/update_data.py --mode full            # full rebuild
    python scripts/update_data.py --dandiset 000017      # specific dandiset(s)
    python scripts/update_data.py --workers 4            # parallel workers
"""

import argparse
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from dandi_helpers import (
    FILTER_IDS,
    build_dandi_regions,
    build_lookup_dicts,
    build_parent_map,
    check_species_mouse,
    compute_mesh_set,
    download_meshes,
    extract_desc,
    extract_electrode_coords,
    extract_locations,
    extract_session,
    extract_subject,
    fetch_allen_structure_graph,
    flatten_structure_graph,
    get_download_url,
    get_nwb_assets_paged,
    iter_all_dandisets,
    iter_dandisets_modified_since,
    match_location,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MESHES_DIR = DATA_DIR / "meshes"
SCRIPT_DIR = Path(__file__).resolve().parent

LABEL_CACHE_FILE = SCRIPT_DIR / "label_cache.jsonl"
ELECTRODE_CACHE_FILE = SCRIPT_DIR / "electrode_cache.jsonl"
LAST_UPDATED_FILE = DATA_DIR / "last_updated.json"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def load_label_cache():
    """Load JSONL label cache; returns dict of (dandiset_id, asset_id) -> entry."""
    cache = {}
    if not LABEL_CACHE_FILE.exists():
        return cache
    with open(LABEL_CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["dandiset_id"], entry["asset_id"])
            cache[key] = entry
    return cache


def load_electrode_cache():
    """Load JSONL electrode cache; returns dict of (dandiset_id, asset_id) -> entry."""
    cache = {}
    if not ELECTRODE_CACHE_FILE.exists():
        return cache
    with open(ELECTRODE_CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["dandiset_id"], entry["asset_id"])
            cache[key] = entry
    return cache


def append_label_cache(entry):
    with open(LABEL_CACHE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def append_electrode_cache(entry):
    with open(ELECTRODE_CACHE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def invalidate_cache_for_dandisets(cache, dandiset_ids):
    """Remove all entries for the given dandiset IDs from cache dict (in-place)."""
    to_remove = [k for k in cache if k[0] in dandiset_ids]
    for k in to_remove:
        del cache[k]
    return len(to_remove)


# ---------------------------------------------------------------------------
# Asset processing
# ---------------------------------------------------------------------------


def process_asset_locations(dandiset_id, asset, lookups):
    """Process a single NWB asset: extract locations and match against Allen CCF.

    Returns a result dict for the label cache.
    """
    asset_id = asset["asset_id"]
    path = asset["path"]

    result = {
        "dandiset_id": dandiset_id,
        "asset_id": asset_id,
        "path": path,
        "status": "unknown",
        "matched_locations": {},
        "unmatched_locations": [],
        "error": None,
    }

    try:
        download_url = get_download_url(dandiset_id, asset_id)
        img_locs, elec_locs, ice_locs = extract_locations(download_url)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    all_locations = set(img_locs) | set(elec_locs) | set(ice_locs)
    if not all_locations:
        result["status"] = "no_locations"
        return result

    for loc in sorted(all_locations):
        structures = match_location(loc, lookups)
        if structures:
            result["matched_locations"][loc] = [
                {"id": s["id"], "acronym": s["acronym"], "name": s["name"]}
                for s in structures
            ]
        else:
            from dandi_helpers import TRIVIAL_LOCATIONS
            if loc.strip().lower() not in TRIVIAL_LOCATIONS:
                result["unmatched_locations"].append(loc)

    if result["matched_locations"]:
        result["status"] = "matched"
    else:
        result["status"] = "no_match"

    return result


def process_asset_electrodes(dandiset_id, asset):
    """Process a single NWB asset: extract electrode coordinates.

    Returns a result dict for the electrode cache.
    """
    asset_id = asset["asset_id"]
    path = asset["path"]

    entry = {
        "dandiset_id": dandiset_id,
        "asset_id": asset_id,
        "path": path,
        "coords": None,
    }

    try:
        url = get_download_url(dandiset_id, asset_id)
        coords = extract_electrode_coords(url)
        entry["coords"] = coords
    except Exception as exc:
        tqdm.write(f"  Electrode error {dandiset_id}/{path}: {exc}")

    return entry


# ---------------------------------------------------------------------------
# Convert label cache to dandiset_assets format
# ---------------------------------------------------------------------------


def build_dandiset_assets(label_cache):
    """Convert label cache entries to dandiset_assets.json format.

    Keeps all assets per subject per dandiset, with session and description
    metadata extracted from BIDS filenames.
    """
    dandisets = defaultdict(lambda: defaultdict(list))

    for (ds_id, _), entry in label_cache.items():
        path = entry["path"]
        asset_id = entry["asset_id"]
        matched = entry.get("matched_locations", {})

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

        subject = extract_subject(path)
        asset_entry = {
            "path": path,
            "asset_id": asset_id,
            "regions": regions,
        }

        session = extract_session(path)
        if session:
            asset_entry["session"] = session

        desc = extract_desc(path)
        if desc:
            asset_entry["desc"] = desc

        dandisets[ds_id][subject].append(asset_entry)

    # Keep ALL assets per subject, sorted by path
    result = {}
    for did in sorted(dandisets):
        subjects = dandisets[did]
        assets = []
        for subj in sorted(subjects):
            sorted_assets = sorted(subjects[subj], key=lambda a: a["path"])
            assets.extend(sorted_assets)
        result[did] = assets

    return result


def build_dandiset_electrodes(electrode_cache):
    """Convert electrode cache entries to dandiset_electrodes.json format."""
    results = defaultdict(dict)

    for (ds_id, _), entry in electrode_cache.items():
        if entry.get("coords"):
            asset_id = entry["asset_id"]
            results[ds_id][asset_id] = entry["coords"]

    output = {}
    for dandiset_id in sorted(results):
        asset_coords = results[dandiset_id]
        if asset_coords:
            output[dandiset_id] = dict(sorted(asset_coords.items()))

    return output


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["incremental", "full"], default="incremental",
        help="Update mode (default: incremental)",
    )
    parser.add_argument(
        "--dandiset", nargs="+", metavar="ID",
        help="Process specific dandiset(s) only",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for NWB streaming (default: 4)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ── Step 1: Fetch Allen CCF structure graph ──────────────────────────
    print("Step 1: Fetching Allen CCF structure graph...")
    graph_msg = fetch_allen_structure_graph()
    structures = flatten_structure_graph(graph_msg)
    print(f"  {len(structures)} structures")

    id_to_structure = {s["id"]: s for s in structures}
    parent_map = build_parent_map(structures)
    lookups = build_lookup_dicts(structures)

    # Save structure graph
    with open(DATA_DIR / "structure_graph.json", "w") as f:
        json.dump(graph_msg, f)
    print("  Saved structure_graph.json")

    # ── Step 2: Determine target dandisets ────────────────────────────────
    is_full = args.mode == "full"

    if args.dandiset:
        # Specific dandisets requested
        target_ids = set(args.dandiset)
        print(f"\nStep 2: Targeting {len(target_ids)} specified dandiset(s)")
    elif is_full:
        print("\nStep 2: Full rebuild — fetching all dandisets...")
        target_ids = set()
        for ds in iter_all_dandisets():
            target_ids.add(ds["identifier"])
        print(f"  Found {len(target_ids)} dandisets")
    else:
        # Incremental: find recently modified
        print("\nStep 2: Incremental — checking for modified dandisets...")
        if LAST_UPDATED_FILE.exists():
            with open(LAST_UPDATED_FILE) as f:
                last_data = json.load(f)
            since = last_data.get("timestamp", "")
            print(f"  Last updated: {since}")
        else:
            print("  No last_updated.json found — falling back to full mode")
            since = ""
            is_full = True

        if since:
            target_ids = set()
            for ds in iter_dandisets_modified_since(since):
                target_ids.add(ds["identifier"])
            print(f"  {len(target_ids)} dandisets modified since last update")
            if not target_ids:
                print("  No changes detected. Writing timestamp and exiting.")
                _write_last_updated("incremental", 0, 0, 0)
                print("Done!")
                return
        else:
            target_ids = set()
            for ds in iter_all_dandisets():
                target_ids.add(ds["identifier"])
            print(f"  Found {len(target_ids)} dandisets for full rebuild")

    # ── Step 3: Filter to mouse-only ──────────────────────────────────────
    print(f"\nStep 3: Filtering to mouse-only dandisets...")
    mouse_ids = set()
    skipped = 0
    for ds_id in tqdm(sorted(target_ids), desc="Species check", unit="ds"):
        if check_species_mouse(ds_id):
            mouse_ids.add(ds_id)
        else:
            skipped += 1

    print(f"  {len(mouse_ids)} mouse dandisets, {skipped} skipped (non-mouse)")

    if not mouse_ids:
        print("  No mouse dandisets to process.")
        _write_last_updated(args.mode, len(target_ids), 0, 0)
        print("Done!")
        return

    # ── Step 4: Load caches ───────────────────────────────────────────────
    print(f"\nStep 4: Loading caches...")
    if is_full:
        label_cache = {}
        electrode_cache = {}
        print("  Full mode — starting with empty caches")
    else:
        label_cache = load_label_cache()
        electrode_cache = load_electrode_cache()
        print(f"  Loaded {len(label_cache)} label entries, {len(electrode_cache)} electrode entries")

        # Invalidate cache entries for modified dandisets
        n_label = invalidate_cache_for_dandisets(label_cache, mouse_ids)
        n_elec = invalidate_cache_for_dandisets(electrode_cache, mouse_ids)
        if n_label or n_elec:
            print(f"  Invalidated {n_label} label + {n_elec} electrode cache entries for modified dandisets")

    # ── Step 5: Process each dandiset ─────────────────────────────────────
    print(f"\nStep 5: Processing {len(mouse_ids)} dandisets ({args.workers} workers)...")
    cache_lock = threading.Lock()
    total_assets_processed = 0
    total_errors = 0

    for ds_id in sorted(mouse_ids):
        print(f"\n--- Dandiset {ds_id} ---")

        # List all NWB assets, group by subject, pick first per subject
        all_assets = list(get_nwb_assets_paged(ds_id, max_assets=None))
        print(f"  {len(all_assets)} NWB assets")

        if not all_assets:
            continue

        subject_assets = defaultdict(list)
        for asset in all_assets:
            # TEMPORARY: IBL dandiset 000409 contains legacy file variants that
            # duplicate data already present in the canonical desc-raw / desc-processed
            # files. Filter them out to avoid displaying files with wrong localization
            if ds_id == "000409":
                fname = asset["path"].split("/")[-1]
                if (fname.endswith("-processed-only_behavior.nwb")
                        or "_behavior+ecephys+image.nwb" in fname
                        or "_ecephys+image.nwb" in fname
                        or "-raw-only_ecephys+image.nwb" in fname
                        or (("_behavior+ecephys.nwb" in fname)
                            and "_desc-processed_" not in fname)):
                    continue
            subj = extract_subject(asset["path"])
            subject_assets[subj].append(asset)

        # Process all assets per subject (sorted by path)
        work_items = []
        for subj in sorted(subject_assets):
            assets = sorted(subject_assets[subj], key=lambda a: a["path"])
            work_items.extend(assets)

        print(f"  {len(work_items)} assets, processing...")

        def process_one(asset):
            """Process a single asset. Returns the label_result status."""
            nonlocal total_assets_processed, total_errors
            asset_id = asset["asset_id"]
            path = asset["path"]
            cache_key = (ds_id, asset_id)

            # Check label cache
            with cache_lock:
                if cache_key in label_cache:
                    # Already cached, skip
                    return label_cache[cache_key]["status"]

            # Process locations
            label_result = process_asset_locations(ds_id, asset, lookups)

            # Process electrodes
            electrode_result = process_asset_electrodes(ds_id, asset)

            with cache_lock:
                # Store in caches
                append_label_cache(label_result)
                label_cache[cache_key] = label_result

                append_electrode_cache(electrode_result)
                electrode_cache[cache_key] = electrode_result

                total_assets_processed += 1
                if label_result["status"] == "error":
                    total_errors += 1
                    tqdm.write(f"  ERROR {ds_id}/{path}: {label_result.get('error', 'unknown')}")
                else:
                    n_regions = len(label_result.get("matched_locations", {}))
                    has_coords = bool(electrode_result.get("coords"))
                    tqdm.write(f"  {path}: {label_result['status']}, {n_regions} regions" +
                               (", has electrodes" if has_coords else ""))

            return label_result["status"]

        # Early stopping: probe first EARLY_STOP_PROBE assets sequentially.
        # If none have matched locations, skip the rest of this dandiset.
        EARLY_STOP_PROBE = 5
        probe_items = work_items[:EARLY_STOP_PROBE]
        remaining_items = work_items[EARLY_STOP_PROBE:]
        found_match = False

        for asset in probe_items:
            status = process_one(asset)
            if status == "matched":
                found_match = True

        if not found_match and remaining_items:
            print(f"  Skipping remaining {len(remaining_items)} assets "
                  f"(no matches in first {len(probe_items)})")
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, asset): asset for asset in remaining_items}
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    asset = futures[future]
                    tqdm.write(f"  Worker error {ds_id}/{asset['path']}: {exc}")

    # ── Step 6: Build data files ──────────────────────────────────────────
    print(f"\n\nStep 6: Building data files...")

    electrodes_dir = DATA_DIR / "electrodes"

    def load_existing_electrodes():
        """Load existing per-dandiset electrode files into a single dict."""
        result = {}
        if electrodes_dir.exists():
            for fp in electrodes_dir.glob("*.json"):
                did = fp.stem
                with open(fp) as f:
                    result[did] = json.load(f)
        return result

    if not is_full and not args.dandiset:
        # Incremental: merge with existing data
        existing_assets_path = DATA_DIR / "dandiset_assets.json"

        if existing_assets_path.exists():
            with open(existing_assets_path) as f:
                existing_assets = json.load(f)
            full_label_cache = load_label_cache()
            dandiset_assets = build_dandiset_assets(full_label_cache)
            for did in existing_assets:
                if did not in dandiset_assets and did not in mouse_ids:
                    dandiset_assets[did] = existing_assets[did]
        else:
            full_label_cache = load_label_cache()
            dandiset_assets = build_dandiset_assets(full_label_cache)

        existing_electrodes = load_existing_electrodes()
        full_electrode_cache = load_electrode_cache()
        dandiset_electrodes = build_dandiset_electrodes(full_electrode_cache)
        for did in existing_electrodes:
            if did not in dandiset_electrodes and did not in mouse_ids:
                dandiset_electrodes[did] = existing_electrodes[did]
    else:
        # Full or specific: build entirely from cache
        if is_full:
            dandiset_assets = build_dandiset_assets(label_cache)
            dandiset_electrodes = build_dandiset_electrodes(electrode_cache)
        else:
            # Specific dandisets: merge with existing
            existing_assets_path = DATA_DIR / "dandiset_assets.json"

            dandiset_assets = {}
            if existing_assets_path.exists():
                with open(existing_assets_path) as f:
                    dandiset_assets = json.load(f)

            dandiset_electrodes = load_existing_electrodes()

            # Replace entries for specified dandisets
            new_assets = build_dandiset_assets(label_cache)
            new_electrodes = build_dandiset_electrodes(electrode_cache)
            dandiset_assets.update(new_assets)
            dandiset_electrodes.update(new_electrodes)

    # Write dandiset_assets.json
    with open(DATA_DIR / "dandiset_assets.json", "w") as f:
        json.dump(dandiset_assets, f, separators=(",", ":"))
    total_assets = sum(len(v) for v in dandiset_assets.values())
    print(f"  dandiset_assets.json: {len(dandiset_assets)} dandisets, {total_assets} assets")

    # Write per-dandiset electrode files
    electrodes_dir.mkdir(parents=True, exist_ok=True)
    total_electrode_assets = 0
    for dandiset_id, asset_coords in dandiset_electrodes.items():
        with open(electrodes_dir / f"{dandiset_id}.json", "w") as f:
            json.dump(asset_coords, f, separators=(",", ":"))
        total_electrode_assets += len(asset_coords)
    print(f"  electrodes/: {len(dandiset_electrodes)} files, {total_electrode_assets} assets")

    # ── Step 7: Rebuild dandi_regions.json ─────────────────────────────────
    print("\nStep 7: Building dandi_regions.json...")
    dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)
    with open(DATA_DIR / "dandi_regions.json", "w") as f:
        json.dump(dandi_regions, f, indent=2)
    print(f"  {len(dandi_regions)} structures with DANDI data")

    # ── Step 8: Download meshes if needed ──────────────────────────────────
    print("\nStep 8: Checking meshes...")
    data_ids, ancestor_ids, all_mesh_ids = compute_mesh_set(dandi_regions, parent_map)

    # Check which meshes are missing
    missing = [sid for sid in all_mesh_ids if not (MESHES_DIR / f"{sid}.obj").exists()]
    if missing:
        print(f"  Downloading {len(missing)} new meshes...")
        failed_ids = download_meshes(set(missing), MESHES_DIR)
    else:
        print("  All meshes present")
        failed_ids = []

    # Check for any missing meshes overall
    all_failed = []
    for sid in sorted(all_mesh_ids):
        if not (MESHES_DIR / f"{sid}.obj").exists():
            all_failed.append(sid)

    # ── Step 9: Rebuild mesh_manifest.json ─────────────────────────────────
    print("\nStep 9: Building mesh_manifest.json...")
    mesh_manifest = {
        "data_structures": sorted(data_ids),
        "ancestor_structures": sorted(ancestor_ids - data_ids),
        "no_mesh": sorted(all_failed),
        "root_id": 997,
    }
    with open(DATA_DIR / "mesh_manifest.json", "w") as f:
        json.dump(mesh_manifest, f, indent=2)
    print(f"  data: {len(data_ids)}, ancestors: {len(ancestor_ids - data_ids)}, no_mesh: {len(all_failed)}")

    # ── Step 10: Write last_updated.json ───────────────────────────────────
    _write_last_updated(args.mode, len(target_ids), len(mouse_ids), total_assets_processed)

    elapsed = time.time() - start_time
    print(f"\nDone! ({elapsed:.0f}s)")
    print(f"  Dandisets checked: {len(target_ids)}")
    print(f"  Dandisets updated: {len(mouse_ids)}")
    print(f"  Assets processed: {total_assets_processed}")
    print(f"  Errors: {total_errors}")


def _write_last_updated(mode, dandisets_checked, dandisets_updated, assets_processed):
    """Write the last_updated.json file."""
    data = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "dandisets_checked": dandisets_checked,
        "dandisets_updated": dandisets_updated,
        "assets_processed": assets_processed,
    }
    with open(LAST_UPDATED_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Updated {LAST_UPDATED_FILE}")


if __name__ == "__main__":
    main()
