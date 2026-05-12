#!/usr/bin/env python3
"""Update macaque DANDI data files (D99, NMT, MEBRAINS).

Refreshes the DANDI-derived JSON files for a single macaque atlas without
needing the local atlas-source files (NIFTI parcellations, label tables, etc.)
that build_macaque_atlas.py requires. Works in CI runners that only have the
repo checked out.

The structure tree is reconstructed from the already-committed
data/atlases/{atlas}/structure_graph.json. Electrode coordinates and per-asset
brain_region_id are fetched from DANDI 001636 by streaming each NWB file via
HTTP (same path as build_macaque_atlas.py's fetch_dandi_data).

Files written (per atlas):
  - dandiset_assets.json
  - dandisets_with_electrodes.json
  - dandi_regions.json
  - electrodes/001636.json   (via fetch_dandi_data)
  - mesh_manifest.json       (only the data_structures and ancestor_structures
                              fields; all_meshes, no_mesh, root_id preserved)

Files NOT written:
  - structure_graph.json     (read-only here; comes from local atlas source via
                              build_macaque_atlas.py)
  - meshes/*.glb             (atlas-source-derived; built by build_macaque_atlas.py)

Usage:
    python scripts/update_macaque_data.py --atlas d99
    python scripts/update_macaque_data.py --atlas nmt --mode full
    python scripts/update_macaque_data.py --atlas mebrains
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# Reuse helpers from macaque_atlas_lib (atlas configs + DANDI fetch path)
# and dandi_helpers (parent map + ancestor walk).
from macaque_atlas_lib import (
    ATLAS_CONFIGS, DANDISET_ID, MACAQUE_LOCATION_ALIASES,
    fetch_dandi_data, fetch_macaque_implicit_data, _normalize_region_name,
)
from dandi_helpers import build_dandi_regions
from dandi_helpers import build_parent_map, get_ancestors


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAST_UPDATED_FILE = PROJECT_ROOT / "data" / "last_updated.json"


def reconstruct_lookups_from_graph(structure_graph):
    """Walk the saved structure_graph.json and rebuild the lookups that
    fetch_dandi_data needs (id_to_structure, abbrev_to_id, name_to_id).

    The saved tree carries one node per structure with id, acronym, name,
    parent_structure_id, and children. We rebuild:
      - id_to_structure: dict {id: structure_record}, where structure_record is
        the minimal {id, acronym, name, parent_structure_id} shape that
        downstream helpers (_map_regions_for_macaque, build_dandi_regions)
        consume.
      - abbrev_to_id: dict {abbreviation_or_id_str: id}. Includes acronym → id
        (used as a fallback when the NWB file does not also ship the full
        region name) and str(id) → id (defensive fallback for files that
        store the integer ID as a string).
      - name_to_id: dict {normalized_full_name: id}. The primary lookup for
        the macaque resolver, since full names are collision-free across
        CHARM and SARM.
    """
    id_to_structure = {}
    abbrev_to_id = {}
    name_to_id = {}

    def walk(node):
        sid = node["id"]
        # Preserve every scalar field downstream helpers may read
        # (color_hex_triplet, parent_structure_id, etc.). Only the recursive
        # `children` field is dropped — id_to_structure is meant to be flat.
        record = {k: v for k, v in node.items() if k != "children"}
        id_to_structure[sid] = record
        if "acronym" in node:
            abbrev_to_id[node["acronym"]] = sid
        abbrev_to_id[str(sid)] = sid
        norm_name = _normalize_region_name(node.get("name"))
        if norm_name is not None and norm_name not in name_to_id:
            name_to_id[norm_name] = sid
        for child in node.get("children", []):
            walk(child)

    if isinstance(structure_graph, list):
        for n in structure_graph:
            walk(n)
    else:
        walk(structure_graph)

    return id_to_structure, abbrev_to_id, name_to_id


def update_mesh_manifest(data_dir, data_structures, parent_map):
    """Update only the DANDI-derived fields of mesh_manifest.json.

    Preserves all_meshes, no_mesh, root_id (those come from the atlas-source
    build path). Updates data_structures (= regions any current asset
    references) and ancestor_structures (= union of ancestors of those regions).
    """
    manifest_path = data_dir / "mesh_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    data_ids = set(data_structures)
    ancestor_ids = set()
    for sid in data_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

    manifest["data_structures"] = sorted(data_ids)
    manifest["ancestor_structures"] = sorted(ancestor_ids - data_ids)
    # all_meshes, no_mesh, root_id preserved

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def write_last_updated(atlas_key, mode, asset_count):
    """Append/update last_updated.json with this atlas's refresh timestamp."""
    record = {}
    if LAST_UPDATED_FILE.exists():
        try:
            record = json.load(open(LAST_UPDATED_FILE))
        except Exception:
            record = {}

    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    record["mode"] = mode
    # Preserve previous per-atlas fields and add this run's
    per_atlas = record.get("per_atlas", {})
    per_atlas[atlas_key] = {
        "timestamp": record["timestamp"],
        "mode": mode,
        "asset_count": asset_count,
    }
    record["per_atlas"] = per_atlas

    LAST_UPDATED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_UPDATED_FILE, "w") as f:
        json.dump(record, f, indent=2)


def update_atlas(atlas_key, mode):
    if atlas_key not in ATLAS_CONFIGS:
        raise SystemExit(f"Unknown atlas: {atlas_key}")

    config = ATLAS_CONFIGS[atlas_key]
    data_dir = config["output_dir"]
    structure_graph_path = data_dir / "structure_graph.json"

    if not structure_graph_path.exists():
        raise SystemExit(
            f"Missing {structure_graph_path}. Run build_macaque_atlas.py first to "
            "produce the structure graph from atlas source files."
        )

    print(f"Updating {atlas_key} (mode={mode})")
    print(f"  Reading structure tree from {structure_graph_path.relative_to(PROJECT_ROOT)}")
    structure_graph = json.load(open(structure_graph_path))
    id_to_structure, abbrev_to_id, name_to_id = reconstruct_lookups_from_graph(
        structure_graph
    )
    parent_map = build_parent_map(list(id_to_structure.values()))

    # In full mode, clear the cache so every asset is re-fetched.
    if mode == "full":
        cache_file = config["cache_file"]
        if cache_file.exists():
            print(f"  Full mode: clearing cache {cache_file.name}")
            cache_file.unlink()

    # Reuse the existing DANDI fetch + aggregation. fetch_dandi_data writes
    # electrodes/{DANDISET_ID}.json directly; we write the other three JSONs
    # from its return values below.
    dandiset_assets, dandisets_with_electrodes, dandi_regions = fetch_dandi_data(
        config, abbrev_to_id, id_to_structure, parent_map,
        name_to_id=name_to_id,
    )

    # Implicit-routing pass: discover all other public macaque dandisets and
    # add region-tag-only records wherever their free-text location strings
    # resolve in this atlas's vocabulary. Embargoed dandisets are NOT covered
    # here (they don't appear in the public listing); inject those manually
    # via ongoing_issues/inject_001693_d99.py if needed.
    implicit_addition = fetch_macaque_implicit_data(
        abbrev_to_id, id_to_structure, name_to_id,
        aliases=MACAQUE_LOCATION_ALIASES,
    )
    for ds_id, recs in implicit_addition.items():
        dandiset_assets[ds_id] = recs
    # Rebuild dandi_regions over the merged dandiset_assets so counts and
    # ancestor propagation account for the implicit additions.
    dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)

    with open(data_dir / "dandiset_assets.json", "w") as f:
        json.dump(dandiset_assets, f)
    with open(data_dir / "dandisets_with_electrodes.json", "w") as f:
        json.dump(dandisets_with_electrodes, f)
    with open(data_dir / "dandi_regions.json", "w") as f:
        json.dump(dandi_regions, f)

    # Update mesh_manifest.json's DANDI-derived fields only.
    data_ids = {int(sid) for sid in dandi_regions.keys()}
    update_mesh_manifest(data_dir, data_ids, parent_map)

    asset_count = len(dandiset_assets.get(DANDISET_ID, []))
    write_last_updated(atlas_key, mode, asset_count)
    print(f"  Wrote DANDI-derived files for {atlas_key} ({asset_count} assets)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        required=True,
        choices=["d99", "nmt", "mebrains"],
        help="Which macaque atlas to refresh",
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Update mode: incremental skips cached assets; full re-fetches all (default: incremental)",
    )
    args = parser.parse_args()
    update_atlas(args.atlas, args.mode)


if __name__ == "__main__":
    main()
