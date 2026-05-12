#!/usr/bin/env python3
"""Rebuild mesh_manifest.json for one macaque atlas from on-disk state.

Reads:
  data/atlases/{atlas}/structure_graph.json
  data/atlases/{atlas}/dandi_regions.json    (optional; supplies data_structures)
  data/atlases/{atlas}/meshes/*.glb

Writes:
  data/atlases/{atlas}/mesh_manifest.json with fields:
    data_structures      - region IDs referenced by any electrode in DANDI
    ancestor_structures  - union of ancestors of data_structures
    no_mesh              - structures whose GLB is missing on disk
    all_meshes           - every GLB present, sorted by integer ID
    root_id              - ROOT_ID

Use this after build_region_meshes.py / build_root_mesh.py to refresh the
manifest the viewer reads at load time. update_macaque_data.py also keeps the
DANDI-derived fields in sync after every refetch; this script is the
mesh-anchored counterpart for when on-disk meshes change without DANDI.

Usage:
    uv run python scripts/build_mesh_manifest.py --atlas d99
"""

import argparse
import json

from macaque_atlas_lib import (
    ATLAS_CONFIGS,
    CATEGORY_ID_START,
    OUTSIDE_ID,
    ROOT_ID,
    get_ancestors,
)
from dandi_helpers import build_parent_map


def reconstruct_lookups_from_graph(structure_graph):
    """Walk a saved structure_graph.json and produce id_to_structure dict.

    Mirrors update_macaque_data.reconstruct_lookups_from_graph but only the
    id_to_structure half (no abbrev lookup needed here).
    """
    id_to_structure = {}

    def walk(node):
        sid = node["id"]
        record = {k: v for k, v in node.items() if k != "children"}
        id_to_structure[sid] = record
        for child in node.get("children", []):
            walk(child)

    if isinstance(structure_graph, list):
        for n in structure_graph:
            walk(n)
    else:
        walk(structure_graph)
    return id_to_structure


def build_manifest(atlas):
    config = ATLAS_CONFIGS[atlas]
    data_dir = config["output_dir"]
    meshes_dir = data_dir / "meshes"

    structure_graph_path = data_dir / "structure_graph.json"
    if not structure_graph_path.exists():
        raise SystemExit(
            f"Missing {structure_graph_path}. Run build_macaque_atlas.py "
            "(or build_region_meshes.py / build_root_mesh.py and then make "
            "sure structure_graph.json exists) before refreshing the manifest."
        )
    structure_graph = json.load(open(structure_graph_path))
    id_to_structure = reconstruct_lookups_from_graph(structure_graph)
    parent_map = build_parent_map(list(id_to_structure.values()))

    # data_structures comes from dandi_regions.json when present. If a fresh
    # checkout has no DANDI snapshot yet, we still produce a valid manifest
    # with empty data_structures so the viewer can load static meshes.
    dandi_regions_path = data_dir / "dandi_regions.json"
    if dandi_regions_path.exists():
        dandi_regions = json.load(open(dandi_regions_path))
        data_ids = {int(sid) for sid in dandi_regions.keys()}
    else:
        print(f"Note: {dandi_regions_path.name} missing; data_structures will be empty.")
        data_ids = set()

    ancestor_ids = set()
    for sid in data_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

    all_mesh_ids = sorted(
        int(p.stem) for p in meshes_dir.glob("*.glb") if p.stem.lstrip("-").isdigit()
    )
    on_disk = set(all_mesh_ids)

    no_mesh = {OUTSIDE_ID}
    for sid in id_to_structure:
        if sid in (ROOT_ID, OUTSIDE_ID):
            continue
        if sid not in on_disk:
            no_mesh.add(sid)
    # Belt-and-braces: any synthetic category node still missing is reported
    # explicitly even if the loop above already added it.
    for sid in id_to_structure:
        if sid >= CATEGORY_ID_START and sid not in on_disk:
            no_mesh.add(sid)

    manifest = {
        "data_structures": sorted(data_ids),
        "ancestor_structures": sorted(ancestor_ids - data_ids),
        "no_mesh": sorted(no_mesh),
        "all_meshes": all_mesh_ids,
        "root_id": ROOT_ID,
    }
    out = data_dir / "mesh_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f)
    print(
        f"Wrote {out.relative_to(data_dir.parent.parent)}: "
        f"{len(all_mesh_ids)} on disk, {len(data_ids)} data, "
        f"{len(ancestor_ids - data_ids)} ancestors, {len(no_mesh)} no_mesh"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        required=True,
        choices=["d99", "nmt", "mebrains"],
        help="Which atlas to refresh the manifest for",
    )
    args = parser.parse_args()
    build_manifest(args.atlas)


if __name__ == "__main__":
    main()
