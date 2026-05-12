#!/usr/bin/env python3
"""Orchestrator: build all macaque atlas data for one atlas in a single run.

Equivalent to running, in order:
  scripts/build_root_mesh.py --atlas {atlas}
  scripts/build_region_meshes.py --atlas {atlas}
  scripts/update_macaque_data.py --atlas {atlas}   (DANDI fetch)
  scripts/build_mesh_manifest.py --atlas {atlas}

Use the focused sub-scripts when iterating on one stage; this orchestrator is
the single command for a clean from-scratch build.

Supported atlases (share DANDI 001636):
  - D99 v2.0 (Saleem & Logothetis)
  - NMT v2.0 sym (D99 labels warped into NMT space)
  - MEBRAINS (EBRAINS macaque parcellation)

Outputs (per atlas, under data/atlases/{atlas}/):
  structure_graph.json, meshes/*.glb, dandiset_assets.json,
  electrodes/001636.json, dandisets_with_electrodes.json,
  dandi_regions.json, mesh_manifest.json

Usage:
    uv run python scripts/build_macaque_atlas.py --atlas d99 [--skip-meshes] [--skip-dandi]
    uv run python scripts/build_macaque_atlas.py --atlas nmt [--skip-meshes] [--skip-dandi]
    uv run python scripts/build_macaque_atlas.py --atlas mebrains [--skip-meshes] [--skip-dandi]
"""

import argparse
import json

from macaque_atlas_lib import (
    ATLAS_CONFIGS,
    CATEGORY_ID_START,
    DANDISET_ID,
    MACAQUE_LOCATION_ALIASES,
    OUTSIDE_ID,
    ROOT_ID,
    build_charm_structure_graph,
    build_nmt_unified_structure_graph,
    build_structure_graph,
    ensure_d99_pial_cache,
    fetch_dandi_data,
    fetch_macaque_implicit_data,
    generate_meshes,
    generate_meshes_from_gifti,
    get_ancestors,
    load_mebrains_palette_from_siibra,
    parse_charm_labels,
    parse_d99_labels,
    parse_mebrains_labels,
    parse_sarm_labels,
)
from dandi_helpers import build_dandi_regions


def main():
    parser = argparse.ArgumentParser(description="Build macaque atlas data")
    parser.add_argument(
        "--atlas",
        required=True,
        choices=["d99", "nmt", "mebrains"],
        help="Which atlas to build",
    )
    parser.add_argument("--skip-meshes", action="store_true", help="Skip mesh generation")
    parser.add_argument("--skip-dandi", action="store_true", help="Skip DANDI data extraction")
    args = parser.parse_args()

    config = ATLAS_CONFIGS[args.atlas]
    data_dir = config["output_dir"]
    meshes_dir = data_dir / "meshes"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Parse labels
    print(f"Building {args.atlas} atlas data...")
    charm_entries = None
    sarm_entries = None
    if config["labels_type"] == "d99":
        entries = parse_d99_labels()
        print(f"  Parsed {len(entries)} D99 label entries")
    elif config["labels_type"] == "charm":
        entries = parse_charm_labels()
        charm_entries = entries
        print(f"  Parsed {len(entries)} CHARM label entries")
    elif config["labels_type"] == "nmt_unified":
        charm_entries = parse_charm_labels()
        sarm_entries = parse_sarm_labels()
        entries = {"charm": charm_entries, "sarm": sarm_entries}
        print(f"  Parsed {len(charm_entries)} CHARM + {len(sarm_entries)} SARM label entries")
    else:
        entries = parse_mebrains_labels()
        print(f"  Parsed {len(entries)} MEBRAINS label entries")

    if config["labels_type"] == "charm":
        tree, id_to_structure, parent_map, abbrev_to_id, name_to_id = (
            build_charm_structure_graph(entries, root_name=config["root_name"])
        )
    elif config["labels_type"] == "nmt_unified":
        tree, id_to_structure, parent_map, abbrev_to_id, name_to_id = (
            build_nmt_unified_structure_graph(
                charm_entries, sarm_entries,
                root_name=config["root_name"],
            )
        )
    elif config["labels_type"] == "mebrains":
        mebrains_colors = load_mebrains_palette_from_siibra()
        tree, id_to_structure, parent_map, abbrev_to_id, name_to_id = (
            build_structure_graph(
                entries,
                root_name=config["root_name"],
                color_overrides=mebrains_colors,
                labels_type="mebrains",
            )
        )
    else:
        tree, id_to_structure, parent_map, abbrev_to_id, name_to_id = (
            build_structure_graph(
                entries,
                root_name=config["root_name"],
                labels_type=config["labels_type"],
            )
        )

    with open(data_dir / "structure_graph.json", "w") as f:
        json.dump(tree, f)
    print("  Wrote structure_graph.json")

    # Generate meshes
    if args.skip_meshes:
        print("Skipping mesh generation")
        # Only mark nodes as no_mesh if they don't have a mesh file on disk
        no_mesh = [OUTSIDE_ID]
        for nid in id_to_structure:
            if nid >= CATEGORY_ID_START and not (meshes_dir / f"{nid}.glb").exists():
                no_mesh.append(nid)
    elif config.get("gifti_surfaces") is not None:
        print("Generating meshes from upstream GIFTI surfaces...")
        # D99 whole-brain pial from RheMAP-Surf is fetched on-demand (the AFNI
        # D99 distribution itself has no whole-brain surface).
        if args.atlas == "d99":
            ensure_d99_pial_cache()
        no_mesh = generate_meshes_from_gifti(
            args.atlas, config["nifti"], meshes_dir, id_to_structure,
            config["gifti_surfaces"],
            charm_entries=charm_entries,
            sarm_entries=sarm_entries,
            template_nifti=config.get("template_nifti"),
        )
    else:
        print("Generating meshes from NIfTI volume...")
        no_mesh = generate_meshes(
            config["nifti"], meshes_dir, id_to_structure,
            template_nifti=config.get("template_nifti"),
        )

    # DANDI data
    if args.skip_dandi:
        print("Skipping DANDI data extraction; reusing existing JSON if present")
        try:
            with open(data_dir / "dandiset_assets.json") as f:
                dandiset_assets = json.load(f)
            with open(data_dir / "dandisets_with_electrodes.json") as f:
                dandisets_with_electrodes = json.load(f)
            with open(data_dir / "dandi_regions.json") as f:
                dandi_regions = json.load(f)
            asset_count = sum(len(v) for v in dandiset_assets.values())
            print(f"  Reused {asset_count} asset entries, {len(dandi_regions)} regions")

            # Robustness: re-sync colours/names/acronyms from the freshly-built
            # structure graph so palette edits propagate without a full DANDI
            # refetch. The viewer reads region colour from dandi_regions.json
            # first (loadMesh in app.js).
            synced_regions = 0
            for rid_str, region_data in dandi_regions.items():
                try:
                    rid = int(rid_str)
                except (TypeError, ValueError):
                    continue
                s = id_to_structure.get(rid)
                if s is None:
                    continue
                if region_data.get("color_hex_triplet") != s.get("color_hex_triplet"):
                    region_data["color_hex_triplet"] = s.get("color_hex_triplet")
                    synced_regions += 1
                if "acronym" in s:
                    region_data["acronym"] = s["acronym"]
                if "name" in s:
                    region_data["name"] = s["name"]
            if synced_regions:
                print(f"  Re-synced colours for {synced_regions} regions from current structure graph")

            for atlas_dandisets in dandiset_assets.values():
                for asset in atlas_dandisets:
                    for r in asset.get("regions") or []:
                        s = id_to_structure.get(r.get("id"))
                        if s is None:
                            continue
                        if "acronym" in s:
                            r["acronym"] = s["acronym"]
                        if "name" in s:
                            r["name"] = s["name"]
        except FileNotFoundError:
            print("  No existing DANDI outputs to reuse; writing empty placeholders")
            dandiset_assets = {DANDISET_ID: []}
            dandisets_with_electrodes = []
            dandi_regions = {}
    else:
        dandiset_assets, dandisets_with_electrodes, dandi_regions = fetch_dandi_data(
            config, abbrev_to_id, id_to_structure, parent_map,
            name_to_id=name_to_id,
        )
        # Implicit-routing pass: discover all other public macaque dandisets
        # and add region-tag-only records for any whose free-text location
        # strings resolve in this atlas. Embargoed dandisets are skipped (they
        # don't appear in the public listing); inject those manually via the
        # one-off scripts in ongoing_issues/ and let CI overwrite them later.
        implicit_addition = fetch_macaque_implicit_data(
            abbrev_to_id, id_to_structure, name_to_id,
            aliases=MACAQUE_LOCATION_ALIASES,
        )
        for ds_id, recs in implicit_addition.items():
            dandiset_assets[ds_id] = recs
        # Rebuild dandi_regions over the merged dandiset_assets so counts and
        # ancestor propagation account for the implicit additions.
        dandi_regions = build_dandi_regions(
            dandiset_assets, id_to_structure, parent_map,
        )

    with open(data_dir / "dandiset_assets.json", "w") as f:
        json.dump(dandiset_assets, f)

    with open(data_dir / "dandisets_with_electrodes.json", "w") as f:
        json.dump(dandisets_with_electrodes, f)

    with open(data_dir / "dandi_regions.json", "w") as f:
        json.dump(dandi_regions, f)

    # Mesh manifest
    data_ids = set(int(sid) for sid in dandi_regions.keys())
    ancestor_ids = set()
    for sid in data_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

    # `all_meshes` is the concrete list of GLB files this build produced. The
    # viewer uses it at load time to populate the macaque anatomical context
    # without relying on the server to generate an HTML directory listing,
    # which Netlify and most CDNs do not.
    all_mesh_ids = sorted(
        int(p.stem) for p in meshes_dir.glob("*.glb") if p.stem.lstrip("-").isdigit()
    )

    mesh_manifest = {
        "data_structures": sorted(data_ids),
        "ancestor_structures": sorted(ancestor_ids - data_ids),
        "no_mesh": sorted(set(no_mesh)),
        "all_meshes": all_mesh_ids,
        "root_id": ROOT_ID,
    }
    with open(data_dir / "mesh_manifest.json", "w") as f:
        json.dump(mesh_manifest, f)

    print(f"Done! All {args.atlas} data written to {data_dir}")


if __name__ == "__main__":
    main()
