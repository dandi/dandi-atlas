#!/usr/bin/env python3
"""Orchestrator: build all WHS-SD rat atlas data in a single run.

Stages:
  1. Parse the MILF (.ilf) hierarchy into a structure graph
  2. Generate region meshes from the parcellation NIfTI (marching cubes)
  3. Fetch DANDI 001699 location strings and match to WHS-SD regions
  4. Aggregate region counts + write mesh manifest

Inputs (downloaded from https://www.nitrc.org/projects/whs-sd-atlas, MBAT pack;
place under atlas_sources/MBAT_WHS_SD_rat_atlas_v4_pack/Data/):
  WHS_SD_rat_atlas_v4.nii.gz       (parcellation)
  WHS_SD_rat_atlas_v4_labels.ilf   (hierarchy)
  WHS_SD_rat_T2star_v1.01.nii.gz   (root template)

Override the source directory with --whs-data or the WHS_SD_DATA env var.

Outputs (under data/atlases/whs_sd/):
  structure_graph.json, meshes/*.glb, dandiset_assets.json,
  dandisets_with_electrodes.json, dandi_regions.json, mesh_manifest.json

Usage:
    uv run python scripts/build_rat_atlas.py [--skip-meshes] [--skip-dandi]
"""

import argparse
import json
import sys
from pathlib import Path

from dandi_helpers import build_dandi_regions, get_ancestors
from macaque_atlas_lib import OUTSIDE_ID, ROOT_ID
import rat_atlas_lib
from rat_atlas_lib import (
    ATLAS_CONFIGS,
    DANDISET_IDS,
    EMBARGOED_DANDISETS,
    fetch_local_rat_data,
    fetch_rat_dandi_data,
    fetch_rat_dandi_sweep,
    generate_meshes_with_progress,
    parse_whs_ilf,
)


def _validate_inputs(config):
    missing = []
    for key in ("parcellation", "hierarchy", "template_nifti"):
        path = config[key]
        if not path.exists():
            missing.append(f"  {key}: {path}")
    if missing:
        sys.stderr.write(
            "Missing required WHS-SD source files:\n"
            + "\n".join(missing)
            + "\n\nDownload the MBAT WHS-SD v4 pack from "
            "https://www.nitrc.org/projects/whs-sd-atlas\n"
            "and place it at the path above (or set WHS_SD_DATA / --whs-data).\n"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build WHS-SD rat atlas data")
    parser.add_argument(
        "--whs-data",
        type=Path,
        help="Override the WHS-SD MBAT pack Data directory",
    )
    parser.add_argument("--skip-meshes", action="store_true", help="Skip mesh generation")
    parser.add_argument("--skip-dandi", action="store_true", help="Skip DANDI data extraction")
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the DANDI-wide rat sweep; only process the explicit "
             "DANDISET_IDS list.",
    )
    parser.add_argument(
        "--sweep-limit",
        type=int,
        default=None,
        help="Stop the sweep after this many rat dandisets have been "
             "processed (post-asset-cap, i.e. actually streamed). "
             "Useful for smoke-testing the streaming pass before running "
             "the full sweep.",
    )
    parser.add_argument(
        "--sweep-max-assets",
        type=int,
        default=1000,
        help="Skip swept dandisets with more than this many NWB assets "
             "(default 1000). Very large dandisets tend to be auto-generated "
             "low-quality dumps that dominate sweep wall time. Pass a large "
             "number (e.g. 100000) or 0 to disable.",
    )
    parser.add_argument(
        "--local-nwb",
        type=Path,
        help="Read locations from local NWB files in this directory instead of "
             "streaming from DANDI (use while dandiset 001699 is embargoed).",
    )
    args = parser.parse_args()

    if args.whs_data:
        rat_atlas_lib.WHS_DATA = args.whs_data
        ATLAS_CONFIGS["whs_sd"]["parcellation"] = args.whs_data / "WHS_SD_rat_atlas_v4.nii.gz"
        ATLAS_CONFIGS["whs_sd"]["hierarchy"] = args.whs_data / "WHS_SD_rat_atlas_v4_labels.ilf"
        ATLAS_CONFIGS["whs_sd"]["template_nifti"] = args.whs_data / "WHS_SD_rat_T2star_v1.01.nii.gz"

    config = ATLAS_CONFIGS["whs_sd"]
    _validate_inputs(config)

    data_dir = config["output_dir"]
    meshes_dir = data_dir / "meshes"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building WHS-SD rat atlas data into {data_dir}...")

    tree, id_to_structure, parent_map, abbrev_to_id, name_to_id = parse_whs_ilf(
        config["hierarchy"], root_name=config["root_name"],
    )
    print(f"  Parsed {len(id_to_structure)} structures from ILF "
          f"(including ROOT and OUTSIDE sentinels)")
    with open(data_dir / "structure_graph.json", "w") as f:
        json.dump(tree, f)
    print("  Wrote structure_graph.json")

    if args.skip_meshes:
        print("Skipping mesh generation")
        no_mesh = [OUTSIDE_ID]
        for nid in id_to_structure:
            if nid == ROOT_ID:
                continue
            if not (meshes_dir / f"{nid}.glb").exists():
                no_mesh.append(nid)
    else:
        print("Generating meshes from NIfTI parcellation...")
        no_mesh = generate_meshes_with_progress(
            config["parcellation"], meshes_dir, id_to_structure,
            template_nifti=config["template_nifti"],
        )

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
        except FileNotFoundError:
            print("  No existing DANDI outputs to reuse; writing empty placeholders")
            dandiset_assets = {dsid: [] for dsid in DANDISET_IDS}
            dandisets_with_electrodes = []
            dandi_regions = {}
    elif args.local_nwb is not None:
        # Additive: feed embargoed dandisets (e.g. 001699) from local NWB files,
        # then stream every non-embargoed entry in DANDISET_IDS from DANDI.
        local_assets, _, _ = fetch_local_rat_data(
            args.local_nwb, name_to_id, abbrev_to_id, id_to_structure, parent_map,
        )
        streaming_ids = [d for d in DANDISET_IDS if d not in EMBARGOED_DANDISETS]
        if streaming_ids:
            stream_assets, _, _ = fetch_rat_dandi_data(
                config, name_to_id, abbrev_to_id, id_to_structure, parent_map,
                dandiset_ids=streaming_ids,
            )
        else:
            stream_assets = {}
        dandiset_assets = {**local_assets, **stream_assets}
        dandisets_with_electrodes = []
    else:
        dandiset_assets, dandisets_with_electrodes, _ = fetch_rat_dandi_data(
            config, name_to_id, abbrev_to_id, id_to_structure, parent_map,
        )

    if not args.skip_dandi and not args.skip_sweep:
        print("Sweeping DANDI for additional rat dandisets...")
        sweep_assets = fetch_rat_dandi_sweep(
            config, name_to_id, abbrev_to_id, id_to_structure, parent_map,
            exclude_ids=set(dandiset_assets),
            limit=args.sweep_limit,
            max_assets_per_dandiset=args.sweep_max_assets if args.sweep_max_assets > 0 else None,
        )
        dandiset_assets.update(sweep_assets)
        print(f"  Sweep added {len(sweep_assets)} new dandiset(s)")

    if not args.skip_dandi:
        dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)

    with open(data_dir / "dandiset_assets.json", "w") as f:
        json.dump(dandiset_assets, f)
    with open(data_dir / "dandisets_with_electrodes.json", "w") as f:
        json.dump(dandisets_with_electrodes, f)
    with open(data_dir / "dandi_regions.json", "w") as f:
        json.dump(dandi_regions, f)
    print(f"  Wrote DANDI outputs ({len(dandi_regions)} regions, "
          f"{sum(len(v) for v in dandiset_assets.values())} assets)")

    data_ids = set(int(sid) for sid in dandi_regions.keys())
    ancestor_ids = set()
    for sid in data_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

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
    print(f"  Wrote mesh_manifest.json ({len(all_mesh_ids)} meshes on disk)")

    print(f"Done! All WHS-SD data written to {data_dir}")


if __name__ == "__main__":
    main()
