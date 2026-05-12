#!/usr/bin/env python3
"""Build per-region (and synthetic-parent) meshes for one macaque atlas.

Skips the root mesh - use build_root_mesh.py for that. The lib's mesh
generators preserve any GLB already on disk, so this script can rebuild a
subset by deleting just the targets up front:

  --force            Delete every non-root GLB before building (full rebuild).
  --regions ID,ID,...  Delete only the listed region GLBs before building.

With neither flag set, only missing GLBs are produced (catch-up mode).

Usage:
    uv run python scripts/build_region_meshes.py --atlas d99
    uv run python scripts/build_region_meshes.py --atlas d99 --force
    uv run python scripts/build_region_meshes.py --atlas mebrains --regions 1,2,3
"""

import argparse

from macaque_atlas_lib import (
    ROOT_ID,
    build_atlas_graph,
    ensure_d99_pial_cache,
    generate_meshes,
    generate_meshes_from_gifti,
)


def parse_region_ids(arg):
    if not arg:
        return None
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def build_regions(atlas, force=False, region_ids=None):
    config, _tree, id_to_structure, _pmap, _amap, _nmap, entries = build_atlas_graph(atlas)
    meshes_dir = config["output_dir"] / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Targeted deletion before delegating to the lib's generators (which skip
    # files already on disk).
    if region_ids is not None:
        removed = 0
        for rid in region_ids:
            p = meshes_dir / f"{rid}.glb"
            if p.exists():
                p.unlink()
                removed += 1
        print(f"--regions: removed {removed} of {len(region_ids)} requested GLBs")
    elif force:
        removed = 0
        for p in meshes_dir.glob("*.glb"):
            if p.stem == str(ROOT_ID):
                continue
            p.unlink()
            removed += 1
        print(f"--force: removed {removed} non-root GLBs")

    if config.get("gifti_surfaces") is not None:
        if atlas == "d99":
            ensure_d99_pial_cache()
        no_mesh = generate_meshes_from_gifti(
            atlas, config["nifti"], meshes_dir, id_to_structure,
            config["gifti_surfaces"],
            charm_entries=entries if config["labels_type"] == "charm" else None,
            template_nifti=config.get("template_nifti"),
        )
    else:
        no_mesh = generate_meshes(
            config["nifti"], meshes_dir, id_to_structure,
            template_nifti=config.get("template_nifti"),
        )

    print(f"Done. {len(no_mesh)} structures still without mesh.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        required=True,
        choices=["d99", "nmt", "mebrains"],
        help="Which atlas to build region meshes for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete every non-root GLB before building (full rebuild)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Comma-separated structure IDs to rebuild (overrides --force scope)",
    )
    args = parser.parse_args()
    region_ids = parse_region_ids(args.regions)
    build_regions(args.atlas, force=args.force, region_ids=region_ids)


if __name__ == "__main__":
    main()
