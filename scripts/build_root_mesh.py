#!/usr/bin/env python3
"""Build only the root (whole-brain) mesh for one macaque atlas.

Useful when iterating on root-mesh quality (face count, smoothing, hole
filling) without rebuilding hundreds of per-region GLBs. Pair with
build_region_meshes.py for per-region work and build_mesh_manifest.py to
refresh the manifest after either pass.

Source per atlas:
  d99      - GIFTI: lh + rh pial cached by ensure_d99_pial_cache (RheMAP-Surf).
             Falls back to template marching cubes if the cache is unreachable.
  nmt      - GIFTI: lh + rh gray_surface from NMT_v2.0_sym_surfaces.
  mebrains - NIFTI: marching cubes on MEBRAINS_T1.nii.gz template.

Usage:
    uv run python scripts/build_root_mesh.py --atlas d99
    uv run python scripts/build_root_mesh.py --atlas d99 --force
"""

import argparse
from pathlib import Path

from macaque_atlas_lib import (
    ATLAS_CONFIGS,
    ROOT_ID,
    ROOT_TARGET_FACES,
    ROOT_TARGET_FACES_GIFTI,
    _export_merged_gifti_glb,
    _export_template_root_glb,
    ensure_d99_pial_cache,
)


def build_root(atlas, force=False):
    config = ATLAS_CONFIGS[atlas]
    meshes_dir = config["output_dir"] / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    root_glb = meshes_dir / f"{ROOT_ID}.glb"

    if root_glb.exists():
        if force:
            print(f"--force: removing existing {root_glb.name}")
            root_glb.unlink()
        else:
            print(f"Root mesh already exists: {root_glb}. Use --force to overwrite.")
            return

    gifti_config = config.get("gifti_surfaces")
    template_nifti = config.get("template_nifti")

    if atlas == "d99":
        # Pull the RheMAP-Surf pial pair on first build before consulting the
        # whole-brain paths in gifti_config.
        ensure_d99_pial_cache()

    lh = gifti_config.get("whole_brain_lh") if gifti_config else None
    rh = gifti_config.get("whole_brain_rh") if gifti_config else None
    have_gifti = lh is not None and rh is not None and Path(lh).exists() and Path(rh).exists()

    if have_gifti:
        print(f"Generating {atlas} root mesh from {Path(lh).name} + {Path(rh).name}")
        mesh = _export_merged_gifti_glb(
            root_glb, [lh, rh], mirror=False,
            target_faces=ROOT_TARGET_FACES_GIFTI,
        )
        if mesh is None:
            raise SystemExit(f"GIFTI root export returned no mesh for {atlas}")
        print(f"  Root mesh: {len(mesh.faces)} faces")
        return

    if template_nifti is not None and Path(template_nifti).exists():
        target = ROOT_TARGET_FACES_GIFTI if gifti_config else ROOT_TARGET_FACES
        print(f"Generating {atlas} root mesh from template: {Path(template_nifti).name}")
        mesh = _export_template_root_glb(template_nifti, root_glb, target)
        if mesh is None:
            raise SystemExit(f"Template root export returned no mesh for {atlas}")
        print(f"  Root mesh: {len(mesh.faces)} faces")
        return

    raise SystemExit(
        f"No GIFTI whole-brain surfaces and no template_nifti for atlas {atlas}; "
        "cannot build root mesh."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        required=True,
        choices=["d99", "nmt", "mebrains"],
        help="Which atlas to build the root mesh for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing root GLB if present",
    )
    args = parser.parse_args()
    build_root(args.atlas, force=args.force)


if __name__ == "__main__":
    main()
