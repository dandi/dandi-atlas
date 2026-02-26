#!/usr/bin/env python3
"""Convert OBJ mesh files to GLB (binary glTF) format.

Reads each data/meshes/{id}.obj and writes data/meshes/{id}.glb using trimesh.
Skips conversion if the GLB file already exists and is newer than the OBJ.
Removes the OBJ file after successful conversion.

Usage:
    python scripts/convert_meshes.py
"""

from pathlib import Path

import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MESHES_DIR = PROJECT_ROOT / "data" / "meshes"


def convert_obj_to_glb(obj_path: Path) -> bool:
    """Convert a single OBJ file to GLB. Returns True on success."""
    glb_path = obj_path.with_suffix(".glb")

    # Skip if GLB already exists and is newer than OBJ
    if glb_path.exists() and glb_path.stat().st_mtime >= obj_path.stat().st_mtime:
        return True

    try:
        mesh = trimesh.load(obj_path, process=False)
        mesh.export(glb_path, file_type="glb")
        return True
    except Exception as exc:
        print(f"  ERROR converting {obj_path.name}: {exc}")
        return False


def main():
    if not MESHES_DIR.exists():
        print("No meshes directory found.")
        return

    obj_files = sorted(MESHES_DIR.glob("*.obj"))
    if not obj_files:
        print("No OBJ files to convert.")
        return

    print(f"Converting {len(obj_files)} OBJ files to GLB...")
    converted = 0
    skipped = 0
    failed = 0

    for obj_path in obj_files:
        glb_path = obj_path.with_suffix(".glb")
        if glb_path.exists() and glb_path.stat().st_mtime >= obj_path.stat().st_mtime:
            skipped += 1
            continue

        if convert_obj_to_glb(obj_path):
            obj_path.unlink()
            converted += 1
        else:
            failed += 1

    # Remove OBJ files that already have up-to-date GLB counterparts
    for obj_path in MESHES_DIR.glob("*.obj"):
        glb_path = obj_path.with_suffix(".glb")
        if glb_path.exists():
            obj_path.unlink()

    print(f"  Converted: {converted}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
