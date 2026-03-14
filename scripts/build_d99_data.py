#!/usr/bin/env python3
"""Generate all D99 atlas data for the dandi-atlas viewer.

Produces:
  data/atlases/d99/structure_graph.json   - hierarchical region tree
  data/atlases/d99/meshes/*.glb           - brain region meshes from NIfTI
  data/atlases/d99/dandiset_assets.json   - DANDI 001636 assets with regions
  data/atlases/d99/electrodes/001636.json - electrode coordinates per asset
  data/atlases/d99/dandisets_with_electrodes.json
  data/atlases/d99/dandi_regions.json     - region metadata for data regions
  data/atlases/d99/mesh_manifest.json     - mesh availability info

Usage:
    uv run python scripts/build_d99_data.py [--skip-meshes] [--skip-dandi]
"""

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dandi_helpers import (
    get_nwb_assets_paged,
    get_download_url,
    get_ancestors,
    build_dandi_regions,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "atlases" / "d99"
MESHES_DIR = DATA_DIR / "meshes"
ELECTRODES_DIR = DATA_DIR / "electrodes"
CACHE_FILE = Path(__file__).resolve().parent / "d99_electrode_cache.jsonl"

D99_LABELS_FILE = Path(
    "/home/heberto/development/conversions/turner-lab-to-nwb"
    "/data/d99_atlas/D99_v2.0_dist/D99_v2.0_labels_semicolon.txt"
)
D99_NIFTI_FILE = Path(
    "/home/heberto/development/conversions/turner-lab-to-nwb"
    "/data/d99_atlas/D99_v2.0_dist/D99_atlas_v2.0.nii.gz"
)

DANDISET_ID = "001636"
ROOT_ID = 9999
OUTSIDE_ID = 9998
CATEGORY_ID_START = 10001
SUBCATEGORY_ID_START = 10100
TARGET_FACES = 10_000
MIN_VOXELS = 50

# Category color hues (HSL hue in degrees)
CATEGORY_HUES = {
    "Basal ganglia": 0,
    "Thalamus": 30,
    "Brainstem": 60,
    "Hypothalamus": 90,
    "Cerebellum": 120,
    "Fiber bundle": 180,
    "Cortex": 210,
    "Hippocampus": 240,
    "Amygdala": 270,
    "Basal forebrain": 300,
    "Bed nucleus of stria terminalis": 330,
    "Other": 150,
}


# ---------------------------------------------------------------------------
# 2a. Parse labels and build hierarchy
# ---------------------------------------------------------------------------


def parse_labels(labels_file):
    """Parse D99 semicolon-delimited labels file.

    Returns list of dicts with keys: index, abbreviation, name, category, subcategory.
    """
    entries = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle quoted fields by using csv reader
            reader = csv.reader(io.StringIO(line), delimiter=";")
            fields = next(reader)
            fields = [field.strip() for field in fields]

            label_index = int(fields[0])
            abbreviation = fields[1].strip()
            name = fields[2].strip()

            category = fields[3].strip() if len(fields) > 3 else ""
            subcategory = fields[4].strip() if len(fields) > 4 else ""

            # Infer category for entries missing it
            if not category:
                name_lower = name.lower()
                if any(
                    kw in name_lower
                    for kw in [
                        "cortex",
                        "cortical",
                        "area",
                        "gyrus",
                        "sulcus",
                        "opercul",
                        "prefrontal",
                        "parietal",
                        "temporal",
                        "visual",
                        "auditory",
                        "somato",
                        "insula",
                        "cingulate",
                        "retrosplenial",
                        "parahippocampal",
                        "perirhinal",
                        "entorhinal",
                        "precentral",
                        "frontal area",
                        "belt region",
                        "core region",
                    ]
                ):
                    category = "Cortex"
                elif any(
                    kw in name_lower for kw in ["amygdal", "periamygdal"]
                ):
                    category = "Amygdala"
                elif any(
                    kw in name_lower
                    for kw in ["hippocamp", "subicul", "fascia dentata", "ca1", "ca2", "ca3", "ca4"]
                ):
                    category = "Hippocampus"
                elif "claustrum" in name_lower:
                    category = "Basal forebrain"
                elif "olfactory" in name_lower:
                    category = "Cortex"
                else:
                    category = "Other"

            entries.append(
                {
                    "index": label_index,
                    "abbreviation": abbreviation,
                    "name": name,
                    "category": category,
                    "subcategory": subcategory,
                }
            )
    return entries


def _hsl_to_hex(h, s, l):
    """Convert HSL (h in degrees, s and l in 0-1) to hex color string."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    return f"{r:02X}{g:02X}{b:02X}"


def build_structure_graph(entries):
    """Build a hierarchical structure graph from parsed D99 labels.

    Returns (tree_root, id_to_structure, parent_map, abbrev_to_id).
    """
    # Assign colors by category with brightness variation
    category_counts = {}
    entry_colors = {}
    for entry in entries:
        cat = entry["category"]
        count = category_counts.get(cat, 0)
        category_counts[cat] = count + 1

    category_index = {}
    for entry in entries:
        cat = entry["category"]
        current = category_index.get(cat, 0)
        category_index[cat] = current + 1
        hue = CATEGORY_HUES.get(cat, 150)
        # Vary lightness between 0.35 and 0.75
        total = category_counts[cat]
        lightness = 0.35 + 0.4 * (current / max(total - 1, 1))
        entry_colors[entry["index"]] = _hsl_to_hex(hue, 0.6, lightness)

    # Collect unique categories and subcategories
    categories = {}
    for entry in entries:
        cat = entry["category"]
        subcat = entry["subcategory"]
        if cat not in categories:
            categories[cat] = set()
        if subcat:
            # Handle underscore-separated subcategories like "Striatum_ventral striatum"
            # or "Dorsal thalamus_anterior group"
            primary_subcat = subcat.split("_")[0]
            categories[cat].add(primary_subcat)

    # Assign synthetic IDs
    category_ids = {}
    next_cat_id = CATEGORY_ID_START
    for cat in sorted(categories.keys()):
        category_ids[cat] = next_cat_id
        next_cat_id += 1

    subcategory_ids = {}
    next_subcat_id = SUBCATEGORY_ID_START
    for cat in sorted(categories.keys()):
        for subcat in sorted(categories[cat]):
            key = (cat, subcat)
            subcategory_ids[key] = next_subcat_id
            next_subcat_id += 1

    # Build flat structure list
    id_to_structure = {}
    parent_map = {}

    # Root node
    root = {
        "id": ROOT_ID,
        "acronym": "root",
        "name": "D99 Atlas",
        "color_hex_triplet": "FFFFFF",
        "parent_structure_id": None,
        "children": [],
    }
    id_to_structure[ROOT_ID] = root
    parent_map[ROOT_ID] = None

    # "Outside atlas" node for electrodes that fall outside labeled regions
    outside_node = {
        "id": OUTSIDE_ID,
        "acronym": "outside",
        "name": "Outside atlas",
        "color_hex_triplet": "888888",
        "parent_structure_id": ROOT_ID,
        "children": [],
    }
    id_to_structure[OUTSIDE_ID] = outside_node
    parent_map[OUTSIDE_ID] = ROOT_ID

    # Category nodes
    for cat, cat_id in category_ids.items():
        hue = CATEGORY_HUES.get(cat, 150)
        node = {
            "id": cat_id,
            "acronym": cat.replace(" ", "_"),
            "name": cat,
            "color_hex_triplet": _hsl_to_hex(hue, 0.5, 0.5),
            "parent_structure_id": ROOT_ID,
            "children": [],
        }
        id_to_structure[cat_id] = node
        parent_map[cat_id] = ROOT_ID

    # Subcategory nodes
    for (cat, subcat), subcat_id in subcategory_ids.items():
        cat_id = category_ids[cat]
        hue = CATEGORY_HUES.get(cat, 150)
        node = {
            "id": subcat_id,
            "acronym": subcat.replace(" ", "_"),
            "name": subcat,
            "color_hex_triplet": _hsl_to_hex(hue, 0.5, 0.55),
            "parent_structure_id": cat_id,
            "children": [],
        }
        id_to_structure[subcat_id] = node
        parent_map[subcat_id] = cat_id

    # Leaf nodes (actual regions)
    abbrev_to_id = {}
    for entry in entries:
        label_id = entry["index"]
        cat = entry["category"]
        subcat = entry["subcategory"]

        # Determine parent: subcategory if available, else category
        if subcat:
            primary_subcat = subcat.split("_")[0]
            parent_id = subcategory_ids.get((cat, primary_subcat), category_ids.get(cat, ROOT_ID))
        else:
            parent_id = category_ids.get(cat, ROOT_ID)

        node = {
            "id": label_id,
            "acronym": entry["abbreviation"],
            "name": entry["name"],
            "color_hex_triplet": entry_colors.get(label_id, "AAAAAA"),
            "parent_structure_id": parent_id,
            "children": [],
        }
        id_to_structure[label_id] = node
        parent_map[label_id] = parent_id
        abbrev_to_id[entry["abbreviation"]] = label_id

    # Build tree by nesting children
    for node_id, node in id_to_structure.items():
        pid = node.get("parent_structure_id")
        if pid is not None and pid in id_to_structure:
            id_to_structure[pid]["children"].append(node)

    return [root], id_to_structure, parent_map, abbrev_to_id


# ---------------------------------------------------------------------------
# 2b. Generate meshes from NIfTI
# ---------------------------------------------------------------------------


def generate_meshes(entries, id_to_structure):
    """Generate GLB meshes from the D99 NIfTI volume."""
    import nibabel as nib
    from skimage.measure import marching_cubes
    import trimesh

    MESHES_DIR.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(D99_NIFTI_FILE))
    affine = img.affine
    voxel_sizes = img.header.get_zooms()[:3]
    atlas_data = np.asarray(img.dataobj, dtype=np.int16)

    # Get all label IDs present in the volume
    unique_labels = set(np.unique(atlas_data)) - {0}
    print(f"Found {len(unique_labels)} non-zero labels in NIfTI volume")

    no_mesh = []
    generated = 0
    skipped_existing = 0
    skipped_small = 0

    # Generate root mesh (union of all non-zero)
    root_glb = MESHES_DIR / f"{ROOT_ID}.glb"
    if not root_glb.exists():
        print("Generating root mesh (whole brain)...")
        root_mask = atlas_data > 0
        verts, faces, _, _ = marching_cubes(root_mask, level=0.5)
        # Transform to world mm, then negate X for neurological display
        # convention (left hemisphere on screen-left in anterior view).
        # Three.js camera-right = -X, so we flip X so left hemi = +X.
        verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
        verts_world = (affine @ verts_homogeneous.T).T[:, :3]
        verts_world[:, 0] *= -1
        faces = faces[:, ::-1]  # reverse winding after X flip
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        if len(mesh.faces) > TARGET_FACES:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        mesh.export(str(root_glb), file_type="glb")
        print(f"  Root mesh: {len(mesh.faces)} faces")
        generated += 1
    else:
        skipped_existing += 1

    # Generate per-region meshes
    label_ids_in_volume = sorted(unique_labels)
    for label_id in label_ids_in_volume:
        glb_path = MESHES_DIR / f"{label_id}.glb"
        if glb_path.exists():
            skipped_existing += 1
            continue

        mask = atlas_data == label_id
        voxel_count = np.sum(mask)
        if voxel_count < MIN_VOXELS:
            skipped_small += 1
            no_mesh.append(int(label_id))
            continue

        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5)
            verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
            verts_world = (affine @ verts_homogeneous.T).T[:, :3]
            verts_world[:, 0] *= -1
            faces = faces[:, ::-1]
            mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
            if len(mesh.faces) > TARGET_FACES:
                mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
            mesh.export(str(glb_path), file_type="glb")
            generated += 1
        except Exception as exc:
            print(f"  Failed mesh for label {label_id}: {exc}")
            no_mesh.append(int(label_id))

        if generated > 0 and generated % 50 == 0:
            print(f"  Generated {generated} meshes...")

    # Add synthetic nodes to no_mesh (they have no volume data)
    no_mesh.append(OUTSIDE_ID)
    for node_id in id_to_structure:
        if node_id >= CATEGORY_ID_START:
            no_mesh.append(node_id)

    print(
        f"Meshes: {generated} generated, {skipped_existing} existing, "
        f"{skipped_small} too small, {len(no_mesh)} no mesh"
    )
    return sorted(set(no_mesh))


# ---------------------------------------------------------------------------
# 2c. Extract Turner data from DANDI 001636
# ---------------------------------------------------------------------------


def _load_cache():
    """Load cached electrode data from JSONL file."""
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["asset_id"]] = entry
    return cache


def _append_cache(entry):
    """Append a single entry to the cache file."""
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def extract_d99_coords(url, asset_id):
    """Extract D99 coordinates from the localization extension in an NWB file.

    Returns dict with keys: asset_id, coords, brain_region, brain_region_id
    or None if no D99 localization found.
    """
    import h5py
    import remfile

    rf = remfile.File(url)
    with h5py.File(rf, "r") as f:
        d99_path = "general/localization/D99AtlasCoordinates"
        if d99_path not in f:
            return None

        t = f[d99_path]
        if not all(col in t for col in ("x", "y", "z")):
            return None

        x = t["x"][()]
        y = t["y"][()]
        z = t["z"][()]

        brain_region = []
        if "brain_region" in t:
            raw = t["brain_region"][()]
            if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
                brain_region = [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in raw
                ]
            elif isinstance(raw, bytes):
                brain_region = [raw.decode("utf-8")]
            else:
                brain_region = [str(raw)]

        brain_region_id = []
        if "brain_region_id" in t:
            raw = t["brain_region_id"][()]
            if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
                brain_region_id = [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in raw
                ]
            elif isinstance(raw, bytes):
                brain_region_id = [raw.decode("utf-8")]
            else:
                brain_region_id = [str(raw)]

    coords = []
    for xi, yi, zi in zip(x, y, z):
        xi, yi, zi = float(xi), float(yi), float(zi)
        if xi != xi or yi != yi or zi != zi:
            continue
        coords.append([round(-xi, 3), round(yi, 3), round(zi, 3)])

    if not coords:
        return None

    return {
        "asset_id": asset_id,
        "coords": coords,
        "brain_region": brain_region,
        "brain_region_id": brain_region_id,
    }


def fetch_dandi_data(abbrev_to_id, id_to_structure, parent_map):
    """Fetch electrode data from DANDI 001636 and build all output files."""
    from dandi_helpers import extract_subject, extract_session

    print(f"Fetching assets from dandiset {DANDISET_ID}...")
    assets = list(get_nwb_assets_paged(DANDISET_ID))
    print(f"  Found {len(assets)} NWB assets")

    cache = _load_cache()
    print(f"  Cache has {len(cache)} entries")

    # Process assets with thread pool
    results = {}
    to_process = []
    for asset in assets:
        asset_id = asset["asset_id"]
        if asset_id in cache:
            results[asset_id] = cache[asset_id]
        else:
            to_process.append(asset)

    print(f"  Need to fetch {len(to_process)} new assets")

    def _process_one(asset):
        asset_id = asset["asset_id"]
        url = get_download_url(DANDISET_ID, asset_id)
        try:
            result = extract_d99_coords(url, asset_id)
            if result is None:
                result = {"asset_id": asset_id, "coords": None, "brain_region": [], "brain_region_id": []}
            return result
        except Exception as exc:
            print(f"  Error processing {asset['path']}: {exc}")
            return {"asset_id": asset_id, "coords": None, "brain_region": [], "brain_region_id": [], "error": str(exc)}

    if to_process:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_one, a): a for a in to_process}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                asset_id = result["asset_id"]
                results[asset_id] = result
                _append_cache(result)
                done += 1
                if done % 50 == 0:
                    print(f"  Processed {done}/{len(to_process)} assets")

    # Build output data
    # 1. dandiset_assets.json
    dandiset_assets = {DANDISET_ID: []}
    electrodes_data = {}  # asset_id -> [[x,y,z], ...]
    has_electrodes = False

    skipped_no_loc = 0
    for asset in assets:
        asset_id = asset["asset_id"]
        path = asset["path"]
        result = results.get(asset_id, {})

        # Skip assets that lack D99 localization entirely (stale uploads)
        if result.get("coords") is None and not result.get("brain_region_id"):
            skipped_no_loc += 1
            continue

        # Map brain_region_id abbreviations to D99 label indices
        regions = []
        for abbrev in result.get("brain_region_id", []):
            if abbrev == "outside":
                regions.append({
                    "id": OUTSIDE_ID,
                    "acronym": "outside",
                    "name": "Outside atlas",
                })
                continue
            label_id = abbrev_to_id.get(abbrev)
            if label_id is not None and label_id in id_to_structure:
                s = id_to_structure[label_id]
                regions.append({
                    "id": label_id,
                    "acronym": s["acronym"],
                    "name": s["name"],
                })

        session = extract_session(path)
        subject = extract_subject(path)

        dandiset_assets[DANDISET_ID].append({
            "path": path,
            "asset_id": asset_id,
            "regions": regions,
            "session": session,
        })

        coords = result.get("coords")
        if coords:
            electrodes_data[asset_id] = coords
            has_electrodes = True

    if skipped_no_loc:
        print(f"  Skipped {skipped_no_loc} assets without D99 localization")

    # 2. electrodes file
    ELECTRODES_DIR.mkdir(parents=True, exist_ok=True)
    if electrodes_data:
        with open(ELECTRODES_DIR / f"{DANDISET_ID}.json", "w") as f:
            json.dump(electrodes_data, f)

    # 3. dandisets_with_electrodes.json
    dandisets_with_electrodes = [DANDISET_ID] if has_electrodes else []

    # 4. dandi_regions.json
    dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)

    return dandiset_assets, dandisets_with_electrodes, dandi_regions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build D99 atlas data")
    parser.add_argument("--skip-meshes", action="store_true", help="Skip mesh generation")
    parser.add_argument("--skip-dandi", action="store_true", help="Skip DANDI data extraction")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2a: Parse labels and build hierarchy
    print("Parsing D99 labels...")
    entries = parse_labels(D99_LABELS_FILE)
    print(f"  Parsed {len(entries)} label entries")

    tree, id_to_structure, parent_map, abbrev_to_id = build_structure_graph(entries)

    with open(DATA_DIR / "structure_graph.json", "w") as f:
        json.dump(tree, f)
    print("  Wrote structure_graph.json")

    # Step 2b: Generate meshes
    if args.skip_meshes:
        print("Skipping mesh generation")
        no_mesh = [OUTSIDE_ID] + [nid for nid in id_to_structure if nid >= CATEGORY_ID_START]
    else:
        print("Generating meshes from NIfTI volume...")
        no_mesh = generate_meshes(entries, id_to_structure)

    # Step 2c: DANDI data
    if args.skip_dandi:
        print("Skipping DANDI data extraction")
        dandiset_assets = {DANDISET_ID: []}
        dandisets_with_electrodes = []
        dandi_regions = {}
    else:
        dandiset_assets, dandisets_with_electrodes, dandi_regions = fetch_dandi_data(
            abbrev_to_id, id_to_structure, parent_map
        )

    # Write remaining outputs
    with open(DATA_DIR / "dandiset_assets.json", "w") as f:
        json.dump(dandiset_assets, f)

    with open(DATA_DIR / "dandisets_with_electrodes.json", "w") as f:
        json.dump(dandisets_with_electrodes, f)

    with open(DATA_DIR / "dandi_regions.json", "w") as f:
        json.dump(dandi_regions, f)

    # Mesh manifest
    data_ids = set(int(sid) for sid in dandi_regions.keys())
    ancestor_ids = set()
    for sid in data_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

    mesh_manifest = {
        "data_structures": sorted(data_ids),
        "ancestor_structures": sorted(ancestor_ids - data_ids),
        "no_mesh": sorted(set(no_mesh)),
        "root_id": ROOT_ID,
    }
    with open(DATA_DIR / "mesh_manifest.json", "w") as f:
        json.dump(mesh_manifest, f)

    print("Done! All D99 data written to", DATA_DIR)


if __name__ == "__main__":
    main()
