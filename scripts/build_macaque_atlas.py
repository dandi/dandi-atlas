#!/usr/bin/env python3
"""Generate macaque atlas data for the dandi-atlas viewer.

Supports three macaque brain atlases that share the same DANDI 001636 dataset:
  - D99 v2.0 (Saleem & Logothetis)
  - NMT v2.0 sym (D99 labels warped into NMT space)
  - MEBRAINS (EBRAINS macaque parcellation)

Produces (per atlas):
  data/atlases/{atlas}/structure_graph.json
  data/atlases/{atlas}/meshes/*.glb
  data/atlases/{atlas}/dandiset_assets.json
  data/atlases/{atlas}/electrodes/001636.json
  data/atlases/{atlas}/dandisets_with_electrodes.json
  data/atlases/{atlas}/dandi_regions.json
  data/atlases/{atlas}/mesh_manifest.json

Usage:
    uv run python scripts/build_macaque_atlas.py --atlas d99 [--skip-meshes] [--skip-dandi]
    uv run python scripts/build_macaque_atlas.py --atlas nmt [--skip-meshes] [--skip-dandi]
    uv run python scripts/build_macaque_atlas.py --atlas mebrains [--skip-meshes] [--skip-dandi]
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
    extract_session,
    extract_subject,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

TURNER_DATA = Path(
    "/home/heberto/development/conversions/turner-lab-to-nwb/data"
)

D99_LABELS_FILE = TURNER_DATA / "d99_atlas/D99_v2.0_dist/D99_v2.0_labels_semicolon.txt"
NMT_LABELS_FILE = TURNER_DATA / "nmt_v2/NMT_v2.0_sym/tables_D99/D99_labeltable.txt"
MEBRAINS_LABELS_FILE = TURNER_DATA / "mebrains/MEBRAINS_labels.json"

ATLAS_CONFIGS = {
    "d99": {
        "nifti": TURNER_DATA / "d99_atlas/D99_v2.0_dist/D99_atlas_v2.0.nii.gz",
        "labels_type": "d99",
        "hdf5_path": "general/localization/D99AtlasCoordinates",
        "output_dir": PROJECT_ROOT / "data/atlases/d99",
        "cache_file": SCRIPTS_DIR / "d99_electrode_cache.jsonl",
        "root_name": "D99 Atlas",
    },
    "nmt": {
        "nifti": TURNER_DATA / "nmt_v2/NMT_v2.0_sym/NMT_v2.0_sym/D99_atlas_in_NMT_v2.0_sym.nii.gz",
        "labels_type": "nmt",
        "hdf5_path": "general/localization/NMTv2symAtlasCoordinates",
        "output_dir": PROJECT_ROOT / "data/atlases/nmt",
        "cache_file": SCRIPTS_DIR / "nmt_electrode_cache.jsonl",
        "root_name": "NMT v2.0 sym Atlas",
    },
    "mebrains": {
        "nifti": TURNER_DATA / "mebrains/MEBRAINS_parcellation.nii.gz",
        "labels_type": "mebrains",
        "hdf5_path": "general/localization/MEBRAINSAtlasCoordinates",
        "output_dir": PROJECT_ROOT / "data/atlases/mebrains",
        "cache_file": SCRIPTS_DIR / "mebrains_electrode_cache.jsonl",
        "root_name": "MEBRAINS Atlas",
    },
}

DANDISET_ID = "001636"
ROOT_ID = 9999
OUTSIDE_ID = 9998
CATEGORY_ID_START = 10001
SUBCATEGORY_ID_START = 10100
TARGET_FACES = 10_000
MIN_VOXELS = 50

# Category color hues (HSL hue in degrees), covering both D99 and MEBRAINS
CATEGORY_HUES = {
    # D99 categories
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
    # MEBRAINS categories
    "Motor cortex": 0,
    "Prefrontal cortex": 30,
    "Parietal cortex": 60,
    "Visual cortex": 120,
    "Temporal cortex": 180,
    "Claustrum": 300,
    "White matter": 150,
    "Ventricle": 200,
    "Other": 150,
}


# ---------------------------------------------------------------------------
# D99 label parsing (reused by NMT)
# ---------------------------------------------------------------------------


def parse_d99_labels():
    """Parse D99 semicolon-delimited labels file.

    Returns list of dicts with keys: index, abbreviation, name, category, subcategory.
    """
    entries = []
    with open(D99_LABELS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reader = csv.reader(io.StringIO(line), delimiter=";")
            fields = next(reader)
            fields = [field.strip() for field in fields]

            label_index = int(fields[0])
            abbreviation = fields[1].strip()
            name = fields[2].strip()

            category = fields[3].strip() if len(fields) > 3 else ""
            subcategory = fields[4].strip() if len(fields) > 4 else ""

            if not category:
                name_lower = name.lower()
                if any(
                    kw in name_lower
                    for kw in [
                        "cortex", "cortical", "area", "gyrus", "sulcus",
                        "opercul", "prefrontal", "parietal", "temporal",
                        "visual", "auditory", "somato", "insula", "cingulate",
                        "retrosplenial", "parahippocampal", "perirhinal",
                        "entorhinal", "precentral", "frontal area",
                        "belt region", "core region",
                    ]
                ):
                    category = "Cortex"
                elif any(kw in name_lower for kw in ["amygdal", "periamygdal"]):
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

            entries.append({
                "index": label_index,
                "abbreviation": abbreviation,
                "name": name,
                "category": category,
                "subcategory": subcategory,
            })
    return entries


# ---------------------------------------------------------------------------
# NMT label parsing (D99 labels in NMT numbering)
# ---------------------------------------------------------------------------


def parse_nmt_labels():
    """Parse NMT D99 label table and enrich with D99 metadata.

    The NMT volume uses its own label IDs (2-224) that differ from D99's (1-522).
    This function reads the NMT label table, then looks up each abbreviation in
    the D99 labels file for name, category, and subcategory metadata.

    Returns list of dicts with keys: index, abbreviation, name, category,
    subcategory, d99_abbreviation (the original D99 abbreviation for alias mapping).
    """
    # Build D99 abbreviation -> metadata lookup
    d99_entries = parse_d99_labels()
    d99_by_abbrev = {e["abbreviation"]: e for e in d99_entries}

    # Also build a normalized lookup for fuzzy matching
    d99_by_norm = {}
    for e in d99_entries:
        # Normalize: lowercase, strip "area_" prefix, remove special chars
        norm = e["abbreviation"].lower().replace("area_", "").replace("'", "").replace("?", "")
        d99_by_norm[norm] = e

    # Read NMT label table: "  NMT_ID  abbreviation"
    entries = []
    with open(NMT_LABELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            nmt_id = int(parts[0])
            abbrev = parts[1].strip()

            # Try exact match first
            d99_entry = d99_by_abbrev.get(abbrev)

            # Try normalized fuzzy match
            if d99_entry is None:
                norm = abbrev.lower().replace("area_", "").replace("'", "").replace("?", "")
                d99_entry = d99_by_norm.get(norm)

            if d99_entry is not None:
                # Store the D99 abbreviation so we can add it as an alias
                d99_abbrev = d99_entry["abbreviation"]
                entries.append({
                    "index": nmt_id,
                    "abbreviation": abbrev,
                    "name": d99_entry["name"],
                    "category": d99_entry["category"],
                    "subcategory": d99_entry["subcategory"],
                    "d99_abbreviation": d99_abbrev,
                })
            else:
                # No D99 match: use abbreviation as name, infer category
                name_lower = abbrev.lower()
                if "cerebellum" in name_lower:
                    category = "Cerebellum"
                else:
                    category = "Other"
                entries.append({
                    "index": nmt_id,
                    "abbreviation": abbrev,
                    "name": abbrev,
                    "category": category,
                    "subcategory": "",
                    "d99_abbreviation": None,
                })

    return entries


# ---------------------------------------------------------------------------
# MEBRAINS label parsing
# ---------------------------------------------------------------------------


def _categorize_mebrains(base_name):
    """Infer category for a MEBRAINS region from its base name."""
    # Motor cortex (area 4 variants and frontal motor areas F2-F7)
    if base_name in ("4a", "4p", "4m"):
        return "Motor cortex"
    if base_name.startswith(("F2", "F3", "F4", "F5", "F6", "F7")):
        return "Motor cortex"
    # Prefrontal cortex (check specific names before digit-based catch-all)
    if base_name in ("44", "45A", "45B"):
        return "Prefrontal cortex"
    if base_name.startswith(("a46", "p46")):
        return "Prefrontal cortex"
    # Remaining digit-prefixed names: 8xx, 9x, 10x, 11x, 12x, 13x, 14x
    if base_name[:1].isdigit():
        return "Prefrontal cortex"
    # Parietal cortex (check before visual since VIP starts with V)
    if base_name.startswith(("PE", "PF", "PG")):
        return "Parietal cortex"
    if base_name in ("AIP", "DP", "LOP", "PIP"):
        return "Parietal cortex"
    if base_name.startswith(("VIP", "LIP", "MIP")):
        return "Parietal cortex"
    # Visual cortex
    if base_name.startswith("V") or base_name == "Opt":
        return "Visual cortex"
    # Temporal cortex
    if base_name == "TSA":
        return "Temporal cortex"
    # Subcortical
    if base_name in ("caudate nucleus", "putamen", "globus pallidus", "nucleus accumbens"):
        return "Basal ganglia"
    if base_name == "amygdala":
        return "Amygdala"
    if base_name == "claustrum":
        return "Claustrum"
    if base_name == "anterior commissure":
        return "White matter"
    if base_name == "lateral ventricle":
        return "Ventricle"
    return "Other"


def parse_mebrains_labels():
    """Parse MEBRAINS JSON labels file.

    Returns list of dicts with keys: index, abbreviation, name, category, subcategory.
    """
    with open(MEBRAINS_LABELS_FILE) as f:
        data = json.load(f)

    entries = []
    for label_id_str, name in data.items():
        label_id = int(label_id_str)

        if name.endswith(" left"):
            hemisphere = "L"
            base_name = name[:-5]
        elif name.endswith(" right"):
            hemisphere = "R"
            base_name = name[:-6]
        else:
            hemisphere = ""
            base_name = name

        category = _categorize_mebrains(base_name)

        # Abbreviation: base_name with hemisphere suffix for uniqueness
        if hemisphere:
            abbreviation = f"{base_name}_{hemisphere}"
        else:
            abbreviation = base_name

        entries.append({
            "index": label_id,
            "abbreviation": abbreviation,
            "name": name,
            "category": category,
            "subcategory": "",
        })
    return entries


# ---------------------------------------------------------------------------
# Structure graph building (shared by all atlases)
# ---------------------------------------------------------------------------


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


def build_structure_graph(entries, root_name="D99 Atlas"):
    """Build a hierarchical structure graph from parsed label entries.

    Returns (tree_root, id_to_structure, parent_map, abbrev_to_id).
    """
    # Assign colors by category with brightness variation
    category_counts = {}
    for entry in entries:
        cat = entry["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    entry_colors = {}
    category_index = {}
    for entry in entries:
        cat = entry["category"]
        current = category_index.get(cat, 0)
        category_index[cat] = current + 1
        hue = CATEGORY_HUES.get(cat, 150)
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

    root = {
        "id": ROOT_ID,
        "acronym": "root",
        "name": root_name,
        "color_hex_triplet": "FFFFFF",
        "parent_structure_id": None,
        "children": [],
    }
    id_to_structure[ROOT_ID] = root
    parent_map[ROOT_ID] = None

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

    abbrev_to_id = {}
    for entry in entries:
        label_id = entry["index"]
        cat = entry["category"]
        subcat = entry["subcategory"]

        if subcat:
            primary_subcat = subcat.split("_")[0]
            parent_id = subcategory_ids.get(
                (cat, primary_subcat), category_ids.get(cat, ROOT_ID)
            )
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
        # For NMT entries, also add the D99 abbreviation as an alias so that
        # brain_region_id values from NWB files (which use D99 abbreviations)
        # resolve to the correct NMT structure IDs.
        d99_abbrev = entry.get("d99_abbreviation")
        if d99_abbrev and d99_abbrev != entry["abbreviation"]:
            abbrev_to_id[d99_abbrev] = label_id
        # Add string label ID as alias (e.g., "303" -> 303) so that
        # brain_region_id values stored as string IDs (MEBRAINS) resolve correctly.
        str_id = str(label_id)
        if str_id not in abbrev_to_id:
            abbrev_to_id[str_id] = label_id

    # Build tree by nesting children
    for node_id, node in id_to_structure.items():
        pid = node.get("parent_structure_id")
        if pid is not None and pid in id_to_structure:
            id_to_structure[pid]["children"].append(node)

    return [root], id_to_structure, parent_map, abbrev_to_id


# ---------------------------------------------------------------------------
# Mesh generation from NIfTI
# ---------------------------------------------------------------------------


def generate_meshes(nifti_file, meshes_dir, id_to_structure):
    """Generate GLB meshes from a NIfTI atlas volume.

    Returns list of label IDs that have no mesh.
    """
    import nibabel as nib
    from skimage.measure import marching_cubes
    import trimesh

    meshes_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(nifti_file))
    affine = img.affine
    atlas_data = np.asarray(img.dataobj, dtype=np.int16)

    unique_labels = set(np.unique(atlas_data)) - {0}
    print(f"Found {len(unique_labels)} non-zero labels in NIfTI volume")

    no_mesh = []
    generated = 0
    skipped_existing = 0
    skipped_small = 0

    # Root mesh (union of all non-zero)
    root_glb = meshes_dir / f"{ROOT_ID}.glb"
    if not root_glb.exists():
        print("Generating root mesh (whole brain)...")
        root_mask = atlas_data > 0
        verts, faces, _, _ = marching_cubes(root_mask, level=0.5)
        verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
        verts_world = (affine @ verts_homogeneous.T).T[:, :3]
        verts_world[:, 0] *= -1
        faces = faces[:, ::-1]
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        if len(mesh.faces) > TARGET_FACES:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        mesh.export(str(root_glb), file_type="glb")
        print(f"  Root mesh: {len(mesh.faces)} faces")
        generated += 1
    else:
        skipped_existing += 1

    # Per-region meshes
    for label_id in sorted(unique_labels):
        glb_path = meshes_dir / f"{label_id}.glb"
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
# Coordinate extraction from NWB files
# ---------------------------------------------------------------------------


def _read_hdf5_strings(group, key):
    """Read a string column from an HDF5 group, returning a list of strings."""
    if key not in group:
        return []
    raw = group[key][()]
    if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw]
    if isinstance(raw, bytes):
        return [raw.decode("utf-8")]
    return [str(raw)]


def extract_atlas_coords(url, asset_id, hdf5_path):
    """Extract atlas coordinates from the localization extension in an NWB file.

    Returns dict with keys: asset_id, coords, brain_region, brain_region_id,
    raw_coords (original coords before X negation), or None if not found.
    """
    import h5py
    import remfile

    rf = remfile.File(url)
    with h5py.File(rf, "r") as f:
        if hdf5_path not in f:
            return None

        t = f[hdf5_path]
        if not all(col in t for col in ("x", "y", "z")):
            return None

        x = t["x"][()]
        y = t["y"][()]
        z = t["z"][()]

        brain_region = _read_hdf5_strings(t, "brain_region")
        brain_region_id = _read_hdf5_strings(t, "brain_region_id")

    coords = []
    raw_coords = []
    for xi, yi, zi in zip(x, y, z):
        xi, yi, zi = float(xi), float(yi), float(zi)
        if xi != xi or yi != yi or zi != zi:
            continue
        raw_coords.append([round(xi, 3), round(yi, 3), round(zi, 3)])
        coords.append([round(-xi, 3), round(yi, 3), round(zi, 3)])

    if not coords:
        return None

    return {
        "asset_id": asset_id,
        "coords": coords,
        "raw_coords": raw_coords,
        "brain_region": brain_region,
        "brain_region_id": brain_region_id,
    }


def lookup_voxel_regions(raw_coords, nifti_data, inv_affine):
    """Look up NIfTI label IDs for a list of world-space coordinates.

    Returns list of integer label IDs (0 = outside volume).
    """
    labels = []
    for coord in raw_coords:
        voxel = inv_affine @ [coord[0], coord[1], coord[2], 1.0]
        vi, vj, vk = int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
        if (
            0 <= vi < nifti_data.shape[0]
            and 0 <= vj < nifti_data.shape[1]
            and 0 <= vk < nifti_data.shape[2]
        ):
            labels.append(int(nifti_data[vi, vj, vk]))
        else:
            labels.append(0)
    return labels


# ---------------------------------------------------------------------------
# DANDI data extraction
# ---------------------------------------------------------------------------


def _load_cache(cache_file):
    """Load cached electrode data from JSONL file."""
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["asset_id"]] = entry
    return cache


def _append_cache(cache_file, entry):
    """Append a single entry to the cache file."""
    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _map_regions_by_abbreviation(brain_region_ids, abbrev_to_id, id_to_structure):
    """Map D99 abbreviations to structure graph regions."""
    regions = []
    for abbrev in brain_region_ids:
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
    return regions


def _map_regions_by_voxel(raw_coords, nifti_data, inv_affine, id_to_structure):
    """Map electrode coordinates to structure graph regions via voxel lookup."""
    labels = lookup_voxel_regions(raw_coords, nifti_data, inv_affine)
    regions = []
    seen = set()
    for label_id in labels:
        if label_id in seen:
            continue
        seen.add(label_id)
        if label_id == 0:
            regions.append({
                "id": OUTSIDE_ID,
                "acronym": "outside",
                "name": "Outside atlas",
            })
        elif label_id in id_to_structure:
            s = id_to_structure[label_id]
            regions.append({
                "id": label_id,
                "acronym": s["acronym"],
                "name": s["name"],
            })
    return regions


def fetch_dandi_data(
    config, abbrev_to_id, id_to_structure, parent_map,
    nifti_data=None, inv_affine=None,
):
    """Fetch electrode data from DANDI 001636 and build all output files.

    For D99/NMT atlases, uses brain_region_id abbreviations for region mapping.
    For MEBRAINS, uses voxel lookup (nifti_data + inv_affine must be provided).
    """
    hdf5_path = config["hdf5_path"]
    cache_file = config["cache_file"]
    data_dir = config["output_dir"]
    electrodes_dir = data_dir / "electrodes"
    use_voxel_lookup = nifti_data is not None

    print(f"Fetching assets from dandiset {DANDISET_ID}...")
    assets = list(get_nwb_assets_paged(DANDISET_ID))
    print(f"  Found {len(assets)} NWB assets")

    cache = _load_cache(cache_file)
    print(f"  Cache has {len(cache)} entries")

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
            result = extract_atlas_coords(url, asset_id, hdf5_path)
            if result is None:
                result = {
                    "asset_id": asset_id,
                    "coords": None,
                    "raw_coords": None,
                    "brain_region": [],
                    "brain_region_id": [],
                }
            return result
        except Exception as exc:
            print(f"  Error processing {asset['path']}: {exc}")
            return {
                "asset_id": asset_id,
                "coords": None,
                "raw_coords": None,
                "brain_region": [],
                "brain_region_id": [],
                "error": str(exc),
            }

    if to_process:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_one, a): a for a in to_process}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                asset_id = result["asset_id"]
                results[asset_id] = result
                _append_cache(cache_file, result)
                done += 1
                if done % 50 == 0:
                    print(f"  Processed {done}/{len(to_process)} assets")

    # Build output data
    dandiset_assets = {DANDISET_ID: []}
    electrodes_data = {}
    has_electrodes = False

    skipped_no_loc = 0
    for asset in assets:
        asset_id = asset["asset_id"]
        path = asset["path"]
        result = results.get(asset_id, {})

        if result.get("coords") is None and not result.get("brain_region_id"):
            skipped_no_loc += 1
            continue

        # Map electrodes to regions
        if use_voxel_lookup and result.get("raw_coords"):
            regions = _map_regions_by_voxel(
                result["raw_coords"], nifti_data, inv_affine, id_to_structure,
            )
        else:
            regions = _map_regions_by_abbreviation(
                result.get("brain_region_id", []), abbrev_to_id, id_to_structure,
            )

        session = extract_session(path)

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
        print(f"  Skipped {skipped_no_loc} assets without localization")

    electrodes_dir.mkdir(parents=True, exist_ok=True)
    if electrodes_data:
        with open(electrodes_dir / f"{DANDISET_ID}.json", "w") as f:
            json.dump(electrodes_data, f)

    dandisets_with_electrodes = [DANDISET_ID] if has_electrodes else []
    dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)

    return dandiset_assets, dandisets_with_electrodes, dandi_regions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    if config["labels_type"] == "d99":
        entries = parse_d99_labels()
        print(f"  Parsed {len(entries)} D99 label entries")
    elif config["labels_type"] == "nmt":
        entries = parse_nmt_labels()
        print(f"  Parsed {len(entries)} NMT label entries")
    else:
        entries = parse_mebrains_labels()
        print(f"  Parsed {len(entries)} MEBRAINS label entries")

    tree, id_to_structure, parent_map, abbrev_to_id = build_structure_graph(
        entries, root_name=config["root_name"],
    )

    with open(data_dir / "structure_graph.json", "w") as f:
        json.dump(tree, f)
    print("  Wrote structure_graph.json")

    # Generate meshes
    if args.skip_meshes:
        print("Skipping mesh generation")
        no_mesh = [OUTSIDE_ID] + [nid for nid in id_to_structure if nid >= CATEGORY_ID_START]
    else:
        print("Generating meshes from NIfTI volume...")
        no_mesh = generate_meshes(config["nifti"], meshes_dir, id_to_structure)

    # DANDI data
    if args.skip_dandi:
        print("Skipping DANDI data extraction")
        dandiset_assets = {DANDISET_ID: []}
        dandisets_with_electrodes = []
        dandi_regions = {}
    else:
        # For MEBRAINS, load NIfTI for voxel-based region lookup
        nifti_data = None
        inv_affine = None
        if config["labels_type"] == "mebrains":
            import nibabel as nib
            print("Loading MEBRAINS NIfTI for voxel lookup...")
            img = nib.load(str(config["nifti"]))
            nifti_data = np.asarray(img.dataobj, dtype=np.int16)
            inv_affine = np.linalg.inv(img.affine)

        dandiset_assets, dandisets_with_electrodes, dandi_regions = fetch_dandi_data(
            config, abbrev_to_id, id_to_structure, parent_map,
            nifti_data=nifti_data, inv_affine=inv_affine,
        )

    # Write outputs
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

    mesh_manifest = {
        "data_structures": sorted(data_ids),
        "ancestor_structures": sorted(ancestor_ids - data_ids),
        "no_mesh": sorted(set(no_mesh)),
        "root_id": ROOT_ID,
    }
    with open(data_dir / "mesh_manifest.json", "w") as f:
        json.dump(mesh_manifest, f)

    print(f"Done! All {args.atlas} data written to {data_dir}")


if __name__ == "__main__":
    main()
