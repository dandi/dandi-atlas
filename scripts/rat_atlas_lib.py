#!/usr/bin/env python3
"""Shared library for the WHS-SD rat-atlas build pipeline.

Holds the rat-atlas config, ILF (MILF) hierarchy parser, and the rat-specific
DANDI fetch (DANDI 001699 NWB files store location strings in the standard
`general/extracellular_ephys/electrodes` table; no AnatomicalCoordinatesTable).

Reuses mesh generation and aggregation helpers from `macaque_atlas_lib` and
`dandi_helpers` — those are atlas-agnostic.

Run `scripts/build_rat_atlas.py` to invoke the pipeline.
"""

import json
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import remfile
from tqdm import tqdm

from dandi_helpers import (
    RAT_TAXON_IDS,
    build_dandi_regions,
    check_species_rat,
    extract_session,
    get_download_url,
    get_nwb_assets_paged,
    iter_all_dandisets,
)
from macaque_atlas_lib import (
    MIN_VOXELS,
    OUTSIDE_ID,
    ROOT_ID,
    ROOT_TARGET_FACES,
    TARGET_FACES,
    _normalize_region_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# WHS-SD source pack. The MBAT pack is the canonical source — it contains the
# parcellation NIfTI, the ILF hierarchy file the raw pack lacks, and the T2*
# template used for the root mesh. Atlas source files live under
# atlas_sources/<atlas_name>/ at the repo root by convention (gitignored).
# Override with the WHS_SD_DATA env var or the --whs-data CLI flag.
DEFAULT_WHS_DATA = PROJECT_ROOT / "atlas_sources" / "MBAT_WHS_SD_rat_atlas_v4_pack" / "Data"
WHS_DATA = Path(os.environ.get("WHS_SD_DATA", DEFAULT_WHS_DATA))

# Explicit list of rat dandisets to ingest. 001699 (Dudchenko postsubiculum) is
# embargoed and built via fetch_local_rat_data; everything else streams from
# DANDI via fetch_rat_dandi_data.
DANDISET_IDS = ["001699", "001754"]
EMBARGOED_DANDISETS = {"001699"}

# Alias map for free-text NWB location strings whose canonical WHS-SD region
# name differs from the upstream label. Keys are normalized via
# _normalize_region_name (lowercased, underscores → spaces). Values are
# canonical full names that must exist in name_to_id.
#
# Postsubiculum: WHS-SD v4 does not ship a separate postsubiculum parcel;
# modern rat neuroanatomy (Kleven et al. 2023, Boccara et al. 2010) folds the
# region historically labelled "postsubiculum" into the **Presubiculum**
# complex (the dorsal cap of PrS). Crediting Presubiculum here preserves the
# region attribution without inventing a separate parcel.
WHS_LOCATION_ALIASES = {
    "postsubiculum": "Presubiculum",
    "hippocampal area ca1": "Cornu ammonis 1",
}

ATLAS_CONFIGS = {
    "whs_sd": {
        "parcellation": WHS_DATA / "WHS_SD_rat_atlas_v4.nii.gz",
        "hierarchy": WHS_DATA / "WHS_SD_rat_atlas_v4_labels.ilf",
        "template_nifti": WHS_DATA / "WHS_SD_rat_T2star_v1.01.nii.gz",
        "output_dir": PROJECT_ROOT / "data/atlases/whs_sd",
        "cache_file": SCRIPTS_DIR / "whs_sd_electrode_cache.jsonl",
        "root_name": "WHS-SD v4 (Rat)",
    },
}


def parse_whs_ilf(hierarchy_file, root_name="WHS-SD v4 (Rat)"):
    """Parse the WHS-SD MILF (.ilf) XML hierarchy file.

    The file is an XML tree of nested <label> elements under a single
    <structure> wrapper. The WHS-SD v4 ILF has three top-level labels: the
    main "Brain" hierarchy (id=1000, with most regions), a standalone
    "spinal cord" leaf (id=45), and "Inner ear" (id=1049). All three are
    parented under a synthesized atlas root using ROOT_ID (same convention
    as macaque_atlas_lib, so `generate_meshes` writes the root mesh to
    f"{ROOT_ID}.glb").

    Group nodes (white matter, hippocampal white matter, etc.) live at
    id >= 1000 and have no voxels in the parcellation volume. Leaf nodes
    (ids 1-249) match the NIfTI parcellation labels.

    Returns (tree_root_list, id_to_structure, parent_map, abbrev_to_id,
    name_to_id) — same shape as macaque_atlas_lib.build_structure_graph.
    """
    tree = ET.parse(str(hierarchy_file))
    structure = tree.getroot().find("structure")
    if structure is None:
        raise ValueError(f"No <structure> element in {hierarchy_file}")
    top_labels = list(structure.findall("label"))
    if not top_labels:
        raise ValueError(f"No top-level <label> elements in {hierarchy_file}")

    id_to_structure = {}
    parent_map = {}
    abbrev_to_id = {}
    name_to_id = {}

    def _color_for(element):
        color = element.get("color", "#AAAAAA").lstrip("#").upper()
        if len(color) == 3:
            color = "".join(c * 2 for c in color)
        return color

    def _walk(element, parent_id):
        node_id = int(element.get("id"))
        name = element.get("name", "")
        abbreviation = element.get("abbreviation", "") or name
        node = {
            "id": node_id,
            "acronym": abbreviation,
            "name": name,
            "color_hex_triplet": _color_for(element),
            "parent_structure_id": parent_id,
            "children": [],
        }
        id_to_structure[node_id] = node
        parent_map[node_id] = parent_id

        if abbreviation and abbreviation not in abbrev_to_id:
            abbrev_to_id[abbreviation] = node_id
        abbrev_to_id.setdefault(str(node_id), node_id)
        norm_name = _normalize_region_name(name)
        if norm_name and norm_name not in name_to_id:
            name_to_id[norm_name] = node_id

        for child in element.findall("label"):
            child_node = _walk(child, node_id)
            node["children"].append(child_node)
        return node

    root_node = {
        "id": ROOT_ID,
        "acronym": "root",
        "name": root_name,
        "color_hex_triplet": "FFFFFF",
        "parent_structure_id": None,
        "children": [],
    }
    id_to_structure[ROOT_ID] = root_node
    parent_map[ROOT_ID] = None

    for top in top_labels:
        child_node = _walk(top, ROOT_ID)
        root_node["children"].append(child_node)

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
    root_node["children"].append(outside_node)

    return [root_node], id_to_structure, parent_map, abbrev_to_id, name_to_id


def generate_meshes_with_progress(nifti_file, meshes_dir, id_to_structure, template_nifti):
    """Generate GLB meshes from a NIfTI parcellation, with tqdm progress bars.

    Behaves like macaque_atlas_lib.generate_meshes but reports progress at each
    stage (loading NIfTI, root mesh from template, per-region leaf meshes,
    parent meshes by merged voxel masks). Returns the list of label IDs with
    no mesh on disk.

    Excludes the GIFTI-surface code path (rat atlas has none) and the
    macaque-specific synthetic category accounting (rat .ilf supplies its own
    group nodes at id >= 1000).
    """
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    import trimesh

    meshes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading parcellation: {nifti_file.name}")
    image = nib.load(str(nifti_file))
    affine = image.affine
    atlas_data = np.asarray(image.dataobj, dtype=np.int16)
    unique_labels = sorted(set(np.unique(atlas_data)) - {0})
    print(f"  {atlas_data.shape} volume, {len(unique_labels)} non-zero labels")

    no_mesh = []
    skipped_small = 0

    root_glb = meshes_dir / f"{ROOT_ID}.glb"
    if root_glb.exists():
        print(f"Root mesh already exists at {root_glb.name}; skipping")
    else:
        print("Generating root mesh from parcellation binary mask...")
        root_mask = atlas_data > 0
        smoothed = gaussian_filter(root_mask.astype(float), sigma=1.0)
        verts, faces, _, _ = marching_cubes(smoothed, level=0.5)
        verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
        verts_world = (affine @ verts_homogeneous.T).T[:, :3]
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        if len(mesh.faces) > ROOT_TARGET_FACES:
            mesh = mesh.simplify_quadric_decimation(face_count=ROOT_TARGET_FACES)
        _ = mesh.vertex_normals
        mesh.export(str(root_glb), file_type="glb")
        print(f"  Root mesh: {len(mesh.faces)} faces, saved to {root_glb.name}")

    leaf_bar = tqdm(unique_labels, desc="Leaf meshes", unit="region")
    for label_id in leaf_bar:
        glb_path = meshes_dir / f"{label_id}.glb"
        if glb_path.exists():
            continue
        mask = atlas_data == label_id
        voxel_count = int(mask.sum())
        leaf_bar.set_postfix(label=int(label_id), voxels=voxel_count)
        if voxel_count < MIN_VOXELS:
            skipped_small += 1
            no_mesh.append(int(label_id))
            continue
        smoothed = gaussian_filter(mask.astype(float), sigma=1.0)
        if smoothed.max() <= 0.5:
            smoothed = mask.astype(float)
        verts, faces, _, _ = marching_cubes(smoothed, level=0.5)
        verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
        verts_world = (affine @ verts_homogeneous.T).T[:, :3]
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        if len(mesh.faces) > TARGET_FACES:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        _ = mesh.vertex_normals
        mesh.export(str(glb_path), file_type="glb")

    leaf_label_set = set(unique_labels)

    children_map = {}
    for node_id, node in id_to_structure.items():
        parent_id = node.get("parent_structure_id")
        if parent_id is not None:
            children_map.setdefault(parent_id, []).append(node_id)

    parent_to_leaves = {}
    for node_id in id_to_structure:
        if node_id in (ROOT_ID, OUTSIDE_ID) or node_id in leaf_label_set:
            continue
        stack = list(children_map.get(node_id, []))
        leaves = []
        while stack:
            cid = stack.pop()
            if cid in leaf_label_set:
                leaves.append(cid)
            stack.extend(children_map.get(cid, []))
        if leaves:
            parent_to_leaves[node_id] = leaves

    parent_bar = tqdm(sorted(parent_to_leaves.items()), desc="Parent meshes", unit="region")
    for parent_id, leaf_ids in parent_bar:
        glb_path = meshes_dir / f"{parent_id}.glb"
        if glb_path.exists():
            continue
        merged_mask = np.zeros_like(atlas_data, dtype=bool)
        for leaf_id in leaf_ids:
            merged_mask |= (atlas_data == leaf_id)
        voxel_count = int(merged_mask.sum())
        parent_bar.set_postfix(parent=parent_id, leaves=len(leaf_ids), voxels=voxel_count)
        if voxel_count < MIN_VOXELS:
            skipped_small += 1
            no_mesh.append(parent_id)
            continue
        smoothed = gaussian_filter(merged_mask.astype(float), sigma=1.0)
        if smoothed.max() <= 0.5:
            smoothed = merged_mask.astype(float)
        verts, faces, _, _ = marching_cubes(smoothed, level=0.5)
        verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
        verts_world = (affine @ verts_homogeneous.T).T[:, :3]
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        if len(mesh.faces) > TARGET_FACES:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        _ = mesh.vertex_normals
        mesh.export(str(glb_path), file_type="glb")

    no_mesh.append(OUTSIDE_ID)
    print(f"Meshes: {len(unique_labels) - skipped_small} leaves + "
          f"{len(parent_to_leaves)} parents, {skipped_small} too small")
    return sorted(set(no_mesh))


def _resolve_location(location_string, name_to_id, abbrev_to_id, id_to_structure):
    """Match a free-text NWB location string against the WHS-SD vocabulary.

    Strategy:
      1. Strip leading hemisphere modifiers (`left`, `right`, `bilateral`) — the
         WHS-SD parcellation is bilateral and uses unsided names ("postsubiculum"
         not "left postsubiculum"). NWB convention is to prefix the hemisphere.
      2. Try full-name match (normalized) against name_to_id.
      3. Try abbreviation match against abbrev_to_id.

    Returns matched structure dict or None.
    """
    raw = location_string.strip()
    if not raw:
        return None
    candidates = [raw]
    lower = raw.lower()
    for prefix in ("left ", "right ", "bilateral "):
        if lower.startswith(prefix):
            candidates.append(raw[len(prefix):])
            break

    for cand in candidates:
        norm = _normalize_region_name(cand)
        if norm and norm in name_to_id:
            sid = name_to_id[norm]
            s = id_to_structure[sid]
            return {"id": sid, "acronym": s["acronym"], "name": s["name"]}
    for cand in candidates:
        cand_clean = cand.strip()
        if cand_clean and cand_clean in abbrev_to_id:
            sid = abbrev_to_id[cand_clean]
            s = id_to_structure[sid]
            return {"id": sid, "acronym": s["acronym"], "name": s["name"]}
    for cand in candidates:
        norm = _normalize_region_name(cand)
        if norm and norm in WHS_LOCATION_ALIASES:
            target_norm = _normalize_region_name(WHS_LOCATION_ALIASES[norm])
            if target_norm in name_to_id:
                sid = name_to_id[target_norm]
                s = id_to_structure[sid]
                return {"id": sid, "acronym": s["acronym"], "name": s["name"]}
    return None


def _extract_rat_location_strings(url):
    """Read electrode location strings from a rat NWB file via HTTP streaming.

    DANDI 001699 stores per-electrode anatomical locations in
    `general/extracellular_ephys/electrodes/location` as free-text strings
    (e.g. "left postsubiculum"). Returns a deduplicated list.
    """
    seen = set()
    out = []
    rf = remfile.File(url)
    with h5py.File(rf, "r") as f:
        if "general/extracellular_ephys/electrodes" not in f:
            return out
        electrodes = f["general/extracellular_ephys/electrodes"]
        if "location" not in electrodes:
            return out
        raw = electrodes["location"][()]
        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
            for value in raw:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                value = str(value).strip()
                if value and value not in seen:
                    seen.add(value)
                    out.append(value)
        else:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            value = str(raw).strip()
            if value:
                out.append(value)
    return out


def _load_cache(cache_file):
    """Load the per-asset location cache keyed by (dandiset_id, asset_id).

    Legacy entries without a `dandiset_id` field are from the single-dandiset
    era and are assumed to belong to 001699.
    """
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                dandiset_id = entry.get("dandiset_id", "001699")
                cache[(dandiset_id, entry["asset_id"])] = entry
    return cache


def _append_cache(cache_file, entry):
    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _select_local_nwb_files(local_dir):
    """Pick one NWB file per session from a local directory.

    Dudchenko-lab raw folder ships paired files per session: the canonical
    `<session>.nwb` and a `<session>_.nwb` Spyglass-pipeline artifact. We
    prefer the canonical (no trailing underscore) and ignore the artifact.
    Session key = basename without trailing `_`.
    """
    by_session = {}
    for path in sorted(Path(local_dir).glob("*.nwb")):
        stem = path.stem
        is_artifact = stem.endswith("_")
        session_key = stem[:-1] if is_artifact else stem
        existing = by_session.get(session_key)
        if existing is None:
            by_session[session_key] = path
            continue
        existing_is_artifact = existing.stem.endswith("_")
        if existing_is_artifact and not is_artifact:
            by_session[session_key] = path
    return [by_session[k] for k in sorted(by_session)]


def _extract_local_location_strings(file_path):
    """Read electrode location strings from a local NWB file (no streaming)."""
    seen = set()
    out = []
    with h5py.File(str(file_path), "r") as f:
        if "general/extracellular_ephys/electrodes" not in f:
            return out
        electrodes = f["general/extracellular_ephys/electrodes"]
        if "location" not in electrodes:
            return out
        raw = electrodes["location"][()]
        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
            iterable = raw
        else:
            iterable = [raw]
        for value in iterable:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            value = str(value).strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
    return out


def fetch_local_rat_data(local_dir, name_to_id, abbrev_to_id, id_to_structure, parent_map,
                        dandiset_id="001699"):
    """Build dandiset_assets/regions from local NWB files instead of DANDI.

    Useful while a dandiset is embargoed: the data is the same as what will
    eventually appear on DANDI, so we tag assets with the eventual dandiset_id
    and let the viewer wire things correctly once 001699 unembargoes.

    Asset IDs are synthesized from filename stems (deterministic, stable across
    runs). One asset per session is emitted (paired heavy/light files are
    deduplicated by _select_local_nwb_files).

    Returns (dandiset_assets, dandisets_with_electrodes, dandi_regions).
    """
    files = _select_local_nwb_files(local_dir)
    print(f"Reading {len(files)} unique sessions from {local_dir}")

    dandiset_assets = {dandiset_id: []}
    progress = tqdm(files, desc="Local NWB", unit="file")
    for file_path in progress:
        progress.set_postfix(file=file_path.name)
        try:
            locations = _extract_local_location_strings(file_path)
        except Exception as exc:
            print(f"  Error reading {file_path.name}: {exc}")
            continue
        if not locations:
            continue

        regions = []
        seen_ids = set()
        unmatched = []
        for loc in locations:
            match = _resolve_location(loc, name_to_id, abbrev_to_id, id_to_structure)
            if match is None:
                if loc not in unmatched:
                    unmatched.append(loc)
                continue
            if match["id"] in seen_ids:
                continue
            seen_ids.add(match["id"])
            regions.append(match)

        session_stem = file_path.stem.rstrip("_")
        dandiset_assets[dandiset_id].append({
            "path": file_path.name,
            "asset_id": f"local-{session_stem}",
            "regions": regions,
            "unmatched_locations": unmatched,
            "session": session_stem,
        })

    dandisets_with_electrodes = []
    dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)
    return dandiset_assets, dandisets_with_electrodes, dandi_regions


def _process_dandiset_assets(dandiset_id, assets, cache, cache_file,
                             name_to_id, abbrev_to_id, id_to_structure,
                             progress_position=None):
    """Stream + cache + resolve locations for every NWB asset in one dandiset.

    Shared between fetch_rat_dandi_data (explicit list) and
    fetch_rat_dandi_sweep (DANDI-wide discovery). Returns the asset-record
    list for this dandiset (matching the dandiset_assets[<id>] shape) plus
    a skip count for the caller to log.
    """
    results = {}
    to_process = []
    for asset in assets:
        asset_id = asset["asset_id"]
        cache_key = (dandiset_id, asset_id)
        if cache_key in cache:
            results[asset_id] = cache[cache_key]
        else:
            to_process.append(asset)

    def _process_one(asset):
        asset_id = asset["asset_id"]
        url = get_download_url(dandiset_id, asset_id)
        try:
            locations = _extract_rat_location_strings(url)
            return {"dandiset_id": dandiset_id, "asset_id": asset_id, "locations": locations}
        except Exception as exc:
            tqdm.write(f"  Error processing {asset['path']}: {exc}")
            return {"dandiset_id": dandiset_id, "asset_id": asset_id, "locations": [], "error": str(exc)}

    if to_process:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_one, a): a for a in to_process}
            inner_bar = tqdm(
                total=len(to_process),
                desc=f"  {dandiset_id}",
                unit="asset",
                leave=False,
                position=progress_position,
            )
            for future in as_completed(futures):
                result = future.result()
                results[result["asset_id"]] = result
                _append_cache(cache_file, result)
                inner_bar.update(1)
            inner_bar.close()

    records = []
    skipped_no_loc = 0
    for asset in assets:
        asset_id = asset["asset_id"]
        path = asset["path"]
        result = results.get(asset_id, {})
        locations = result.get("locations") or []
        if not locations:
            skipped_no_loc += 1
            continue

        regions = []
        seen_ids = set()
        unmatched = []
        for loc in locations:
            match = _resolve_location(loc, name_to_id, abbrev_to_id, id_to_structure)
            if match is None:
                if loc not in unmatched:
                    unmatched.append(loc)
                continue
            if match["id"] in seen_ids:
                continue
            seen_ids.add(match["id"])
            regions.append(match)

        records.append({
            "path": path,
            "asset_id": asset_id,
            "regions": regions,
            "unmatched_locations": unmatched,
            "session": extract_session(path),
        })

    return records, skipped_no_loc


def fetch_rat_dandi_data(config, name_to_id, abbrev_to_id, id_to_structure, parent_map,
                         dandiset_ids=None):
    """Stream location data from each rat dandiset and aggregate.

    For each dandiset in `dandiset_ids` (default: DANDISET_IDS), iterate its
    NWB assets, read `general/extracellular_ephys/electrodes/location` via
    `remfile` + h5py, and resolve each unique string against the WHS-SD
    vocabulary. Results are cached per (dandiset_id, asset_id) in
    `config["cache_file"]`.

    Electrode coordinate rendering is intentionally skipped: the rat NWB files
    encountered so far have no atlas-space x/y/z (only probe-relative coords),
    so `dandisets_with_electrodes` is always empty.

    Returns (dandiset_assets, dandisets_with_electrodes, dandi_regions).
    """
    if dandiset_ids is None:
        dandiset_ids = DANDISET_IDS

    cache_file = config["cache_file"]
    cache = _load_cache(cache_file)
    print(f"  Cache has {len(cache)} entries")

    dandiset_assets = {}
    outer_bar = tqdm(dandiset_ids, desc="Explicit rat dandisets", unit="dandiset")
    for dandiset_id in outer_bar:
        outer_bar.set_postfix(dandiset=dandiset_id)
        assets = list(get_nwb_assets_paged(dandiset_id))
        records, skipped_no_loc = _process_dandiset_assets(
            dandiset_id, assets, cache, cache_file,
            name_to_id, abbrev_to_id, id_to_structure,
        )
        dandiset_assets[dandiset_id] = records
        tqdm.write(
            f"  {dandiset_id}: {len(assets)} assets, "
            f"{len(records)} with location data"
            + (f" (skipped {skipped_no_loc} without locations)" if skipped_no_loc else "")
        )
    outer_bar.close()

    dandisets_with_electrodes = []
    dandi_regions = build_dandi_regions(dandiset_assets, id_to_structure, parent_map)
    return dandiset_assets, dandisets_with_electrodes, dandi_regions


def _is_rat_metadata(dandiset_metadata):
    """Cheap species check using the metadata dict yielded by
    iter_all_dandisets, falling back to a per-dandiset API call if the listing
    payload doesn't include species inline.

    Mirrors macaque_atlas_lib._is_macaque_metadata.
    """
    summary = (
        (dandiset_metadata.get("most_recent_published_version") or {})
        .get("metadata", {})
        .get("assetsSummary", {})
    )
    species = summary.get("species") if isinstance(summary, dict) else None
    if species:
        for sp in species:
            identifier = sp.get("identifier", "") or ""
            name = sp.get("name", "") or ""
            if any(tid in identifier for tid in RAT_TAXON_IDS):
                return True
            if "rattus" in name.lower():
                return True
        return False
    dandiset_id = dandiset_metadata.get("identifier")
    return bool(dandiset_id) and check_species_rat(dandiset_id)


def fetch_rat_dandi_sweep(config, name_to_id, abbrev_to_id, id_to_structure, parent_map,
                          exclude_ids=(), limit=None, max_assets_per_dandiset=1000):
    """Discover all public rat dandisets via iter_all_dandisets() and return
    per-asset region records resolved against the WHS-SD vocabulary.

    Streams only `general/extracellular_ephys/electrodes/location` per asset
    via `_extract_rat_location_strings` — no full-NWB downloads. Results are
    cached per (dandiset_id, asset_id) in `config["cache_file"]`, the same
    cache used by `fetch_rat_dandi_data`.

    Pre-filtering uses the inline `assetsSummary.species` metadata from the
    DANDI listing (via _is_rat_metadata), so the discovery pass costs ~one
    paginated API call total. Dandisets in `exclude_ids` (typically the
    explicit DANDISET_IDS the caller already handled) are skipped.

    Dandisets with more than `max_assets_per_dandiset` NWB assets are skipped
    entirely (just the cheap asset listing, never the streams). These tend to
    be auto-generated low-quality dumps (e.g. 001836's 2494 NWBs all carry
    location="None") and dominate sweep wall time. Pass None to disable.

    When `limit` is set, the discovery scan short-circuits as soon as `limit`
    rat dandisets have been found, and only those are streamed. Useful for
    smoke tests before committing to a full multi-hour sweep.

    Returns {dandiset_id: [asset_record, ...]} in the same shape as
    fetch_rat_dandi_data[0], suitable for `dict.update()` merging.
    """
    exclude_ids = set(exclude_ids)
    cache_file = config["cache_file"]
    cache = _load_cache(cache_file)
    print(f"  Cache has {len(cache)} entries")

    rat_dandiset_ids = []
    scan_bar = tqdm(iter_all_dandisets(), desc="Scanning DANDI", unit="dandiset")
    for dandiset_metadata in scan_bar:
        dandiset_id = dandiset_metadata.get("identifier")
        if not dandiset_id or dandiset_id in exclude_ids:
            continue
        if _is_rat_metadata(dandiset_metadata):
            rat_dandiset_ids.append(dandiset_id)
            scan_bar.set_postfix(rat_found=len(rat_dandiset_ids))
            if limit is not None and len(rat_dandiset_ids) >= limit:
                break
    scan_bar.close()
    if limit is not None:
        print(f"  [rat-sweep] discovered {len(rat_dandiset_ids)} rat dandisets "
              f"(stopped early; --sweep-limit={limit})")
    else:
        print(f"  [rat-sweep] discovered {len(rat_dandiset_ids)} candidate rat dandisets")

    out = {}
    outer_bar = tqdm(rat_dandiset_ids, desc="Rat sweep", unit="dandiset")
    for dandiset_id in outer_bar:
        outer_bar.set_postfix(dandiset=dandiset_id)
        try:
            assets = list(get_nwb_assets_paged(dandiset_id))
        except Exception as exc:
            tqdm.write(f"  [rat-sweep] failed to list {dandiset_id}: {exc}")
            continue
        if not assets:
            continue
        if max_assets_per_dandiset is not None and len(assets) > max_assets_per_dandiset:
            tqdm.write(
                f"  [rat-sweep] {dandiset_id}: skipping ({len(assets)} assets > "
                f"max {max_assets_per_dandiset}); raise --sweep-max-assets to include."
            )
            continue
        records, skipped_no_loc = _process_dandiset_assets(
            dandiset_id, assets, cache, cache_file,
            name_to_id, abbrev_to_id, id_to_structure,
        )
        if not records:
            continue
        if not any(record["regions"] for record in records):
            continue
        out[dandiset_id] = records
        tqdm.write(
            f"  [rat-sweep] {dandiset_id}: {len(assets)} assets, "
            f"{len(records)} with location data"
            + (f" (skipped {skipped_no_loc} without locations)" if skipped_no_loc else "")
        )
    outer_bar.close()

    return out
