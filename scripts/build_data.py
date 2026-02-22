#!/usr/bin/env python3
"""Build static data for the brain atlas viewer.

Downloads Allen CCF structure graph, matches DANDI locations to CCF terms,
and downloads OBJ meshes for relevant structures.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MESHES_DIR = DATA_DIR / "meshes"

LABEL_RESULTS_PATH = Path(
    os.environ.get(
        "LABEL_RESULTS_PATH",
        os.path.expanduser(
            "~/dev/sandbox/analyze-locations/label_results_full.json"
        ),
    )
)

STRUCTURE_GRAPH_URL = (
    "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
)
MESH_URL_TEMPLATE = (
    "http://download.alleninstitute.org/informatics-archive/"
    "current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/{structure_id}.obj"
)


def download_json(url: str) -> dict:
    """Download and parse JSON from a URL."""
    print(f"  Downloading {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "brain-atlas-viewer/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest: Path) -> bool:
    """Download a file to disk. Returns True if successful."""
    if dest.exists():
        return True
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "brain-atlas-viewer/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return False


def flatten_structure_graph(msg: list) -> list:
    """Recursively flatten the Allen structure graph tree into a flat list."""
    result = []
    for node in msg:
        result.append(node)
        if node.get("children"):
            result.extend(flatten_structure_graph(node["children"]))
    return result


def build_parent_map(structures: list) -> dict:
    """Build a mapping from structure_id -> parent_structure_id."""
    return {s["id"]: s.get("parent_structure_id") for s in structures}


def get_ancestors(structure_id: int, parent_map: dict) -> list:
    """Get all ancestor structure IDs (excluding self)."""
    ancestors = []
    current = parent_map.get(structure_id)
    while current is not None:
        ancestors.append(current)
        current = parent_map.get(current)
    return ancestors


def main():
    MESHES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load label results (mouse-only, pre-matched to Allen CCF)
    print("Step 1: Loading label results...")
    with open(LABEL_RESULTS_PATH) as f:
        label_data = json.load(f)
    print(f"  {label_data['summary']['dandisets_processed']} dandisets, "
          f"{label_data['summary']['dandisets_skipped_species']} skipped (non-mouse)")

    # 2. Download Allen structure graph
    print("Step 2: Downloading Allen structure graph...")
    graph_data = download_json(STRUCTURE_GRAPH_URL)
    structures = flatten_structure_graph(graph_data["msg"])
    print(f"  Found {len(structures)} structures in graph")

    # Build lookups
    id_to_structure = {}
    for s in structures:
        id_to_structure[s["id"]] = s

    parent_map = build_parent_map(structures)

    # Save structure graph (the full tree for the hierarchy view)
    structure_graph_path = DATA_DIR / "structure_graph.json"
    with open(structure_graph_path, "w") as f:
        json.dump(graph_data["msg"], f)
    print(f"  Saved structure graph to {structure_graph_path}")

    # 3. Aggregate label results by structure
    print("Step 3: Aggregating structures from label results...")
    # structure_id -> {acronym, name, dandisets: set, file_count: int}
    region_data = {}

    for result in label_data["results"]:
        if result["status"] not in ("would_update", "updated", "no_change"):
            continue
        dandiset_id = result["dandiset_id"]
        for location, matches in result.get("matched_locations", {}).items():
            for match in matches:
                structure_id = match["id"]
                if structure_id not in id_to_structure:
                    continue
                if structure_id not in region_data:
                    s = id_to_structure[structure_id]
                    region_data[structure_id] = {
                        "acronym": s["acronym"],
                        "name": s["name"],
                        "color_hex_triplet": s.get("color_hex_triplet", "AAAAAA"),
                        "dandisets": set(),
                        "file_count": 0,
                    }
                region_data[structure_id]["dandisets"].add(dandiset_id)
                region_data[structure_id]["file_count"] += 1

    print(f"  Matched {len(region_data)} unique CCF structures with DANDI data (mouse only)")

    # 4. Compute aggregate stats (descendants included) for every ancestor
    print("Step 4: Computing aggregate stats including descendants...")

    # Build children map from structure graph
    children_map = {}  # id -> [child_ids]
    for s in structures:
        pid = s.get("parent_structure_id")
        if pid is not None:
            children_map.setdefault(pid, []).append(s["id"])

    # For each structure, recursively collect all descendant structure IDs
    def get_all_descendants(sid):
        result = set()
        stack = [sid]
        while stack:
            current = stack.pop()
            for child in children_map.get(current, []):
                result.add(child)
                stack.append(child)
        return result

    # Compute total (aggregate) dandisets and file counts for every node
    # that has data itself or has any descendant with data
    aggregate_data = {}  # structure_id -> {total_dandisets: set, total_file_count: int}

    # Start with direct data
    for sid, data in region_data.items():
        aggregate_data[sid] = {
            "total_dandisets": set(data["dandisets"]),
            "total_file_count": data["file_count"],
        }

    # Propagate upward: for each structure with data, add to all ancestors
    for sid in list(region_data.keys()):
        for anc_id in get_ancestors(sid, parent_map):
            if anc_id not in aggregate_data:
                aggregate_data[anc_id] = {
                    "total_dandisets": set(),
                    "total_file_count": 0,
                }
            aggregate_data[anc_id]["total_dandisets"].update(region_data[sid]["dandisets"])
            aggregate_data[anc_id]["total_file_count"] += region_data[sid]["file_count"]

    print(f"  {len(aggregate_data)} structures have data (direct or descendant)")

    # Convert to JSON-serializable output
    dandi_regions = {}
    for sid, agg in aggregate_data.items():
        s = id_to_structure[sid]
        direct = region_data.get(sid)
        dandi_regions[str(sid)] = {
            "acronym": s["acronym"],
            "name": s["name"],
            "color_hex_triplet": s.get("color_hex_triplet", "AAAAAA"),
            "file_count": direct["file_count"] if direct else 0,
            "dandiset_count": len(direct["dandisets"]) if direct else 0,
            "dandisets": sorted(direct["dandisets"]) if direct else [],
            "total_file_count": agg["total_file_count"],
            "total_dandiset_count": len(agg["total_dandisets"]),
            "total_dandisets": sorted(agg["total_dandisets"]),
        }

    dandi_regions_path = DATA_DIR / "dandi_regions.json"
    with open(dandi_regions_path, "w") as f:
        json.dump(dandi_regions, f, indent=2)
    print(f"  Saved dandi_regions.json ({len(dandi_regions)} structures)")

    # 5. Determine which meshes to download
    print("Step 5: Determining meshes to download...")
    data_structure_ids = set(int(sid) for sid in dandi_regions.keys())

    # Get ancestors for context meshes
    ancestor_ids = set()
    for sid in data_structure_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

    # Always include root brain outline (structure 997)
    all_mesh_ids = data_structure_ids | ancestor_ids
    all_mesh_ids.add(997)  # root brain outline

    print(
        f"  Data structures: {len(data_structure_ids)}, "
        f"Ancestors: {len(ancestor_ids)}, "
        f"Total meshes to download: {len(all_mesh_ids)}"
    )

    # 6. Download meshes
    print("Step 6: Downloading OBJ meshes...")
    downloaded = 0
    failed = 0
    skipped = 0
    for i, sid in enumerate(sorted(all_mesh_ids)):
        dest = MESHES_DIR / f"{sid}.obj"
        if dest.exists():
            skipped += 1
            continue

        url = MESH_URL_TEMPLATE.format(structure_id=sid)
        if download_file(url, dest):
            downloaded += 1
        else:
            failed += 1

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(
                f"  Progress: {i + 1}/{len(all_mesh_ids)} "
                f"(downloaded: {downloaded}, skipped: {skipped}, failed: {failed})"
            )

        # Be nice to the server
        if downloaded > 0 and downloaded % 10 == 0:
            time.sleep(0.5)

    print(
        f"  Done: {downloaded} downloaded, {skipped} already existed, {failed} failed"
    )

    # 7. Save a mesh manifest (so the frontend knows which meshes are available)
    mesh_manifest = {
        "data_structures": sorted(data_structure_ids),
        "ancestor_structures": sorted(ancestor_ids - data_structure_ids),
        "root_id": 997,
    }
    manifest_path = DATA_DIR / "mesh_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(mesh_manifest, f, indent=2)
    print(f"  Saved mesh manifest to {manifest_path}")

    print("\nBuild complete!")
    print(f"  Structures with DANDI data: {len(dandi_regions)}")
    print(f"  Total meshes available: {downloaded + skipped}")
    all_dandisets = set()
    for v in dandi_regions.values():
        all_dandisets.update(v["dandisets"])
    print(f"  Total unique dandisets: {len(all_dandisets)}")


if __name__ == "__main__":
    main()
