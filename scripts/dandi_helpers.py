"""DANDI API, NWB streaming, and Allen CCF helper functions.

Internalized from scan_locations.py, label_anatomy.py, extract_electrodes.py,
and build_data.py for self-contained CI usage.
"""

import ast
import json
import re
import time
import urllib.request

import h5py
import remfile
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DANDI_API = "https://api.dandiarchive.org/api"

ALLEN_STRUCTURE_GRAPH_URL = (
    "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
)
MESH_URL_TEMPLATE = (
    "http://download.alleninstitute.org/informatics-archive/"
    "current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/{structure_id}.obj"
)

TRIVIAL_LOCATIONS = {
    "unknown", "none", "", " ", "n/a", "void", "unspecific",
    "na", "not applicable", "other", "nan",
}

FILTER_IDS = {997, 8}  # root, grey — not useful to display

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, multiplied by attempt number

# Allen CCF bounds (micrometers) for electrode validation
ALLEN_X_MAX = 13200
ALLEN_Y_MAX = 8000
ALLEN_Z_MAX = 11400


# ---------------------------------------------------------------------------
# DANDI API helpers
# ---------------------------------------------------------------------------


def _request_with_retry(method, url, **kwargs):
    """Make an HTTP request with retry on 429 and 5xx errors."""
    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = method(url, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * attempt
                    tqdm.write(f"  Retry {attempt}/{MAX_RETRIES} after {resp.status_code}, waiting {wait}s")
                    time.sleep(wait)
                    continue
            return resp
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                tqdm.write(f"  Retry {attempt}/{MAX_RETRIES} after error: {exc}, waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    return resp


def check_species_mouse(dandiset_id):
    """Check if a dandiset's assetsSummary.species includes Mus musculus."""
    url = f"{DANDI_API}/dandisets/{dandiset_id}/versions/draft/"
    resp = _request_with_retry(requests.get, url, timeout=30)
    if resp.status_code != 200:
        return False
    data = resp.json()
    assets_summary = data.get("metadata", data).get("assetsSummary", {})
    species = assets_summary.get("species", [])
    for sp in species:
        identifier = sp.get("identifier", "")
        if "NCBITaxon_10090" in identifier or "10090" in identifier:
            return True
    return False


def get_nwb_assets_paged(dandiset_id, version="draft", max_assets=None):
    """Yield NWB asset dicts for a dandiset, with pagination."""
    url = (
        f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}"
        f"/assets/?page_size=100&glob=*.nwb"
    )
    count = 0
    while url:
        resp = _request_with_retry(requests.get, url, timeout=30)
        if resp.status_code != 200:
            break
        data = resp.json()
        for asset in data["results"]:
            if asset["path"].endswith(".nwb"):
                yield asset
                count += 1
                if max_assets and count >= max_assets:
                    return
        url = data.get("next")


def get_download_url(dandiset_id, asset_id, version="draft"):
    """Build the DANDI download URL for an asset."""
    return (
        f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}"
        f"/assets/{asset_id}/download/"
    )


def iter_dandisets_modified_since(since_iso):
    """Yield dandiset metadata dicts modified after the given ISO timestamp.

    Iterates the DANDI API ordered by most-recently-modified, stopping
    once a dandiset older than the cutoff is encountered.
    """
    url = f"{DANDI_API}/dandisets/?page_size=200&ordering=-modified"
    while url:
        resp = _request_with_retry(requests.get, url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for ds in data["results"]:
            modified = ds.get("modified", "")
            if modified <= since_iso:
                return
            yield ds
        url = data.get("next")


def iter_all_dandisets():
    """Yield all dandiset metadata dicts from the DANDI API."""
    url = f"{DANDI_API}/dandisets/?page_size=200&ordering=-modified"
    while url:
        resp = _request_with_retry(requests.get, url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        yield from data["results"]
        url = data.get("next")


# ---------------------------------------------------------------------------
# NWB streaming: location extraction
# ---------------------------------------------------------------------------


def _read_scalar_or_array(dataset):
    """Read an HDF5 dataset and return a list of decoded strings."""
    raw = dataset[()]
    if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        return [
            v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            for v in raw
        ]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return [raw]


def extract_locations(url):
    """Open an NWB file via HTTP streaming and return location values.

    Returns (imaging_locations, electrode_locations, icephys_locations).
    """
    imaging_locations = []
    electrode_locations = []
    icephys_locations = []

    rf = remfile.File(url)
    with h5py.File(rf, "r") as f:
        # ImagingPlane objects
        if "general/optophysiology" in f:
            opto = f["general/optophysiology"]
            for name in opto:
                plane = opto[name]
                if isinstance(plane, h5py.Group) and "location" in plane:
                    val = plane["location"][()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    imaging_locations.append(val)

        # Extracellular electrodes table
        if "general/extracellular_ephys/electrodes" in f:
            electrodes = f["general/extracellular_ephys/electrodes"]
            if "location" in electrodes:
                electrode_locations = _read_scalar_or_array(electrodes["location"])

        # IntracellularElectrode objects
        if "general/intracellular_ephys" in f:
            icephys = f["general/intracellular_ephys"]
            for name in icephys:
                item = icephys[name]
                if isinstance(item, h5py.Group) and "location" in item:
                    val = item["location"][()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    icephys_locations.append(val)

    return imaging_locations, electrode_locations, icephys_locations


def extract_electrode_coords(url):
    """Open an NWB file via HTTP streaming and return electrode x,y,z coords.

    Returns list of [x, y, z] or None if no electrode coordinates found.
    """
    rf = remfile.File(url)
    with h5py.File(rf, "r") as f:
        if "general/extracellular_ephys/electrodes" not in f:
            return None
        electrodes = f["general/extracellular_ephys/electrodes"]
        if not all(col in electrodes for col in ("x", "y", "z")):
            return None

        x = electrodes["x"][()]
        y = electrodes["y"][()]
        z = electrodes["z"][()]

    coords = []
    for xi, yi, zi in zip(x, y, z):
        xi, yi, zi = float(xi), float(yi), float(zi)
        # Skip NaN values
        if xi != xi or yi != yi or zi != zi:
            continue
        coords.append([round(xi, 1), round(yi, 1), round(zi, 1)])

    if not coords:
        return None

    # Filter out non-atlas coordinates. Many NWB files store probe-relative
    # positions or placeholders instead of Allen CCF coordinates. Valid CCF
    # positions are inside the brain (~0-13200 µm per axis), so the median
    # electrode position should have at least 2 axes with |value| > 1000.
    # Some files use 10 µm voxel units (max ~1320); for those, require
    # at least 2 axes with |median| > 100.
    xs = sorted(abs(c[0]) for c in coords)
    ys = sorted(abs(c[1]) for c in coords)
    zs = sorted(abs(c[2]) for c in coords)
    med = [xs[len(xs) // 2], ys[len(ys) // 2], zs[len(zs) // 2]]
    max_val = max(xs[-1], ys[-1], zs[-1])
    if max_val > 1500:
        # Likely µm: require 2 axes with median > 1000
        if sum(1 for m in med if m > 1000) < 2:
            return None
    else:
        # Likely 10 µm voxel units: require 2 axes with median > 100
        if sum(1 for m in med if m > 100) < 2:
            return None

    return coords


# ---------------------------------------------------------------------------
# Allen CCF helpers
# ---------------------------------------------------------------------------


def fetch_allen_structure_graph():
    """Download the Allen CCF structure graph and return the raw message list."""
    print("Fetching Allen CCF structure graph...")
    resp = requests.get(ALLEN_STRUCTURE_GRAPH_URL, timeout=60)
    resp.raise_for_status()
    return resp.json()["msg"]


def flatten_structure_graph(msg):
    """Recursively flatten the Allen structure graph tree into a flat list."""
    result = []
    for node in msg:
        result.append(node)
        if node.get("children"):
            result.extend(flatten_structure_graph(node["children"]))
    return result


def build_lookup_dicts(structures):
    """Build lookup dictionaries from Allen CCF structures.

    Returns (by_acronym, by_name, by_acronym_lower, by_name_lower).
    """
    by_acronym = {}
    by_name = {}
    by_acronym_lower = {}
    by_name_lower = {}

    for s in structures:
        by_acronym[s["acronym"]] = s
        by_name[s["name"]] = s
        acr_lower = s["acronym"].lower()
        name_lower = s["name"].lower()
        if acr_lower not in by_acronym_lower:
            by_acronym_lower[acr_lower] = s
        if name_lower not in by_name_lower:
            by_name_lower[name_lower] = s

    return by_acronym, by_name, by_acronym_lower, by_name_lower


def build_parent_map(structures):
    """Build a mapping from structure_id -> parent_structure_id."""
    return {s["id"]: s.get("parent_structure_id") for s in structures}


def get_ancestors(structure_id, parent_map):
    """Get all ancestor structure IDs (excluding self)."""
    ancestors = []
    current = parent_map.get(structure_id)
    while current is not None:
        ancestors.append(current)
        current = parent_map.get(current)
    return ancestors


# ---------------------------------------------------------------------------
# Location matching
# ---------------------------------------------------------------------------


def _extract_area(location):
    """Try to extract a brain-area value from structured location strings."""
    # Python dict repr: {'area': 'VISp', 'depth': '20'}
    if location.startswith("{"):
        try:
            d = ast.literal_eval(location)
            if isinstance(d, dict) and "area" in d:
                return str(d["area"]).strip()
        except (ValueError, SyntaxError):
            pass

    # Comma-separated key: value pairs: area: VISp,depth: 175
    m = re.match(r"area:\s*([^,]+)", location)
    if m:
        return m.group(1).strip()

    return None


def _match_single(loc, by_acronym, by_name, by_acronym_lower, by_name_lower):
    """Match a single token against Allen CCF structures."""
    if loc in by_acronym:
        return by_acronym[loc]
    if loc in by_name:
        return by_name[loc]
    loc_lower = loc.lower()
    if loc_lower in by_acronym_lower:
        return by_acronym_lower[loc_lower]
    if loc_lower in by_name_lower:
        return by_name_lower[loc_lower]
    return None


def match_location(location, lookups):
    """Match a location string against Allen CCF structures.

    Args:
        location: Raw location string from NWB file.
        lookups: Tuple of (by_acronym, by_name, by_acronym_lower, by_name_lower).

    Returns a list of matched structure dicts (may be empty).
    """
    by_acronym, by_name, by_acronym_lower, by_name_lower = lookups
    loc = location.strip()
    if loc.lower() in TRIVIAL_LOCATIONS:
        return []

    # Try as a single value first
    result = _match_single(loc, by_acronym, by_name, by_acronym_lower, by_name_lower)
    if result:
        return [result]

    # Try extracting an area value from structured strings
    area = _extract_area(loc)
    if area:
        return match_location(area, lookups)

    # Try comma-separated list (e.g. "VISp,VISrl,VISlm,VISal")
    if "," in loc:
        parts = [p.strip() for p in loc.split(",")]
        matches = []
        for part in parts:
            if part and part.lower() not in TRIVIAL_LOCATIONS:
                s = _match_single(part, by_acronym, by_name, by_acronym_lower, by_name_lower)
                if s:
                    matches.append(s)
        if matches:
            return matches

    return []


# ---------------------------------------------------------------------------
# Subject / session extraction
# ---------------------------------------------------------------------------


def extract_subject(path):
    """Extract subject directory from asset path."""
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else path.split("_")[0]


def extract_session(path):
    """Extract session ID from a BIDS-style NWB filename.

    Strips known non-standard suffixes like '-processed-only' that some
    datasets (e.g., IBL) append to the session UUID without an underscore.
    """
    import re
    match = re.search(r"_ses-([^_/]+)", path)
    if not match:
        return None
    session = match.group(1)
    session = re.sub(r"-processed-only$", "", session)
    return session


def extract_desc(path):
    """Extract description label from a BIDS-style NWB filename."""
    import re
    match = re.search(r"_desc-([^_/]+)", path)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Aggregation and data building
# ---------------------------------------------------------------------------


def build_dandi_regions(dandiset_assets, id_to_structure, parent_map):
    """Build dandi_regions dict from dandiset_assets.

    Aggregates per-structure counts and propagates to ancestors.
    Returns a dict of structure_id_str -> region info.
    """
    # Direct counts: structure_id -> {dandisets: set, file_count: int}
    region_data = {}

    for dandiset_id, assets in dandiset_assets.items():
        for asset in assets:
            for r in asset.get("regions", []):
                sid = r["id"]
                if sid not in id_to_structure:
                    continue
                if sid not in region_data:
                    region_data[sid] = {"dandisets": set(), "file_count": 0}
                region_data[sid]["dandisets"].add(dandiset_id)
                region_data[sid]["file_count"] += 1

    # Build children map
    children_map = {}
    for s in id_to_structure.values():
        pid = s.get("parent_structure_id")
        if pid is not None:
            children_map.setdefault(pid, []).append(s["id"])

    # Aggregate: propagate upward to ancestors
    aggregate_data = {}
    for sid, data in region_data.items():
        aggregate_data[sid] = {
            "total_dandisets": set(data["dandisets"]),
            "total_file_count": data["file_count"],
        }

    for sid in list(region_data.keys()):
        for anc_id in get_ancestors(sid, parent_map):
            if anc_id not in aggregate_data:
                aggregate_data[anc_id] = {
                    "total_dandisets": set(),
                    "total_file_count": 0,
                }
            aggregate_data[anc_id]["total_dandisets"].update(region_data[sid]["dandisets"])
            aggregate_data[anc_id]["total_file_count"] += region_data[sid]["file_count"]

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

    return dandi_regions


def compute_mesh_set(dandi_regions, parent_map):
    """Determine which mesh IDs are needed.

    Returns (data_ids, ancestor_ids, all_ids).
    """
    data_ids = set(int(sid) for sid in dandi_regions.keys())
    ancestor_ids = set()
    for sid in data_ids:
        for anc in get_ancestors(sid, parent_map):
            ancestor_ids.add(anc)

    all_ids = data_ids | ancestor_ids
    all_ids.add(997)  # root brain outline
    return data_ids, ancestor_ids, all_ids


def download_meshes(mesh_ids, meshes_dir):
    """Download OBJ meshes for the given structure IDs.

    Returns list of IDs that failed to download.
    """
    meshes_dir.mkdir(parents=True, exist_ok=True)
    failed_ids = []
    downloaded = 0

    for i, sid in enumerate(sorted(mesh_ids)):
        dest = meshes_dir / f"{sid}.obj"
        if dest.exists():
            continue

        url = MESH_URL_TEMPLATE.format(structure_id=sid)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "brain-atlas-viewer/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            downloaded += 1
        except Exception as e:
            print(f"  Failed to download mesh {sid}: {e}")
            failed_ids.append(sid)

        if downloaded > 0 and downloaded % 10 == 0:
            time.sleep(0.5)

    # Check for any missing meshes
    for sid in sorted(mesh_ids):
        dest = meshes_dir / f"{sid}.obj"
        if not dest.exists() and sid not in failed_ids:
            failed_ids.append(sid)

    return failed_ids
