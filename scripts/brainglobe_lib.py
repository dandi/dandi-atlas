#!/usr/bin/env python3
"""Read atlases from the BrainGlobe V3 public S3 bucket.

BrainGlobe V3 splits an atlas into independently versioned components. An
atlas manifest points at the coordinate space, template, terminology and
annotation set it is assembled from:

    atlas/atlases/<atlas>/<version>/manifest.json
    atlas/terminologies/<name>/<version>/terminology.csv
    atlas/annotation-sets/<name>/<version>/annotation.precomputed/<value>

The bucket is public (no credentials), sets `Access-Control-Allow-Origin: *`
and supports range requests, so everything here could equally be done from the
browser at runtime - see decode_precomputed_mesh for the wire format.

Meshes are neuroglancer "precomputed" fragments, but only in the legacy
single-resolution flavour: no Draco, no LOD, no sharding, one flat file per
segment named after its annotation value.
"""

import csv
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree

import numpy as np
import trimesh

BUCKET_URL = "https://brainglobe.s3.us-west-2.amazonaws.com"
ATLAS_ROOT = "atlas"  # sibling prefix "atlas-rc2" holds an older release candidate
LAST_VERSIONS_KEY = f"{ATLAS_ROOT}/atlases/last_versions.conf"

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dandi-atlas/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _key_url(key: str) -> str:
    return f"{BUCKET_URL}/{key}"


def _location_key(location: str) -> str:
    """Manifest locations are bucket-root-relative ("/terminologies/x/1_2")."""
    return f"{ATLAS_ROOT}/{location.lstrip('/')}"


def list_keys(prefix: str) -> list:
    """List every key under a prefix, following continuation tokens."""
    keys = []
    token = None
    while True:
        url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        root = ElementTree.fromstring(_get(url))
        for contents in root.findall(f"{S3_NS}Contents"):
            keys.append(contents.findtext(f"{S3_NS}Key"))
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return keys
        token = root.findtext(f"{S3_NS}NextContinuationToken")


def latest_version(atlas_key: str) -> str:
    """Resolve e.g. "allen_mouse_25um" -> "3_0" from last_versions.conf."""
    conf = _get(_key_url(LAST_VERSIONS_KEY)).decode("utf-8")
    match = re.search(rf"^{re.escape(atlas_key)}\s*=\s*(\S+)$", conf, re.MULTILINE)
    if not match:
        raise SystemExit(f"Atlas {atlas_key!r} not listed in last_versions.conf")
    return match.group(1).replace(".", "_")


def fetch_manifest(atlas_key: str, version: str = None) -> dict:
    version = version or latest_version(atlas_key)
    key = f"{ATLAS_ROOT}/atlases/{atlas_key}/{version}/manifest.json"
    return json.loads(_get(_key_url(key)))


# ── Terminology (the structure hierarchy) ───────────────────────────────────

def fetch_terminology(manifest: dict) -> list:
    """Return terminology.csv rows as dicts, in file order (root first)."""
    key = f"{_location_key(manifest['terminology']['location'])}/terminology.csv"
    text = _get(_key_url(key)).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def build_structure_graph(rows: list) -> list:
    """Convert terminology rows to the Allen-shaped tree the viewer reads.

    app.js walks structure_graph.json for `id`, `acronym`, `name`,
    `color_hex_triplet` (bare hex, no leading '#') and `parent_structure_id`,
    so those names are preserved rather than passing BrainGlobe's through.
    """
    nodes = {}
    for row in rows:
        sid = int(row["identifier"])
        parent = row["parent_identifier"].strip()
        nodes[sid] = {
            "id": sid,
            "acronym": row["abbreviation"],
            "name": row["name"],
            "color_hex_triplet": row["color_hex_triplet"].lstrip("#").upper(),
            "parent_structure_id": int(parent) if parent else None,
            "st_level": len(json.loads(row["root_identifier_path"])) - 1,
            # Segment id the mesh file is named after. Equal to `id` in every
            # atlas checked so far, but they are distinct fields upstream.
            "annotation_value": int(row["annotation_value"]),
            "children": [],
        }

    roots = []
    for node in nodes.values():
        parent_id = node["parent_structure_id"]
        if parent_id is None:
            roots.append(node)
        elif parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            raise SystemExit(f"Structure {node['id']} has unknown parent {parent_id}")
    if not roots:
        raise SystemExit("Terminology has no root structure")
    return roots


# ── Meshes ─────────────────────────────────────────────────────────────────

def mesh_prefix(manifest: dict) -> str:
    return f"{_location_key(manifest['annotation_set']['location'])}/annotation.precomputed/"


def list_mesh_values(manifest: dict) -> list:
    """Annotation values that have a mesh in the bucket."""
    prefix = mesh_prefix(manifest)
    values = []
    for key in list_keys(prefix):
        name = key[len(prefix):]
        if name.isdigit():
            values.append(int(name))
    return sorted(values)


def decode_precomputed_mesh(buf: bytes):
    """Decode a legacy single-resolution neuroglancer mesh fragment.

        uint32   num_vertices
        float32  positions[3 * num_vertices]   (micrometres, atlas space)
        uint32   indices[3 * num_triangles]

    Returns (vertices, faces) as float32 (N, 3) and uint32 (M, 3) arrays.
    """
    if len(buf) < 4:
        raise ValueError("mesh fragment shorter than its header")
    num_vertices = int(np.frombuffer(buf, dtype="<u4", count=1)[0])
    vertex_bytes = 12 * num_vertices
    if len(buf) < 4 + vertex_bytes:
        raise ValueError(f"mesh fragment truncated: want {4 + vertex_bytes} bytes, got {len(buf)}")
    vertices = np.frombuffer(buf, dtype="<f4", count=3 * num_vertices, offset=4).reshape(-1, 3)

    index_bytes = len(buf) - 4 - vertex_bytes
    if index_bytes % 12:
        raise ValueError(f"index block is not a whole number of triangles ({index_bytes} bytes)")
    faces = np.frombuffer(buf, dtype="<u4", offset=4 + vertex_bytes).reshape(-1, 3)
    if faces.size and faces.max() >= num_vertices:
        raise ValueError(f"index {faces.max()} out of range for {num_vertices} vertices")
    return vertices, faces


def encode_precomputed_mesh(vertices, faces) -> bytes:
    """Inverse of decode_precomputed_mesh, for meshes we generate ourselves."""
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3), got {faces.shape}")
    return b"".join([
        np.uint32(len(vertices)).tobytes(),
        vertices.tobytes(),
        faces.tobytes(),
    ])


def fetch_mesh_bytes(manifest: dict, annotation_value: int):
    """Raw precomputed fragment as served. Returns None when absent."""
    url = _key_url(f"{mesh_prefix(manifest)}{annotation_value}")
    try:
        return _get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_mesh(manifest: dict, annotation_value: int):
    """Fetch and decode one region mesh. Returns None when absent."""
    buf = fetch_mesh_bytes(manifest, annotation_value)
    return None if buf is None else decode_precomputed_mesh(buf)


def to_trimesh(vertices, faces) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def download_fragments(atlas_key: str, structure_ids, meshes_dir, version: str = None):
    """Fetch precomputed fragments for the given structure IDs.

    Files are written under their bare structure ID, byte-identical to the
    bucket. Returns the IDs that have no mesh upstream.
    """
    meshes_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(atlas_key, version)
    values = {
        int(row["identifier"]): int(row["annotation_value"])
        for row in fetch_terminology(manifest)
    }
    available = set(list_mesh_values(manifest))

    missing = []
    for sid in sorted(structure_ids):
        dest = meshes_dir / str(sid)
        if dest.exists():
            continue
        value = values.get(sid, sid)
        if value not in available:
            missing.append(sid)
            continue
        raw = fetch_mesh_bytes(manifest, value)
        if raw is None:
            missing.append(sid)
            continue
        dest.write_bytes(raw)
    return missing


def decimate(vertices, faces, face_count: int):
    """Quadric-decimate to a face budget. Returns the inputs when already under."""
    if not face_count or len(faces) <= face_count:
        return vertices, faces
    mesh = to_trimesh(vertices, faces).simplify_quadric_decimation(face_count=face_count)
    return mesh.vertices, mesh.faces


def write_glb(vertices, faces, path, decimate_to: int = None):
    """Write a region mesh as GLB, optionally decimating to a face budget."""
    vertices, faces = decimate(vertices, faces, decimate_to)
    mesh = to_trimesh(vertices, faces)
    path.write_bytes(mesh.export(file_type="glb"))
    return mesh


def write_precomputed(vertices, faces, path, decimate_to: int = None):
    """Write a region mesh as a precomputed fragment, optionally decimated."""
    vertices, faces = decimate(vertices, faces, decimate_to)
    path.write_bytes(encode_precomputed_mesh(vertices, faces))
