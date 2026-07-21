#!/usr/bin/env python3
"""Prototype: build viewer meshes (and optionally a hierarchy) from BrainGlobe V3.

Writes meshes/<id> - and, unless --terminology keep, structure_graph.json - into
an output directory. Meshes default to the neuroglancer precomputed fragment
format the bucket serves, stored under the same bare-ID filenames, so an atlas
can later be pointed at S3 instead of our own copy without any other change.
Pass --format glb for the older GLB layout. The DANDI half (dandi_regions.json,
electrodes) is untouched.

For allen_ccf we keep the Allen API structure graph and take geometry only:
BrainGlobe's terminology is a strict subset of the Allen graph (840 of 1327
structures) and drops 18 structures that dandi_regions.json references - the
cortical layer subdivisions (VIS1, VIS2/3, ...), the CA3 strata, and the
VISrll/VISmma/VISmmp areas. Those have no mesh either way, but they do appear
in the hierarchy tree, so the Allen graph stays authoritative there. build_data.py
and update_data.py do the same fetch inline via brainglobe_lib.download_fragments;
this script is the standalone version, and the only one that can also compare
against what is already on disk:

  python scripts/build_brainglobe_atlas.py --atlas allen_mouse_25um \
      --out data/atlases/allen_ccf --terminology keep --like data/atlases/allen_ccf

For atlases with no Allen-style graph of their own (rat, human, developmental
mouse), BrainGlobe supplies the hierarchy too - that is the default:

  python scripts/build_brainglobe_atlas.py --atlas allen_human_500um --out data/atlases/allen_human

Add --compare to diff fetched geometry against the GLBs already on disk.

Requires numpy and trimesh (scripts/requirements.txt).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

import brainglobe_lib as bg

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rel(path: Path) -> Path:
    """Path relative to the repo when it lives there, absolute otherwise."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def flatten(nodes):
    out = []
    for node in nodes:
        out.append(node)
        out.extend(flatten(node.get("children", [])))
    return out


def write_structure_graph(graph, out_dir):
    path = out_dir / "structure_graph.json"
    with open(path, "w") as f:
        json.dump(graph, f)
    return path


def report_terminology_gap(out_dir: Path, by_id: dict):
    """Warn about structures the local graph has that BrainGlobe does not.

    Only informational under --terminology keep: those structures keep their
    place in the tree, they just cannot gain a mesh from BrainGlobe.
    """
    graph_path = out_dir / "structure_graph.json"
    regions_path = out_dir / "dandi_regions.json"
    if not graph_path.exists():
        print(f"  No {rel(graph_path)} to keep - run build_data.py first, or drop --terminology keep")
        return
    local = flatten(json.load(open(graph_path)))
    missing = [s for s in local if s["id"] not in by_id]
    print(f"  local graph has {len(local)} structures, {len(missing)} of them absent upstream")
    if missing and regions_path.exists():
        regions = json.load(open(regions_path))
        with_data = [s for s in missing if str(s["id"]) in regions]
        if with_data:
            print(f"  {len(with_data)} of those are referenced by dandi_regions.json and stay mesh-less:")
            print("    " + ", ".join(s["acronym"] for s in with_data[:20]))


def existing_mesh_ids(directory: Path):
    """Structure IDs an atlas directory already ships, in either mesh format.

    GLBs are named <id>.glb; precomputed fragments are named <id> with no
    extension, exactly as BrainGlobe serves them.
    """
    ids = set()
    for path in (directory / "meshes").iterdir():
        if not path.is_file():
            continue
        stem = path.stem if path.suffix == ".glb" else path.name
        if stem.lstrip("-").isdigit():
            ids.add(int(stem))
    return sorted(ids)


def load_glb(path: Path):
    """Load a GLB as a single concatenated Trimesh."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_geometry()
    return loaded


def load_reference(reference_dir: Path, sid: int):
    """Load a shipped mesh in whichever format the directory uses."""
    glb_path = reference_dir / "meshes" / f"{sid}.glb"
    if glb_path.exists():
        mesh = load_glb(glb_path)
        return mesh.vertices, mesh.faces
    fragment_path = reference_dir / "meshes" / str(sid)
    if fragment_path.exists():
        return bg.decode_precomputed_mesh(fragment_path.read_bytes())
    return None


def compare_to_existing(sid, vertices, faces, reference_dir: Path):
    """Compare a fetched mesh against the one we currently ship."""
    reference = load_reference(reference_dir, sid)
    if reference is None:
        return {"id": sid, "status": "missing_reference"}
    ref_vertices, ref_faces = reference
    result = {
        "id": sid,
        "bg_vertices": len(vertices),
        "ref_vertices": len(ref_vertices),
        "bg_faces": len(faces),
        "ref_faces": len(ref_faces),
    }
    if len(vertices) != len(ref_vertices) or len(faces) != len(ref_faces):
        result["status"] = "different_topology"
        return result
    # Same counts: compare geometry directly. Vertex order is preserved by both
    # the precomputed fragment and trimesh's GLB round-trip, so an elementwise
    # comparison is meaningful; fall back to bbox distance if it is not exact.
    max_delta = float(np.abs(np.asarray(ref_vertices, dtype=np.float64) - vertices).max())
    result["max_vertex_delta_um"] = max_delta
    faces_equal = np.array_equal(np.asarray(ref_faces, dtype=np.int64), faces.astype(np.int64))
    result["faces_equal"] = faces_equal
    # GLB stores positions as float32, so exact equality is the expected result
    # for an unmodified round-trip; anything under a nanometre is still a match.
    result["status"] = "identical" if (max_delta < 1e-3 and faces_equal) else "different_geometry"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", required=True, help='BrainGlobe atlas key, e.g. "allen_mouse_25um"')
    parser.add_argument("--version", help="Atlas version (default: latest from last_versions.conf)")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--like", type=Path, help="Only build meshes an existing atlas directory ships")
    parser.add_argument("--ids", help="Comma-separated structure IDs to build")
    parser.add_argument("--limit", type=int, help="Stop after N meshes (smoke test)")
    parser.add_argument("--compare", type=Path, nargs="?", const=True,
                        help="Diff fetched geometry against an atlas directory (defaults to --like)")
    parser.add_argument("--decimate-to", type=int, help="Quadric-decimate meshes above this face count")
    parser.add_argument("--skip-existing", action="store_true", help="Leave GLBs already on disk alone")
    parser.add_argument("--format", choices=["precomputed", "glb"], default="precomputed",
                        help="Mesh format to write (default: precomputed, the format the viewer "
                             "and the bucket share)")
    parser.add_argument("--terminology", choices=["brainglobe", "keep"], default="brainglobe",
                        help='"keep" leaves structure_graph.json alone and takes meshes only '
                             "(what allen_ccf wants: the Allen graph stays authoritative)")
    args = parser.parse_args()

    reference_dir = args.compare if isinstance(args.compare, Path) else (args.like if args.compare else None)
    if args.compare and reference_dir is None:
        parser.error("--compare needs a directory, or use it together with --like")
    if args.terminology == "keep" and not (args.like or args.ids):
        parser.error("--terminology keep needs --like or --ids to say which meshes to build")

    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: Resolving {args.atlas} on BrainGlobe...")
    manifest = bg.fetch_manifest(args.atlas, args.version)
    print(f"  {manifest['name']} v{manifest['version']} - {manifest['species']}, "
          f"{manifest['resolution']} um, orientation {manifest['orientation']}, shape {manifest['shape']}")
    print(f"  terminology    {manifest['terminology']['name']} v{manifest['terminology']['version']}")
    print(f"  annotation set {manifest['annotation_set']['name']} v{manifest['annotation_set']['version']}")

    # The terminology is always read: it maps structure IDs to the annotation
    # values the mesh files are named after, even when we do not adopt the tree.
    print("Step 2: Reading terminology.csv...")
    rows = bg.fetch_terminology(manifest)
    graph = bg.build_structure_graph(rows)
    structures = flatten(graph)
    by_id = {s["id"]: s for s in structures}
    if args.terminology == "keep":
        print(f"  {len(structures)} structures upstream; keeping the existing structure_graph.json")
        report_terminology_gap(out_dir, by_id)
    else:
        path = write_structure_graph(graph, out_dir)
        print(f"  {len(structures)} structures, root {graph[0]['id']} ({graph[0]['acronym']}) -> "
              f"{rel(path)}")

    print("Step 3: Selecting meshes...")
    available = set(bg.list_mesh_values(manifest))
    print(f"  {len(available)} meshes in the bucket")
    if args.ids:
        wanted = [int(x) for x in args.ids.split(",") if x.strip()]
    elif args.like:
        like_dir = args.like if args.like.is_absolute() else PROJECT_ROOT / args.like
        wanted = existing_mesh_ids(like_dir)
        print(f"  {len(wanted)} meshes shipped by {args.like}")
    else:
        wanted = sorted(available)
    if args.limit:
        wanted = wanted[: args.limit]

    unknown = [sid for sid in wanted if sid not in by_id]
    if unknown:
        print(f"  {len(unknown)} requested IDs are not in this terminology: {unknown[:10]}")

    print(f"Step 4: Fetching {len(wanted)} meshes...")
    written = skipped = 0
    absent = []
    comparisons = []
    for i, sid in enumerate(wanted):
        # Precomputed fragments keep the bucket's own naming (bare structure
        # ID, no extension) so that pointing an atlas at the S3 bucket later is
        # a base-URL change and nothing else.
        dest = meshes_dir / (f"{sid}.glb" if args.format == "glb" else str(sid))
        if args.skip_existing and dest.exists() and not reference_dir:
            skipped += 1
            continue

        value = by_id[sid]["annotation_value"] if sid in by_id else sid
        if value not in available:
            absent.append(sid)
            continue

        raw = bg.fetch_mesh_bytes(manifest, value)
        if raw is None:
            absent.append(sid)
            continue
        vertices, faces = bg.decode_precomputed_mesh(raw)

        if reference_dir:
            ref = reference_dir if reference_dir.is_absolute() else PROJECT_ROOT / reference_dir
            comparisons.append(compare_to_existing(sid, vertices, faces, ref))

        if not (args.skip_existing and dest.exists()):
            if args.format == "glb":
                bg.write_glb(vertices, faces, dest, decimate_to=args.decimate_to)
            elif args.decimate_to:
                bg.write_precomputed(vertices, faces, dest, decimate_to=args.decimate_to)
            else:
                # Undecimated: store the fragment exactly as served, so our copy
                # is byte-identical to the bucket's.
                dest.write_bytes(raw)
            written += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(wanted)} ({written} written, {len(absent)} absent)")

    print(f"  Done: {written} written, {skipped} skipped, {len(absent)} with no mesh upstream")
    if absent:
        print(f"  No mesh for: {absent[:20]}{' ...' if len(absent) > 20 else ''}")

    extra = sorted(available - {by_id[s]["annotation_value"] for s in wanted if s in by_id})
    if not args.ids and not args.limit:
        print(f"  {len(extra)} further meshes available upstream that were not requested")

    if comparisons:
        print(f"\nStep 5: Comparing geometry against {reference_dir}...")
        buckets = {}
        for c in comparisons:
            buckets.setdefault(c["status"], []).append(c)
        for status, items in sorted(buckets.items()):
            print(f"  {status}: {len(items)}")
            if status != "identical":
                for c in items[:5]:
                    print(f"    {c}")
        deltas = [c["max_vertex_delta_um"] for c in comparisons if "max_vertex_delta_um" in c]
        if deltas:
            print(f"  worst vertex delta across {len(deltas)} matched meshes: {max(deltas):.6g} um")
        if set(buckets) == {"identical"}:
            print("  -> BrainGlobe geometry is a drop-in replacement for this atlas.")
        else:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
