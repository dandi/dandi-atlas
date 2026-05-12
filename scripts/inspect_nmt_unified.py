#!/usr/bin/env python3
"""Print a tree summary of the unified NMT (CHARM + SARM) structure graph.

Diagnostic tool used during the SARM rollout (Plan B in
ongoing_issues/plan_add_sarm_subcortex.md). Builds the unified graph in memory
and reports per-depth node and leaf counts, top-level subtree sizes, sample
SARM nuclei (substantia nigra in particular, since that motivated the
addition), and any abbreviation collisions between CHARM and SARM.

Does not write files, does not regenerate meshes. Run before any mesh-pipeline
changes to confirm the graph looks right.

Usage:
    uv run python scripts/inspect_nmt_unified.py
"""

import re
from collections import Counter

from macaque_atlas_lib import (
    NMT_UNIFIED_CORTEX_ID,
    NMT_UNIFIED_SUBCORTEX_ID,
    OUTSIDE_ID,
    ROOT_ID,
    SARM_OFFSET,
    build_nmt_unified_structure_graph,
    parse_charm_labels,
    parse_sarm_labels,
)


def walk_depths(node, depth, depths, leaf_depths):
    depths[depth] += 1
    children = node.get("children", [])
    if not children:
        leaf_depths[depth] += 1
    for child in children:
        walk_depths(child, depth + 1, depths, leaf_depths)


def count_subtree(root_node):
    total = 0
    leaves = 0
    stack = [root_node]
    while stack:
        n = stack.pop()
        total += 1
        kids = n.get("children", [])
        if not kids:
            leaves += 1
        stack.extend(kids)
    return total, leaves


def main():
    charm_entries = parse_charm_labels()
    sarm_entries = parse_sarm_labels()
    print(f"Parsed {len(charm_entries)} CHARM entries, {len(sarm_entries)} SARM entries")
    print(f"SARM offset applied: +{SARM_OFFSET}")

    tree, id_to_structure, parent_map, abbrev_to_id, name_to_id = (
        build_nmt_unified_structure_graph(charm_entries, sarm_entries)
    )

    root = tree[0]
    print()
    print(f"Root: {root['name']} (id={root['id']})")
    print(f"Total nodes (incl. synthetic + outside): {len(id_to_structure)}")

    # Per-depth distribution
    depths = Counter()
    leaf_depths = Counter()
    walk_depths(root, 0, depths, leaf_depths)
    print()
    print("Depth   total_nodes   leaves")
    for d in sorted(depths):
        print(f"  {d:3d}   {depths[d]:11d}   {leaf_depths[d]:6d}")
    print(f"Total leaves: {sum(leaf_depths.values())}")

    # Top-level breakdown
    print()
    print("=== Top-level subtrees ===")
    for child in root["children"]:
        total, leaves = count_subtree(child)
        print(f"  {child['name']:30s}  id={child['id']:>5d}  total={total:4d}  leaves={leaves:4d}")

    # CHARM-side category breakdown (one level under Cortex container)
    cortex = id_to_structure[NMT_UNIFIED_CORTEX_ID]
    print()
    print("=== CHARM lobes (under Cortex) ===")
    for child in cortex["children"]:
        total, leaves = count_subtree(child)
        print(f"  {child['name']:30s}  total={total:4d}  leaves={leaves:4d}")

    # SARM-side category breakdown (one level under Subcortex container)
    subcortex = id_to_structure[NMT_UNIFIED_SUBCORTEX_ID]
    print()
    print("=== SARM systems (under Subcortex) ===")
    for child in subcortex["children"]:
        total, leaves = count_subtree(child)
        print(f"  {child['name']:30s}  total={total:4d}  leaves={leaves:4d}")

    # Verify substantia nigra subdivisions are present (the motivating example)
    print()
    print("=== Substantia nigra under SARM ===")
    nigra_pat = re.compile(r"nigra|nigral", re.IGNORECASE)
    hits = []
    for nid, node in id_to_structure.items():
        if node.get("source") != "sarm":
            continue
        if nigra_pat.search(node.get("name", "")) or nigra_pat.search(node.get("acronym", "")):
            hits.append(node)
    if hits:
        for node in sorted(hits, key=lambda n: n["id"]):
            print(f"  id={node['id']:>5d}  native={node['native_index']:>4d}  "
                  f"{node['acronym']:>8s}  {node['name']}")
    else:
        print("  (none found, expect at least one SN entry)")

    # Verify full-name lookup table disambiguates the abbreviation collisions.
    print()
    print("=== Resolver disambiguation check (collision abbreviations) ===")
    print(f"abbrev_to_id size: {len(abbrev_to_id)}, name_to_id size: {len(name_to_id)}")
    for ab, charm_full, sarm_full in [
        ("R", "rostral core region", "red nucleus"),
        ("RM", "rostromedial belt region", "retromammillary hypothalamus"),
        ("CM", "caudomedial belt region", "central medial thalamus"),
        ("Pi", "parainsula", "pineal gland"),
        ("PM", "premotor cortex", "paramedian lobule"),
    ]:
        ab_id = abbrev_to_id.get(ab)
        charm_id = name_to_id.get(charm_full)
        sarm_id = name_to_id.get(sarm_full)
        ab_node = id_to_structure.get(ab_id, {})
        charm_node = id_to_structure.get(charm_id, {})
        sarm_node = id_to_structure.get(sarm_id, {})
        print(f"  '{ab}' (abbrev) -> id={ab_id} ({ab_node.get('source','?')}: {ab_node.get('name','?')})")
        print(f"    full '{charm_full}'  -> id={charm_id} ({charm_node.get('source','?')})")
        print(f"    full '{sarm_full}'   -> id={sarm_id} ({sarm_node.get('source','?')})")

    # Sources represented in id_to_structure
    sources = Counter(n.get("source", "synthetic") for n in id_to_structure.values())
    print()
    print("=== Source breakdown ===")
    for src, count in sorted(sources.items()):
        print(f"  {src:>10s}: {count}")

    # Sanity: every non-root, non-outside node has a parent in the graph
    orphans = []
    for nid, node in id_to_structure.items():
        if nid in (ROOT_ID, OUTSIDE_ID):
            continue
        pid = node.get("parent_structure_id")
        if pid is None or pid not in id_to_structure:
            orphans.append((nid, node["name"], pid))
    print()
    print(f"Orphan nodes: {len(orphans)}")
    if orphans:
        for nid, name, pid in orphans[:10]:
            print(f"  id={nid} parent={pid} name={name}")


if __name__ == "__main__":
    main()
