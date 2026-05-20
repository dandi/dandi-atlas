#!/usr/bin/env python3
"""Generate UBERON canonical mappings for WHS-SD rat atlas regions.

One-off generator: queries the OLS4 search API once per WHS-SD canonical region
name and writes `data/atlases/whs_sd/uberon_mapping.json`. The build pipeline
reads the committed JSON; it never touches the network itself.

Strategy per region:
  1. Query OLS4 with `exact=true` against the `uberon` ontology. If we get an
     exact label match (or exact-synonym match), record it as `match_quality:
     "exact"`.
  2. Otherwise re-query with `exact=false`, take the top hit, record it as
     `match_quality: "fuzzy"` for human review.
  3. Apply WHS_UBERON_OVERRIDES last so manually curated entries win over the
     OLS4 result. An override value of `None` marks a region as intentionally
     unmapped (synthetic root, outside-atlas sentinel, etc.).

Run by hand whenever the WHS-SD hierarchy changes:
    uv run python scripts/build_whs_uberon_mapping.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from rat_atlas_lib import ATLAS_CONFIGS, WHS_UBERON_OVERRIDES, parse_whs_ilf
from macaque_atlas_lib import OUTSIDE_ID, ROOT_ID


OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
REQUEST_TIMEOUT = 30
MAX_WORKERS = 5  # OLS4 handles concurrent requests fine; keep it modest


def _ols4_search(session, query, exact):
    parameters = {
        "q": query,
        "ontology": "uberon",
        "exact": "true" if exact else "false",
        "rows": "5",
    }
    response = session.get(OLS4_SEARCH, params=parameters, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("response", {}).get("docs", [])


def _pick_uberon_hit(documents):
    """Return the first hit that is actually a UBERON class, else None."""
    for document in documents:
        if document.get("ontology_name") != "uberon":
            continue
        if not document.get("obo_id", "").startswith("UBERON:"):
            continue
        return document
    return None


def _lookup_uberon(session, name):
    """Look up a WHS-SD region name in UBERON via OLS4. Returns
    (entry_dict, match_quality) where entry_dict has uberon_id / uberon_label /
    uberon_iri (or is None when nothing matched), and match_quality is one of
    "exact" | "fuzzy" | "none".
    """
    exact_hits = _ols4_search(session, name, exact=True)
    hit = _pick_uberon_hit(exact_hits)
    if hit is not None:
        return {
            "uberon_id": hit["obo_id"],
            "uberon_label": hit["label"],
            "uberon_iri": hit["iri"],
        }, "exact"

    fuzzy_hits = _ols4_search(session, name, exact=False)
    hit = _pick_uberon_hit(fuzzy_hits)
    if hit is not None:
        return {
            "uberon_id": hit["obo_id"],
            "uberon_label": hit["label"],
            "uberon_iri": hit["iri"],
        }, "fuzzy"

    return None, "none"


def main():
    config = ATLAS_CONFIGS["whs_sd"]
    hierarchy_file = config["hierarchy"]
    if not hierarchy_file.exists():
        sys.stderr.write(
            f"Missing WHS-SD hierarchy file: {hierarchy_file}\n"
            "Download the MBAT WHS-SD v4 pack and set WHS_SD_DATA if needed.\n"
        )
        sys.exit(1)

    _, id_to_structure, _, _, _ = parse_whs_ilf(
        hierarchy_file, root_name=config["root_name"],
    )

    # Sentinel nodes have no anatomical counterpart in UBERON.
    skip_ids = {ROOT_ID, OUTSIDE_ID}

    mapping = {}
    exact_count = 0
    fuzzy_rows = []
    unmapped_rows = []

    sorted_items = sorted(id_to_structure.items())
    queryable_items = [(whs_id, s) for whs_id, s in sorted_items if whs_id not in skip_ids]
    for whs_id, structure in sorted_items:
        if whs_id in skip_ids:
            mapping[str(whs_id)] = {
                "whs_id": whs_id,
                "whs_name": structure["name"],
                "uberon_id": None,
                "uberon_label": None,
                "uberon_iri": None,
                "match_quality": "skipped",
            }

    session = requests.Session()

    def _query_one(item):
        whs_id, structure = item
        name = structure["name"]
        entry, quality = _lookup_uberon(session, name)
        return whs_id, name, entry, quality

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results_iter = pool.map(_query_one, queryable_items)
        progress = tqdm(results_iter, total=len(queryable_items),
                        desc="OLS4 UBERON lookup", unit="region")
        for whs_id, name, entry, quality in progress:
            row = {
                "whs_id": whs_id,
                "whs_name": name,
                "uberon_id": entry["uberon_id"] if entry else None,
                "uberon_label": entry["uberon_label"] if entry else None,
                "uberon_iri": entry["uberon_iri"] if entry else None,
                "match_quality": quality,
            }
            mapping[str(whs_id)] = row

            if quality == "exact":
                exact_count += 1
            elif quality == "fuzzy":
                fuzzy_rows.append(row)
            else:
                unmapped_rows.append(row)

    # Apply curated overrides. An override of None marks the region as
    # intentionally unmapped (and replaces any OLS4 result).
    override_count = 0
    for whs_id, override in WHS_UBERON_OVERRIDES.items():
        key = str(whs_id)
        if key not in mapping:
            continue
        row = mapping[key]
        if override is None:
            row["uberon_id"] = None
            row["uberon_label"] = None
            row["uberon_iri"] = None
            row["match_quality"] = "override-unmapped"
        else:
            row["uberon_id"] = override["id"]
            row["uberon_label"] = override["label"]
            row["uberon_iri"] = override.get(
                "iri", f"http://purl.obolibrary.org/obo/{override['id'].replace(':', '_')}",
            )
            row["match_quality"] = "override"
        override_count += 1

    output_path = config["output_dir"] / "uberon_mapping.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"\nWrote {output_path}")
    print(
        f"  exact: {exact_count}, "
        f"fuzzy: {len(fuzzy_rows)}, "
        f"unmapped: {len(unmapped_rows)}, "
        f"overrides applied: {override_count}, "
        f"skipped sentinels: {len(skip_ids)}"
    )

    if fuzzy_rows:
        print("\nFuzzy matches (review these against OLS4):")
        for row in fuzzy_rows:
            print(f"  [{row['whs_id']}] {row['whs_name']!r} -> "
                  f"{row['uberon_id']} ({row['uberon_label']!r})")

    if unmapped_rows:
        print("\nUnmapped (no UBERON candidate found):")
        for row in unmapped_rows:
            print(f"  [{row['whs_id']}] {row['whs_name']!r}")


if __name__ == "__main__":
    main()
