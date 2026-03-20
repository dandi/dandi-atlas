"""
Update data/electrodes/DEMO-FOV.json from an NWB file.

Reads per-pixel CCF coordinates from AnatomicalCoordinatesImageCCFv3,
colors pixels by brain region using Allen CCF colors, and writes landmarks
from AnatomicalCoordinatesCCFv3.

Usage:
    python scripts/update_demo_fov.py E:\IBL-data-share\sub-CSK-im-009_ses-2864dca1-38d8-464c-9777-f6fdfd5e63b5_desc-processed_ophys+behavior.nwb [--step N]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def collect_colors(node, result):
    result[node.get("acronym", "")] = node.get("color_hex_triplet", "AAAAAA")
    for child in node.get("children", []):
        collect_colors(child, result)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def main():
    parser = argparse.ArgumentParser(
        description="Update DEMO-FOV.json from an NWB file."
    )
    parser.add_argument("nwb_file", help="Path to the NWB file")
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Subsampling step (every Nth pixel, default: 4)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    structure_graph_path = repo_root / "data" / "structure_graph.json"
    output_path = repo_root / "data" / "electrodes" / "DEMO-FOV.json"

    # Build acronym -> Allen color map
    with open(structure_graph_path) as f:
        graph = json.load(f)
    acronym_to_color = {}
    for node in graph:
        collect_colors(node, acronym_to_color)

    step = args.step
    with h5py.File(args.nwb_file, "r") as f:
        ccf = f["general/localization/AnatomicalCoordinatesImageCCFv3"]
        x = ccf["x"][::step, ::step].ravel()
        y = ccf["y"][::step, ::step].ravel()
        z = ccf["z"][::step, ::step].ravel()
        br = ccf["brain_region"][::step, ::step].ravel()

        # Corners (full resolution)
        cx, cy, cz = ccf["x"][:], ccf["y"][:], ccf["z"][:]

        # Landmarks
        lm = f["general/localization/AnatomicalCoordinatesCCFv3"]
        lm_x = lm["x"][:]
        lm_y = lm["y"][:]
        lm_z = lm["z"][:]
        lm_labels = [s.decode() for s in lm["brain_region"][:]]

        lm_src = f["general/atlas_registration/landmarks"]
        lm_colors = [s.decode() for s in lm_src["color"][:]]

    print(f"Image shape after subsampling: {len(x)} points (step={step})")
    print(f"x range: {np.nanmin(x):.1f} – {np.nanmax(x):.1f} µm")
    print(f"y range: {np.nanmin(y):.1f} – {np.nanmax(y):.1f} µm")
    print(f"z range: {np.nanmin(z):.1f} – {np.nanmax(z):.1f} µm")

    coords = [[float(xi), float(yi), float(zi)] for xi, yi, zi in zip(x, y, z)]
    colors = [
        hex_to_rgb(
            acronym_to_color.get(r.decode() if isinstance(r, bytes) else r, "888888")
        )
        for r in br
    ]

    landmarks = []
    for label, lx, ly, lz, color in zip(lm_labels, lm_x, lm_y, lm_z, lm_colors):
        print(f"  landmark {label}: ({lx:.1f}, {ly:.1f}, {lz:.1f})")
        landmarks.append(
            {"label": label, "color": color, "xyz": [float(lx), float(ly), float(lz)]}
        )

    corners = [
        [float(cx[0, 0]), float(cy[0, 0]), float(cz[0, 0])],
        [float(cx[0, -1]), float(cy[0, -1]), float(cz[0, -1])],
        [float(cx[-1, 0]), float(cy[-1, 0]), float(cz[-1, 0])],
        [float(cx[-1, -1]), float(cy[-1, -1]), float(cz[-1, -1])],
    ]

    out = {
        "demo-fov-pixels": coords,
        "demo-fov-colors": colors,
        "demo-fov-corners": corners,
        "demo-fov-landmarks": landmarks,
    }
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"Saved {len(coords)} points to {output_path}")


if __name__ == "__main__":
    main()
