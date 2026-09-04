<p align="center">
  <img src="logo.png" alt="DANDI Atlas Explorer logo" width="120">
</p>

<h1 align="center">DANDI Brain Atlas Explorer</h1>

<p align="center">
  <a href="https://github.com/dandi/dandi-atlas/actions/workflows/update-data.yml"><img src="https://github.com/dandi/dandi-atlas/actions/workflows/update-data.yml/badge.svg" alt="Nightly data update"></a>
  <a href="data/last_updated.json"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdandi%2Fdandi-atlas%2Fmain%2Fdata%2Flast_updated.json&query=%24.date&label=data%20updated&color=blue" alt="Data last updated"></a>
</p>

Interactive 3D viewer for finding data in the [DANDI Archive](https://dandiarchive.org) by brain region. Regions come from standard reference atlases and are shaded by how much data covers them, and selecting one lists the dandisets, subjects, and sessions that recorded there. Where a session provides electrode coordinates, the individual contacts are drawn in the same 3D space.

Live at **[atlas.dandiarchive.org](https://atlas.dandiarchive.org)**. There is a [blog post](https://about.dandiarchive.org/blog/2026/02/24/introducing-dandi-atlas-explorer-explore-the-dandi-archive-in-3d/) introducing the project, and a [demo video](https://www.youtube.com/watch?v=D8514CLVXYo) walking through what the viewer can do.

<p align="center">
  <a href="https://www.youtube.com/watch?v=D8514CLVXYo">
    <img src="https://img.youtube.com/vi/D8514CLVXYo/maxresdefault.jpg" alt="DANDI Atlas Explorer Demo" width="500">
  </a>
</p>

## Atlases

The landing page offers a card per atlas, and the selector in the left sidebar switches between them without a reload.

| Atlas | Species | Source |
|---|---|---|
| Allen CCF v3 | Mouse | Allen Institute Common Coordinate Framework |
| D99 v2.0 | Macaque | Saleem and Logothetis |
| NMT v2.0 sym | Macaque | D99 labels warped into NMT space |
| MEBRAINS | Macaque | EBRAINS macaque parcellation |
| WHS-SD v4 | Rat | Waxholm Space Sprague Dawley |

Coverage differs a great deal between them, and the Allen CCF is by far the most populated. Current per-atlas counts of dandisets, files, and regions with data are in [`data/atlases_index.json`](data/atlases_index.json), which the landing page reads directly. They are not repeated here because the nightly job changes them.

## What the Viewer Does

- Renders atlas region meshes with Three.js, colored by the atlas convention and shaded by how many files are associated with each region.
- Lets you click a region to isolate it, dimming everything else, and shows the associated dandisets in the right sidebar with direct links to DANDI.
- Provides a searchable, collapsible hierarchy tree in the left sidebar, with badges for direct and total counts per structure.
- Drills from a dandiset down to its subjects and then to individual sessions, filtering the tree as you go.
- Draws electrode positions as points for sessions that record them, with separate opacity sliders for regions and electrodes so contacts inside a structure stay visible.
- Snaps the camera to dorsal, ventral, anterior, posterior, left, and right views.

The current view lives in the URL hash, so any state can be linked or bookmarked and the browser back button works:

```
#region=672                                        region view
#dandiset=000017                                   dandiset view
#dandiset=000017&region=672                        dandiset filtered by region
#dandiset=000017&subject=sub-M1                    subject view
#dandiset=000017&subject=sub-M1&session=<assetId>  session view
```

## Running Locally

The viewer is static. Three.js loads from a CDN through an import map, so there is no build step and no npm install. The data files are already committed, including the meshes, so a clone is all you need. Note that this makes the clone large, on the order of 400 MB.

```bash
git clone https://github.com/dandi/dandi-atlas.git
cd dandi-atlas
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000). A server is required rather than opening `index.html` from disk, because the app uses ES modules and `fetch`.

## Updating the Data

A [nightly GitHub Actions workflow](.github/workflows/update-data.yml) runs at 03:00 UTC, refreshes each atlas in turn, rebuilds the landing-page index and the dandiset title map, and commits any changes. Each atlas is allowed to fail without taking the others down, and the run is only marked failed if every atlas fails. You can also trigger it manually from the Actions tab and choose `full` instead of `incremental` for a complete rebuild.

The same scripts run locally:

```bash
pip install -r scripts/requirements.txt

python scripts/update_data.py --mode incremental          # Allen CCF
python scripts/update_macaque_data.py --atlas d99         # also nmt, mebrains
python scripts/update_rat_data.py --mode incremental      # WHS-SD
python scripts/build_atlases_index.py                     # landing-page index
```

These only refresh the DANDI-derived files. They read the committed `structure_graph.json` for each atlas and never touch the meshes, which is what lets them run on a CI runner with nothing but the repository checked out.

The rat atlas draws on an embargoed dandiset that CI cannot reach, so its previously committed records are preserved rather than dropped on each run.

## Rebuilding an Atlas from Source

Regenerating the structure graph and meshes needs the upstream atlas packs, which are not committed. Download them and place them under `atlas_sources/`, which is gitignored. The [WHS-SD pack](https://www.nitrc.org/projects/whs-sd-atlas) comes from NITRC; the macaque sources are described in the header of each build script.

```bash
python scripts/build_macaque_atlas.py --atlas d99    # also nmt, mebrains
python scripts/build_rat_atlas.py
```

Both orchestrators accept `--skip-meshes` and `--skip-dandi` so you can iterate on one stage at a time, and both wrap focused sub-scripts you can call directly.

The Allen pipeline predates this arrangement. `update_data.py --mode full` rebuilds everything derived from DANDI, which is normally what you want. `build_data.py` and `convert_meshes.py` were the original bootstrap, downloading OBJ meshes from the Allen API and converting them to GLB, and they have not been fully migrated to the per-atlas layout; `build_data.py` also expects a `label_results_full.json` from [dandi-location-analysis](https://github.com/catalystneuro/dandi-location-analysis) via `LABEL_RESULTS_PATH`. The Allen meshes are committed, so this path is rarely needed.

## Repository Layout

```
├── index.html                       # Landing page and viewer shell
├── style.css
├── app.js                           # Three.js scene, hierarchy tree, panels, routing
├── data/
│   ├── atlases_index.json           # Landing-page summary, one record per atlas
│   ├── dandiset_titles.json         # Dandiset ID to title, for every dandiset any atlas lists
│   ├── last_updated.json            # Timestamps and counts from the last nightly run
│   └── atlases/<atlas>/
│       ├── structure_graph.json     # Region hierarchy
│       ├── dandi_regions.json       # Per-region dandiset and file counts
│       ├── dandiset_assets.json     # Per-asset region assignments
│       ├── mesh_manifest.json       # Which regions have meshes
│       ├── meshes/*.glb             # Region meshes (committed)
│       └── electrodes/*.json        # Electrode coordinates per dandiset
└── scripts/                         # Build and update pipeline
```

## Data Sources

Region assignments are derived from the metadata inside the NWB files themselves rather than from dandiset-level annotation, which is why coverage grows as the archive does. Meshes and hierarchies come from the atlas providers listed above, and the Allen CCF meshes come from the [Allen Brain Atlas API](https://atlas.brain-map.org/).

Built by [CatalystNeuro](https://catalystneuro.com) and maintained under the DANDI organization.
