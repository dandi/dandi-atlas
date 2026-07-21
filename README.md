# 3D Allen CCF Brain Atlas — DANDI Explorer

Interactive 3D viewer of the Allen Common Coordinate Framework (CCF) mouse brain atlas, highlighting regions that have associated datasets on the [DANDI Archive](https://dandiarchive.org).

Read the [blog post](https://about.dandiarchive.org/blog/2026/02/24/introducing-dandi-atlas-explorer-explore-the-dandi-archive-in-3d/) for more details.

<p align="center">
  <a href="https://www.youtube.com/watch?v=D8514CLVXYo">
    <img src="https://img.youtube.com/vi/D8514CLVXYo/maxresdefault.jpg" alt="DANDI Atlas Explorer Demo" width="500">
  </a>
</p>

## Features

- **3D brain visualization** using Three.js with Allen CCF mesh data
- **Region coloring** by Allen CCF color scheme, opacity scaled by dataset count
- **Click to isolate** — selecting a region dims everything else, showing only the selected structure at full opacity
- **Hierarchy tree** — collapsible Allen CCF structure tree with search; badges show direct/total dandiset counts
- **Dandiset panel** — click a region to see associated DANDI datasets with direct links
- **Orientation buttons** — snap to dorsal, ventral, anterior, posterior, left, right views
- **Resizable sidebar** — drag to expand the hierarchy panel for deep navigation
- **Mouse-only data** — filters to Mus musculus datasets only (48 dandisets, 353 brain structures)

## Quick Start

### 1. Generate data

```bash
python scripts/build_data.py
```

This downloads the Allen structure graph and matches DANDI locations to CCF terms.

Region meshes come from the [BrainGlobe](https://brainglobe.info) V3 public S3 bucket, in the neuroglancer precomputed format it serves them in:

```bash
python scripts/build_brainglobe_atlas.py --atlas allen_mouse_25um \
    --out data/atlases/allen_ccf --terminology keep --like data/atlases/allen_ccf
```

Files are stored under their bare structure ID, byte-identical to the bucket, so setting `meshBaseUrl` on an atlas in `app.js` streams the meshes from S3 instead of our copy. The hierarchy still comes from the Allen API — BrainGlobe's terminology covers 840 structures to the Allen graph's 1327, dropping the cortical layer subdivisions and CA3 strata that `dandi_regions.json` references.

Requires `label_results_full.json` from the [DANDI location analysis](https://github.com/catalystneuro/dandi-location-analysis). Set the path via:

```bash
LABEL_RESULTS_PATH=/path/to/label_results_full.json python scripts/build_data.py
```

### 2. Serve locally

```bash
python -m http.server 8000
```

### 3. Open

Navigate to [http://localhost:8000](http://localhost:8000)

## Project Structure

```
├── index.html                  # Main app
├── style.css                   # Styles
├── app.js                      # Three.js scene, hierarchy tree, dandiset panel
├── data/
│   ├── structure_graph.json    # Allen hierarchy tree (from Allen API)
│   ├── dandi_regions.json      # Structure data with direct + aggregate dandiset counts
│   ├── mesh_manifest.json      # Index of available meshes
│   └── meshes/                 # Precomputed mesh fragments, named by structure ID
└── scripts/
    ├── build_data.py           # Generates all static data
    ├── brainglobe_lib.py       # Reads atlases from the BrainGlobe V3 S3 bucket
    └── build_brainglobe_atlas.py  # Builds meshes (and hierarchies) from BrainGlobe
```

## Data Sources

- **Allen Brain Atlas API** — structure graph
- **BrainGlobe** — region meshes, as neuroglancer precomputed fragments
- **DANDI Archive** — dataset-to-brain-region mappings from NWB file metadata
