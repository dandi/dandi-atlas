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

This downloads the Allen structure graph, matches DANDI locations to CCF terms, and downloads OBJ meshes (~190 MB) to `data/meshes/`.

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
│   └── meshes/                 # OBJ files by structure ID (generated, not in git)
└── scripts/
    └── build_data.py           # Generates all static data
```

## Data Sources

- **Allen Brain Atlas API** — structure graph and OBJ meshes
- **DANDI Archive** — dataset-to-brain-region mappings from NWB file metadata
