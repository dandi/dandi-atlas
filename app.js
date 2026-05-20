import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Bump this string after any data rebuild to force browsers to drop cached GLBs and JSONs.
const DATA_VERSION = '20260519';

// ── State ──────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls, raycaster, mouse;
// Group that wraps every atlas-anchored object (meshes + electrode Points).
// Lights stay on `scene` so a negative scale on this root mirrors only the
// rendered geometry, not the light directions. Used to apply a viewer-side
// X mirror for macaque atlases without touching on-disk vertex coordinates;
// see neurological_vs_radiological_convention.md for the rationale.
let worldRoot;
let structureGraph = [];   // Allen hierarchy tree
let dandiRegions = {};     // structure_id -> {acronym, name, dandisets, ...}
let meshManifest = {};     // {data_structures, ancestor_structures, root_id}
let idToStructure = {};    // structure_id -> flat structure object
let meshObjects = {};      // structure_id -> THREE.Mesh
let selectedId = null;
let hoveredId = null;
let loadingCount = 0;
let brainCenter = new THREE.Vector3();

// ── Helpers ────────────────────────────────────────────────────────────────
function recreateOrbitControls(target) {
  if (controls) controls.dispose();
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.rotateSpeed = 0.8;
  controls.zoomSpeed = 1.2;
  if (target) {
    controls.target.copy(target);
    controls.update();
  }
}

// ── Atlas Configuration ────────────────────────────────────────────────────
const ATLAS_CONFIGS = {
  allen_ccf: {
    name: "Allen CCF (Mouse)",
    dataPrefix: "data/atlases/allen_ccf/",
    camDist: 18000,
    cameraUp: [0, -1, 0],
    camOffset: [0, 0, 1],  // Z=Right in PIR, lateral view
    nearPlane: 1,
    farPlane: 100000,
    electrodeSize: 150,
    electrodePickThreshold: 180,
    rootOpacity: 0.06,
    coordSystem: 'allen',
    attribution: 'Atlas: Allen Institute CCF',
    attributionUrl: 'https://atlas.brain-map.org/',
    regionLinkTemplate: 'https://atlas.brain-map.org/atlas#atlas=2&structure={id}',
  },
  d99: {
    name: "D99 v2.0 (Macaque)",
    dataPrefix: "data/atlases/d99/",
    camDist: 200,
    cameraUp: [0, 0, 1],
    camOffset: [1, 0, 0],  // X=Right in RAS, lateral view
    nearPlane: 0.1,
    farPlane: 1000,
    // Electrode visual size: 0.75 mm matches Allen CCF's ~1.5% of brain width
    // proportion (Allen uses 150 µm in a ~10,000 µm-wide brain). This is a
    // proportional calibration to keep visual identity consistent across
    // atlases — it is not a measurement of real electrode hardware (real
    // Neuropixels contacts are ~12 µm; we exaggerate so users can see them).
    // Re-calibrate when more macaque electrode datasets become available and
    // we have a clearer picture of typical session electrode counts and
    // recording-site distributions on macaque atlases. The pick threshold is
    // ~1.33× the visual size to keep hover picking forgiving without being
    // misleading. Same values across all three macaque atlases by convention
    // (D99, NMT, MEBRAINS share the mm-equivalent RAS coordinate scale).
    electrodeSize: 0.75,
    electrodePickThreshold: 1.0,
    rootOpacity: 0.3,
    coordSystem: 'ras',
    attribution: 'Atlas: D99 v2 (Saleem & Logothetis)',
    attributionUrl: 'https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/atlas_d99v2.html',
    regionLinkTemplate: null,
  },
  nmt: {
    name: "NMT v2.0 sym (Macaque)",
    dataPrefix: "data/atlases/nmt/",
    camDist: 200,
    cameraUp: [0, 0, 1],
    camOffset: [1, 0, 0],  // X=Right in RAS, lateral view
    nearPlane: 0.1,
    farPlane: 1000,
    electrodeSize: 0.75,
    electrodePickThreshold: 1.0,
    rootOpacity: 0.3,
    coordSystem: 'ras',
    attribution: 'Atlas: NMT v2 (Jung et al. 2021)',
    attributionUrl: 'https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/template_nmtv2.html',
    regionLinkTemplate: null,
  },
  mebrains: {
    name: "MEBRAINS (Macaque)",
    dataPrefix: "data/atlases/mebrains/",
    camDist: 200,
    cameraUp: [0, 0, 1],
    camOffset: [1, 0, 0],  // X=Right in RAS, lateral view
    nearPlane: 0.1,
    farPlane: 1000,
    electrodeSize: 0.75,
    electrodePickThreshold: 1.0,
    rootOpacity: 0.25,
    coordSystem: 'ras',
    attribution: 'Atlas: MEBRAINS (EBRAINS)',
    attributionUrl: 'https://ebrains.eu/tools/mebrains',
    regionLinkTemplate: null,
  },
  whs_sd: {
    name: "WHS-SD v4 (Rat)",
    dataPrefix: "data/atlases/whs_sd/",
    // Waxholm space is RAS in mm; rat brain spans ~20 mm AP. Camera distance
    // and clipping picked to comfortably fit the brain at lateral view; tune
    // empirically once the meshes render.
    camDist: 60,
    cameraUp: [0, 0, 1],
    camOffset: [1, 0, 0],  // X=Right in RAS, lateral view
    nearPlane: 0.1,
    farPlane: 500,
    // Electrode dots calibrated proportionally for the rat brain (~half the
    // linear size of a macaque brain at the same atlas scale). DANDI 001699
    // doesn't ship atlas-frame coordinates anyway, so these are placeholders
    // for any future rat dandiset that does.
    electrodeSize: 0.3,
    electrodePickThreshold: 0.5,
    rootOpacity: 0.25,
    coordSystem: 'ras',
    attribution: 'Atlas: WHS-SD v4 (Kleven et al. 2023)',
    attributionUrl: 'https://www.nitrc.org/projects/whs-sd-atlas',
    regionLinkTemplate: null,
  },
};

// Short label shown before electrode coordinates in the hover tooltip.
// Each atlas has its own coordinate frame (Allen uses CCF µm in PIR;
// macaque atlases use native RAS mm), so the tooltip should not lie by
// calling everything "CCF".
const ATLAS_COORD_LABELS = {
  allen_ccf: 'CCF',
  d99: 'D99',
  nmt: 'NMT',
  mebrains: 'MEBRAINS',
  whs_sd: 'WHS',
};

let activeAtlasKey = 'allen_ccf';
let activeAtlas = ATLAS_CONFIGS.allen_ccf;

// Single source of truth for which view the user is currently in. Set by the
// navigation functions (enterRegionView, enterDandisetView, enterSubjectViewFromURL,
// applyURLState's no-hash branch). Read by view-conditional logic such as
// the electrode auto-drop in showElectrodePointsForAssets.
//
// Values:
//   'init'     — atlas init view (root selected, no dandiset).
//   'region'   — a non-root region is selected (no dandiset).
//   'dandiset' — a dandiset is selected, no specific subject yet.
//   'subject'  — a subject within a dandiset is selected, no specific session.
//   'session'  — a specific session/asset is selected (electrodes typically shown).
//
// This is somewhat redundant with selectedId / selectedDandiset / etc. but
// centralizes the "what view" question so callers don't have to reconstruct
// it from multiple flags. The flags remain authoritative for their specific
// data (selectedId is the actual region ID); currentView is the categorical
// summary.
let currentView = 'init';

// Sets for quick lookups
let dataStructureIds = new Set();
let ancestorStructureIds = new Set();
let noMeshIds = new Set();
let dandisetToStructures = {};  // dandiset_id -> [structure_ids]
let dandisetTitles = {};        // dandiset_id -> title string
let dandisetAssets = {};        // dandiset_id -> [{path, asset_id, regions}]
let selectedDandiset = null;
let dandisetElectrodes = {};  // cache: dandiset_id -> {asset_id: [[x,y,z], ...]}
let electrodePoints = null;   // THREE.Points object
let sliderRegionOpacity = 1;          // mirrors the Regions slider value (range 0-1)
let sliderElectrodeOpacity = 1;       // mirrors the Electrodes slider value (range 0-1)

// Default Regions slider value when entering a view that displays electrodes.
// Auto-applied in showElectrodePointsForAssets so electrodes inside the active
// region aren't occluded by it. User can re-raise after; the auto-drop only
// fires when the slider was higher than this value.
const ELECTRODE_VIEW_DEFAULT_REGION_OPACITY = 0.5;
let dandisetRegionFilter = null; // structure_id when filtering subjects by region within a dandiset
let dandisetSubjectCounts = null; // { directSubjects, totalSubjects } when a dandiset is selected
let hiddenRegionIds = new Set();  // regions toggled off by user in dandiset/subject view
let dandisetsWithElectrodes = new Set();  // dandiset IDs that have electrode coordinate data
const SESSION_ELECTRODE_COLORS = [
  'ff4466',
  '4dd6ff',
  'ffd166',
  '8aff80',
  'c084fc',
  'ff9f43',
  '00d1b2',
  'f78fb3',
];

// ── Atlas Loading ──────────────────────────────────────────────────────────
async function loadAtlas(atlasKey) {
  activeAtlasKey = atlasKey;
  activeAtlas = ATLAS_CONFIGS[atlasKey];

  // Re-tune the electrode picking tolerance for the new atlas's world scale
  // (µm vs mm). Without this the old threshold carries over and hovers pick
  // incorrectly on the new atlas.
  if (raycaster) raycaster.params.Points.threshold = activeAtlas.electrodePickThreshold;

  showLoading();
  updateLoadingText('Fetching data...');

  // Clear existing state
  clearElectrodePoints();
  selectedId = null;
  hoveredId = null;
  selectedDandiset = null;
  dandisetElectrodes = {};
  dandisetRegionFilter = null;
  dandisetSubjectCounts = null;
  hiddenRegionIds = new Set();
  idToStructure = {};
  dandisetToStructures = {};

  // Reset the Regions slider to full so any auto-drop from the previous
  // atlas's electrode view doesn't bleed into the new atlas's init view.
  // Same reasoning as the reset in enterInitView; loadAtlas is the other
  // entry point that lands the user on a fresh "everything" state.
  const regionSlider = document.getElementById('region-opacity');
  if (regionSlider && parseFloat(regionSlider.value) !== 1) {
    regionSlider.value = 1;
    sliderRegionOpacity = 1;
  }

  // Remove existing meshes from worldRoot
  for (const [id, mesh] of Object.entries(meshObjects)) {
    worldRoot.remove(mesh);
    mesh.geometry.dispose();
    if (mesh.material.dispose) mesh.material.dispose();
  }
  meshObjects = {};
  failedMeshIds.clear();

  const v = `?v=${DATA_VERSION}`;
  const [graphResp, regionsResp, manifestResp, assetsResp, electrodeManifestResp] = await Promise.all([
    fetch(`${activeAtlas.dataPrefix}structure_graph.json${v}`).then(r => r.json()),
    fetch(`${activeAtlas.dataPrefix}dandi_regions.json${v}`).then(r => r.json()),
    fetch(`${activeAtlas.dataPrefix}mesh_manifest.json${v}`).then(r => r.json()),
    fetch(`${activeAtlas.dataPrefix}dandiset_assets.json${v}`).then(r => r.json()),
    fetch(`${activeAtlas.dataPrefix}dandisets_with_electrodes.json${v}`).then(r => r.json()).catch(() => []),
  ]);

  structureGraph = graphResp;
  dandiRegions = regionsResp;
  meshManifest = manifestResp;
  dandisetAssets = assetsResp;

  dandisetsWithElectrodes = new Set(electrodeManifestResp);
  dataStructureIds = new Set(meshManifest.data_structures);
  ancestorStructureIds = new Set(meshManifest.ancestor_structures);
  noMeshIds = new Set(meshManifest.no_mesh || []);

  // Build flat lookup from the tree
  flattenTree(structureGraph);

  // Build reverse lookup: dandiset -> structure IDs (direct only)
  for (const [sid, region] of Object.entries(dandiRegions)) {
    for (const did of region.dandisets) {
      if (!dandisetToStructures[did]) dandisetToStructures[did] = [];
      dandisetToStructures[did].push(parseInt(sid));
    }
  }

  // Reconfigure camera for this atlas
  camera.near = activeAtlas.nearPlane;
  camera.far = activeAtlas.farPlane;
  camera.up.set(...activeAtlas.cameraUp);
  camera.updateProjectionMatrix();

  // Update attribution
  const attrEl = document.querySelector('.ccf-attribution');
  if (attrEl) {
    attrEl.textContent = activeAtlas.attribution;
    attrEl.href = activeAtlas.attributionUrl;
  }

  buildHierarchyTree();

  updateLoadingText('Loading brain meshes...');
  await loadInitialMeshes();

  // Select root BEFORE hiding the loading overlay so the user never sees
  // the undimmed "speckled brain" state between the last mesh load and
  // syncSceneToSelection. Late-arriving ancestor meshes (triggered by
  // spotlightRegion's own toLoad queue) dim themselves via loadMesh's
  // post-add isolation check.
  const rootNode = structureGraph[0];
  if (rootNode) enterRegionView(rootNode.id, { expandTree: true, pushState: false });

  hideLoading();

  // Fetch dandiset titles in background
  fetchDandisetTitles();
}

// ── Initialization ─────────────────────────────────────────────────────────
let sceneInitialized = false;

async function init() {
  const urlParams = new URLSearchParams(window.location.search);
  const atlasParam = urlParams.get('atlas');
  const hasHash = !!window.location.hash.slice(1);
  const directEntry = (atlasParam && ATLAS_CONFIGS[atlasParam]) || hasHash;

  if (directEntry) {
    if (atlasParam && ATLAS_CONFIGS[atlasParam]) {
      activeAtlasKey = atlasParam;
      activeAtlas = ATLAS_CONFIGS[atlasParam];
    }
    await enterAtlas(activeAtlasKey, { pushState: false });
    if (hasHash) applyURLState();
  } else {
    // Bare load: render the landing and defer scene/atlas setup until a card
    // is clicked. The loading overlay is for atlas mesh fetches, not for the
    // landing itself.
    hideLoading();
    setupLanding();
  }
}

function ensureSceneInitialized() {
  if (sceneInitialized) return;
  setupScene();
  setupSearch();
  animate();

  window.addEventListener('hashchange', () => applyURLState());

  const selector = document.getElementById('atlas-selector');
  if (selector) {
    selector.addEventListener('change', (e) => {
      const newAtlas = e.target.value;
      if (newAtlas !== activeAtlasKey) {
        const url = new URL(window.location);
        url.searchParams.set('atlas', newAtlas);
        url.hash = '';
        window.history.replaceState({}, '', url);
        loadAtlas(newAtlas);
      }
    });
  }

  fetch('data/last_updated.json').then(r => r.json()).catch(() => null).then(resp => {
    if (resp && resp.timestamp) {
      const date = new Date(resp.timestamp);
      const formatted = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      const el = document.getElementById('last-updated');
      if (el) el.textContent = `Data updated ${formatted}`;
    }
  });

  sceneInitialized = true;
}

async function enterAtlas(atlasKey, { pushState = true } = {}) {
  document.body.classList.remove('landing-active');
  ensureSceneInitialized();

  const selector = document.getElementById('atlas-selector');
  if (selector) selector.value = atlasKey;

  if (pushState) {
    const url = new URL(window.location);
    url.searchParams.set('atlas', atlasKey);
    window.history.pushState({}, '', url);
  }

  await loadAtlas(atlasKey);
}

async function setupLanding() {
  const grid = document.getElementById('atlas-landing-grid');
  if (!grid) return;
  let index;
  try {
    const resp = await fetch('data/atlases_index.json');
    index = await resp.json();
  } catch (err) {
    console.error('Failed to load atlases_index.json', err);
    return;
  }

  const fmt = (n) => n.toLocaleString('en-US');
  const plural = (n, singular, pluralForm) => (n === 1 ? singular : pluralForm);
  grid.innerHTML = '';
  for (const atlas of index.atlases) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'atlas-card';
    card.dataset.atlasKey = atlas.key;
    card.innerHTML = `
      <img class="atlas-card-image" src="${atlas.preview}" alt="${atlas.name} brain preview" loading="lazy">
      <div class="atlas-card-body">
        <h3 class="atlas-card-title">${atlas.name}</h3>
        <div class="atlas-card-species">${atlas.species}</div>
        <div class="atlas-card-stats">
          <div class="atlas-card-stat">
            <span class="atlas-card-stat-value">${fmt(atlas.dandiset_count)}</span>
            <span class="atlas-card-stat-label">${plural(atlas.dandiset_count, 'dandiset', 'dandisets')}</span>
          </div>
          <div class="atlas-card-stat">
            <span class="atlas-card-stat-value">${fmt(atlas.file_count)}</span>
            <span class="atlas-card-stat-label">${plural(atlas.file_count, 'NWB file', 'NWB files')}</span>
          </div>
          <div class="atlas-card-stat">
            <span class="atlas-card-stat-value">${fmt(atlas.regions_with_data)}</span>
            <span class="atlas-card-stat-label">${plural(atlas.regions_with_data, 'region with data', 'regions with data')}</span>
          </div>
        </div>
      </div>`;
    card.addEventListener('click', () => {
      if (ATLAS_CONFIGS[atlas.key]) {
        activeAtlasKey = atlas.key;
        activeAtlas = ATLAS_CONFIGS[atlas.key];
        enterAtlas(atlas.key);
      }
    });
    grid.appendChild(card);
  }
}

function flattenTree(nodes) {
  for (const node of nodes) {
    idToStructure[node.id] = node;
    if (node.children) flattenTree(node.children);
  }
}

async function fetchDandisetTitles() {
  const ids = Object.keys(dandisetToStructures);
  const batchSize = 10;
  for (let i = 0; i < ids.length; i += batchSize) {
    const batch = ids.slice(i, i + batchSize);
    await Promise.all(batch.map(async (did) => {
      try {
        const resp = await fetch(`https://api.dandiarchive.org/api/dandisets/${did}/`);
        if (!resp.ok) return;
        const data = await resp.json();
        const title = data.draft_version?.name || data.most_recent_published_version?.name;
        if (title) dandisetTitles[did] = title;
      } catch { /* skip failures silently */ }
    }));
    // After each batch, refresh any currently-displayed dandiset info
    refreshDandisetDisplays();
  }
}

function refreshDandisetDisplays() {
  // Update dandiset card titles in the right panel
  document.querySelectorAll('.dandiset-card-title').forEach(el => {
    const did = el.dataset.dandisetId;
    if (dandisetTitles[did]) el.textContent = dandisetTitles[did];
  });
  // Update dandiset detail panel title if viewing a dandiset
  const detailTitle = document.getElementById('dandiset-detail-title');
  if (detailTitle && selectedDandiset && dandisetTitles[selectedDandiset]) {
    detailTitle.textContent = dandisetTitles[selectedDandiset];
  }
  // Update filter bar label
  const filterLabel = document.getElementById('dandiset-filter-label');
  if (filterLabel && selectedDandiset && dandisetTitles[selectedDandiset]) {
    filterLabel.textContent = `${dandisetTitles[selectedDandiset]} (${selectedDandiset})`;
  }
}

// ── Three.js Scene Setup ───────────────────────────────────────────────────
function setupScene() {
  const canvas = document.getElementById('brain-canvas');
  const viewer = document.getElementById('viewer');

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  renderer.setClearColor(0x1a1a2e, 1);

  scene = new THREE.Scene();
  worldRoot = new THREE.Group();
  worldRoot.name = 'worldRoot';
  scene.add(worldRoot);

  camera = new THREE.PerspectiveCamera(
    45,
    viewer.clientWidth / viewer.clientHeight,
    activeAtlas.nearPlane,
    activeAtlas.farPlane
  );
  camera.position.set(0, 0, activeAtlas.coordSystem === 'allen' ? 20000 : activeAtlas.camDist);
  camera.up.set(...activeAtlas.cameraUp);

  recreateOrbitControls();

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(10000, 10000, 10000);
  scene.add(dirLight);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dirLight2.position.set(-10000, -5000, -10000);
  scene.add(dirLight2);

  // Raycaster for picking. Points threshold is the world-space distance
  // tolerance for Three.js to register an electrode as "hit" by the ray,
  // so it must match the scale of the atlas (µm for Allen, mm for macaque).
  // The value is read from activeAtlas on each atlas switch in loadAtlas.
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = activeAtlas.electrodePickThreshold;
  mouse = new THREE.Vector2();

  // Events
  canvas.addEventListener('mousemove', onMouseMove);
  canvas.addEventListener('pointerdown', (e) => { pointerDownPos = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener('click', onClick);
  window.addEventListener('resize', onResize);
}

function onResize() {
  const viewer = document.getElementById('viewer');
  camera.aspect = viewer.clientWidth / viewer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
}

// ── Mesh Loading ───────────────────────────────────────────────────────────
const gltfLoader = new GLTFLoader();
const failedMeshIds = new Set();

function loadMesh(structureId) {
  return new Promise((resolve) => {
    if (meshObjects[structureId]) {
      resolve(meshObjects[structureId]);
      return;
    }

    const path = `${activeAtlas.dataPrefix}meshes/${structureId}.glb?v=${DATA_VERSION}`;
    gltfLoader.load(
      path,
      (gltf) => {
        let mesh = null;
        gltf.scene.traverse((child) => {
          if (!mesh && child.isMesh) mesh = child;
        });
        if (!mesh) { resolve(null); return; }

        // Get color and style based on whether this structure has data
        const isRoot = structureId === meshManifest.root_id;
        const region = dandiRegions[String(structureId)];
        // Root is never treated as a data region even if a stale
        // dandi_regions.json carries a zero-count entry for it — that was
        // visible in NMT as a "0 dandisets" tooltip on hover of the outline.
        const hasData = !isRoot && !!region;
        const s = idToStructure[structureId];

        let color = 0xaaaaaa;
        if (region && region.color_hex_triplet) {
          color = parseInt(region.color_hex_triplet, 16);
        } else if (s && s.color_hex_triplet) {
          color = parseInt(s.color_hex_triplet, 16);
        }

        let material;
        if (isRoot) {
          material = new THREE.MeshPhongMaterial({
            color: 0xcccccc,
            transparent: true,
            opacity: activeAtlas.rootOpacity,
            side: THREE.FrontSide,
            depthWrite: false,
          });
        } else if (hasData) {
          // Opacity scaled by log of total (including descendants) dandiset count
          const count = region.total_dandiset_count || region.dandiset_count || 1;
          const opacity = Math.min(0.3 + 0.2 * Math.log2(count + 1), 0.9);
          material = new THREE.MeshPhongMaterial({
            color,
            transparent: true,
            opacity,
            side: THREE.DoubleSide,
          });
        } else if (activeAtlas.coordSystem === 'allen') {
          // CCF: wireframe context (original behavior)
          material = new THREE.MeshPhongMaterial({
            color,
            transparent: true,
            opacity: 0.05,
            wireframe: true,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
        } else {
          // Macaque: low-opacity solid for anatomical context
          material = new THREE.MeshPhongMaterial({
            color,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
        }

        mesh.material = material;
        mesh.userData.structureId = structureId;
        mesh.userData.isData = hasData;
        mesh.userData.isRoot = isRoot;
        mesh.userData.originalMaterial = material.clone();

        worldRoot.add(mesh);
        meshObjects[structureId] = mesh;

        // When meshes load asynchronously mid-selection, give them the same
        // display mode syncSceneToSelection would assign so late arrivals don't
        // flash in visible. Uses the same decideDisplayMode/applyDisplayMode primitives
        // as the orchestrator, so policy lives in one place.
        if (selectedId !== null) {
          applyDisplayMode(mesh, decideDisplayMode(structureId, new Set([selectedId])));
        } else if (selectedDandiset !== null) {
          const dandiStructures = new Set(dandisetToStructures[selectedDandiset] || []);
          applyDisplayMode(mesh, decideDisplayMode(structureId, dandiStructures));
        }

        resolve(mesh);
      },
      undefined,
      () => { failedMeshIds.add(structureId); resolve(null); }
    );
  });
}

async function ensureMeshLoaded(structureId) {
  if (meshObjects[structureId]) return meshObjects[structureId];
  if (failedMeshIds.has(structureId) || noMeshIds.has(structureId)) return null;
  return loadMesh(structureId);
}

async function loadInitialMeshes() {
  // Load root brain outline first
  await loadMesh(meshManifest.root_id);

  // Determine which meshes to load
  let allToLoad;
  if (activeAtlas.coordSystem === 'allen') {
    // CCF: only load data structures (original behavior)
    allToLoad = meshManifest.data_structures.filter(id => id !== meshManifest.root_id);
  } else {
    // Macaque: load every GLB produced by the build so the anatomical tree
    // is fully populated even when the user navigates to a non-data region.
    // The list is precomputed into meshManifest.all_meshes by the build
    // script. We used to scrape the server's HTML directory index at runtime,
    // but that relies on auto-indexing which Netlify (and most CDNs) don't
    // serve, so the deployed site was falling back to data_structures only.
    const allMeshIds = meshManifest.all_meshes || meshManifest.data_structures;
    allToLoad = allMeshIds.filter(id =>
      id !== meshManifest.root_id && !noMeshIds.has(id)
    );
  }
  const batchSize = 20;
  for (let i = 0; i < allToLoad.length; i += batchSize) {
    const batch = allToLoad.slice(i, i + batchSize);
    await Promise.all(batch.map(id => loadMesh(id)));
    updateLoadingText(`Loading meshes... ${Math.min(i + batchSize, allToLoad.length)}/${allToLoad.length}`);
  }

  // Center camera on the brain
  if (meshObjects[meshManifest.root_id]) {
    const box = new THREE.Box3().setFromObject(meshObjects[meshManifest.root_id]);
    brainCenter = box.getCenter(new THREE.Vector3());

    // Position camera along the atlas's offset axis
    const off = activeAtlas.camOffset;
    camera.position.set(
      brainCenter.x + off[0] * activeAtlas.camDist,
      brainCenter.y + off[1] * activeAtlas.camDist,
      brainCenter.z + off[2] * activeAtlas.camDist
    );
    camera.up.set(...activeAtlas.cameraUp);

    // Always recreate OrbitControls so it caches a fresh quaternion
    // from the current up vector. Without this, switching atlases
    // leaves the old atlas's quaternion baked into the controls.
    recreateOrbitControls(brainCenter);
  }
}


// ── Raycasting & Interaction ───────────────────────────────────────────────
function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const tooltip = document.getElementById('tooltip');

  if (electrodePoints?.parent) {
    const electrodeHits = raycaster.intersectObject(electrodePoints, false);
    if (electrodeHits.length > 0) {
      if (hoveredId !== null) {
        unhighlightMesh(hoveredId);
        hoveredId = null;
      }

      const hit = electrodeHits[0];
      const point = electrodePoints.userData.pointInfo?.[hit.index];
      if (point) {
        tooltip.classList.remove('hidden');
        tooltip.innerHTML = `
          <div class="tooltip-name">Electrode ${point.index + 1}</div>
          <div class="tooltip-acronym">${escapeHtml(point.sessionLabel)}</div>
          <div class="tooltip-info">${escapeHtml(point.subjectLabel)} &middot; ${escapeHtml(point.assetLabel)}</div>
          <div class="tooltip-info">${ATLAS_COORD_LABELS[activeAtlasKey] || activeAtlasKey}: ${formatCoord(point.coord[0])}, ${formatCoord(point.coord[1])}, ${formatCoord(point.coord[2])}</div>
        `;
        tooltip.style.left = (event.clientX - rect.left + 15) + 'px';
        tooltip.style.top = (event.clientY - rect.top + 15) + 'px';
        renderer.domElement.style.cursor = 'crosshair';
        return;
      }
    }
  }

  const intersects = raycaster.intersectObjects(getHoverPickables(), false);
  const brainHit = pickBrainRegionHit(intersects);

  if (brainHit) {
    const hit = brainHit.object;
    const sid = hit.userData.structureId;

    if (hoveredId !== sid) {
      // Un-highlight previous
      unhighlightMesh(hoveredId);
      hoveredId = sid;
      highlightMesh(sid);
    }

    // Update tooltip. Use the aggregate (total across descendants) counts —
    // intermediate nodes like "Motor cortex" have 0 direct annotations but
    // non-zero descendant counts, and showing "0 dandisets" there would be
    // misleading. Leaf regions have total == direct so the value is the same.
    // Matches how the tree badge renders totals via renderBadge.
    const region = dandiRegions[String(sid)];
    if (region) {
      const dsCount = region.total_dandiset_count ?? region.dandiset_count ?? 0;
      const fileCount = region.total_file_count ?? region.file_count ?? 0;
      tooltip.classList.remove('hidden');
      tooltip.innerHTML = `
        <div class="tooltip-name">${region.name}</div>
        <div class="tooltip-acronym">${region.acronym}</div>
        <div class="tooltip-info">${dsCount} dandiset${dsCount !== 1 ? 's' : ''} &middot; ${fileCount} files</div>
      `;
      tooltip.style.left = (event.clientX - renderer.domElement.getBoundingClientRect().left + 15) + 'px';
      tooltip.style.top = (event.clientY - renderer.domElement.getBoundingClientRect().top + 15) + 'px';
    }
    renderer.domElement.style.cursor = 'pointer';
  } else {
    if (hoveredId !== null) {
      unhighlightMesh(hoveredId);
      hoveredId = null;
    }
    tooltip.classList.add('hidden');
    renderer.domElement.style.cursor = 'default';
  }
}

function highlightMesh(structureId) {
  const mesh = meshObjects[structureId];
  if (!mesh || !mesh.visible) return;
  mesh.material.emissive = new THREE.Color(0x335577);
  mesh.material.emissiveIntensity = 0.5;
}

function unhighlightMesh(structureId) {
  if (structureId === null) return;
  const mesh = meshObjects[structureId];
  if (!mesh || !mesh.visible) return;
  if (structureId === selectedId) return;
  mesh.material.emissive = new THREE.Color(0x000000);
  mesh.material.emissiveIntensity = 0;
}

let pointerDownPos = null;

function onClick(event) {
  if (pointerDownPos) {
    const dx = event.clientX - pointerDownPos.x;
    const dy = event.clientY - pointerDownPos.y;
    if (dx * dx + dy * dy > 9) return; // ignore drags (>3px)
  }
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  const intersects = raycaster.intersectObjects(getClickPickables(), false);
  const brainHit = pickBrainRegionHit(intersects);

  if (brainHit) {
    const sid = brainHit.object.userData.structureId;
    if (selectedDandiset) {
      filterDandisetPanelByRegion(sid);
    } else {
      enterRegionView(sid);
    }
  }
}

// Hover includes root when it's the solid init-view mesh, so hovering the
// whole brain at the atlas view surfaces the aggregate tooltip. Root is
// excluded while dimmed (fresnel silhouette during selection) — the rim is
// pure context, not a UI target.
function getHoverPickables() {
  return Object.values(meshObjects).filter(
    m => m.visible && (m.userData.isData || (m.userData.isRoot && !m.userData.isDimmed))
  );
}

// Click never targets root. Navigation back to the init view goes through
// the hierarchy tree's root node, not the 3D mesh — giving root a click
// action would create a large accidental-click surface during camera drags.
function getClickPickables() {
  return Object.values(meshObjects).filter(m => m.userData.isData && m.visible);
}

function pickBrainRegionHit(intersects) {
  if (intersects.length === 0) return null;
  return intersects.find(hit => hit.object.userData.structureId !== meshManifest.root_id) || intersects[0];
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatCoord(value) {
  return Number.isInteger(value) ? value.toString() : Number(value).toFixed(1);
}

function getDescendantIds(structureId) {
  const ids = new Set();
  function walk(node) {
    ids.add(node.id);
    if (node.children) node.children.forEach(walk);
  }
  const s = idToStructure[structureId];
  if (s) walk(s);
  return ids;
}

// ── Mesh display mode: policy + mechanism ───────────────────────────────────
//
// The "display mode" of a mesh is its visual role in the current scene state.
// Four possible values:
//   'active'     — spotlight; the focused content of the current view.
//   'glass'      — translucent original material; Allen root only, always.
//   'silhouette' — fresnel-rim outline; macaque root when not active.
//   'hidden'     — visible=false; anything else not in the active set.
//
// Each mesh stores its current display mode in mesh.userData.displayMode.
//
// Two functions cleanly split the concerns:
//
// - decideDisplayMode(meshId, activeIds): pure policy. Given the current
//   selection's active set, returns the display mode for a mesh ID.
//
// - applyDisplayMode(mesh, mode): the single mutator. Given a display mode,
//   sets all relevant material/visibility/depth properties, derived from the
//   current sliderRegionOpacity and atlas. Every code path that mutates a mesh's
//   visual state goes through this function — there is no other place that
//   knows about depthWrite, transparent, etc. This prevents the class of bug
//   where one call site (e.g. the slider handler) updates opacity but forgets
//   depthWrite, leaving the mesh in an inconsistent state.

// Build the fresnel-rim ShaderMaterial used for the macaque root in dimmed
// state. Cached on mesh.userData.outlineMaterial so the shader is compiled
// once per mesh, not per dim cycle.
function buildFresnelMaterial(origMaterial) {
  const color = origMaterial.color ? origMaterial.color.clone() : new THREE.Color(0xcccccc);
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color },
      uRimPower: { value: 2.5 },
      uRimStrength: { value: 0.9 },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewDir = normalize(-mvPosition.xyz);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uRimPower;
      uniform float uRimStrength;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        float facing = abs(dot(normalize(vNormal), vViewDir));
        float rim = pow(1.0 - facing, uRimPower);
        gl_FragColor = vec4(uColor, rim * uRimStrength);
      }
    `,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });
}

// Pure policy: which display mode should this mesh have given the active set?
// Returns one of:
//   'active'     — spotlight (visible, opacity from slider, depthWrite if opaque).
//   'glass'      — translucent original material; currently only Allen root.
//   'silhouette' — fresnel-rim outline; currently only non-active macaque root.
//   'hidden'     — mesh.visible = false; everything else not in the active set.
function decideDisplayMode(meshId, activeIds) {
  const isAllen = activeAtlas.coordSystem === 'allen';
  const isRoot = meshId === meshManifest.root_id;
  // Allen root is always glass, regardless of selection state.
  if (isRoot && isAllen) return 'glass';
  // In the active set → spotlight.
  if (activeIds.has(meshId)) return 'active';
  // Macaque root, not in active set → fresnel-rim silhouette for context.
  if (isRoot) return 'silhouette';
  // Anything else not active → completely hidden.
  return 'hidden';
}

// Single mutator. Apply a display mode to a mesh, deriving all properties from
// current sliderRegionOpacity and atlas. Sets mesh.userData.displayMode so callers
// that re-apply (slider handler) can read back the current mode.
function applyDisplayMode(mesh, mode) {
  const orig = mesh.userData.originalMaterial;
  if (!orig) return;
  const isAllen = activeAtlas.coordSystem === 'allen';
  const isRoot = mesh.userData.isRoot;

  if (mode === 'active') {
    if (sliderRegionOpacity === 0) {
      mesh.visible = false;
    } else {
      const mat = orig.clone();
      mat.opacity = sliderRegionOpacity;
      mat.transparent = sliderRegionOpacity < 1;
      mat.depthWrite = sliderRegionOpacity >= 1;
      if (isRoot && !isAllen) {
        // Macaque root coming out of fresnel mode: reset state the
        // outline material left on the mesh so root renders as a solid brain.
        mat.depthTest = true;
        mesh.renderOrder = 0;
      }
      mat.needsUpdate = true;
      mesh.material = mat;
      mesh.visible = true;
    }
  } else if (mode === 'glass') {
    // Allen root only: original (translucent) material at slider-modulated alpha.
    if (sliderRegionOpacity === 0) {
      mesh.visible = false;
    } else {
      const mat = orig.clone();
      mat.opacity = orig.opacity * sliderRegionOpacity;
      mat.transparent = mat.opacity < 1;
      mat.needsUpdate = true;
      mesh.material = mat;
      mesh.visible = true;
    }
  } else if (mode === 'silhouette') {
    // Fresnel-rim outline. Currently only used for non-active macaque root,
    // to provide spatial context without the "fog" that a flat-alpha outline
    // produced.
    if (!mesh.userData.outlineMaterial) {
      mesh.userData.outlineMaterial = buildFresnelMaterial(orig);
    }
    mesh.material = mesh.userData.outlineMaterial;
    mesh.renderOrder = -1;
    mesh.visible = true;
  } else if (mode === 'hidden') {
    // Not drawn. Used for everything not in the active set on Allen, and for
    // macaque non-root meshes not in the active set.
    mesh.visible = false;
  }

  mesh.userData.displayMode = mode;
  // Back-compat: existing readers of `isDimmed` (hover-pickable filter at
  // ~line 734, region-visibility overlay handler) check this flag. Maintained
  // as a derived value from `displayMode` — true when the mesh is in any
  // non-focal state (silhouette or hidden) — so we don't have to update
  // every reader at once. New code should read `displayMode` directly.
  mesh.userData.isDimmed = (mode === 'silhouette' || mode === 'hidden');
}

// Thin wrappers for existing call sites (e.g. updateDandisetPanel's checkbox
// handlers). New code should prefer applyDisplayMode(mesh, mode) directly. The
// applyDimmed wrapper dispatches to 'hidden' because its callers (right-panel
// region toggles) only ever target data meshes, never the macaque root.
function applyActive(mesh)     { applyDisplayMode(mesh, 'active'); }
function applyDimmed(mesh)     { applyDisplayMode(mesh, 'hidden'); }
function restoreOriginal(mesh) { applyDisplayMode(mesh, 'glass');  }

function findNearestAncestorWithMesh(structureId) {
  // Walk up the hierarchy to find the closest ancestor that has a loaded mesh
  let current = idToStructure[structureId]?.parent_structure_id;
  while (current != null) {
    if (meshObjects[current]) return current;
    current = idToStructure[current]?.parent_structure_id;
  }
  return null;
}

function spotlightRegion(structureId) {
  // Resolve the mesh ID to spotlight: the structure itself if its mesh is loaded,
  // otherwise the nearest ancestor that has a loaded mesh.
  const targetId = meshObjects[structureId] ? structureId : findNearestAncestorWithMesh(structureId);
  const activeSet = targetId != null ? new Set([targetId]) : new Set();

  // Speculatively pre-load descendant meshes so navigation into the subtree is
  // instant. Note: descendants are NOT made visible in region view (only the
  // clicked mesh is — see syncSceneToSelection). The pre-load is a cache warm,
  // not a visibility decision.
  const subtreeIds = getDescendantIds(structureId);
  if (targetId != null) subtreeIds.add(targetId);
  const toLoad = [];
  for (const id of subtreeIds) {
    if (!meshObjects[id] && (dataStructureIds.has(id) || ancestorStructureIds.has(id))) {
      toLoad.push(ensureMeshLoaded(id));
    }
  }
  Promise.all(toLoad).then(() => syncSceneToSelection(activeSet));

  // Apply immediately to already-loaded meshes
  syncSceneToSelection(activeSet);
}

// Scene-level orchestrator. Brings every loaded mesh's display mode into
// sync with the given selection: walks meshObjects, asks decideDisplayMode
// for each mesh's intended mode, optionally demotes 'active' to 'hidden'
// when the user-override callback says so, and dispatches to applyDisplayMode
// to mutate the material. This single function is called by every view that
// changes scene state — region, dandiset, subject, session, atlas init — so
// the visibility policy lives in exactly one place.
//
// activeIds: Set of structure IDs that should be displayed as 'active'.
// For region view this is a singleton; for dandiset/subject/session views
// it's the set of regions covered by the dandiset/subject/session. For the
// atlas init view it contains just the root.
//
// shouldHideMesh: optional callback (meshId) => boolean. If provided and
// returns true for a given mesh, the mesh's mode is demoted to 'hidden'
// even if it's in activeIds. Used by dandiset/subject/session views to
// honor per-region visibility toggles in the right-panel overlay.
//
// Display mode dispatch (per mesh, computed by decideDisplayMode):
// - Allen root: always 'glass', regardless of selection.
// - Macaque root in activeIds (atlas init view only): 'active' (solid opaque).
// - Macaque root not in activeIds: 'silhouette' (fresnel rim).
// - Active data mesh (and not user-hidden): 'active'.
// - Active data mesh that is user-hidden: 'hidden'.
// - Non-active mesh: 'hidden' (on both atlases).
function syncSceneToSelection(activeIds, shouldHideMesh = null) {
  for (const [idStr, mesh] of Object.entries(meshObjects)) {
    const id = parseInt(idStr);
    let mode = decideDisplayMode(id, activeIds);
    // Honor per-region user hide toggles (right-panel checkboxes): demote
    // an 'active' display mode to 'hidden' when the caller says this mesh is
    // user-hidden.
    if (mode === 'active' && shouldHideMesh && shouldHideMesh(id)) {
      mode = 'hidden';
    }
    applyDisplayMode(mesh, mode);
  }
}

function showAllRegions() {
  for (const mesh of Object.values(meshObjects)) {
    restoreOriginal(mesh);
  }
}

function computeDandisetSubjectCounts(dandisetId) {
  const assets = dandisetAssets[dandisetId] || [];

  // Build regionId -> Set<subjectDir> for direct counts
  const directSubjects = {};
  for (const asset of assets) {
    const parts = asset.path.split('/');
    const subjectDir = parts.length > 1 ? parts[0] : asset.path.split('_')[0];
    for (const r of asset.regions) {
      if (!directSubjects[r.id]) directSubjects[r.id] = new Set();
      directSubjects[r.id].add(subjectDir);
    }
  }

  // Build total counts by walking up ancestry
  const totalSubjects = {};
  for (const [ridStr, subjects] of Object.entries(directSubjects)) {
    const rid = parseInt(ridStr);
    if (!totalSubjects[rid]) totalSubjects[rid] = new Set();
    for (const s of subjects) totalSubjects[rid].add(s);
    let current = idToStructure[rid]?.parent_structure_id;
    while (current != null) {
      if (!totalSubjects[current]) totalSubjects[current] = new Set();
      for (const s of subjects) totalSubjects[current].add(s);
      current = idToStructure[current]?.parent_structure_id;
    }
  }

  return { directSubjects, totalSubjects };
}

function renderBadge(badge, nodeId) {
  const region = dandiRegions[String(nodeId)];
  if (!region) { badge.innerHTML = ''; badge.title = ''; return; }

  let direct, total, unit;
  if (dandisetSubjectCounts) {
    direct = dandisetSubjectCounts.directSubjects[nodeId]?.size || 0;
    total = dandisetSubjectCounts.totalSubjects[nodeId]?.size || 0;
    unit = 'subjects';
  } else {
    direct = region.dandiset_count;
    total = region.total_dandiset_count || direct;
    unit = 'dandisets';
  }

  if (total === 0) {
    badge.innerHTML = '';
    badge.title = '';
  } else if (direct > 0 && direct !== total) {
    badge.innerHTML = `<span class="badge-direct">${direct}</span><span class="badge-sep">/</span><span class="badge-total">${total}</span>`;
    badge.title = `${direct} direct, ${total} incl. sub-regions ${unit}`;
  } else if (direct > 0) {
    badge.innerHTML = `<span class="badge-direct">${direct}</span>`;
    badge.title = `${direct} ${unit}`;
  } else {
    badge.innerHTML = `<span class="badge-total">${total}</span>`;
    badge.title = `${total} ${unit} in sub-regions`;
  }
}

function updateTreeBadges() {
  document.getElementById('hierarchy-tree').querySelectorAll('.tree-badge').forEach(badge => {
    const content = badge.closest('.tree-node-content');
    if (content) renderBadge(badge, parseInt(content.dataset.id));
  });
}

async function enterDandisetView(dandisetId, { pushState = true } = {}) {
  return transitionView('dandiset', async () => {
    selectedDandiset = dandisetId;
    selectedId = null;
    dandisetSubjectCounts = computeDandisetSubjectCounts(dandisetId);
    hiddenRegionIds = new Set();
    clearElectrodePoints();

    if (pushState) setHash(`dandiset=${dandisetId}`);

    const structureIds = dandisetToStructures[dandisetId] || [];
    const activeSet = new Set(structureIds);

    // Ensure meshes are loaded for all structures in this dandiset
    const toLoad = [];
    for (const sid of structureIds) {
      if (!meshObjects[sid] && !failedMeshIds.has(sid) && !noMeshIds.has(sid)) {
        toLoad.push(ensureMeshLoaded(sid));
      }
    }
    if (toLoad.length > 0) await Promise.all(toLoad);

    // Build mapping: meshId -> [regionIds it represents]
    const meshToRegions = new Map();
    for (const sid of structureIds) {
      const meshId = meshObjects[sid] ? sid : findNearestAncestorWithMesh(sid);
      if (meshId) {
        activeSet.add(meshId);
        if (!meshToRegions.has(meshId)) meshToRegions.set(meshId, []);
        meshToRegions.get(meshId).push(sid);
      }
    }

    // Sync scene: active regions get 'active' display mode, root gets
    // atlas-appropriate mode, everything else is hidden. Honors per-region
    // visibility toggles via the shouldHideMesh callback.
    syncSceneToSelection(activeSet, (meshId) => {
      const regions = meshToRegions.get(meshId) || [meshId];
      return regions.every(rid => hiddenRegionIds.has(rid));
    });

    updateDandisetPanel(dandisetId, structureIds);
    filterTreeByDandiset(dandisetId);
    updateTreeBadges();
  });
}

function filterTreeByDandiset(dandisetId) {
  const structureIds = dandisetToStructures[dandisetId] || [];
  const activeIds = new Set(structureIds);

  // Add all ancestors so tree paths remain visible
  for (const sid of structureIds) {
    let current = idToStructure[sid]?.parent_structure_id;
    while (current != null) {
      activeIds.add(current);
      current = idToStructure[current]?.parent_structure_id;
    }
  }

  // Show the filter bar
  const filterBar = document.getElementById('dandiset-filter-bar');
  const filterLabel = document.getElementById('dandiset-filter-label');
  filterBar.classList.remove('hidden');
  filterLabel.textContent = dandisetTitles[dandisetId]
    ? `${dandisetTitles[dandisetId]} (${dandisetId})`
    : `Dandiset ${dandisetId}`;

  // Expand tree paths to matching nodes so they're visible
  for (const sid of structureIds) {
    expandToNode(sid);
  }

  // Apply dandiset-inactive class to tree nodes NOT in the active set
  const container = document.getElementById('hierarchy-tree');
  container.querySelectorAll('.tree-node').forEach(node => {
    const id = parseInt(node.dataset.id);
    const label = node.querySelector(':scope > .tree-node-content .tree-label');
    const dot = node.querySelector(':scope > .tree-node-content .tree-color-dot');
    if (!activeIds.has(id)) {
      if (label) { label.classList.add('dandiset-inactive'); label.classList.remove('dandiset-active'); }
      if (dot) { dot.classList.add('dandiset-inactive'); dot.classList.remove('dandiset-active'); }
    } else {
      if (label) { label.classList.remove('dandiset-inactive'); label.classList.add('dandiset-active'); }
      if (dot) { dot.classList.remove('dandiset-inactive'); dot.classList.add('dandiset-active'); }
    }
  });
}

// Cross-cutting boundary discipline that runs on every view transition.
// Per-view enter*View functions wrap their body in transitionView(newView,
// () => { ... }) so the shared cleanup happens in exactly one place and
// currentView gets updated by exactly one writer.
//
// What runs BEFORE per-view work:
// - currentView = newView. Single writer guarantees this categorical state
//   is always consistent with what just happened.
// - dandisetRegionFilter = null. Only the dandiset view ever uses this field
//   to mean anything, and entering any view starts that filter fresh.
// - Deselect any tree node carrying .selected. Per-view work re-adds it if
//   the new view wants a node selected (e.g. enterRegionView selects the
//   target region's node).
//
// What stays in per-view work:
// - selectedId / selectedDandiset writes (each view owns its primary identifier).
// - hiddenRegionIds reset (different rule per view: reset on init/region/dandiset,
//   inherit on subject/session).
// - clearElectrodePoints (region/dandiset/init clear; subject/session re-render
//   via showElectrodePoints*).
// - region-visibility-overlay show/hide (dandiset re-shows via panel render;
//   region/init hide).
// - Filter bar visibility.
// - Mesh display modes, panel render, hash push.
function transitionView(newView, work) {
  currentView = newView;
  dandisetRegionFilter = null;
  const sel = document.querySelector('.tree-node-content.selected');
  if (sel) sel.classList.remove('selected');
  return work();
}

// Reset selection state, DOM overlays, and the 3D scene back to the atlas-init
// look. Shared by every code path that returns the user to the "no selection"
// state: the dandiset-filter clear button (clearDandisetFilter) and the
// no-hash branch of applyURLState (popstate / external hash clear).
//
// Does NOT touch the URL hash — callers decide whether to push a clean URL
// (clearDandisetFilter does; applyURLState is reacting to a hash that's
// already empty).
//
// The Allen vs macaque 3D-view split is intentional: showAllRegions restores
// Allen's "frosted brain" look but on macaque would leak previously loaded
// region meshes, so macaque routes through enterRegionView(root) so only root
// is visible.
function enterInitView() {
  transitionView('init', () => {
    selectedId = null;
    selectedDandiset = null;
    dandisetSubjectCounts = null;
    hiddenRegionIds = new Set();

    // Restore the Regions slider to full. Any prior auto-drop from an
    // electrode view should not bleed into the "everything" view — root
    // mesh becomes 'active' here and would otherwise inherit the dropped
    // value as its applied opacity.
    const regionSlider = document.getElementById('region-opacity');
    if (regionSlider && parseFloat(regionSlider.value) !== 1) {
      regionSlider.value = 1;
      regionSlider.dispatchEvent(new Event('input', { bubbles: true }));
    }

    clearElectrodePoints();
    document.getElementById('region-visibility-overlay').classList.add('hidden');
    document.getElementById('dandiset-filter-bar').classList.add('hidden');
    hideSubjectFilter();

    // Tree: strip dandiset filter classes (selected node already deselected by transitionView)
    const tree = document.getElementById('hierarchy-tree');
    tree.querySelectorAll('.dandiset-inactive').forEach(el => el.classList.remove('dandiset-inactive'));
    tree.querySelectorAll('.dandiset-active').forEach(el => el.classList.remove('dandiset-active'));

    // 3D scene
    if (activeAtlas.coordSystem === 'allen') {
      showAllRegions();
    } else if (meshManifest && meshManifest.root_id != null) {
      enterRegionView(meshManifest.root_id, { pushState: false, expandTree: false });
    }

    updateTreeBadges();

    document.getElementById('region-panel').innerHTML =
      '<p class="placeholder-text">Click a brain region to view details and associated DANDI datasets.</p>';
  });
}

function clearDandisetFilter() {
  enterInitView();
  history.pushState(null, '', window.location.pathname);
}

function filterDandisetPanelByRegion(structureId, { pushState = true } = {}) {
  if (!selectedDandiset) return;

  dandisetRegionFilter = structureId;

  // Update URL hash to include region
  if (pushState) {
    setHash(`dandiset=${selectedDandiset}&region=${structureId}`);
  }

  // Re-render the panel with the region filter active
  const structureIds = dandisetToStructures[selectedDandiset] || [];
  updateDandisetPanel(selectedDandiset, structureIds);

  // Show region filter indicator
  const s = idToStructure[structureId];
  const regionName = s ? (s.name || s.acronym) : `Region ${structureId}`;
  showSubjectFilter(`Region: ${regionName}`);

  // Clear electrodes (no specific subject selected)
  clearElectrodePoints();

  // Update 3D view: isolate to matching structures within this dandiset
  const descendantIds = getDescendantIds(structureId);
  const dandiStructures = dandisetToStructures[selectedDandiset] || [];
  const matchingStructures = dandiStructures.filter(id => descendantIds.has(id));
  if (matchingStructures.length > 0) {
    spotlightRegions(matchingStructures);
  } else {
    spotlightRegions([structureId]);
  }

  // Highlight region in tree
  const prevEl = document.querySelector('.tree-node-content.selected');
  if (prevEl) prevEl.classList.remove('selected');
  const el = document.querySelector(`.tree-node-content[data-id="${structureId}"]`);
  if (el) el.classList.add('selected');
}

function showSubjectFilter(filterText) {
  const bar = document.getElementById('subject-filter-bar');
  const label = document.getElementById('subject-filter-label');
  bar.classList.remove('hidden');
  label.textContent = filterText;
}

function hideSubjectFilter() {
  document.getElementById('subject-filter-bar').classList.add('hidden');
}

async function updateDandisetPanel(dandisetId, structureIds) {
  const panel = document.getElementById('region-panel');
  const assets = dandisetAssets[dandisetId] || [];

  // Pre-fetch electrode data for this dandiset (lazy, cached)
  await fetchElectrodes(dandisetId);

  const title = dandisetTitles[dandisetId] || '';

  // Group assets by subject, merging regions across sessions
  const subjectMap = new Map();
  for (const asset of assets) {
    const parts = asset.path.split('/');
    const subjectDir = parts.length > 1 ? parts[0] : asset.path.split('_')[0];
    const subjectId = subjectDir.replace(/^sub-/, '');

    if (!subjectMap.has(subjectId)) {
      subjectMap.set(subjectId, { regions: new Map(), assets: [], subjectDir });
    }
    const entry = subjectMap.get(subjectId);
    entry.assets.push(asset);
    for (const r of asset.regions) {
      if (!entry.regions.has(r.id)) {
        entry.regions.set(r.id, r);
      }
    }
  }

  const uniqueRegionIds = new Set();
  for (const [, entry] of subjectMap) {
    for (const rid of entry.regions.keys()) uniqueRegionIds.add(rid);
  }

  const allRegionIds = [...uniqueRegionIds];
  // Natural sort: Intl.Collator with numeric: true sorts "2" before "10".
  // JS has no natsort equivalent, so this is the idiomatic zero-dependency approach.
  const natCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
  const allSubjects = [...subjectMap.entries()].sort((a, b) => natCollator.compare(a[0], b[0]));
  for (const [, entry] of allSubjects) {
    entry.assets.sort((a, b) => natCollator.compare(a.session || a.path, b.session || b.path));
  }

  // Filter subjects by region if a region filter is active
  let displaySubjects = allSubjects;
  if (dandisetRegionFilter) {
    const filterDescendants = getDescendantIds(dandisetRegionFilter);
    displaySubjects = allSubjects.filter(([, entry]) => {
      return [...entry.regions.keys()].some(rid => filterDescendants.has(rid));
    });
  }

  const subjects = displaySubjects;
  const PAGE_SIZE = 20;
  let currentPage = 0;
  const totalPages = Math.ceil(subjects.length / PAGE_SIZE);

  function render(page) {
    currentPage = page;
    const start = page * PAGE_SIZE;
    const end = Math.min(start + PAGE_SIZE, subjects.length);
    const pageSubjects = subjects.slice(start, end);

    let html = `
      <div class="region-header">
        <div class="region-name">Dandiset ${dandisetId}</div>
        <div class="dandiset-detail-title" id="dandiset-detail-title">${title}</div>
        <a class="dandiset-external-link" href="https://dandiarchive.org/dandiset/${dandisetId}" target="_blank" rel="noopener">
          View on DANDI Archive &#8599;
        </a>
      </div>
      <div class="region-stats">
        <div class="stat-item">
          <div class="stat-value">${dandisetRegionFilter ? `${subjects.length} / ${subjectMap.size}` : subjectMap.size}</div>
          <div class="stat-label">Subjects</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${uniqueRegionIds.size || structureIds.length}</div>
          <div class="stat-label">Brain Regions</div>
        </div>
      </div>
    `;

    if (subjectMap.size === 0) {
      html += '<p class="no-data-msg">No asset data available for this dandiset.</p>';
    } else {
      // Region visibility toggles
      const displayRegionIds = new Set();
      for (const [, entry] of subjects) {
        for (const rid of entry.regions.keys()) displayRegionIds.add(rid);
      }
      const regionList = [...displayRegionIds].map(rid => {
        const r = dandiRegions[String(rid)] || {};
        const s = idToStructure[rid] || {};
        return {
          id: rid,
          name: r.name || s.name || `Region ${rid}`,
          acronym: r.acronym || s.acronym || '',
          color: r.color_hex_triplet || s.color_hex_triplet || 'aaaaaa',
        };
      }).sort((a, b) => a.name.localeCompare(b.name));

      // Render region toggles into the 3D viewer overlay
      const toggleOverlay = document.getElementById('region-visibility-overlay');
      if (regionList.length > 0) {
        const wasCollapsed = toggleOverlay.classList.contains('collapsed');
        let toggleHtml = `<div class="region-visibility-header"><label class="region-visibility-all-label"><input type="checkbox" id="toggle-all-region-visibility"> Brain Regions (${regionList.length})</label><button class="region-visibility-collapse-btn" title="Collapse">${wasCollapsed ? '&#x25B6;' : '&#x25BC;'}</button></div>`;
        toggleHtml += `<div class="region-visibility-list">`;
        for (const r of regionList) {
          const checked = !hiddenRegionIds.has(r.id);
          toggleHtml += `<label class="region-visibility-row" title="${r.name}">`;
          toggleHtml += `<input type="checkbox" data-region-id="${r.id}" ${checked ? 'checked' : ''}>`;
          toggleHtml += `<span class="region-visibility-dot" style="background:#${r.color}"></span>`;
          toggleHtml += `<span class="region-visibility-name">${r.acronym || r.name}</span>`;
          toggleHtml += `</label>`;
        }
        toggleHtml += `</div>`;
        toggleOverlay.innerHTML = toggleHtml;
        toggleOverlay.classList.remove('hidden');
      } else {
        toggleOverlay.innerHTML = '';
        toggleOverlay.classList.add('hidden');
      }

      // "All Subjects" button above the header
      html += `<div class="asset-card${dandisetRegionFilter ? '' : ' asset-card-selected'}" data-region-ids='${JSON.stringify(allRegionIds)}' data-all="true">`;
      html += `<span class="asset-card-filename">All Subjects</span>`;
      html += `<span class="asset-card-region-count">${uniqueRegionIds.size} regions</span>`;
      html += `</div>`;

      html += `<div class="dandiset-list-header">Subjects <span class="dandiset-list-hint">click to filter</span></div>`;

      for (const [subjectId, entry] of pageSubjects) {
        const regionIds = JSON.stringify([...entry.regions.keys()]);
        const dandiFilesUrl = `https://dandiarchive.org/dandiset/${dandisetId}/draft/files?location=${encodeURIComponent(entry.subjectDir)}`;
        const regionCount = entry.regions.size;
        const electrodeData = dandisetElectrodes[dandisetId] || {};
        const hasAnyElectrodes = entry.assets.some(a => electrodeData[a.asset_id]?.length > 0);
        const isMultiSession = entry.assets.length > 1;

        if (isMultiSession) {
          const sessionKeys = [...new Set(entry.assets.map(a => a.session || a.path))];
          const electrodeAssets = entry.assets.map(a => {
            const sessionKey = a.session || a.path;
            return {
              id: a.asset_id,
              label: a.session ? `ses-${a.session}` : a.path.split('/').pop(),
              sessionKey,
              color: SESSION_ELECTRODE_COLORS[sessionKeys.indexOf(sessionKey) % SESSION_ELECTRODE_COLORS.length],
            };
          });

          // Expandable subject card with session rows
          html += `<div class="subject-group">`;
          html += `<div class="asset-card subject-card-expandable" data-region-ids='${regionIds}' data-electrode-assets='${JSON.stringify(electrodeAssets)}' data-subject-dir="${entry.subjectDir}">`;
          if (hasAnyElectrodes) html += `<span class="electrode-indicator" title="Has electrode coordinates"></span>`;
          html += `<span class="expand-arrow">&#x25B6;</span>`;
          html += `<span class="asset-card-filename">${subjectId}</span>`;
          const uniqueSessions = new Set(entry.assets.map(a => a.session || a.path)).size;
          const sessionWord = uniqueSessions === 1 ? 'session' : 'sessions';
          const fileInfo = entry.assets.length > uniqueSessions ? `, ${entry.assets.length} files` : '';
          html += `<span class="asset-card-region-count">${uniqueSessions} ${sessionWord}${fileInfo}, ${regionCount} region${regionCount !== 1 ? 's' : ''}</span>`;
          html += `<a class="asset-card-ext" href="${dandiFilesUrl}" target="_blank" rel="noopener" title="View on DANDI Archive">&#8599;</a>`;
          html += `</div>`;
          html += `<div class="session-list hidden">`;
          for (const asset of entry.assets) {
            const sessionLabel = asset.session || '';
            let descLabel = '';
            if (asset.desc) {
              descLabel = ` (${asset.desc})`;
            } else {
              // Derive a short type label from the filename
              const fname = asset.path.split('/').pop();
              if (fname.includes('processed-only_behavior')) descLabel = ' (behavior-only)';
              else if (fname.includes('behavior+ecephys+image')) descLabel = ' (processed)';
              else if (fname.includes('behavior+ecephys')) descLabel = ' (processed)';
              else if (fname.includes('ecephys+image')) descLabel = ' (ecephys+image)';
            }
            const label = sessionLabel ? `ses-${sessionLabel}${descLabel}` : asset.path.split('/').pop();
            const assetRegionIds = JSON.stringify(asset.regions.map(r => r.id));
            const sessionHasElectrodes = electrodeData[asset.asset_id]?.length > 0;
            const tooltip = `Session: ${sessionLabel || 'unknown'}\nPath: ${asset.path}\nAsset ID: ${asset.asset_id}`;
            html += `<div class="session-row" title="${tooltip.replace(/"/g, '&quot;')}" data-asset-id="${asset.asset_id}" data-region-ids='${assetRegionIds}' data-subject-dir="${entry.subjectDir}">`;
            if (sessionHasElectrodes) {
              const sessionKey = asset.session || asset.path;
              const color = SESSION_ELECTRODE_COLORS[sessionKeys.indexOf(sessionKey) % SESSION_ELECTRODE_COLORS.length];
              html += `<span class="electrode-indicator" title="Has electrode coordinates" style="background:#${color}; box-shadow:0 0 4px #${color}99"></span>`;
            }
            html += `<span class="session-row-label">${label}</span>`;
            html += `<span class="asset-card-region-count">${asset.regions.length} region${asset.regions.length !== 1 ? 's' : ''}</span>`;
            html += `</div>`;
          }
          html += `</div>`;
          html += `</div>`;
        } else {
          // Single-session subject card (original behavior)
          const singleAsset = entry.assets[0];
          const singleHasElectrodes = electrodeData[singleAsset.asset_id]?.length > 0;
          html += `<div class="asset-card" data-region-ids='${regionIds}' data-subject-dir="${entry.subjectDir}" data-asset-id="${singleAsset.asset_id}">`;
          if (singleHasElectrodes) html += `<span class="electrode-indicator" title="Has electrode coordinates"></span>`;
          html += `<span class="asset-card-filename">${subjectId}</span>`;
          html += `<span class="asset-card-region-count">${regionCount} region${regionCount !== 1 ? 's' : ''}</span>`;
          html += `<a class="asset-card-ext" href="${dandiFilesUrl}" target="_blank" rel="noopener" title="View on DANDI Archive">&#8599;</a>`;
          html += `</div>`;
        }
      }

      // Pagination controls
      if (totalPages > 1) {
        html += `<div class="pagination">`;
        html += `<button class="pagination-btn" data-page="${page - 1}" ${page === 0 ? 'disabled' : ''}>&laquo; Prev</button>`;
        html += `<span class="pagination-info">${start + 1}&ndash;${end} of ${subjects.length}</span>`;
        html += `<button class="pagination-btn" data-page="${page + 1}" ${page >= totalPages - 1 ? 'disabled' : ''}>Next &raquo;</button>`;
        html += `</div>`;
      }
    }

    panel.innerHTML = html;
    attachCardListeners();
  }

  function attachCardListeners() {
    // Multi-session subject card: expand/collapse the inline session list,
    // then route to enterSubjectView. The expand/collapse is panel chrome
    // (specific to this card layout); the navigation work lives in
    // enterSubjectView and is shared with enterSubjectViewFromURL.
    panel.querySelectorAll('.subject-card-expandable').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.asset-card-ext')) return;

        const group = card.closest('.subject-group');
        const sessionList = group.querySelector('.session-list');
        const arrow = card.querySelector('.expand-arrow');
        const isExpanded = group.classList.contains('expanded');
        if (isExpanded) {
          group.classList.remove('expanded');
          sessionList.classList.add('hidden');
          arrow.classList.remove('expanded');
        } else {
          group.classList.add('expanded');
          sessionList.classList.remove('hidden');
          arrow.classList.add('expanded');
        }

        enterSubjectView(subjectViewParamsFromCard(card, dandisetId));

        panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
        panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
        card.classList.add('asset-card-selected');
      });
    });

    // Session row inside an expanded subject group: route to enterSessionView.
    panel.querySelectorAll('.session-row').forEach(row => {
      row.addEventListener('click', () => {
        enterSessionView(sessionViewParamsFromRow(row, dandisetId));

        panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
        panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
        row.classList.add('session-row-selected');
      });
    });

    // Single-session subject card or "All Subjects" card. The "All Subjects"
    // branch is structurally a return-to-dandiset-view (not a subject/session
    // entry) so it stays inline rather than going through enterSubjectView.
    // The single-session subject branch routes to enterSessionView (assetId
    // present) or enterSubjectView (assetId absent).
    panel.querySelectorAll('.asset-card:not(.subject-card-expandable)').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.asset-card-ext')) return;

        if (card.dataset.all) {
          // "All Subjects" — return to the full dandiset view from a subject /
          // session / region-filter sub-state. Routes through transitionView
          // so currentView and the cross-cutting cleanup land in one place.
          const hadRegionFilter = dandisetRegionFilter !== null;
          transitionView('dandiset', () => {
            filterTreeByDandiset(dandisetId);
            hideSubjectFilter();
            clearElectrodePoints();
            setHash(`dandiset=${dandisetId}`);
            if (hadRegionFilter) {
              spotlightRegions(structureIds);
              updateDandisetPanel(dandisetId, structureIds);
              const newAllCard = panel.querySelector('.asset-card[data-all]');
              if (newAllCard) newAllCard.classList.add('asset-card-selected');
              return;
            }
            spotlightRegions(JSON.parse(card.dataset.regionIds || '[]'));
            filterRegionVisibilityRows(null);
          });
          if (hadRegionFilter) return;
        } else if (card.dataset.assetId) {
          enterSessionView({
            dandisetId,
            subjectDir: card.dataset.subjectDir,
            assetId: card.dataset.assetId,
            regionIds: JSON.parse(card.dataset.regionIds || '[]'),
            sessionLabel: '',
          });
        } else {
          enterSubjectView({
            dandisetId,
            subjectDir: card.dataset.subjectDir,
            regionIds: JSON.parse(card.dataset.regionIds || '[]'),
            electrodeAssets: [],
            subjectName: card.querySelector('.asset-card-filename')?.textContent
              || (card.dataset.subjectDir || '').replace(/^sub-/, ''),
          });
        }

        panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
        panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
        card.classList.add('asset-card-selected');
      });
    });

    // Pagination buttons
    panel.querySelectorAll('.pagination-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const page = parseInt(btn.dataset.page);
        if (page >= 0 && page < totalPages) {
          render(page);
        }
      });
    });

    // Region visibility toggles
    // Find the actual mesh for a region (direct or fallback ancestor)
    function meshIdForRegion(rid) {
      if (meshObjects[rid]) return rid;
      return findNearestAncestorWithMesh(rid);
    }

    // After updating hiddenRegionIds, update the mesh for a given region
    function applyVisibilityToMesh(rid) {
      const meshId = meshIdForRegion(rid);
      if (!meshId || !meshObjects[meshId]) return;
      const overlay = document.getElementById('region-visibility-overlay');
      // Check if ANY visible toggle region sharing this mesh is still checked
      const shouldShow = [...overlay.querySelectorAll('.region-visibility-row:not(.row-filtered) input[type="checkbox"]')].some(otherCb => {
        const otherRid = parseInt(otherCb.dataset.regionId);
        if (hiddenRegionIds.has(otherRid)) return false;
        return meshIdForRegion(otherRid) === meshId;
      });
      if (shouldShow) {
        applyActive(meshObjects[meshId]);
      } else {
        applyDimmed(meshObjects[meshId]);
      }
    }

    const overlay = document.getElementById('region-visibility-overlay');
    overlay.querySelectorAll('.region-visibility-row input[type="checkbox"]').forEach(cb => {
      cb.addEventListener('change', () => {
        const rid = parseInt(cb.dataset.regionId);
        if (cb.checked) {
          hiddenRegionIds.delete(rid);
        } else {
          hiddenRegionIds.add(rid);
        }
        applyVisibilityToMesh(rid);
        filterRegionVisibilityRows();  // update toggle-all state
      });
    });

    const collapseBtn = overlay.querySelector('.region-visibility-collapse-btn');
    if (collapseBtn) {
      collapseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        overlay.classList.toggle('collapsed');
        collapseBtn.innerHTML = overlay.classList.contains('collapsed') ? '&#x25B6;' : '&#x25BC;';
      });
    }

    const toggleAll = overlay.querySelector('#toggle-all-region-visibility');
    if (toggleAll) {
      filterRegionVisibilityRows();  // set initial toggle-all state
      toggleAll.addEventListener('change', () => {
        const checked = toggleAll.checked;
        // Collect affected mesh IDs to update after all checkboxes are set
        const affectedMeshIds = new Set();
        overlay.querySelectorAll('.region-visibility-row:not(.row-filtered) input[type="checkbox"]').forEach(cb => {
          const rid = parseInt(cb.dataset.regionId);
          cb.checked = checked;
          if (checked) {
            hiddenRegionIds.delete(rid);
          } else {
            hiddenRegionIds.add(rid);
          }
          const meshId = meshIdForRegion(rid);
          if (meshId) affectedMeshIds.add(meshId);
        });
        // Apply visibility for all affected meshes
        for (const meshId of affectedMeshIds) {
          if (!meshObjects[meshId]) continue;
          if (checked) {
            applyActive(meshObjects[meshId]);
          } else {
            applyDimmed(meshObjects[meshId]);
          }
        }
      });
    }
  }

  render(0);
}

function filterTreeByStructureIds(structureIds) {
  const activeIds = new Set(structureIds);

  // Add all ancestors so tree paths remain visible
  for (const sid of structureIds) {
    let current = idToStructure[sid]?.parent_structure_id;
    while (current != null) {
      activeIds.add(current);
      current = idToStructure[current]?.parent_structure_id;
    }
  }

  // Expand tree paths to matching nodes
  for (const sid of structureIds) {
    expandToNode(sid);
  }

  // Apply active/inactive styling
  const container = document.getElementById('hierarchy-tree');
  container.querySelectorAll('.tree-node').forEach(node => {
    const id = parseInt(node.dataset.id);
    const label = node.querySelector(':scope > .tree-node-content .tree-label');
    const dot = node.querySelector(':scope > .tree-node-content .tree-color-dot');
    if (!activeIds.has(id)) {
      if (label) { label.classList.add('dandiset-inactive'); label.classList.remove('dandiset-active'); }
      if (dot) { dot.classList.add('dandiset-inactive'); dot.classList.remove('dandiset-active'); }
    } else {
      if (label) { label.classList.remove('dandiset-inactive'); label.classList.add('dandiset-active'); }
      if (dot) { dot.classList.remove('dandiset-inactive'); dot.classList.add('dandiset-active'); }
    }
  });
}

async function spotlightRegions(structureIds) {
  const activeSet = new Set(structureIds);

  // Ensure meshes are loaded
  const toLoad = [];
  for (const sid of structureIds) {
    if (!meshObjects[sid] && !failedMeshIds.has(sid) && !noMeshIds.has(sid)) {
      toLoad.push(ensureMeshLoaded(sid));
    }
  }
  if (toLoad.length > 0) await Promise.all(toLoad);

  // Build mapping: meshId -> [regionIds it represents]
  const meshToRegions = new Map();
  for (const sid of structureIds) {
    const meshId = meshObjects[sid] ? sid : findNearestAncestorWithMesh(sid);
    if (meshId) {
      activeSet.add(meshId);
      if (!meshToRegions.has(meshId)) meshToRegions.set(meshId, []);
      meshToRegions.get(meshId).push(sid);
    }
  }

  // Apply scene visibility: active regions get applyActive, root gets atlas-
  // appropriate display mode (Allen=glass, macaque=silhouette), everything else
  // is hidden. Honors per-region hide toggles via the shouldHideMesh callback.
  syncSceneToSelection(activeSet, (meshId) => {
    const regions = meshToRegions.get(meshId) || [meshId];
    return regions.every(rid => hiddenRegionIds.has(rid));
  });
}

function filterRegionVisibilityRows(regionIds) {
  const overlay = document.getElementById('region-visibility-overlay');
  const activeSet = regionIds ? new Set(regionIds) : null;
  let visibleCount = 0;
  const rows = overlay.querySelectorAll('.region-visibility-row');
  rows.forEach(row => {
    const rid = parseInt(row.querySelector('input[type="checkbox"]').dataset.regionId);
    if (!activeSet || activeSet.has(rid)) {
      row.classList.remove('row-filtered');
      visibleCount++;
    } else {
      row.classList.add('row-filtered');
    }
  });
  const headerLabel = overlay.querySelector('.region-visibility-all-label');
  if (headerLabel) {
    const total = rows.length;
    headerLabel.lastChild.textContent = ` Brain Regions (${activeSet ? visibleCount : total})`;
  }
  // Update toggle-all checkbox state
  const ta = overlay.querySelector('#toggle-all-region-visibility');
  if (ta) {
    const cbs = [...overlay.querySelectorAll('.region-visibility-row:not(.row-filtered) input[type="checkbox"]')];
    if (cbs.length === 0) { ta.checked = false; ta.indeterminate = false; }
    else {
      const allChecked = cbs.every(c => c.checked);
      const noneChecked = cbs.every(c => !c.checked);
      ta.checked = allChecked;
      ta.indeterminate = !allChecked && !noneChecked;
    }
  }
}

// Enter the subject view: a dandiset's recordings narrowed to one subject.
// Updates state, scene, tree filter, electrodes, URL hash, and visibility
// overlay. Does NOT touch panel-card selection styling — callers (the
// inline click handlers in attachCardListeners and enterSubjectViewFromURL) own
// that because it depends on which DOM element was the click target.
function enterSubjectView({ dandisetId, subjectDir, regionIds, electrodeAssets, subjectName }) {
  transitionView('subject', () => {
    filterTreeByStructureIds(regionIds);
    showSubjectFilter(`Subject: ${subjectName}`);

    showElectrodePointsForAssets(dandisetId, electrodeAssets, { colorBySession: true });
    setHash(`dandiset=${dandisetId}&subject=${subjectDir}`);

    spotlightRegions(regionIds);
    filterRegionVisibilityRows(regionIds);
  });
}

// Enter the session view: a single recording session (asset) within a subject.
// Same shape as enterSubjectView but renders single-session electrodes and
// includes the session asset ID in the URL hash. If assetId is missing the
// view degrades to "subject" semantics (no electrodes); kept on this entry
// point so the single-session-card click can route through it without a
// per-case branch.
function enterSessionView({ dandisetId, subjectDir, assetId, regionIds, sessionLabel }) {
  transitionView('session', () => {
    filterTreeByStructureIds(regionIds);
    showSubjectFilter(`${subjectDir.replace(/^sub-/, '')} / ${sessionLabel}`);

    if (assetId) {
      showElectrodePoints(dandisetId, assetId);
      setHash(`dandiset=${dandisetId}&subject=${subjectDir}&session=${assetId}`);
    } else {
      clearElectrodePoints();
      setHash(`dandiset=${dandisetId}&subject=${subjectDir}`);
    }

    spotlightRegions(regionIds);
    filterRegionVisibilityRows(regionIds);
  });
}

// Helpers: extract enterSubjectView/enterSessionView params from a panel
// DOM element. Used by both the inline click handlers and enterSubjectViewFromURL
// so all four entry paths read the same data attributes the same way.
function subjectViewParamsFromCard(card, dandisetId) {
  return {
    dandisetId,
    subjectDir: card.dataset.subjectDir,
    regionIds: JSON.parse(card.dataset.regionIds || '[]'),
    electrodeAssets: JSON.parse(card.dataset.electrodeAssets || '[]'),
    subjectName: card.querySelector('.asset-card-filename')?.textContent
      || (card.dataset.subjectDir || '').replace(/^sub-/, ''),
  };
}

function sessionViewParamsFromRow(row, dandisetId) {
  return {
    dandisetId,
    subjectDir: row.dataset.subjectDir,
    assetId: row.dataset.assetId,
    regionIds: JSON.parse(row.dataset.regionIds || '[]'),
    sessionLabel: row.querySelector('.session-row-label')?.textContent || '',
  };
}

// Deep-link entry point. Called from applyURLState when the URL contains
// &subject= (and optionally &session=). Looks up the matching DOM element
// in the dandiset panel (rendered by the prior enterDandisetView call) and
// calls enterSubjectView / enterSessionView directly. Replaces the prior
// pattern of synthesizing element.click() to delegate to inline handlers.
function enterSubjectViewFromURL(dandisetId, subjectDir, sessionAssetId) {
  const panel = document.getElementById('region-panel');

  // Specific session: find the row, expand the parent group for visibility,
  // route to enterSessionView.
  if (sessionAssetId) {
    const sessionRow = panel.querySelector(`.session-row[data-asset-id="${sessionAssetId}"]`);
    if (sessionRow) {
      const group = sessionRow.closest('.subject-group');
      if (group) {
        group.classList.add('expanded');
        const sessionList = group.querySelector('.session-list');
        const arrow = group.querySelector('.expand-arrow');
        if (sessionList) sessionList.classList.remove('hidden');
        if (arrow) arrow.classList.add('expanded');
      }
      enterSessionView(sessionViewParamsFromRow(sessionRow, dandisetId));
      panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
      panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
      sessionRow.classList.add('session-row-selected');
      return;
    }
  }

  // Subject only: find the subject card.
  const card = panel.querySelector(`.asset-card[data-subject-dir="${subjectDir}"]`);
  if (!card) return;

  if (card.classList.contains('subject-card-expandable')) {
    // Multi-session subject: expand and enter subject view.
    const group = card.closest('.subject-group');
    if (group) {
      group.classList.add('expanded');
      const sessionList = group.querySelector('.session-list');
      const arrow = card.querySelector('.expand-arrow');
      if (sessionList) sessionList.classList.remove('hidden');
      if (arrow) arrow.classList.add('expanded');
    }
    enterSubjectView(subjectViewParamsFromCard(card, dandisetId));
  } else {
    // Single-session card: assetId presence selects between session and subject view.
    const assetId = card.dataset.assetId;
    const regionIds = JSON.parse(card.dataset.regionIds || '[]');
    const subjectName = card.querySelector('.asset-card-filename')?.textContent || '';
    if (assetId) {
      enterSessionView({ dandisetId, subjectDir, assetId, regionIds, sessionLabel: '' });
    } else {
      enterSubjectView({ dandisetId, subjectDir, regionIds, electrodeAssets: [], subjectName });
    }
  }

  panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
  panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
  card.classList.add('asset-card-selected');
}

// ── Electrode Points ───────────────────────────────────────────────────────
async function fetchElectrodes(dandisetId) {
  if (dandisetElectrodes[dandisetId]) return dandisetElectrodes[dandisetId];
  try {
    const resp = await fetch(`${activeAtlas.dataPrefix}electrodes/${dandisetId}.json`);
    if (!resp.ok) return null;
    const data = await resp.json();
    dandisetElectrodes[dandisetId] = data;
    return data;
  } catch { return null; }
}

async function showElectrodePoints(dandisetId, assetId) {
  return showElectrodePointsForAssets(dandisetId, [assetId]);
}

async function showElectrodePointsForAssets(dandisetId, assetRefs, { colorBySession = false } = {}) {
  clearElectrodePoints();
  const assetCoords = await fetchElectrodes(dandisetId);
  if (!assetCoords) return;

  const subjectLabel = document.getElementById('subject-filter-label')?.textContent || `Dandiset ${dandisetId}`;
  const assets = assetRefs.map(ref => (
    typeof ref === 'string'
      ? { id: ref, label: 'Selected session', sessionKey: ref, color: 'ff4466' }
      : ref
  ));
  const coords = [];
  const colors = [];
  const pointInfo = [];
  for (const asset of assets) {
    const assetPoints = assetCoords[asset.id];
    if (!assetPoints?.length) continue;
    const validPoints = assetPoints.filter(coord => !isZeroCoordinate(coord));
    if (validPoints.length === 0) continue;

    const startIndex = coords.length;
    coords.push(...validPoints);
    const sessionLabel = asset.label || asset.sessionKey || 'Session';
    const assetLabel = `Asset ${asset.id.slice(0, 8)}`;
    for (let i = 0; i < validPoints.length; i++) {
      pointInfo.push({
        index: startIndex + i,
        sessionLabel,
        subjectLabel,
        assetLabel,
        coord: validPoints[i],
      });
    }

    if (colorBySession) {
      const color = new THREE.Color(`#${asset.color || 'ff4466'}`);
      for (let i = 0; i < validPoints.length; i++) {
        colors.push(color.r, color.g, color.b);
      }
    }
  }
  if (!coords || coords.length === 0) return;

  // Detect coordinate unit for Allen CCF (micrometers vs 10um voxels).
  // D99 coordinates are already in mm matching the meshes, no scaling needed.
  let scale = 1;
  if (activeAtlas.coordSystem === 'allen') {
    let maxVal = 0;
    for (const c of coords) {
      for (let j = 0; j < 3; j++) {
        if (Math.abs(c[j]) > maxVal) maxVal = Math.abs(c[j]);
      }
    }
    if (maxVal > 100 && maxVal < 1500) scale = 10;
  }

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(coords.length * 3);
  for (let i = 0; i < coords.length; i++) {
    positions[i * 3] = coords[i][0] * scale;
    positions[i * 3 + 1] = coords[i][1] * scale;
    positions[i * 3 + 2] = coords[i][2] * scale;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (colorBySession && colors.length === coords.length * 3) {
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  }

  const material = new THREE.PointsMaterial({
    color: 0xff4466,
    size: activeAtlas.electrodeSize,
    sizeAttenuation: true,
    vertexColors: colorBySession,
    transparent: true,
    opacity: sliderElectrodeOpacity,
  });

  electrodePoints = new THREE.Points(geometry, material);
  electrodePoints.userData.pointInfo = pointInfo;
  worldRoot.add(electrodePoints);
  document.getElementById('electrode-control-row').classList.remove('hidden');

  // Auto-drop the Regions slider when electrodes appear so the active region
  // mesh stops occluding them. User can still raise the slider if they want
  // the region fully opaque. Only drops, never raises — respects users who
  // had set the slider lower already. Triggered here (rather than at view
  // transitions) because this is the precise moment electrodes become
  // visible; views without electrode data don't need the auto-drop.
  const regionSlider = document.getElementById('region-opacity');
  if (regionSlider && parseFloat(regionSlider.value) > ELECTRODE_VIEW_DEFAULT_REGION_OPACITY) {
    regionSlider.value = ELECTRODE_VIEW_DEFAULT_REGION_OPACITY;
    regionSlider.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

function isZeroCoordinate(coord) {
  return coord?.length >= 3 && (
    (coord[0] === 0 && coord[1] === 0 && coord[2] === 0) ||
    (coord[0] === -1 && coord[1] === -1 && coord[2] === -1)
  );
}

function clearElectrodePoints() {
  if (electrodePoints) {
    worldRoot.remove(electrodePoints);
    electrodePoints.geometry.dispose();
    electrodePoints.material.dispose();
    electrodePoints = null;
  }
  document.getElementById('electrode-control-row').classList.add('hidden');
}

function enterRegionView(structureId, { expandTree = true, pushState = true } = {}) {
  // The atlas-init view is "enterRegionView(root)" — same code path, different
  // categorical view. Compute newView before transitionView so the wrapper
  // sets currentView correctly.
  const newView = (structureId === meshManifest.root_id) ? 'init' : 'region';
  transitionView(newView, () => {
    // Drop the highlight tint from the previously selected region (if any).
    // The .selected tree-node class is already cleared by transitionView.
    if (selectedId !== null) unhighlightMesh(selectedId);

    selectedId = structureId;
    selectedDandiset = null;
    hiddenRegionIds = new Set();
    document.getElementById('region-visibility-overlay').classList.add('hidden');
    clearElectrodePoints();

    if (pushState) setHash(`region=${structureId}`);

    spotlightRegion(structureId);
    highlightMesh(structureId);

    if (expandTree) expandToNode(structureId);
    // Query after expandToNode so lazily-rendered nodes exist in DOM
    const el = document.querySelector(`.tree-node-content[data-id="${structureId}"]`);
    if (el) {
      el.classList.add('selected');
      if (expandTree) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    updateRegionPanel(structureId);
  });
}

// ── Region Panel (Right Sidebar) ───────────────────────────────────────────
function updateRegionPanel(structureId) {
  const panel = document.getElementById('region-panel');
  const region = dandiRegions[String(structureId)];
  const s = idToStructure[structureId];

  if (!region && !s) {
    panel.innerHTML = '<p class="placeholder-text">No data for this region.</p>';
    return;
  }

  const name = region ? region.name : s.name;
  const acronym = region ? region.acronym : s.acronym;
  const color = region ? region.color_hex_triplet : (s.color_hex_triplet || 'aaaaaa');
  const atlasLinkTemplate = activeAtlas.regionLinkTemplate;
  const atlasUrl = atlasLinkTemplate ? atlasLinkTemplate.replace('{id}', structureId) : null;

  // Merge all dandisets (direct + sub-region) into one deduplicated list
  let allDandisets = [];
  if (region) {
    const seen = new Set();
    for (const did of (region.dandisets || [])) { seen.add(did); allDandisets.push(did); }
    for (const did of (region.total_dandisets || [])) { if (!seen.has(did)) { seen.add(did); allDandisets.push(did); } }
  }

  const PAGE_SIZE = 20;
  const totalPages = Math.ceil(allDandisets.length / PAGE_SIZE);
  let currentPage = 0;

  function render(page) {
    currentPage = page;
    const start = page * PAGE_SIZE;
    const end = Math.min(start + PAGE_SIZE, allDandisets.length);
    const pageDandisets = allDandisets.slice(start, end);

    const extLink = atlasUrl
      ? ` <a class="region-ext-link" href="${atlasUrl}" target="_blank" rel="noopener" title="View on atlas">&#8599;</a>`
      : '';
    let html = `
      <div class="region-header">
        <div class="region-name">${name}${extLink}</div>
        <div class="region-acronym">${acronym}</div>
        <div class="region-color-bar" style="background: #${color}"></div>
      </div>
    `;

    if (region) {
      const totalCount = region.total_dandiset_count || region.dandiset_count || allDandisets.length;
      html += `<div class="region-stats">
        <div class="stat-item">
          <div class="stat-value">${totalCount}</div>
          <div class="stat-label">Dandisets</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${(region.total_file_count || region.file_count).toLocaleString()}</div>
          <div class="stat-label">NWB Files</div>
        </div>
      </div>`;

      if (allDandisets.length > 0) {
        html += `<div class="dandiset-list-header">Dandisets <span class="dandiset-list-hint">click to view in 3D</span></div>`;
        for (const did of pageDandisets) {
          const regionCount = (dandisetToStructures[did] || []).length;
          const hasElectrodes = dandisetsWithElectrodes.has(did);
          html += `
            <div class="dandiset-card" data-dandiset-id="${did}">
              <div class="dandiset-card-top">
                ${hasElectrodes ? '<span class="electrode-indicator" title="Has electrode coordinates"></span>' : ''}
                <span class="dandiset-card-id">${did}</span>
                <span class="dandiset-card-count">${regionCount} region${regionCount !== 1 ? 's' : ''}</span>
                <a class="dandiset-card-ext" href="https://dandiarchive.org/dandiset/${did}" target="_blank" rel="noopener" title="Open on DANDI Archive">&#8599;</a>
              </div>
              <div class="dandiset-card-title" data-dandiset-id="${did}">${dandisetTitles[did] || ''}</div>
            </div>`;
        }

        if (totalPages > 1) {
          html += `<div class="pagination">`;
          html += `<button class="pagination-btn" data-page="${page - 1}" ${page === 0 ? 'disabled' : ''}>&laquo; Prev</button>`;
          html += `<span class="pagination-info">${start + 1}&ndash;${end} of ${allDandisets.length}</span>`;
          html += `<button class="pagination-btn" data-page="${page + 1}" ${page >= totalPages - 1 ? 'disabled' : ''}>Next &raquo;</button>`;
          html += `</div>`;
        }
      }
    } else {
      html += '<p class="no-data-msg">No DANDI datasets reference this region.</p>';
    }

    panel.innerHTML = html;

    // Click dandiset cards
    panel.querySelectorAll('.dandiset-card').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.dandiset-card-ext')) return;
        enterDandisetView(card.dataset.dandisetId);
      });
    });

    // Pagination buttons
    panel.querySelectorAll('.pagination-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const p = parseInt(btn.dataset.page);
        if (p >= 0 && p < totalPages) render(p);
      });
    });
  }

  render(0);
}

// ── Hierarchy Tree (Left Sidebar) ──────────────────────────────────────────
function buildHierarchyTree() {
  const container = document.getElementById('hierarchy-tree');
  container.innerHTML = '';

  for (const rootNode of structureGraph) {
    container.appendChild(createTreeNode(rootNode, 0));
  }
}

function createTreeNode(node, depth) {
  const el = document.createElement('div');
  el.className = 'tree-node';
  el.dataset.id = node.id;

  const hasChildren = node.children && node.children.length > 0;
  const region = dandiRegions[String(node.id)];
  const hasData = !!region;
  const hasMesh = !noMeshIds.has(node.id);
  const color = node.color_hex_triplet || 'aaaaaa';

  // Content row
  const content = document.createElement('div');
  content.className = 'tree-node-content';
  content.dataset.id = node.id;
  content.style.paddingLeft = (depth * 16 + 8) + 'px';

  // Toggle arrow
  const toggle = document.createElement('span');
  toggle.className = `tree-toggle ${hasChildren ? '' : 'leaf'}`;
  toggle.textContent = '\u25B6'; // right triangle

  // Color dot — hollow ring if no mesh
  const dot = document.createElement('span');
  dot.className = 'tree-color-dot';
  if (hasData && !hasMesh) {
    dot.style.background = 'transparent';
    dot.style.border = `2px solid #${color}`;
  } else {
    dot.style.background = `#${color}`;
  }
  if (!hasData) dot.style.opacity = '0.3';

  // Label
  const label = document.createElement('span');
  let labelClass = hasData ? 'has-data' : 'no-data';
  if (hasData && !hasMesh) labelClass += ' no-mesh';
  label.className = `tree-label ${labelClass}`;
  label.textContent = node.acronym ? `${node.name} (${node.acronym})` : node.name;
  label.title = hasData && !hasMesh ? `${node.name} (no 3D mesh available)` : node.name;

  content.appendChild(toggle);
  content.appendChild(dot);
  content.appendChild(label);

  // Badge: show counts (dandisets normally, subjects when a dandiset is selected)
  if (hasData) {
    const badge = document.createElement('span');
    badge.className = 'tree-badge';
    renderBadge(badge, node.id);
    content.appendChild(badge);
  }

  el.appendChild(content);

  // Children container (lazy: only rendered when expanded)
  if (hasChildren) {
    const childrenEl = document.createElement('div');
    childrenEl.className = 'tree-children';
    childrenEl.dataset.parentId = node.id;
    el.appendChild(childrenEl);

    // Click on toggle or content to expand/collapse
    content.addEventListener('click', (e) => {
      e.stopPropagation();

      // If children not yet rendered, render them
      if (childrenEl.children.length === 0) {
        for (const child of node.children) {
          childrenEl.appendChild(createTreeNode(child, depth + 1));
        }
      }

      const isExpanded = childrenEl.classList.toggle('expanded');
      toggle.classList.toggle('expanded', isExpanded);

      // Tree root → full init view, even when a dandiset is selected. Clicking
      // "everything" should always exit the current scope; otherwise it would
      // be a dandiset-region filter on root, which is meaningless.
      if (node.id === meshManifest.root_id) {
        enterInitView();
      } else if (selectedDandiset) {
        filterDandisetPanelByRegion(node.id);
      } else {
        enterRegionView(node.id, { expandTree: false });
      }
      ensureMeshLoaded(node.id);
    });
  } else {
    content.addEventListener('click', (e) => {
      e.stopPropagation();
      if (node.id === meshManifest.root_id) {
        enterInitView();
      } else if (selectedDandiset) {
        filterDandisetPanelByRegion(node.id);
      } else {
        enterRegionView(node.id, { expandTree: false });
      }
      ensureMeshLoaded(node.id);
    });
  }

  return el;
}

function expandToNode(structureId) {
  // Walk up the tree to find ancestors, then expand each
  const path = [];
  let current = structureId;
  while (current != null) {
    path.unshift(current);
    const s = idToStructure[current];
    current = s ? s.parent_structure_id : null;
  }

  for (const id of path) {
    const nodeEl = document.querySelector(`.tree-node[data-id="${id}"]`);
    if (!nodeEl) continue;
    const childrenEl = nodeEl.querySelector(':scope > .tree-children');
    if (!childrenEl) continue;
    const contentEl = nodeEl.querySelector(':scope > .tree-node-content');
    const toggle = contentEl?.querySelector('.tree-toggle');

    // Lazy-render children if needed
    if (childrenEl.children.length === 0) {
      const s = idToStructure[id];
      if (s && s.children) {
        const depth = path.indexOf(id);
        for (const child of s.children) {
          childrenEl.appendChild(createTreeNode(child, depth + 1));
        }
      }
    }

    childrenEl.classList.add('expanded');
    if (toggle) toggle.classList.add('expanded');
  }
}

// ── Search ─────────────────────────────────────────────────────────────────
function setupSearch() {
  const searchBox = document.getElementById('search-box');
  let debounceTimer;

  searchBox.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const query = searchBox.value.trim().toLowerCase();
      filterTree(query);
    }, 200);
  });
}

function filterTree(query) {
  const container = document.getElementById('hierarchy-tree');

  if (!query) {
    // Show all nodes
    container.querySelectorAll('.tree-node').forEach(n => {
      n.classList.remove('search-hidden');
    });
    return;
  }

  // First expand the entire tree so we can search all nodes
  expandAllForSearch();

  // Mark matching nodes
  const allNodes = container.querySelectorAll('.tree-node');
  const matchingIds = new Set();

  // Find which structure IDs match
  for (const [sid, region] of Object.entries(dandiRegions)) {
    if (
      region.acronym.toLowerCase().includes(query) ||
      region.name.toLowerCase().includes(query)
    ) {
      matchingIds.add(parseInt(sid));
      // Also include ancestors so the path is visible
      let current = idToStructure[parseInt(sid)]?.parent_structure_id;
      while (current != null) {
        matchingIds.add(current);
        current = idToStructure[current]?.parent_structure_id;
      }
    }
  }

  // Also match non-data structures by name/acronym in the tree
  for (const [idStr, s] of Object.entries(idToStructure)) {
    const id = parseInt(idStr);
    if (
      (s.acronym && s.acronym.toLowerCase().includes(query)) ||
      (s.name && s.name.toLowerCase().includes(query))
    ) {
      matchingIds.add(id);
      let current = s.parent_structure_id;
      while (current != null) {
        matchingIds.add(current);
        current = idToStructure[current]?.parent_structure_id;
      }
    }
  }

  allNodes.forEach(n => {
    const id = parseInt(n.dataset.id);
    if (matchingIds.has(id)) {
      n.classList.remove('search-hidden');
    } else {
      n.classList.add('search-hidden');
    }
  });
}

function expandAllForSearch() {
  // Expand top-level nodes so search can find rendered tree nodes
  // We only expand nodes that have DANDI data descendants to keep it manageable
  const container = document.getElementById('hierarchy-tree');

  function expandNode(node) {
    const childrenEl = node.querySelector(':scope > .tree-children');
    if (!childrenEl) return;

    const id = parseInt(node.dataset.id);
    const s = idToStructure[id];

    // Lazy-render children
    if (childrenEl.children.length === 0 && s && s.children) {
      const parentContent = node.querySelector(':scope > .tree-node-content');
      const depth = parseInt(parentContent?.style.paddingLeft || '8') / 16;
      for (const child of s.children) {
        childrenEl.appendChild(createTreeNode(child, depth));
      }
    }

    childrenEl.classList.add('expanded');
    const toggle = node.querySelector(':scope > .tree-node-content .tree-toggle');
    if (toggle) toggle.classList.add('expanded');

    // Recursively expand children
    childrenEl.querySelectorAll(':scope > .tree-node').forEach(expandNode);
  }

  container.querySelectorAll(':scope > .tree-node').forEach(expandNode);
}

// ── Loading UI ─────────────────────────────────────────────────────────────
function updateLoadingText(text) {
  const el = document.getElementById('loading-text');
  if (el) el.textContent = text;
}

function showLoading() {
  document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('loading-overlay').classList.add('hidden');
}

// ── Animation Loop ─────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ── Orientation Buttons ────────────────────────────────────────────────────
function setView(view) {
  const c = brainCenter;
  const d = activeAtlas.camDist;

  if (activeAtlas.coordSystem === 'ras') {
    // RAS: X=Right, Y=Anterior, Z=Superior
    switch (view) {
      case 'dorsal':    camera.position.set(c.x, c.y, c.z + d); camera.up.set(0, 1, 0); break;
      case 'ventral':   camera.position.set(c.x, c.y, c.z - d); camera.up.set(0, -1, 0); break;
      case 'anterior':  camera.position.set(c.x, c.y + d, c.z); camera.up.set(0, 0, 1); break;
      case 'posterior': camera.position.set(c.x, c.y - d, c.z); camera.up.set(0, 0, 1); break;
      case 'left':      camera.position.set(c.x - d, c.y, c.z); camera.up.set(0, 0, 1); break;
      case 'right':     camera.position.set(c.x + d, c.y, c.z); camera.up.set(0, 0, 1); break;
    }
  } else {
    // Allen CCF: X = anterior-posterior, Y = dorsal(low)-ventral(high), Z = left-right
    switch (view) {
      case 'dorsal':    camera.position.set(c.x, c.y - d, c.z); camera.up.set(-1, 0, 0); break;
      case 'ventral':   camera.position.set(c.x, c.y + d, c.z); camera.up.set(1, 0, 0);  break;
      case 'anterior':  camera.position.set(c.x - d, c.y, c.z); camera.up.set(0, -1, 0); break;
      case 'posterior': camera.position.set(c.x + d, c.y, c.z); camera.up.set(0, -1, 0); break;
      case 'left':      camera.position.set(c.x, c.y, c.z - d); camera.up.set(0, -1, 0); break;
      case 'right':     camera.position.set(c.x, c.y, c.z + d); camera.up.set(0, -1, 0); break;
    }
  }
  // Recreate OrbitControls so it caches a fresh quaternion matching
  // the new camera.up. Without this, mouse orbit uses the stale axis.
  recreateOrbitControls(c);
}

document.getElementById('orient-buttons').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (btn) setView(btn.dataset.view);
});

// ── Sidebar Resize ─────────────────────────────────────────────────────────
function setupResize() {
  const handle = document.getElementById('resize-handle');
  const sidebar = document.getElementById('sidebar-left');
  let dragging = false;

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const newWidth = Math.max(200, Math.min(e.clientX, window.innerWidth * 0.5));
    sidebar.style.width = newWidth + 'px';
    onResize();
  });

  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
}

setupResize();

// ── Alpha Sliders ───────────────────────────────────────────────────────────
document.getElementById('electrode-opacity').addEventListener('input', (e) => {
  sliderElectrodeOpacity = parseFloat(e.target.value);
  if (electrodePoints) {
    if (sliderElectrodeOpacity === 0) {
      worldRoot.remove(electrodePoints);
    } else {
      if (!electrodePoints.parent) worldRoot.add(electrodePoints);
      electrodePoints.material.opacity = sliderElectrodeOpacity;
    }
  }
});

document.getElementById('region-opacity').addEventListener('input', (e) => {
  sliderRegionOpacity = parseFloat(e.target.value);
  // In-place opacity update on each non-dimmed mesh's existing material.
  // Deliberately does NOT re-run applyDisplayMode and does NOT touch
  // depthWrite. Reason: applyDisplayMode for 'active' sets
  // `depthWrite = sliderRegionOpacity >= 1`, which turns depth writes off as
  // soon as the user drags the slider below 1. Without depth writes, Three.js
  // falls back to sort-by-mesh-center for transparent rendering, which
  // produces wrong layering when meshes overlap volumetrically (e.g. the
  // hippocampus mesh ends up behind a larger surrounding region whose center
  // is closer to the camera). Production main avoids this by leaving
  // depthWrite at whatever applyDisplayMode set on the last selection change
  // (true at slider=1, the entry value). The cost of leaving depthWrite=true
  // is that geometry strictly inside the active mesh (e.g. electrodes inside
  // a region) gets culled by the depth test even at low alpha; that's an
  // accepted trade-off, see [[depth_rendering_issues]] and the PR #17
  // discussion. Revisit if we adopt order-independent transparency or a
  // hierarchy-aware renderOrder scheme.
  for (const mesh of Object.values(meshObjects)) {
    if (mesh.userData.displayMode === 'hidden' || mesh.userData.displayMode === 'silhouette') continue;
    if (sliderRegionOpacity === 0) {
      mesh.visible = false;
      continue;
    }
    mesh.visible = true;
    const orig = mesh.userData.originalMaterial;
    if (!orig) continue;
    mesh.material.opacity = orig.opacity * sliderRegionOpacity;
    mesh.material.transparent = mesh.material.opacity < 1;
    mesh.material.needsUpdate = true;
  }
});

// ── Dandiset Filter Clear Button ────────────────────────────────────────────
document.getElementById('dandiset-filter-clear').addEventListener('click', clearDandisetFilter);
document.getElementById('subject-filter-clear').addEventListener('click', () => {
  // Return-to-dandiset-view from subject / session / region-filter sub-state.
  // Routes through transitionView so currentView and the cross-cutting cleanup
  // happen in one place. No-op if no dandiset is selected (subject filter
  // shouldn't be visible in that case anyway).
  if (!selectedDandiset) return;
  const hadRegionFilter = dandisetRegionFilter !== null;
  transitionView('dandiset', () => {
    hideSubjectFilter();
    clearElectrodePoints();
    setHash(`dandiset=${selectedDandiset}`);

    filterTreeByDandiset(selectedDandiset);
    const structureIds = dandisetToStructures[selectedDandiset] || [];
    spotlightRegions(structureIds);
    if (hadRegionFilter) {
      updateDandisetPanel(selectedDandiset, structureIds);
    }
    // Card selection: deselect everything, mark "All Subjects" selected.
    const panel = document.getElementById('region-panel');
    panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
    panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
    const allCard = panel.querySelector('.asset-card[data-all]');
    if (allCard) allCard.classList.add('asset-card-selected');
  });
});

// ── URL State ──────────────────────────────────────────────────────────────
//
// The app's deep-link state lives in the URL hash (`location.hash`):
//   #region=672                                           → region view
//   #dandiset=000017                                       → dandiset view
//   #dandiset=000017&region=672                            → dandiset filtered by region
//   #dandiset=000017&subject=sub-M1                        → subject view
//   #dandiset=000017&subject=sub-M1&session=<assetId>      → session view
//   (empty)                                                → atlas init view
//
// setHash writes a new URL via pushState so browser back/forward navigates
// across views. applyURLState reads location.hash and dispatches to the
// matching enter*View function with pushState:false (so the entry function
// doesn't re-push the URL it's reacting to).
function setHash(hash) {
  history.pushState(null, '', '#' + hash);
}

async function applyURLState() {
  // URLSearchParams handles URL-decoding and edge cases like '=' in values.
  // location.hash includes the leading '#' which URLSearchParams doesn't want.
  const params = new URLSearchParams(location.hash.slice(1));

  if (params.size === 0) {
    // No hash — show default (init) view. Guard avoids redundant DOM thrash
    // when a hashchange fires but nothing is actually selected.
    if (selectedId !== null || selectedDandiset !== null) {
      enterInitView();
    }
    return;
  }

  const did = params.get('dandiset');
  const region = params.get('region');
  const subject = params.get('subject');
  const session = params.get('session');

  if (did && dandisetToStructures[did]) {
    await enterDandisetView(did, { pushState: false });
    if (region) {
      const rid = parseInt(region);
      if (idToStructure[rid]) filterDandisetPanelByRegion(rid, { pushState: false });
    }
    if (subject) {
      enterSubjectViewFromURL(did, subject, session || null);
    }
  } else if (region) {
    const sid = parseInt(region);
    if (idToStructure[sid]) enterRegionView(sid, { expandTree: true, pushState: false });
  }
}

window.addEventListener('popstate', () => applyURLState());

// ── Start ──────────────────────────────────────────────────────────────────
init().catch(err => {
  console.error('Failed to initialize:', err);
  updateLoadingText(`Error: ${err.message}`);
});

// Debug hook (opt-in via ?debug=1). Exposes internal state to Playwright and
// DevTools so rendering issues can be diagnosed without UI clicking.
if (new URLSearchParams(window.location.search).get('debug') === '1') {
  window.__debug = {
    get scene() { return scene; },
    get worldRoot() { return worldRoot; },
    get camera() { return camera; },
    get controls() { return controls; },
    get meshObjects() { return meshObjects; },
    get meshManifest() { return meshManifest; },
    get dandiRegions() { return dandiRegions; },
    get activeAtlas() { return activeAtlas; },
    get THREE() { return THREE; },
    meshBounds(id) {
      const m = meshObjects[id];
      if (!m) return null;
      const box = new THREE.Box3().setFromObject(m);
      const c = box.getCenter(new THREE.Vector3());
      return {
        id,
        visible: m.visible,
        worldPosition: m.getWorldPosition(new THREE.Vector3()).toArray(),
        boundsMin: box.min.toArray(),
        boundsMax: box.max.toArray(),
        center: c.toArray(),
      };
    },
    cameraState() {
      return {
        position: camera.position.toArray(),
        up: camera.up.toArray(),
        target: controls ? controls.target.toArray() : null,
        matrixWorld: camera.matrixWorld.elements,
      };
    },
  };
  console.log('[debug] window.__debug exposed');
}
