import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// ── State ──────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls, raycaster, mouse;
let structureGraph = [];   // Allen hierarchy tree
let dandiRegions = {};     // structure_id -> {acronym, name, dandisets, ...}
let meshManifest = {};     // {data_structures, ancestor_structures, root_id}
let idToStructure = {};    // structure_id -> flat structure object
let meshObjects = {};      // structure_id -> THREE.Mesh
let selectedId = null;
let hoveredId = null;
let loadingCount = 0;
let brainCenter = new THREE.Vector3();
const CAM_DIST = 18000;

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
let regionAlpha = 1;          // global opacity multiplier for brain meshes
let dandisetRegionFilter = null; // structure_id when filtering subjects by region within a dandiset
let dandisetSubjectCounts = null; // { directSubjects, totalSubjects } when a dandiset is selected
let hiddenRegionIds = new Set();  // regions toggled off by user in dandiset/subject view
let dandisetsWithElectrodes = new Set();  // dandiset IDs that have electrode coordinate data

// ── Initialization ─────────────────────────────────────────────────────────
async function init() {
  updateLoadingText('Fetching data...');

  const [graphResp, regionsResp, manifestResp, assetsResp, lastUpdatedResp, electrodeManifestResp] = await Promise.all([
    fetch('data/structure_graph.json').then(r => r.json()),
    fetch('data/dandi_regions.json').then(r => r.json()),
    fetch('data/mesh_manifest.json').then(r => r.json()),
    fetch('data/dandiset_assets.json').then(r => r.json()),
    fetch('data/last_updated.json').then(r => r.json()).catch(() => null),
    fetch('data/dandisets_with_electrodes.json').then(r => r.json()).catch(() => []),
  ]);

  structureGraph = graphResp;
  dandiRegions = regionsResp;
  meshManifest = manifestResp;
  dandisetAssets = assetsResp;

  // Show last-updated timestamp
  if (lastUpdatedResp && lastUpdatedResp.timestamp) {
    const date = new Date(lastUpdatedResp.timestamp);
    const formatted = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    const el = document.getElementById('last-updated');
    if (el) el.textContent = `Data updated ${formatted}`;
  }

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

  updateLoadingText('Setting up 3D scene...');
  setupScene();
  buildHierarchyTree();
  setupSearch();

  updateLoadingText('Loading brain meshes...');
  await loadInitialMeshes();

  hideLoading();
  animate();

  // Restore view from URL hash, or default to root selection
  if (!location.hash.slice(1)) {
    const rootNode = structureGraph[0];
    if (rootNode) selectRegion(rootNode.id, { expandTree: true, pushState: false });
  } else {
    applyHashState();
  }
  window.addEventListener('hashchange', () => applyHashState());

  // Fetch dandiset titles in background (non-blocking)
  fetchDandisetTitles();
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

  camera = new THREE.PerspectiveCamera(
    45,
    viewer.clientWidth / viewer.clientHeight,
    1,
    100000
  );
  camera.position.set(0, 0, 20000);
  camera.up.set(0, -1, 0);  // Allen CCF: Y increases ventrally, so flip up

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.rotateSpeed = 0.8;
  controls.zoomSpeed = 1.2;

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(10000, 10000, 10000);
  scene.add(dirLight);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dirLight2.position.set(-10000, -5000, -10000);
  scene.add(dirLight2);

  // Raycaster for picking
  raycaster = new THREE.Raycaster();
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
const objLoader = new OBJLoader();
const failedMeshIds = new Set();

function loadMesh(structureId) {
  return new Promise((resolve) => {
    const path = `data/meshes/${structureId}.obj`;
    objLoader.load(
      path,
      (obj) => {
        const mesh = obj.children[0];
        if (!mesh) { resolve(null); return; }

        // Get color and style based on whether this structure has data
        const isRoot = structureId === meshManifest.root_id;
        const region = dandiRegions[String(structureId)];
        const hasData = !!region;  // has direct or descendant data
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
            opacity: 0.06,
            side: THREE.DoubleSide,
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
        } else {
          // No data at all — wireframe context
          material = new THREE.MeshPhongMaterial({
            color,
            transparent: true,
            opacity: 0.05,
            wireframe: true,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
        }

        mesh.material = material;
        mesh.userData.structureId = structureId;
        mesh.userData.isData = hasData;
        mesh.userData.isRoot = isRoot;
        mesh.userData.originalMaterial = material.clone();

        scene.add(mesh);
        meshObjects[structureId] = mesh;

        // If a region or dandiset is already selected, hide this new mesh unless it's part of the selection
        if (selectedId !== null) {
          const activeIds = getDescendantIds(selectedId);
          if (!activeIds.has(structureId) && structureId !== meshManifest.root_id) {
            applyDimmed(mesh);
          }
        } else if (selectedDandiset !== null) {
          const dandiStructures = new Set(dandisetToStructures[selectedDandiset] || []);
          if (!dandiStructures.has(structureId)) {
            applyDimmed(mesh);
          }
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
  if (failedMeshIds.has(structureId)) return null;
  return loadMesh(structureId);
}

async function loadInitialMeshes() {
  // Load root brain outline first
  await loadMesh(meshManifest.root_id);

  // Load top-level structures (depth-1 children)
  const topLevel = structureGraph.flatMap(root =>
    (root.children || []).map(c => c.id)
  );

  // Load data structures in batches
  const allToLoad = [...meshManifest.data_structures];
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
    controls.target.copy(brainCenter);
    camera.position.set(brainCenter.x, brainCenter.y, brainCenter.z + CAM_DIST);
    controls.update();
  }
}


// ── Raycasting & Interaction ───────────────────────────────────────────────
function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // Only pick visible data meshes
  const pickable = Object.values(meshObjects).filter(
    m => m.userData.isData && m.visible
  );
  const intersects = raycaster.intersectObjects(pickable, false);

  const tooltip = document.getElementById('tooltip');

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    const sid = hit.userData.structureId;

    if (hoveredId !== sid) {
      // Un-highlight previous
      unhighlightMesh(hoveredId);
      hoveredId = sid;
      highlightMesh(sid);
    }

    // Update tooltip
    const region = dandiRegions[String(sid)];
    if (region) {
      tooltip.classList.remove('hidden');
      tooltip.innerHTML = `
        <div class="tooltip-name">${region.name}</div>
        <div class="tooltip-acronym">${region.acronym}</div>
        <div class="tooltip-info">${region.dandiset_count} dandiset${region.dandiset_count !== 1 ? 's' : ''} &middot; ${region.file_count} files</div>
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

  const pickable = Object.values(meshObjects).filter(m => m.userData.isData && m.visible);
  const intersects = raycaster.intersectObjects(pickable, false);

  if (intersects.length > 0) {
    const sid = intersects[0].object.userData.structureId;
    if (selectedDandiset) {
      filterDandisetPanelByRegion(sid);
    } else {
      selectRegion(sid);
    }
  }
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

function applyDimmed(mesh) {
  mesh.visible = false;
  mesh.userData.isDimmed = true;
}

function restoreOriginal(mesh) {
  if (mesh.userData.originalMaterial) {
    mesh.material = mesh.userData.originalMaterial.clone();
    mesh.material.opacity *= regionAlpha;
    mesh.material.transparent = mesh.material.opacity < 1;
    mesh.material.needsUpdate = true;
  }
  mesh.visible = true;
  mesh.userData.isDimmed = false;
}

function applyActive(mesh) {
  const orig = mesh.userData.originalMaterial;
  const mat = orig.clone();
  mat.opacity = regionAlpha;
  mat.transparent = regionAlpha < 1;
  mat.depthWrite = regionAlpha >= 1;
  mat.needsUpdate = true;
  mesh.material = mat;
  mesh.visible = true;
  mesh.userData.isDimmed = false;
}

function findNearestAncestorWithMesh(structureId) {
  // Walk up the hierarchy to find the closest ancestor that has a loaded mesh
  let current = idToStructure[structureId]?.parent_structure_id;
  while (current != null) {
    if (meshObjects[current]) return current;
    current = idToStructure[current]?.parent_structure_id;
  }
  return null;
}

function isolateRegion(structureId) {
  const activeIds = getDescendantIds(structureId);

  // If the selected structure has no loaded mesh, show nearest ancestor that does
  let fallbackId = null;
  if (!meshObjects[structureId]) {
    fallbackId = findNearestAncestorWithMesh(structureId);
    if (fallbackId) activeIds.add(fallbackId);
  }

  // Load any descendant meshes that aren't loaded yet
  const toLoad = [];
  for (const id of activeIds) {
    if (!meshObjects[id] && (dataStructureIds.has(id) || ancestorStructureIds.has(id))) {
      toLoad.push(ensureMeshLoaded(id));
    }
  }
  Promise.all(toLoad).then(() => applyIsolation(structureId, activeIds, fallbackId));

  // Apply immediately to already-loaded meshes
  applyIsolation(structureId, activeIds, fallbackId);
}

function applyIsolation(selectedStructureId, activeIds, fallbackId) {
  // Only show the selected (or fallback) mesh; hide everything else
  const showId = meshObjects[selectedStructureId] ? selectedStructureId : fallbackId;
  for (const [idStr, mesh] of Object.entries(meshObjects)) {
    const id = parseInt(idStr);
    if (id === showId) {
      applyActive(mesh);
    } else {
      applyDimmed(mesh);
    }
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

async function selectDandiset(dandisetId, { pushState = true } = {}) {
  selectedDandiset = dandisetId;
  selectedId = null;
  dandisetRegionFilter = null;
  dandisetSubjectCounts = computeDandisetSubjectCounts(dandisetId);
  hiddenRegionIds = new Set();
  clearElectrodePoints();

  // Update URL hash
  if (pushState) {
    setHash(`dandiset=${dandisetId}`);
  }

  // Deselect tree
  const prevEl = document.querySelector('.tree-node-content.selected');
  if (prevEl) prevEl.classList.remove('selected');

  const structureIds = dandisetToStructures[dandisetId] || [];
  const activeSet = new Set(structureIds);

  // Ensure meshes are loaded for all structures in this dandiset
  const toLoad = [];
  for (const sid of structureIds) {
    if (!meshObjects[sid] && !failedMeshIds.has(sid) && !noMeshIds.has(sid)) {
      toLoad.push(ensureMeshLoaded(sid));
    }
  }
  if (toLoad.length > 0) {
    await Promise.all(toLoad);
  }

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

  // Apply isolation: show active regions, hide everything else (including root)
  activeSet.delete(meshManifest.root_id);
  for (const [idStr, mesh] of Object.entries(meshObjects)) {
    const id = parseInt(idStr);
    if (activeSet.has(id)) {
      const regions = meshToRegions.get(id) || [id];
      const allHidden = regions.every(rid => hiddenRegionIds.has(rid));
      if (!allHidden) {
        applyActive(mesh);
      } else {
        applyDimmed(mesh);
      }
    } else {
      applyDimmed(mesh);
    }
  }

  // Update right panel to show dandiset info
  updateDandisetPanel(dandisetId, structureIds);

  // Filter the left panel tree to highlight matching regions
  filterTreeByDandiset(dandisetId);

  // Update tree badges to show subject counts
  updateTreeBadges();
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

function clearDandisetFilter() {
  // Hide filter bars
  document.getElementById('dandiset-filter-bar').classList.add('hidden');
  hideSubjectFilter();
  clearElectrodePoints();
  dandisetRegionFilter = null;
  hiddenRegionIds = new Set();
  document.getElementById('region-toggles-overlay').classList.add('hidden');

  // Remove dandiset filter classes from all tree nodes
  const container = document.getElementById('hierarchy-tree');
  container.querySelectorAll('.dandiset-inactive').forEach(el => {
    el.classList.remove('dandiset-inactive');
  });
  container.querySelectorAll('.dandiset-active').forEach(el => {
    el.classList.remove('dandiset-active');
  });

  // Restore 3D view
  showAllRegions();
  selectedDandiset = null;
  selectedId = null;
  dandisetSubjectCounts = null;

  // Clear URL hash
  history.pushState(null, '', window.location.pathname);

  // Restore tree badges to dandiset counts
  updateTreeBadges();

  // Reset right panel
  document.getElementById('region-panel').innerHTML =
    '<p class="placeholder-text">Click a brain region to view details and associated DANDI datasets.</p>';
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
    isolateStructureIds(matchingStructures);
  } else {
    isolateStructureIds([structureId]);
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
  const allSubjects = [...subjectMap.entries()];

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
      const toggleOverlay = document.getElementById('region-toggles-overlay');
      if (regionList.length > 0) {
        const wasCollapsed = toggleOverlay.classList.contains('collapsed');
        let toggleHtml = `<div class="region-toggles-header"><label class="region-toggle-all-label"><input type="checkbox" id="toggle-all-regions"> Brain Regions (${regionList.length})</label><button class="region-toggles-collapse-btn" title="Collapse">${wasCollapsed ? '&#x25B6;' : '&#x25BC;'}</button></div>`;
        toggleHtml += `<div class="region-toggle-list">`;
        for (const r of regionList) {
          const checked = !hiddenRegionIds.has(r.id);
          toggleHtml += `<label class="region-toggle-row" title="${r.name}">`;
          toggleHtml += `<input type="checkbox" data-region-id="${r.id}" ${checked ? 'checked' : ''}>`;
          toggleHtml += `<span class="region-toggle-dot" style="background:#${r.color}"></span>`;
          toggleHtml += `<span class="region-toggle-name">${r.acronym || r.name}</span>`;
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
          // Expandable subject card with session rows
          html += `<div class="subject-group">`;
          html += `<div class="asset-card subject-card-expandable" data-region-ids='${regionIds}' data-subject-dir="${entry.subjectDir}">`;
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
            if (sessionHasElectrodes) html += `<span class="electrode-indicator" title="Has electrode coordinates"></span>`;
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
    // Expand/collapse for multi-session subject cards
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
      });
    });

    // Click on session rows within expanded subject cards
    panel.querySelectorAll('.session-row').forEach(row => {
      row.addEventListener('click', () => {
        const regionIds = JSON.parse(row.dataset.regionIds || '[]');
        const assetId = row.dataset.assetId;
        const subjectDir = row.dataset.subjectDir;
        const sessionLabel = row.querySelector('.session-row-label')?.textContent || '';

        dandisetRegionFilter = null;
        filterTreeByStructureIds(regionIds);
        showSubjectFilter(`${subjectDir.replace(/^sub-/, '')} / ${sessionLabel}`);
        const selEl = document.querySelector('.tree-node-content.selected');
        if (selEl) selEl.classList.remove('selected');

        if (assetId) {
          showElectrodePoints(dandisetId, assetId);
          setHash(`dandiset=${dandisetId}&subject=${subjectDir}&session=${assetId}`);
        } else {
          clearElectrodePoints();
          setHash(`dandiset=${dandisetId}&subject=${subjectDir}`);
        }

        isolateStructureIds(regionIds);
        filterRegionToggles(regionIds);

        panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
        panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
        row.classList.add('session-row-selected');
      });
    });

    // Click on single-session subject cards and "All Subjects"
    panel.querySelectorAll('.asset-card:not(.subject-card-expandable)').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.asset-card-ext')) return;
        const regionIds = JSON.parse(card.dataset.regionIds || '[]');

        if (card.dataset.all) {
          const hadRegionFilter = dandisetRegionFilter !== null;
          dandisetRegionFilter = null;
          filterTreeByDandiset(dandisetId);
          hideSubjectFilter();
          clearElectrodePoints();
          setHash(`dandiset=${dandisetId}`);
          const selEl = document.querySelector('.tree-node-content.selected');
          if (selEl) selEl.classList.remove('selected');
          if (hadRegionFilter) {
            isolateStructureIds(structureIds);
            updateDandisetPanel(dandisetId, structureIds);
            const newAllCard = panel.querySelector('.asset-card[data-all]');
            if (newAllCard) newAllCard.classList.add('asset-card-selected');
            return;
          }
        } else {
          dandisetRegionFilter = null;
          const subjectName = card.querySelector('.asset-card-filename')?.textContent || '';
          filterTreeByStructureIds(regionIds);
          showSubjectFilter(`Subject: ${subjectName}`);
          const selEl = document.querySelector('.tree-node-content.selected');
          if (selEl) selEl.classList.remove('selected');
          const assetId = card.dataset.assetId;
          const subjectDir = card.dataset.subjectDir;
          if (assetId) {
            showElectrodePoints(dandisetId, assetId);
            setHash(`dandiset=${dandisetId}&subject=${subjectDir}&session=${assetId}`);
          } else {
            clearElectrodePoints();
            setHash(`dandiset=${dandisetId}`);
          }
        }
        isolateStructureIds(regionIds);
        filterRegionToggles(card.dataset.all ? null : regionIds);

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
    function applyToggleToMesh(rid) {
      const meshId = meshIdForRegion(rid);
      if (!meshId || !meshObjects[meshId]) return;
      const overlay = document.getElementById('region-toggles-overlay');
      // Check if ANY visible toggle region sharing this mesh is still checked
      const shouldShow = [...overlay.querySelectorAll('.region-toggle-row:not(.toggle-hidden) input[type="checkbox"]')].some(otherCb => {
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

    const overlay = document.getElementById('region-toggles-overlay');
    overlay.querySelectorAll('.region-toggle-row input[type="checkbox"]').forEach(cb => {
      cb.addEventListener('change', () => {
        const rid = parseInt(cb.dataset.regionId);
        if (cb.checked) {
          hiddenRegionIds.delete(rid);
        } else {
          hiddenRegionIds.add(rid);
        }
        applyToggleToMesh(rid);
        filterRegionToggles();  // update toggle-all state
      });
    });

    const collapseBtn = overlay.querySelector('.region-toggles-collapse-btn');
    if (collapseBtn) {
      collapseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        overlay.classList.toggle('collapsed');
        collapseBtn.innerHTML = overlay.classList.contains('collapsed') ? '&#x25B6;' : '&#x25BC;';
      });
    }

    const toggleAll = overlay.querySelector('#toggle-all-regions');
    if (toggleAll) {
      filterRegionToggles();  // set initial toggle-all state
      toggleAll.addEventListener('change', () => {
        const checked = toggleAll.checked;
        // Collect affected mesh IDs to update after all checkboxes are set
        const affectedMeshIds = new Set();
        overlay.querySelectorAll('.region-toggle-row:not(.toggle-hidden) input[type="checkbox"]').forEach(cb => {
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

async function isolateStructureIds(structureIds) {
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

  activeSet.delete(meshManifest.root_id);
  for (const [idStr, mesh] of Object.entries(meshObjects)) {
    const id = parseInt(idStr);
    if (activeSet.has(id)) {
      // Show mesh only if at least one of its represented regions is not hidden
      const regions = meshToRegions.get(id) || [id];
      const allHidden = regions.every(rid => hiddenRegionIds.has(rid));
      if (!allHidden) {
        applyActive(mesh);
      } else {
        applyDimmed(mesh);
      }
    } else {
      applyDimmed(mesh);
    }
  }
}

function filterRegionToggles(regionIds) {
  const overlay = document.getElementById('region-toggles-overlay');
  const activeSet = regionIds ? new Set(regionIds) : null;
  let visibleCount = 0;
  const rows = overlay.querySelectorAll('.region-toggle-row');
  rows.forEach(row => {
    const rid = parseInt(row.querySelector('input[type="checkbox"]').dataset.regionId);
    if (!activeSet || activeSet.has(rid)) {
      row.classList.remove('toggle-hidden');
      visibleCount++;
    } else {
      row.classList.add('toggle-hidden');
    }
  });
  const headerLabel = overlay.querySelector('.region-toggle-all-label');
  if (headerLabel) {
    const total = rows.length;
    headerLabel.lastChild.textContent = ` Brain Regions (${activeSet ? visibleCount : total})`;
  }
  // Update toggle-all checkbox state
  const ta = overlay.querySelector('#toggle-all-regions');
  if (ta) {
    const cbs = [...overlay.querySelectorAll('.region-toggle-row:not(.toggle-hidden) input[type="checkbox"]')];
    if (cbs.length === 0) { ta.checked = false; ta.indeterminate = false; }
    else {
      const allChecked = cbs.every(c => c.checked);
      const noneChecked = cbs.every(c => !c.checked);
      ta.checked = allChecked;
      ta.indeterminate = !allChecked && !noneChecked;
    }
  }
}

function selectSubjectByDir(dandisetId, subjectDir, sessionAssetId) {
  const panel = document.getElementById('region-panel');

  // If a specific session asset is requested, find and click the session row
  if (sessionAssetId) {
    const sessionRow = panel.querySelector(`.session-row[data-asset-id="${sessionAssetId}"]`);
    if (sessionRow) {
      // Expand the parent subject group first
      const group = sessionRow.closest('.subject-group');
      if (group) {
        group.classList.add('expanded');
        const sessionList = group.querySelector('.session-list');
        const arrow = group.querySelector('.expand-arrow');
        if (sessionList) sessionList.classList.remove('hidden');
        if (arrow) arrow.classList.add('expanded');
      }
      sessionRow.click();
      return;
    }
  }

  // Fall back to selecting the subject card (single-session or expandable)
  const card = panel.querySelector(`.asset-card[data-subject-dir="${subjectDir}"]`);
  if (!card) return;

  if (card.classList.contains('subject-card-expandable')) {
    // Expand multi-session card
    const group = card.closest('.subject-group');
    if (group) {
      group.classList.add('expanded');
      const sessionList = group.querySelector('.session-list');
      const arrow = card.querySelector('.expand-arrow');
      if (sessionList) sessionList.classList.remove('hidden');
      if (arrow) arrow.classList.add('expanded');
    }
  } else {
    // Single-session card: click to select
    card.click();
  }
}

// ── Electrode Points ───────────────────────────────────────────────────────
async function fetchElectrodes(dandisetId) {
  if (dandisetElectrodes[dandisetId]) return dandisetElectrodes[dandisetId];
  try {
    const resp = await fetch(`data/electrodes/${dandisetId}.json`);
    if (!resp.ok) return null;
    const data = await resp.json();
    dandisetElectrodes[dandisetId] = data;
    return data;
  } catch { return null; }
}

async function showElectrodePoints(dandisetId, assetId) {
  clearElectrodePoints();
  const assetCoords = await fetchElectrodes(dandisetId);
  if (!assetCoords) return;
  const coords = assetCoords[assetId];
  if (!coords || coords.length === 0) return;

  // Detect coordinate unit: Allen CCF is in micrometers (max ~13200).
  // Some NWB files store coords in 10µm voxel units (max ~1320).
  let scale = 1;
  let maxVal = 0;
  for (const c of coords) {
    for (let j = 0; j < 3; j++) {
      if (Math.abs(c[j]) > maxVal) maxVal = Math.abs(c[j]);
    }
  }
  if (maxVal > 100 && maxVal < 1500) scale = 10;

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(coords.length * 3);
  for (let i = 0; i < coords.length; i++) {
    positions[i * 3] = coords[i][0] * scale;
    positions[i * 3 + 1] = coords[i][1] * scale;
    positions[i * 3 + 2] = coords[i][2] * scale;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const alphaSlider = document.getElementById('electrode-alpha');
  const material = new THREE.PointsMaterial({
    color: 0xff4466,
    size: 150,
    sizeAttenuation: true,
    transparent: true,
    opacity: parseFloat(alphaSlider.value),
  });

  electrodePoints = new THREE.Points(geometry, material);
  scene.add(electrodePoints);
  document.getElementById('electrode-control-row').classList.remove('hidden');
}

function clearElectrodePoints() {
  if (electrodePoints) {
    scene.remove(electrodePoints);
    electrodePoints.geometry.dispose();
    electrodePoints.material.dispose();
    electrodePoints = null;
  }
  document.getElementById('electrode-control-row').classList.add('hidden');
}

function selectRegion(structureId, { expandTree = true, pushState = true } = {}) {
  // Deselect previous
  if (selectedId !== null) {
    unhighlightMesh(selectedId);
    const prevEl = document.querySelector(`.tree-node-content[data-id="${selectedId}"]`);
    if (prevEl) prevEl.classList.remove('selected');
  }

  selectedId = structureId;
  selectedDandiset = null;
  hiddenRegionIds = new Set();
  document.getElementById('region-toggles-overlay').classList.add('hidden');
  clearElectrodePoints();

  // Update URL hash
  if (pushState) {
    setHash(`region=${structureId}`);
  }

  // Isolate this region in the 3D view, then highlight
  isolateRegion(structureId);
  highlightMesh(structureId);

  // Update tree selection
  if (expandTree) {
    expandToNode(structureId);
  }
  // Query after expandToNode so lazily-rendered nodes exist in DOM
  const el = document.querySelector(`.tree-node-content[data-id="${structureId}"]`);
  if (el) {
    el.classList.add('selected');
    if (expandTree) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  // Update right panel
  updateRegionPanel(structureId);
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
  const atlasUrl = `https://atlas.brain-map.org/atlas#atlas=2&structure=${structureId}`;

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

    let html = `
      <div class="region-header">
        <div class="region-name">${name} <a class="region-ext-link" href="${atlasUrl}" target="_blank" rel="noopener" title="View on Allen Brain Atlas">&#8599;</a></div>
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
        selectDandiset(card.dataset.dandisetId);
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

      // If in dandiset mode, filter subjects by this region
      if (selectedDandiset) {
        filterDandisetPanelByRegion(node.id);
      } else {
        selectRegion(node.id, { expandTree: false });
      }
      ensureMeshLoaded(node.id);
    });
  } else {
    content.addEventListener('click', (e) => {
      e.stopPropagation();
      if (selectedDandiset) {
        filterDandisetPanelByRegion(node.id);
      } else {
        selectRegion(node.id, { expandTree: false });
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
// Allen CCF OBJ coords: X = anterior-posterior, Y = dorsal(low)-ventral(high), Z = left-right
function setView(view) {
  const c = brainCenter;
  switch (view) {
    case 'dorsal':    camera.position.set(c.x, c.y - CAM_DIST, c.z); camera.up.set(-1, 0, 0); break;
    case 'ventral':   camera.position.set(c.x, c.y + CAM_DIST, c.z); camera.up.set(1, 0, 0);  break;
    case 'anterior':  camera.position.set(c.x - CAM_DIST, c.y, c.z); camera.up.set(0, -1, 0); break;
    case 'posterior': camera.position.set(c.x + CAM_DIST, c.y, c.z); camera.up.set(0, -1, 0); break;
    case 'left':      camera.position.set(c.x, c.y, c.z - CAM_DIST); camera.up.set(0, -1, 0); break;
    case 'right':     camera.position.set(c.x, c.y, c.z + CAM_DIST); camera.up.set(0, -1, 0); break;
  }
  controls.target.copy(c);
  controls.update();
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
document.getElementById('electrode-alpha').addEventListener('input', (e) => {
  const val = parseFloat(e.target.value);
  if (electrodePoints) {
    if (val === 0) {
      scene.remove(electrodePoints);
    } else {
      if (!electrodePoints.parent) scene.add(electrodePoints);
      electrodePoints.material.opacity = val;
    }
  }
});

document.getElementById('region-alpha').addEventListener('input', (e) => {
  regionAlpha = parseFloat(e.target.value);
  for (const mesh of Object.values(meshObjects)) {
    if (mesh.userData.isDimmed) continue;
    if (regionAlpha === 0) {
      mesh.visible = false;
    } else {
      mesh.visible = true;
      const orig = mesh.userData.originalMaterial;
      if (!orig) continue;
      mesh.material.opacity = orig.opacity * regionAlpha;
      mesh.material.transparent = mesh.material.opacity < 1;
      mesh.material.needsUpdate = true;
    }
  }
});

// ── Dandiset Filter Clear Button ────────────────────────────────────────────
document.getElementById('dandiset-filter-clear').addEventListener('click', clearDandisetFilter);
document.getElementById('subject-filter-clear').addEventListener('click', () => {
  hideSubjectFilter();
  clearElectrodePoints();
  const hadRegionFilter = dandisetRegionFilter !== null;
  dandisetRegionFilter = null;
  // Deselect tree node
  const selEl = document.querySelector('.tree-node-content.selected');
  if (selEl) selEl.classList.remove('selected');
  // Restore to full dandiset view
  if (selectedDandiset) {
    setHash(`dandiset=${selectedDandiset}`);
    filterTreeByDandiset(selectedDandiset);
    const structureIds = dandisetToStructures[selectedDandiset] || [];
    isolateStructureIds(structureIds);
    // Re-render panel if region filter was active, to show all subjects
    if (hadRegionFilter) {
      updateDandisetPanel(selectedDandiset, structureIds);
    }
    // Deselect subject card and session rows, select "All Subjects"
    const panel = document.getElementById('region-panel');
    panel.querySelectorAll('.asset-card').forEach(c => c.classList.remove('asset-card-selected'));
    panel.querySelectorAll('.session-row').forEach(r => r.classList.remove('session-row-selected'));
    const allCard = panel.querySelector('.asset-card[data-all]');
    if (allCard) allCard.classList.add('asset-card-selected');
  }
});

// ── URL Hash State ──────────────────────────────────────────────────────────
function setHash(hash) {
  history.pushState(null, '', '#' + hash);
}

async function applyHashState() {
  const hash = location.hash.slice(1); // remove '#'
  if (!hash) {
    // No hash — show default view
    if (selectedId !== null || selectedDandiset !== null) {
      selectedId = null;
      selectedDandiset = null;
      dandisetSubjectCounts = null;
      hiddenRegionIds = new Set();
      document.getElementById('region-toggles-overlay').classList.add('hidden');
      clearElectrodePoints();
      const prevEl = document.querySelector('.tree-node-content.selected');
      if (prevEl) prevEl.classList.remove('selected');
      showAllRegions();
      updateTreeBadges();
      // Clear dandiset filter bar
      document.getElementById('dandiset-filter-bar').classList.add('hidden');
      const tree = document.getElementById('hierarchy-tree');
      tree.querySelectorAll('.dandiset-inactive').forEach(el => el.classList.remove('dandiset-inactive'));
      tree.querySelectorAll('.dandiset-active').forEach(el => el.classList.remove('dandiset-active'));
      document.getElementById('region-panel').innerHTML =
        '<p class="placeholder-text">Click a brain region to view details and associated DANDI datasets.</p>';
    }
    return;
  }

  const params = Object.fromEntries(hash.split('&').map(p => p.split('=')));
  if (params.region) {
    const sid = parseInt(params.region);
    if (idToStructure[sid]) {
      selectRegion(sid, { expandTree: true, pushState: false });
    }
  } else if (params.dandiset) {
    const did = params.dandiset;
    if (dandisetToStructures[did]) {
      await selectDandiset(did, { pushState: false });
      if (params.region) {
        const rid = parseInt(params.region);
        if (idToStructure[rid]) {
          filterDandisetPanelByRegion(rid, { pushState: false });
        }
      }
      if (params.subject) {
        selectSubjectByDir(did, params.subject, params.session || null);
      }
    }
  }
}

window.addEventListener('popstate', () => applyHashState());

// ── Start ──────────────────────────────────────────────────────────────────
init().catch(err => {
  console.error('Failed to initialize:', err);
  updateLoadingText(`Error: ${err.message}`);
});
