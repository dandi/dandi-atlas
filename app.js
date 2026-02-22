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

// ── Initialization ─────────────────────────────────────────────────────────
async function init() {
  updateLoadingText('Fetching data...');

  const [graphResp, regionsResp, manifestResp] = await Promise.all([
    fetch('data/structure_graph.json').then(r => r.json()),
    fetch('data/dandi_regions.json').then(r => r.json()),
    fetch('data/mesh_manifest.json').then(r => r.json()),
  ]);

  structureGraph = graphResp;
  dandiRegions = regionsResp;
  meshManifest = manifestResp;

  dataStructureIds = new Set(meshManifest.data_structures);
  ancestorStructureIds = new Set(meshManifest.ancestor_structures);

  // Build flat lookup from the tree
  flattenTree(structureGraph);

  updateLoadingText('Setting up 3D scene...');
  setupScene();
  buildHierarchyTree();
  setupSearch();

  updateLoadingText('Loading brain meshes...');
  await loadInitialMeshes();

  hideLoading();
  animate();
}

function flattenTree(nodes) {
  for (const node of nodes) {
    idToStructure[node.id] = node;
    if (node.children) flattenTree(node.children);
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

        // If a region is already selected, dim this new mesh unless it's part of the selection
        if (selectedId !== null) {
          const activeIds = getDescendantIds(selectedId);
          if (!activeIds.has(structureId) && structureId !== meshManifest.root_id) {
            applyDimmed(mesh);
          }
        }

        resolve(mesh);
      },
      undefined,
      () => resolve(null)
    );
  });
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

async function ensureMeshLoaded(structureId) {
  if (meshObjects[structureId]) return meshObjects[structureId];
  return loadMesh(structureId);
}

// ── Raycasting & Interaction ───────────────────────────────────────────────
function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // Only pick non-dimmed data meshes
  const pickable = Object.values(meshObjects).filter(
    m => m.userData.isData && !m.userData.isDimmed
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
  if (!mesh) return;
  // Only highlight if the mesh isn't dimmed
  if (mesh.userData.isDimmed) return;
  mesh.material.emissive = new THREE.Color(0x335577);
  mesh.material.emissiveIntensity = 0.5;
}

function unhighlightMesh(structureId) {
  if (structureId === null) return;
  const mesh = meshObjects[structureId];
  if (!mesh) return;
  if (structureId === selectedId) return;
  if (mesh.userData.isDimmed) return;
  mesh.material.emissive = new THREE.Color(0x000000);
  mesh.material.emissiveIntensity = 0;
}

function onClick(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  const pickable = Object.values(meshObjects).filter(m => m.userData.isData && !m.userData.isDimmed);
  const intersects = raycaster.intersectObjects(pickable, false);

  if (intersects.length > 0) {
    const sid = intersects[0].object.userData.structureId;
    selectRegion(sid);
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
  mesh.material = new THREE.MeshPhongMaterial({
    color: 0x666666,
    transparent: true,
    opacity: 0.03,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  mesh.userData.isDimmed = true;
}

function restoreOriginal(mesh) {
  if (mesh.userData.originalMaterial) {
    mesh.material = mesh.userData.originalMaterial.clone();
    mesh.material.needsUpdate = true;
  }
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

  // If the selected structure has no mesh, also include its nearest ancestor with a mesh
  let fallbackId = null;
  if (!meshObjects[structureId] && !dataStructureIds.has(structureId)) {
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
  for (const [idStr, mesh] of Object.entries(meshObjects)) {
    const id = parseInt(idStr);
    if (id === selectedStructureId || id === fallbackId) {
      // Selected region (or fallback ancestor): full color, full opacity
      const orig = mesh.userData.originalMaterial;
      const mat = orig.clone();
      mat.opacity = 1.0;
      mat.transparent = false;
      mat.depthWrite = true;
      mat.needsUpdate = true;
      mesh.material = mat;
      mesh.userData.isDimmed = false;
    } else if (activeIds.has(id)) {
      // Descendants of selected: normal colored appearance
      mesh.material = mesh.userData.originalMaterial.clone();
      mesh.material.needsUpdate = true;
      mesh.userData.isDimmed = false;
    } else {
      // Everything else: very transparent gray
      applyDimmed(mesh);
    }
  }
}

function showAllRegions() {
  for (const mesh of Object.values(meshObjects)) {
    restoreOriginal(mesh);
  }
}

function selectRegion(structureId, { expandTree = true } = {}) {
  // Deselect previous
  if (selectedId !== null) {
    unhighlightMesh(selectedId);
    const prevEl = document.querySelector(`.tree-node-content[data-id="${selectedId}"]`);
    if (prevEl) prevEl.classList.remove('selected');
  }

  selectedId = structureId;

  // Isolate this region in the 3D view, then highlight
  isolateRegion(structureId);
  highlightMesh(structureId);

  // Update tree selection
  const el = document.querySelector(`.tree-node-content[data-id="${structureId}"]`);
  if (el) {
    el.classList.add('selected');
    if (expandTree) {
      // Expand parents and scroll into view (only when selected from 3D view)
      expandToNode(structureId);
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

  let html = `
    <div class="region-header">
      <div class="region-name">${name}</div>
      <div class="region-acronym">${acronym}</div>
      <div class="region-color-bar" style="background: #${color}"></div>
    </div>
  `;

  if (region) {
    const hasDirect = region.dandiset_count > 0;
    const hasDescendant = (region.total_dandiset_count || 0) > region.dandiset_count;

    html += `<div class="region-stats">`;
    if (hasDirect) {
      html += `
        <div class="stat-item">
          <div class="stat-value">${region.dandiset_count}</div>
          <div class="stat-label">Direct</div>
        </div>`;
    }
    if (hasDescendant) {
      html += `
        <div class="stat-item">
          <div class="stat-value">${region.total_dandiset_count}</div>
          <div class="stat-label">Incl. Sub-regions</div>
        </div>`;
    }
    html += `
        <div class="stat-item">
          <div class="stat-value">${(region.total_file_count || region.file_count).toLocaleString()}</div>
          <div class="stat-label">NWB Files</div>
        </div>
      </div>`;

    if (hasDirect) {
      html += `<div class="dandiset-list-header">Direct Dandisets</div>`;
      for (const did of region.dandisets) {
        html += `<a class="dandiset-link" href="https://dandiarchive.org/dandiset/${did}" target="_blank" rel="noopener">${did}</a>`;
      }
    }

    if (hasDescendant) {
      const descendantOnly = (region.total_dandisets || []).filter(d => !region.dandisets.includes(d));
      if (descendantOnly.length > 0) {
        html += `<div class="dandiset-list-header" style="margin-top:12px">Sub-region Dandisets</div>`;
        for (const did of descendantOnly) {
          html += `<a class="dandiset-link" href="https://dandiarchive.org/dandiset/${did}" target="_blank" rel="noopener">${did}</a>`;
        }
      }
    }
  } else {
    html += '<p class="no-data-msg">No DANDI datasets reference this region.</p>';
  }

  panel.innerHTML = html;
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

  // Color dot
  const dot = document.createElement('span');
  dot.className = 'tree-color-dot';
  dot.style.background = `#${color}`;
  if (!hasData) dot.style.opacity = '0.3';

  // Label
  const label = document.createElement('span');
  label.className = `tree-label ${hasData ? 'has-data' : 'no-data'}`;
  label.textContent = node.acronym || node.name;
  label.title = node.name;

  content.appendChild(toggle);
  content.appendChild(dot);
  content.appendChild(label);

  // Badge: show direct / total dandiset counts in different colors
  if (hasData) {
    const direct = region.dandiset_count;
    const total = region.total_dandiset_count || direct;
    const badge = document.createElement('span');
    badge.className = 'tree-badge';
    if (direct > 0 && direct !== total) {
      badge.innerHTML = `<span class="badge-direct">${direct}</span><span class="badge-sep">/</span><span class="badge-total">${total}</span>`;
      badge.title = `${direct} direct, ${total} including sub-regions`;
    } else if (direct > 0) {
      badge.innerHTML = `<span class="badge-direct">${direct}</span>`;
      badge.title = `${direct} dandisets`;
    } else {
      badge.innerHTML = `<span class="badge-total">${total}</span>`;
      badge.title = `${total} dandisets in sub-regions`;
    }
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

      // Also select this region (don't re-expand the tree)
      selectRegion(node.id, { expandTree: false });
      ensureMeshLoaded(node.id);
    });
  } else {
    content.addEventListener('click', (e) => {
      e.stopPropagation();
      selectRegion(node.id, { expandTree: false });
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

// ── Start ──────────────────────────────────────────────────────────────────
init().catch(err => {
  console.error('Failed to initialize:', err);
  updateLoadingText(`Error: ${err.message}`);
});
