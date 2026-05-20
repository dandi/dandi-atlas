#!/usr/bin/env python3
"""Local web app for reviewing the WHS-SD -> UBERON mapping.

Shows fuzzy + unmapped rows from `data/atlases/whs_sd/uberon_mapping.json`
and lets the reviewer either confirm the current pick or replace it via a
live OLS4 search. Saves the curated mapping back to the same file, which
is what `scripts/build_rat_atlas.py` reads via `_load_uberon_lookups`.

Usage:
    uv run python scripts/serve_uberon_review.py [--port 8765]

Then open http://localhost:8765 in a browser.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rat_atlas_lib import ATLAS_CONFIGS


MAPPING_PATH = ATLAS_CONFIGS["whs_sd"]["output_dir"] / "uberon_mapping.json"
OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WHS-SD &harr; UBERON review</title>
<style>
  :root {
    --border: #d0d7de;
    --muted: #57606a;
    --accent: #0969da;
    --confirm: #1f883d;
    --reject: #cf222e;
    --bg-card: #ffffff;
    --bg-page: #f6f8fa;
    --bg-confirmed: #dafbe1;
    --bg-overridden: #ddf4ff;
    --bg-rejected: #ffebe9;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg-page);
    margin: 0;
    padding: 0;
  }
  header {
    position: sticky;
    top: 0;
    background: white;
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    z-index: 10;
  }
  header h1 { font-size: 16px; margin: 0; flex-grow: 1; }
  header .counter { color: var(--muted); font-size: 14px; }
  header button {
    font-size: 14px;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    cursor: pointer;
    background: white;
  }
  header button.save {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  header button:disabled { opacity: 0.5; cursor: not-allowed; }
  #status { font-size: 13px; color: var(--muted); }
  #status.dirty { color: var(--reject); font-weight: 600; }
  main { padding: 16px 24px; max-width: 1100px; margin: 0 auto; }
  .filter-row {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    color: var(--muted);
    font-size: 14px;
  }
  .filter-row label { cursor: pointer; }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 16px;
    align-items: start;
  }
  .card.state-confirmed { background: var(--bg-confirmed); }
  .card.state-overridden { background: var(--bg-overridden); }
  .card.state-rejected { background: var(--bg-rejected); }
  .card .whs .label { color: var(--muted); font-size: 11px; text-transform: uppercase; }
  .card .whs .name { font-weight: 600; font-size: 15px; }
  .card .whs .id { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .card .quality {
    display: inline-block;
    font-size: 11px;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 4px;
    background: #eaeef2;
    color: var(--muted);
    margin-top: 4px;
  }
  .card .quality.fuzzy { background: #fff8c5; color: #9a6700; }
  .card .quality.none { background: #ffebe9; color: var(--reject); }
  .card .quality.confirmed { background: var(--bg-confirmed); color: var(--confirm); }
  .card .quality.override { background: var(--bg-overridden); color: var(--accent); }
  .card .uberon .label { color: var(--muted); font-size: 11px; text-transform: uppercase; }
  .card .uberon .term { font-weight: 600; font-size: 14px; }
  .card .uberon .curie { font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); }
  .card .uberon .curie a { color: var(--accent); text-decoration: none; }
  .card .actions { display: flex; flex-direction: column; gap: 6px; min-width: 140px; }
  .card .actions button {
    font-size: 13px;
    padding: 5px 10px;
    border-radius: 5px;
    border: 1px solid var(--border);
    cursor: pointer;
    background: white;
  }
  .card .actions button.confirm { color: var(--confirm); }
  .card .actions button.reject { color: var(--reject); }
  .search-panel {
    grid-column: 1 / -1;
    margin-top: 8px;
    padding-top: 12px;
    border-top: 1px dashed var(--border);
  }
  .search-panel input {
    width: 100%;
    padding: 6px 10px;
    font-size: 14px;
    border: 1px solid var(--border);
    border-radius: 5px;
    box-sizing: border-box;
  }
  .search-results { margin-top: 8px; max-height: 280px; overflow-y: auto; }
  .search-results .hit {
    padding: 6px 10px;
    cursor: pointer;
    border-radius: 4px;
    font-size: 13px;
  }
  .search-results .hit:hover { background: #eaeef2; }
  .search-results .hit .term { font-weight: 600; }
  .search-results .hit .curie { font-family: ui-monospace, monospace; color: var(--muted); margin-left: 6px; }
  .search-results .hit .description { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .empty { color: var(--muted); padding: 32px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>WHS-SD &harr; UBERON review</h1>
  <span class="counter" id="counter"></span>
  <span id="status">Loaded</span>
  <button class="save" id="save-button" disabled>Save</button>
</header>
<main>
  <div class="filter-row">
    Show:
    <label><input type="checkbox" id="filter-fuzzy" checked> fuzzy</label>
    <label><input type="checkbox" id="filter-none" checked> unmapped</label>
    <label><input type="checkbox" id="filter-exact"> exact</label>
    <label><input type="checkbox" id="filter-skipped"> skipped (ROOT/OUTSIDE)</label>
  </div>
  <div id="cards"></div>
</main>
<script>
const FILTER_QUALITIES = {
  "fuzzy": "filter-fuzzy",
  "none": "filter-none",
  "exact": "filter-exact",
  "skipped": "filter-skipped",
  "override-unmapped": "filter-skipped",
};
let mapping = {};
let dirty = false;

async function load() {
  const response = await fetch("/mapping.json");
  mapping = await response.json();
  render();
}

function setDirty(value) {
  dirty = value;
  document.getElementById("save-button").disabled = !value;
  const status = document.getElementById("status");
  if (value) {
    status.textContent = "Unsaved changes";
    status.classList.add("dirty");
  } else {
    status.textContent = "Saved";
    status.classList.remove("dirty");
  }
}

function rowState(row) {
  return row._state || row.match_quality;
}

function visibleRows() {
  const checked = {};
  for (const q of Object.keys(FILTER_QUALITIES)) {
    checked[q] = document.getElementById(FILTER_QUALITIES[q]).checked;
  }
  return Object.entries(mapping)
    .filter(([, row]) => checked[row.match_quality])
    .sort(([, a], [, b]) => a.whs_id - b.whs_id);
}

function render() {
  const cards = document.getElementById("cards");
  cards.innerHTML = "";
  const rows = visibleRows();
  document.getElementById("counter").textContent =
    `${rows.length} of ${Object.keys(mapping).length} shown`;
  if (rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Nothing to review with the current filters.";
    cards.appendChild(empty);
    return;
  }
  for (const [key, row] of rows) {
    cards.appendChild(renderCard(key, row));
  }
}

function renderCard(key, row) {
  const card = document.createElement("div");
  const state = rowState(row);
  card.className = "card";
  if (row._state === "confirmed") card.classList.add("state-confirmed");
  if (row._state === "overridden") card.classList.add("state-overridden");
  if (row._state === "rejected") card.classList.add("state-rejected");

  const whs = document.createElement("div");
  whs.className = "whs";
  whs.innerHTML =
    `<div class="label">WHS-SD region</div>` +
    `<div class="name"></div>` +
    `<div class="id">id ${row.whs_id}</div>` +
    `<div class="quality ${row.match_quality}">${state}</div>`;
  whs.querySelector(".name").textContent = row.whs_name;

  const uberon = document.createElement("div");
  uberon.className = "uberon";
  if (row.uberon_id) {
    const iri = row.uberon_iri ||
      `http://purl.obolibrary.org/obo/${row.uberon_id.replace(":", "_")}`;
    uberon.innerHTML =
      `<div class="label">Current UBERON pick</div>` +
      `<div class="term"></div>` +
      `<div class="curie"><a target="_blank" rel="noopener"></a></div>`;
    uberon.querySelector(".term").textContent = row.uberon_label || "(no label)";
    const link = uberon.querySelector("a");
    link.href = iri;
    link.textContent = row.uberon_id;
  } else {
    uberon.innerHTML =
      `<div class="label">Current UBERON pick</div>` +
      `<div class="term" style="color:var(--muted)">(none)</div>`;
  }

  const actions = document.createElement("div");
  actions.className = "actions";
  const confirmButton = document.createElement("button");
  confirmButton.className = "confirm";
  confirmButton.textContent = row.uberon_id ? "Confirm pick" : "Leave unmapped";
  confirmButton.onclick = () => onConfirm(key);
  const changeButton = document.createElement("button");
  changeButton.textContent = "Search alternative";
  changeButton.onclick = () => toggleSearch(card, key);
  const rejectButton = document.createElement("button");
  rejectButton.className = "reject";
  rejectButton.textContent = "No UBERON match";
  rejectButton.onclick = () => onReject(key);
  actions.appendChild(confirmButton);
  actions.appendChild(changeButton);
  if (row.uberon_id) actions.appendChild(rejectButton);

  card.appendChild(whs);
  card.appendChild(uberon);
  card.appendChild(actions);
  return card;
}

function onConfirm(key) {
  const row = mapping[key];
  row._state = "confirmed";
  if (row.match_quality === "fuzzy") row.match_quality = "confirmed";
  setDirty(true);
  render();
}

function onReject(key) {
  const row = mapping[key];
  row.uberon_id = null;
  row.uberon_label = null;
  row.uberon_iri = null;
  row.match_quality = "override-unmapped";
  row._state = "rejected";
  setDirty(true);
  render();
}

async function toggleSearch(card, key) {
  let panel = card.querySelector(".search-panel");
  if (panel) { panel.remove(); return; }
  panel = document.createElement("div");
  panel.className = "search-panel";
  panel.innerHTML =
    `<input type="text" placeholder="Search UBERON via OLS4..." autofocus>` +
    `<div class="search-results"></div>`;
  card.appendChild(panel);
  const input = panel.querySelector("input");
  const results = panel.querySelector(".search-results");
  input.value = mapping[key].whs_name;
  input.select();
  let timer = null;
  const runSearch = async () => {
    const query = input.value.trim();
    if (!query) { results.innerHTML = ""; return; }
    results.innerHTML = `<div class="hit">Searching...</div>`;
    const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
    const hits = await response.json();
    results.innerHTML = "";
    if (hits.length === 0) {
      results.innerHTML = `<div class="hit">No UBERON hits.</div>`;
      return;
    }
    for (const hit of hits) {
      const hitElement = document.createElement("div");
      hitElement.className = "hit";
      hitElement.innerHTML =
        `<div><span class="term"></span><span class="curie"></span></div>` +
        `<div class="description"></div>`;
      hitElement.querySelector(".term").textContent = hit.label;
      hitElement.querySelector(".curie").textContent = hit.obo_id;
      hitElement.querySelector(".description").textContent =
        (hit.description || []).join(" ").slice(0, 200);
      hitElement.onclick = () => applyAlternative(key, hit);
      results.appendChild(hitElement);
    }
  };
  input.oninput = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(runSearch, 250);
  };
  runSearch();
}

function applyAlternative(key, hit) {
  const row = mapping[key];
  row.uberon_id = hit.obo_id;
  row.uberon_label = hit.label;
  row.uberon_iri = hit.iri;
  row.match_quality = "override";
  row._state = "overridden";
  setDirty(true);
  render();
}

async function save() {
  const clean = {};
  for (const [key, row] of Object.entries(mapping)) {
    const copy = { ...row };
    delete copy._state;
    clean[key] = copy;
  }
  const response = await fetch("/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(clean),
  });
  if (response.ok) {
    setDirty(false);
  } else {
    document.getElementById("status").textContent = "Save failed";
  }
}

document.getElementById("save-button").onclick = save;
for (const id of Object.values(FILTER_QUALITIES)) {
  document.getElementById(id).onchange = render;
}
load();
</script>
</body>
</html>
"""


def _ols4_search(query):
    parameters = {
        "q": query,
        "ontology": "uberon",
        "exact": "false",
        "rows": "20",
    }
    url = f"{OLS4_SEARCH}?{urllib.parse.urlencode(parameters)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    documents = payload.get("response", {}).get("docs", [])
    return [
        {
            "obo_id": document.get("obo_id"),
            "label": document.get("label"),
            "iri": document.get("iri"),
            "description": document.get("description", []),
        }
        for document in documents
        if document.get("ontology_name") == "uberon"
        and (document.get("obo_id") or "").startswith("UBERON:")
    ]


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter log — print one short line per request, no client IP/timestamp.
        sys.stderr.write(f"  {self.command} {self.path}\n")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/mapping.json":
            with open(MAPPING_PATH) as f:
                payload = json.load(f)
            self._send_json(payload)
            return
        if parsed.path == "/search":
            query_string = urllib.parse.parse_qs(parsed.query)
            query = (query_string.get("q") or [""])[0]
            hits = _ols4_search(query) if query else []
            self._send_json(hits)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/save":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            with open(MAPPING_PATH, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            sys.stderr.write(f"  wrote {MAPPING_PATH}\n")
            self._send_json({"ok": True})
            return
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="Review WHS-SD UBERON mapping in a browser")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not MAPPING_PATH.exists():
        sys.stderr.write(
            f"Mapping file not found: {MAPPING_PATH}\n"
            "Run scripts/build_whs_uberon_mapping.py first.\n"
        )
        sys.exit(1)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    print(f"Review server running at http://127.0.0.1:{args.port}")
    print(f"Editing {MAPPING_PATH}")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
