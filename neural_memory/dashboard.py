"""Interactive ECharts dashboard for neural memory knowledge graph.

Generates a single self-contained HTML file with:
- Sidebar: category/type/importance/status/search filters
- Tab 1: Hierarchy treemap (ECharts treemap)
- Tab 2: Vector space scatter (PCA-projected, ECharts scatter)
- Tab 3: Hierarchical tree graph (ECharts graph, Reingold-Tilford layout)
- Click-to-inspect detail panel
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .models import NodeType, EdgeType
from .storage import Storage

# ---------------------------------------------------------------------------
# ECharts source
# ---------------------------------------------------------------------------

def _get_echarts(project_root: str = ".") -> str:
    """Return ECharts v5 JS string — use cached copy or empty string."""
    cache = Path(project_root) / ".neural-memory" / "echarts.min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    return ""


def _echarts_cdn_loader() -> str:
    """Return JS that loads ECharts from CDN using safe DOM methods."""
    return (
        "var _s = document.createElement('script');"
        " _s.setAttribute('src', 'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js');"
        " document.head.appendChild(_s);"
    )


# ---------------------------------------------------------------------------
# Data extraction (unchanged)
# ---------------------------------------------------------------------------

def _pca_positions(nodes: list[dict]) -> dict[str, list[float]]:
    """Compute 2-D PCA from stored embeddings. Falls back to circular layout."""
    try:
        import numpy as np
        vecs, ids = [], []
        for n in nodes:
            emb = n.get("embedding")
            if emb and len(emb) >= 2:
                vecs.append(emb)
                ids.append(n["id"])
        if len(vecs) < 3:
            raise ValueError("not enough")
        X = np.array(vecs, dtype=np.float32)
        X -= X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ Vt[:2].T
        # normalise to [-1, 1]
        for col in range(2):
            mn, mx = proj[:, col].min(), proj[:, col].max()
            rng = mx - mn or 1.0
            proj[:, col] = (proj[:, col] - mn) / rng * 2 - 1
        return {nid: proj[i].tolist() for i, nid in enumerate(ids)}
    except Exception:
        pass
    # Circular fallback
    result = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / max(n, 1)
        result[node["id"]] = [round(math.cos(angle), 4), round(math.sin(angle), 4)]
    return result


def _build_hierarchy(nodes: list[dict], edges: list[dict]) -> dict:
    """Build a tree dict for ECharts treemap from CONTAINS/PHASE_CONTAINS edges."""
    contains_types = {"contains", "phase_contains", "task_contains"}
    children_map: dict[str, list[str]] = {}
    child_set: set[str] = set()

    for e in edges:
        if e.get("edge_type", "") in contains_types:
            src, tgt = e["source_id"], e["target_id"]
            children_map.setdefault(src, []).append(tgt)
            child_set.add(tgt)

    node_by_id = {n["id"]: n for n in nodes}
    roots = [n for n in nodes if n["id"] not in child_set]

    def _subtree(nid: str, depth: int = 0) -> dict:
        n = node_by_id.get(nid, {})
        entry = {
            "id": nid,
            "name": n.get("name", nid),
            "node_type": n.get("node_type", ""),
            "category": n.get("category", "codebase"),
            "importance": n.get("importance", 0.0),
            "value": max(0.01, n.get("importance", 0.1)),
        }
        kids = children_map.get(nid, [])
        if kids and depth < 6:
            entry["children"] = [_subtree(k, depth + 1) for k in kids]
        return entry

    if not roots:
        roots = nodes[:1]

    if len(roots) == 1:
        return _subtree(roots[0]["id"])

    return {
        "id": "__root__",
        "name": "Project",
        "node_type": "project_overview",
        "category": "codebase",
        "importance": 1.0,
        "value": 1.0,
        "children": [_subtree(r["id"]) for r in roots],
    }


def _build_virtual_tree(nodes: list[dict], edges: list[dict], max_depth: int = 6) -> dict:
    """Build a virtual hierarchy tree for the interactive Graph tab.

    Each real node's children are grouped by relationship type into virtual
    intermediate nodes (e.g. [calls], [imports], [called by]).  This turns the
    multi-edge knowledge graph into an infinitely navigable collapsible tree.

    Virtual group nodes carry ``_virtual: true`` so the JS can style them
    differently.  Real leaf nodes beyond max_depth carry ``_hasMore: true`` so
    the JS can expand them on-demand from RAW.edges without a server round-trip.
    """
    # ── Build adjacency indices ───────────────────────────────────────────────
    contains_types = {"contains", "phase_contains", "task_contains"}
    outgoing: dict[str, dict[str, list[str]]] = {}  # {node_id: {edge_type: [target_ids]}}
    incoming: dict[str, dict[str, list[str]]] = {}  # {node_id: {edge_type: [source_ids]}}
    child_set: set[str] = set()

    for e in edges:
        src = e.get("source_id", "")
        tgt = e.get("target_id", "")
        etype = e.get("edge_type", "")
        if not src or not tgt or not etype:
            continue
        outgoing.setdefault(src, {}).setdefault(etype, []).append(tgt)
        incoming.setdefault(tgt, {}).setdefault(etype, []).append(src)
        if etype in contains_types:
            child_set.add(tgt)

    node_by_id = {n["id"]: n for n in nodes}

    # Groups to show and their display labels — order matters (containment first)
    # Each entry: (edge_type, direction, label)
    GROUP_ORDER = [
        ("contains",       "out", "contains"),
        ("phase_contains", "out", "contains"),
        ("task_contains",  "out", "contains"),
        ("calls",          "out", "calls"),
        ("calls",          "in",  "called by"),
        ("imports",        "out", "imports"),
        ("imports",        "in",  "imported by"),
        ("inherits",       "out", "inherits"),
        ("inherits",       "in",  "inherited by"),
        ("implements",     "out", "implements"),
        ("implements",     "in",  "implemented by"),
        ("uses",           "out", "uses"),
        ("uses",           "in",  "used by"),
        ("defines",        "out", "defines"),
        ("defines",        "in",  "defined by"),
        ("relates_to",     "out", "relates to"),
        ("relates_to",     "in",  "related from"),
        ("fixed_by",       "out", "fixed by"),
        ("fixed_by",       "in",  "fixes"),
        ("references",     "out", "references"),
        ("references",     "in",  "referenced by"),
        ("queries",        "out", "queries"),
        ("queries",        "in",  "queried by"),
        ("writes_to",      "out", "writes to"),
        ("writes_to",      "in",  "written by"),
    ]
    def _make_node_entry(nid: str) -> dict:
        n = node_by_id.get(nid, {})
        return {
            "id": nid,
            "name": n.get("name", nid),
            "node_type": n.get("node_type", "other"),
            "language": n.get("language", ""),
            "importance": n.get("importance", 0.0),
            "_real": True,
        }

    def _build_node(nid: str, visited: frozenset, depth: int) -> dict:
        entry = _make_node_entry(nid)
        n = node_by_id.get(nid, {})
        # Always attach short summary so detail panel can render from virtual tree nodes
        entry["summary_short"] = n.get("summary_short", "")
        entry["file_path"] = n.get("file_path", "")

        if depth >= max_depth:
            # Check if this node has any relationships beyond what's visible
            has_out = any(outgoing.get(nid, {}).get(et) for et, _, _ in GROUP_ORDER)
            has_in = any(incoming.get(nid, {}).get(et) for _, d, _ in GROUP_ORDER if d == "in"
                         for et in [next((g[0] for g in GROUP_ORDER if g[1] == "in" and g[2] == _), None)])
            if outgoing.get(nid) or incoming.get(nid):
                entry["_hasMore"] = True
            return entry

        visited_next = visited | {nid}
        children: list[dict] = []

        for etype, direction, label in GROUP_ORDER:
            if direction == "out":
                targets = outgoing.get(nid, {}).get(etype, [])
            else:
                targets = incoming.get(nid, {}).get(etype, [])

            # Separate normal targets from back-references (ancestors already in path)
            normal_targets = [t for t in targets if t not in visited_next and t in node_by_id]
            back_targets = [t for t in targets if t in visited_next and t in node_by_id]

            if not normal_targets and not back_targets:
                continue

            # Sort by importance descending so top nodes appear first
            normal_targets.sort(key=lambda t: node_by_id.get(t, {}).get("importance", 0.0), reverse=True)

            group_children = [_build_node(t, visited_next, depth + 2) for t in normal_targets]

            # Append back-reference leaf nodes for ancestor connections
            for bt in back_targets:
                bn = node_by_id.get(bt, {})
                group_children.append({
                    "id": bt,
                    "name": bn.get("name", bt) + " \u21a9",
                    "node_type": bn.get("node_type", "other"),
                    "language": bn.get("language", ""),
                    "importance": bn.get("importance", 0.0),
                    "_backRef": True,
                    "_refTargetId": bt,
                })

            children.append({
                "name": f"[{label}]",
                "_virtual": True,
                "_edgeType": etype,
                "_direction": direction,
                "children": group_children,
            })

        if children:
            entry["children"] = children
        elif outgoing.get(nid) or incoming.get(nid):
            entry["_hasMore"] = True

        return entry

    # ── Find containment roots (nodes with no contains-type parent) ───────────
    roots = [n for n in nodes if n["id"] not in child_set]
    if not roots:
        roots = nodes[:1]

    if len(roots) == 1:
        return _build_node(roots[0]["id"], frozenset(), 0)

    # Multiple roots → synthetic project root
    root_children = [_build_node(r["id"], frozenset(), 2) for r in roots]
    return {
        "id": "__root__",
        "name": "Project",
        "node_type": "project_overview",
        "language": "",
        "importance": 1.0,
        "_real": True,
        "children": [{
            "name": "[contains]",
            "_virtual": True,
            "_edgeType": "contains",
            "_direction": "out",
            "children": root_children,
        }],
    }


def _extract_data(storage: Storage) -> dict:
    """Extract all nodes, edges, hierarchy, and PCA positions as JSON-serialisable dict."""
    raw_nodes = storage.get_all_nodes()
    raw_edges = []
    try:
        raw_edges = storage.get_all_edges()
    except Exception:
        pass

    nodes = []
    for n in raw_nodes:
        nodes.append({
            "id": n.id,
            "name": n.name,
            "node_type": n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type),
            "category": getattr(n, "category", "codebase") or "codebase",
            "file_path": n.file_path or "",
            "importance": round(n.importance or 0.0, 4),
            "summary_short": (n.summary_short or "")[:200],
            "summary_detailed": (n.summary_detailed or "")[:800],
            "line_start": n.line_start,
            "line_end": n.line_end,
            "severity": getattr(n, "severity", "") or "",
            "bug_status": getattr(n, "bug_status", "") or "",
            "task_status": getattr(n, "task_status", "") or "",
            "priority": getattr(n, "priority", "") or "",
            "lsp_hover_doc": (getattr(n, "lsp_hover_doc", "") or "")[:300],
            "lsp_diagnostics": getattr(n, "lsp_diagnostics", []) or [],
            "embedding": getattr(n, "embedding", None),
        })

    edges = []
    for e in raw_edges:
        edges.append({
            "source_id": e.source_id,
            "target_id": e.target_id,
            "edge_type": e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type),
        })

    pca = _pca_positions(nodes)

    # Remove embedding from node data (large, not needed in JS)
    for n in nodes:
        n.pop("embedding", None)
        n["px"] = pca.get(n["id"], [0.0, 0.0])[0]
        n["py"] = pca.get(n["id"], [0.0, 0.0])[1]

    hierarchy = _build_hierarchy(nodes, edges)
    virtual_tree = _build_virtual_tree(nodes, edges)

    stats = storage.get_stats() if hasattr(storage, "get_stats") else {}

    return {
        "nodes": nodes,
        "edges": edges,
        "hierarchy": hierarchy,
        "virtual_tree": virtual_tree,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# HTML template builder
# ---------------------------------------------------------------------------

def _build_html(data_json: str, echarts_script: str) -> str:
    """Assemble the final HTML string from safe components."""
    head = _html_head()
    body = _html_body()
    scripts = _html_scripts(data_json, echarts_script)
    return head + body + scripts


def _html_head() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Memory Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;display:flex;flex-direction:column;overflow:hidden}
#topbar{background:#161b22;border-bottom:1px solid #30363d;padding:8px 16px;display:flex;align-items:center;gap:16px;flex-shrink:0}
#topbar h1{font-size:15px;font-weight:600;color:#58a6ff}
.stats{font-size:13px;color:#8b949e;display:flex;gap:12px}
#tabs{display:flex;gap:6px;margin-left:auto}
.tab-btn{background:none;border:1px solid #30363d;color:#8b949e;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}
.tab-btn:hover{background:#21262d;color:#c9d1d9;border-color:#484f58}
.tab-btn.active{background:#58a6ff22;border-color:#58a6ff;color:#58a6ff}
#main{display:flex;flex:1;overflow:hidden}
#sidebar{width:260px;background:#161b22;border-right:1px solid #30363d;overflow-y:auto;padding:14px;flex-shrink:0;font-size:13px}
.sb-head{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#8b949e;margin:14px 0 8px;font-weight:600}
.filter-row{display:flex;align-items:center;gap:8px;margin:4px 0;cursor:pointer;padding:3px 4px;border-radius:4px;transition:background .15s}
.filter-row:hover{background:#21262d}
.filter-row input,.filter-row label{cursor:pointer}
select{width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;padding:4px 8px;font-size:13px}
select:focus{border-color:#58a6ff;outline:none}
input[type=range]{width:100%;margin:4px 0;accent-color:#58a6ff}
input[type=text]{width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;padding:5px 10px;font-size:13px;transition:border-color .15s}
input[type=text]:focus{border-color:#58a6ff;outline:none}
.range-labels{display:flex;justify-content:space-between;font-size:11px;color:#8b949e}
#view-area{flex:1;position:relative;overflow:hidden}
.view-panel{position:absolute;inset:0;display:none}
.view-panel.active{display:block}
.chart-container{width:100%;height:100%}
#detail-panel{position:absolute;right:0;top:0;bottom:0;width:320px;background:#161b22ee;border-left:1px solid #30363d;padding:18px;overflow-y:auto;transform:translateX(100%);transition:transform .25s;font-size:13px;z-index:10;box-shadow:-4px 0 12px rgba(0,0,0,0.3)}
#detail-panel.open{transform:translateX(0)}
.close-btn{position:absolute;top:8px;right:8px;background:none;border:none;color:#8b949e;cursor:pointer;font-size:18px;padding:4px 8px;border-radius:4px;transition:all .15s}
.close-btn:hover{background:#30363d;color:#f0f6fc}
.d-field{margin:10px 0}
.d-label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin-bottom:3px;font-weight:600}
.d-value{color:#c9d1d9;line-height:1.5;word-break:break-word;overflow-y:auto;max-height:150px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500;margin:2px}
.edge-toggle-swatch{display:inline-block;width:16px;height:3px;border-radius:1px;vertical-align:middle;margin-right:4px}
</style>
</head>
"""


def _html_body() -> str:
    return """<body>
<div id="topbar">
  <h1>Neural Memory</h1>
  <div class="stats" id="header-stats"></div>
  <div id="tabs">
    <button class="tab-btn active" data-tab="hierarchy">Hierarchy</button>
    <button class="tab-btn" data-tab="vectors">Semantic</button>
    <button class="tab-btn" data-tab="graph">Graph</button>
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <div class="sb-head">Category</div>
    <div id="cat-filters"></div>
    <div class="sb-head">Language</div>
    <div id="lang-filters"></div>
    <div class="sb-head">Node Type</div>
    <select id="type-filter" multiple size="5"></select>
    <div class="sb-head">Importance</div>
    <input type="range" id="imp-slider" min="0" max="1" step="0.05" value="0">
    <div class="range-labels"><span id="imp-lo">0.0</span><span id="imp-hi">1.0</span></div>
    <div class="sb-head">Status</div>
    <div id="status-filters"></div>
    <div class="sb-head">Search</div>
    <input type="text" id="search-input" placeholder="name / summary...">
    <div class="sb-head">Treemap Depth</div>
    <input type="range" id="depth-slider" min="1" max="6" step="1" value="3">
    <div class="range-labels"><span>1</span><span id="depth-val">3</span><span>6</span></div>
    <div id="graph-controls" style="display:none">
      <div class="sb-head">Layout</div>
      <select id="graph-layout-select" style="width:100%;background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:3px 6px;border-radius:4px;font-size:12px">
        <option value="orthogonal">Orthogonal (L→R)</option>
        <option value="radial">Radial</option>
      </select>
      <div class="sb-head" style="margin-top:8px">Expand Depth</div>
      <input type="range" id="graph-depth-slider" min="1" max="6" step="1" value="3">
      <div class="range-labels"><span>1</span><span id="graph-depth-val">3</span><span>6</span></div>
      <div class="sb-head" style="margin-top:8px">Relationship Types</div>
      <div id="edge-toggles"></div>
      <div class="sb-head" style="margin-top:6px">Options</div>
      <div class="filter-row">
        <input type="checkbox" id="label-toggle" checked>
        <label for="label-toggle">Show labels</label>
      </div>
    </div>
  </div>
  <div id="view-area">
    <div id="view-hierarchy" class="view-panel active">
      <div id="chart-hierarchy" class="chart-container"></div>
    </div>
    <div id="view-vectors" class="view-panel">
      <div id="chart-vectors" class="chart-container"></div>
    </div>
    <div id="view-graph" class="view-panel" style="position:relative">
      <div id="chart-graph" class="chart-container"></div>
      <button id="graph-back-btn" style="display:none;position:absolute;top:10px;left:10px;z-index:10;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:5px 14px;font-size:13px;cursor:pointer;transition:background 0.15s" onmouseover="this.style.background='#30363d'" onmouseout="this.style.background='#21262d'">&#8592; Back</button>
    </div>
    <div id="detail-panel">
      <button class="close-btn" id="close-detail">&#x2715;</button>
      <div id="detail-content"></div>
    </div>
  </div>
</div>
"""


def _html_scripts(data_json: str, echarts_script: str) -> str:
    """Return the closing script block."""
    return (
        "<script>\n"
        + echarts_script
        + "\n</script>\n"
        + "<script>\n"
        + "var RAW = "
        + data_json
        + ";\n"
        + _dashboard_js()
        + "\n</script>\n</body>\n</html>"
    )


def _dashboard_js() -> str:
    """Return the ECharts-based dashboard application JavaScript."""
    return r"""
// Wait for echarts to be available (handles both inline and CDN loading)
function startApp() {
  if (typeof echarts === 'undefined') { setTimeout(startApp, 50); return; }

  // ── Theme ──────────────────────────────────────────────────────────────────
  echarts.registerTheme('nd', {
    backgroundColor: 'transparent',
    textStyle: { color: '#c9d1d9', fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif' },
    legend: { textStyle: { color: '#8b949e', fontSize: 10 } },
    tooltip: {
      backgroundColor: '#1c2128',
      borderColor: '#30363d',
      textStyle: { color: '#c9d1d9', fontSize: 11 }
    }
  });

  // ── Color helpers ──────────────────────────────────────────────────────────
  function catColor(cat) {
    if (cat === 'bugs') return '#f85149';
    if (cat === 'tasks') return '#3fb950';
    if (cat === 'codebase') return '#58a6ff';
    if (cat === 'database') return '#d2a8ff';
    return '#8b949e';
  }

  // ── Node / edge style configs ──────────────────────────────────────────────
  var NODE_STYLE = {
    module:             ['#58a6ff',  56, 22],
    class:              ['#d2a8ff',  62, 24],
    function:           ['#79c0ff',  48, 20],
    method:             ['#79c0ff',  48, 18],
    project_overview:   ['#f0e68c',  66, 24],
    directory_overview: ['#8b949e',  56, 20],
    config:             ['#e3b341',  48, 18],
    export:             ['#56d364',  48, 18],
    type_def:           ['#bc8cff',  50, 18],
    other:              ['#8b949e',  44, 18],
    bug:                ['#f85149',  52, 20],
    phase:              ['#3fb950',  62, 22],
    task:               ['#3fb950',  52, 20],
    subtask:            ['#56d364',  46, 18],
    database:           ['#d2a8ff',  66, 24],
    table:              ['#b28cf5',  56, 20],
    column:             ['#9c6fe0',  46, 18]
  };

  var EDGE_STYLE = {
    calls:          ['#58a6ff', false, 1.5, true],
    imports:        ['#8b949e', true,  1.2, true],
    inherits:       ['#d2a8ff', false, 2.5, true],
    contains:       ['#30363d', true,  1.0, false],
    relates_to:     ['#f0883e', true,  1.5, true],
    fixed_by:       ['#f85149', false, 1.5, true],
    phase_contains: ['#3fb950', true,  1.0, false],
    task_contains:  ['#56d364', true,  1.0, false],
    defines:        ['#79c0ff', false, 1.2, true],
    uses:           ['#e3b341', true,  1.0, true]
  };

  var CATEGORY_LIST = Object.keys(NODE_STYLE);

  function nodeStyleFor(type) { return NODE_STYLE[type] || ['#8b949e', 96, 36]; }
  // Size nodes to fit their label text — ~7px per char at font-size 11, 24px padding
  function nodeW(n) { return Math.max(52, Math.min(180, (n.name || '').length * 7 + 24)); }
  function nodeH(n) { return 26; }
  function nodeColor(n) { return nodeStyleFor(n.node_type)[0]; }
  function edgeStyleFor(type) { return EDGE_STYLE[type] || ['#444d56', true, 1.0, true]; }

  // ── O(1) node lookup ───────────────────────────────────────────────────────
  var NODE_MAP = {};
  RAW.nodes.forEach(function(n) { NODE_MAP[n.id] = n; });

  // ── State ──────────────────────────────────────────────────────────────────
  var state = {
    tab: 'hierarchy',
    cats: { codebase: true, bugs: true, tasks: true },
    types: {},
    languages: {},
    minImp: 0,
    statuses: { open: true, fixed: true, pending: true, in_progress: true, done: true, '': true },
    search: '',
    depth: 3,
    graphDepth: 3,
    selectedId: null
  };

  function visibleNodes() {
    var q = state.search.toLowerCase();
    return RAW.nodes.filter(function(n) {
      if (!state.cats[n.category]) return false;
      var typeKeys = Object.keys(state.types);
      if (typeKeys.length > 0 && !state.types[n.node_type]) return false;
      var langKeys = Object.keys(state.languages);
      if (langKeys.length > 0 && n.language && !state.languages[n.language]) return false;
      if (n.importance < state.minImp) return false;
      var st = n.bug_status || n.task_status || '';
      if (!state.statuses[st]) return false;
      if (q && n.name.toLowerCase().indexOf(q) < 0 && n.summary_short.toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
  }

  function visibleEdges(visSet) {
    return RAW.edges.filter(function(e) { return visSet[e.source_id] && visSet[e.target_id]; });
  }

  // ── Chart instances ────────────────────────────────────────────────────────
  var charts = {};

  function getChart(id) {
    if (charts[id]) return charts[id];
    var el = document.getElementById('chart-' + id);
    if (!el) return null;
    charts[id] = echarts.init(el, 'nd', { renderer: 'canvas' });
    return charts[id];
  }

  // ── DOM helpers (used only for detail panel) ───────────────────────────────
  function mk(tag, attrs, kids) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function(k) {
      var v = attrs[k];
      if (k === 'class') node.className = v;
      else if (k === 'for') node.setAttribute('for', v);
      else if (k === 'style') node.style.cssText = v;
      else node.setAttribute(k, String(v));
    });
    if (kids) kids.forEach(function(c) {
      if (c == null) return;
      if (typeof c === 'string') node.appendChild(document.createTextNode(c));
      else node.appendChild(c);
    });
    return node;
  }

  function clearEl(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  // ── Tooltip formatter ──────────────────────────────────────────────────────
  function tipHtml(n) {
    var s = n.summary_short || '';
    if (s.length > 120) s = s.slice(0, 120) + '\u2026';
    return (
      '<div style="font-size:11px;max-width:240px;line-height:1.5;overflow:hidden;word-break:break-word">' +
      '<div style="font-weight:600;color:#e6edf3;margin-bottom:2px">' + n.name + '</div>' +
      '<div style="color:#8b949e">' + n.node_type + ' \u00b7 ' + n.category + '</div>' +
      (s ? '<div style="color:#c9d1d9;margin-top:4px">' + s + '</div>' : '') +
      '<div style="color:#8b949e;margin-top:4px;font-size:10px">importance: ' + n.importance.toFixed(3) + '</div>' +
      '</div>'
    );
  }

  // ── Detail panel ───────────────────────────────────────────────────────────
  function showDetail(n) {
    var panel = document.getElementById('detail-panel');
    var content = document.getElementById('detail-content');
    clearEl(content);
    state.selectedId = n.id;

    var color = catColor(n.category);

    content.appendChild(mk('div', {style:'font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:10px;padding-right:24px'}, [n.name]));

    var badges = mk('div', {style:'margin-bottom:10px'});
    badges.appendChild(mk('span', {class:'badge', style:'background:'+color+'33;color:'+color}, [n.category]));
    badges.appendChild(mk('span', {class:'badge', style:'background:#30363d;color:#c9d1d9'}, [n.node_type]));
    content.appendChild(badges);

    if (n.importance != null) {
      var pct = Math.round(n.importance * 100);
      var impBlock = mk('div', {class:'d-field'});
      impBlock.appendChild(mk('div', {class:'d-label'}, ['Importance']));
      impBlock.appendChild(mk('div', {class:'d-value'}, [n.importance.toFixed(3) + '  (' + pct + '%)']));
      var bar = mk('div', {style:'background:#30363d;border-radius:4px;height:4px;margin:4px 0 8px'});
      bar.appendChild(mk('div', {style:'background:'+color+';height:4px;border-radius:4px;width:'+pct+'%'}));
      impBlock.appendChild(bar);
      content.appendChild(impBlock);
    }

    var fields = [
      ['File', n.file_path + (n.line_start ? ':' + n.line_start : '')],
      ['Summary', n.summary_short],
      ['Details', n.summary_detailed],
      ['Severity', n.severity],
      ['Bug Status', n.bug_status],
      ['Task Status', n.task_status],
      ['Priority', n.priority],
      ['LSP Info', n.lsp_hover_doc]
    ];
    fields.forEach(function(f) {
      if (!f[1]) return;
      var div = mk('div', {class:'d-field'});
      div.appendChild(mk('div', {class:'d-label'}, [f[0]]));
      div.appendChild(mk('div', {class:'d-value'}, [String(f[1])]));
      content.appendChild(div);
    });

    if (n.lsp_diagnostics && n.lsp_diagnostics.length) {
      var diagBlock = mk('div', {class:'d-field'});
      diagBlock.appendChild(mk('div', {class:'d-label'}, ['Diagnostics']));
      n.lsp_diagnostics.forEach(function(d) {
        diagBlock.appendChild(mk('div', {style:'color:#f85149;font-size:11px;margin:1px 0'}, [d]));
      });
      content.appendChild(diagBlock);
    }

    // Connected nodes — "Neurals to"
    var connected = [];
    var seen = {};
    RAW.edges.forEach(function(e) {
      if (e.source_id === n.id && !seen[e.target_id]) {
        seen[e.target_id] = true;
        var t = NODE_MAP[e.target_id];
        if (t) connected.push({ node: t, type: e.edge_type, dir: 'out' });
      }
      if (e.target_id === n.id && !seen[e.source_id]) {
        seen[e.source_id] = true;
        var s = NODE_MAP[e.source_id];
        if (s) connected.push({ node: s, type: e.edge_type, dir: 'in' });
      }
    });
    if (connected.length) {
      var connBlock = mk('div', {class:'d-field'});
      connBlock.appendChild(mk('div', {class:'d-label'}, ['Neurals to (' + connected.length + ')']));
      var connScroll = mk('div', {style:'max-height:400px;overflow-y:auto'});
      connected.forEach(function(c) {
        var cColor = nodeColor(c.node);
        var arrow = c.dir === 'out' ? '\u2192' : '\u2190';
        var link = mk('div', {style:'cursor:pointer;padding:4px 0;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:6px'});
        link.appendChild(mk('span', {style:'color:#8b949e;font-size:10px;flex-shrink:0'}, [arrow + '\u00a0' + c.type]));
        link.appendChild(mk('span', {style:'color:'+cColor+';font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'}, [c.node.name]));
        link.addEventListener('click', function() {
          showDetail(c.node);
          if (state.tab === 'graph' && charts['graph']) {
            charts['graph'].dispatchAction({ type: 'highlight', seriesIndex: 0, name: c.node.name });
          }
        });
        link.addEventListener('mouseenter', function() { link.style.background = '#21262d'; });
        link.addEventListener('mouseleave', function() { link.style.background = ''; });
        connScroll.appendChild(link);
      });
      connBlock.appendChild(connScroll);
      content.appendChild(connBlock);
    }

    panel.classList.add('open');
    setTimeout(function() {
      var ac = charts[state.tab]; if (ac) ac.resize();
    }, 320);
  }

  // ── Hierarchy treemap ──────────────────────────────────────────────────────

  var _treemapZoom = 1;  // cumulative zoom for label sizing

  function filterHierarchy(node) {
    var visible = {};
    visibleNodes().forEach(function(n) { visible[n.id] = true; });
    function prune(n) {
      if (!n) return null;
      if (!n.children) return visible[n.id] ? n : null;
      var kids = n.children.map(prune).filter(Boolean);
      if (kids.length === 0 && !visible[n.id]) return null;
      return Object.assign({}, n, { children: kids });
    }
    return prune(node) || { id: '__empty__', name: 'No results', value: 1 };
  }

  function colorizeTree(node) {
    var raw = NODE_MAP[node.id];
    var nodeType = raw ? raw.node_type : (node.node_type || 'other');
    var color = nodeStyleFor(nodeType)[0];
    var imp = raw ? (raw.importance || 0) : 0;
    // Scale fill from 30% opacity (low importance) to 60% (high importance)
    var fillHex = Math.round(0x4d + imp * 0x52).toString(16).padStart(2, '0');
    // Scale border from 60% opacity to 100%
    var borderHex = Math.round(0x99 + imp * 0x66).toString(16).padStart(2, '0');
    var out = {
      id: node.id,
      name: node.name,
      value: node.value,
      itemStyle: { color: color + fillHex, borderColor: color + borderHex, borderWidth: 1 },
      emphasis: { itemStyle: { color: color + 'bb', borderColor: color, borderWidth: 2 } }
    };
    if (node.children) out.children = node.children.map(colorizeTree);
    return out;
  }

  function drawHierarchy() {
    var chart = getChart('hierarchy');
    if (!chart) return;
    _treemapZoom = 1;

    var filtered = filterHierarchy(RAW.hierarchy);
    var data = [colorizeTree(filtered)];

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        confine: true,
        formatter: function(params) {
          var n = NODE_MAP[params.data.id];
          return n ? tipHtml(n) : params.name;
        }
      },
      series: [{
        type: 'treemap',
        id: 'treemap',
        data: data,
        visibleMin: 0,
        leafDepth: state.depth,
        roam: true,
        scaleLimit: { min: 0.5, max: 5 },
        breadcrumb: {
          show: true,
          bottom: 10,
          height: 26,
          itemStyle: { color: '#21262d', borderColor: '#484f58', borderWidth: 1, textStyle: { color: '#c9d1d9', fontSize: 12 } },
          emphasis: { itemStyle: { color: '#30363d' } }
        },
        label: {
          show: true,
          formatter: function(params) {
            var raw = NODE_MAP[params.data.id];
            var prefix = '';
            if (raw) {
              var t = raw.node_type;
              if (t === 'module') prefix = '\u25a0 ';
              else if (t === 'class' || t === 'phase') prefix = '\u25c6 ';
              else if (t === 'function' || t === 'method') prefix = '\u0192 ';
              else if (t === 'bug') prefix = '\u26a0 ';
              else if (t === 'task' || t === 'subtask') prefix = '\u2611 ';
            }
            return prefix + params.name;
          },
          fontSize: 14,
          fontWeight: 500,
          color: '#f0f6fc',
          overflow: 'truncate',
          padding: [2, 4]
        },
        upperLabel: {
          show: true,
          height: 26,
          fontSize: 14,
          fontWeight: 600,
          color: '#f0f6fc',
          overflow: 'truncate',
          backgroundColor: '#161b2299',
          padding: [2, 6, 2, 6]
        },
        itemStyle: { gapWidth: 2, borderWidth: 0 },
        levels: [
          { itemStyle: { borderWidth: 3, borderColor: '#0d1117', gapWidth: 5 } },
          { itemStyle: { borderWidth: 2, borderColor: '#0d111788', gapWidth: 3 } },
          { itemStyle: { gapWidth: 1 } }
        ]
      }]
    }, true);

    chart.off('click');
    chart.on('click', function(params) {
      if (!params.data || !params.data.id) return;
      var n = NODE_MAP[params.data.id];
      if (n) showDetail(n);
    });
  }

  // ── Semantic radial tree ───────────────────────────────────────────────────
  // Shows the knowledge graph as a radial tree: hierarchy + importance sizing
  // + semantic neighbors (nearest nodes in PCA embedding space) on hover.

  function computeNeighbors(nodes) {
    // O(n²) nearest-neighbor lookup in 2-D PCA space. Capped at 500 nodes.
    var map = {};
    if (nodes.length > 500) return map;
    nodes.forEach(function(n) {
      var sorted = nodes
        .filter(function(m) { return m.id !== n.id; })
        .map(function(m) {
          var dx = n.px - m.px, dy = n.py - m.py;
          return { name: m.name, dist2: dx * dx + dy * dy };
        })
        .sort(function(a, b) { return a.dist2 - b.dist2; })
        .slice(0, 4);
      map[n.id] = sorted.map(function(x) { return x.name; });
    });
    return map;
  }

  function drawVectors() {
    var chart = getChart('vectors');
    if (!chart) return;

    var nodes = visibleNodes();
    if (!nodes.length) { chart.clear(); return; }

    var visSet = {};
    nodes.forEach(function(n) { visSet[n.id] = true; });

    // Nearest-neighbor map (semantic proximity from PCA positions)
    var neighborMap = computeNeighbors(nodes);

    // Build radial tree data from the stored hierarchy tree,
    // colouring and sizing each node by importance and type.
    function buildNode(hNode) {
      if (!hNode) return null;
      var raw = NODE_MAP[hNode.id];
      var visible = !!visSet[hNode.id];
      var color = raw ? nodeColor(raw) : '#8b949e';
      var imp = raw ? (raw.importance || 0) : 0;
      var size = Math.max(6, 8 + imp * 34);

      var out = {
        id: hNode.id,
        name: hNode.name,
        value: imp,
        symbolSize: size,
        symbol: raw && raw.node_type === 'bug'   ? 'diamond'
               : raw && (raw.node_type === 'class' || raw.node_type === 'phase') ? 'roundRect'
               : 'circle',
        itemStyle: {
          color: color + (visible ? '2a' : '0d'),
          borderColor: visible ? color : color + '33',
          borderWidth: visible ? (1.5 + imp * 2.5) : 0.5
        },
        emphasis: {
          itemStyle: { color: color + '66', borderColor: color, borderWidth: 3,
                       shadowBlur: 8, shadowColor: color + '88' }
        },
        // Show label only for visible, reasonably important nodes
        label: { show: visible && (imp > 0.28 || !hNode.children) }
      };

      if (hNode.children) {
        var kids = hNode.children.map(buildNode).filter(Boolean);
        if (kids.length) out.children = kids;
      }
      return out;
    }

    var treeRoot = buildNode(RAW.hierarchy);
    if (!treeRoot) treeRoot = { name: 'No results', symbolSize: 10, children: [] };

    // Depth slider drives initialTreeDepth
    var initDepth = Math.min(state.depth, 5);

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        confine: true,
        enterable: false,
        formatter: function(params) {
          if (!params.data || !params.data.id) return params.name || '';
          var n = NODE_MAP[params.data.id];
          if (!n) return params.name || '';
          var nbrs = neighborMap[params.data.id];
          var nbStr = nbrs && nbrs.length
            ? '<div style="color:#8b949e;margin-top:5px;font-size:10px">'
              + '\u2248 nearest in vector space: '
              + '<span style="color:#c9d1d9">' + nbrs.slice(0, 3).join(', ') + '</span>'
              + '</div>'
            : '';
          return tipHtml(n) + nbStr;
        }
      },
      series: [{
        type: 'tree',
        id: 'sem-tree',
        data: [treeRoot],
        layout: 'radial',
        roam: true,
        zoom: 0.85,
        animationEasingUpdate: 'cubicOut',
        // Per-node symbolSize is set in the data items above;
        // series-level is a fallback for nodes without it.
        symbolSize: function(val, params) {
          return params.data.symbolSize || 10;
        },
        lineStyle: { color: '#30363d88', width: 1.2, curveness: 0.45 },
        scaleLimit: { min: 0.2, max: 5 },
        labelLayout: { hideOverlap: true },
        emphasis: {
          focus: 'descendant',
          lineStyle: { width: 2.5, color: '#58a6ff55' }
        },
        blur: {
          itemStyle: { opacity: 0.35 },
          lineStyle: { opacity: 0.15 }
        },
        // Series-level label; per-node label.show controls visibility
        label: {
          show: true,
          formatter: function(params) {
            if (params.data.label && params.data.label.show === false) return '';
            var n = params.data.name || '';
            return n.length > 22 ? n.slice(0, 20) + '\u2026' : n;
          },
          fontSize: 12,
          color: '#c9d1d9',
          distance: 12,
          rotate: 0
        },
        leaves: {
          label: {
            show: true,
            position: 'right',
            verticalAlign: 'middle',
            align: 'left',
            fontSize: 12,
            color: '#8b949e',
            distance: 10,
            formatter: function(params) {
              if (params.data.label && params.data.label.show === false) return '';
              var n = params.data.name || '';
              return n.length > 20 ? n.slice(0, 18) + '\u2026' : n;
            }
          }
        },
        expandAndCollapse: false,
        animationDuration: 400,
        animationDurationUpdate: 600,
        initialTreeDepth: initDepth
      }]
    }, true);

    chart.off('click');
    chart.off('dblclick');
    chart.on('click', function(params) {
      if (!params.data || !params.data.id) return;
      var n = NODE_MAP[params.data.id];
      if (n) showDetail(n);
    });
    // Double-click to manually toggle collapse on a subtree node
    chart.on('dblclick', function(params) {
      if (!params.data) return;
      params.data.collapsed = !params.data.collapsed;
      chart.setOption({ series: [{ data: [treeRoot] }] }, false);
    });
  }

  // ── Interactive collapsible tree (Graph tab) ─────────────────────────────
  // Replaces the old static radial graph with a virtual hierarchy tree.
  // Each real node's children are grouped by relationship type into virtual
  // intermediate nodes ([calls], [imports], [inherits], etc.), enabling
  // infinite drill-down.  Nodes beyond pre-computed depth expand on-demand
  // in JS using the already-embedded RAW.edges data.

  var _graphLayout = 'orthogonal'; // 'orthogonal' | 'radial'
  var _graphTreeRoot = null;        // mutable root — updated on JS-side expansion
  var _expandedIds  = {};           // {nodeId: true} — real nodes explicitly expanded
  var _graphFirstDraw = true;       // first draw uses replace mode (auto-fit)
  var _navHistory   = [];           // [{focusId, expandedSnapshot}] for back navigation

  // Relationship-type → display label map (mirrors Python GROUP_ORDER)
  var GROUP_LABELS = {
    'contains:out':    'contains',
    'phase_contains:out': 'contains',
    'task_contains:out':  'contains',
    'calls:out':       'calls',
    'calls:in':        'called by',
    'imports:out':     'imports',
    'imports:in':      'imported by',
    'inherits:out':    'inherits',
    'inherits:in':     'inherited by',
    'implements:out':  'implements',
    'implements:in':   'implemented by',
    'uses:out':        'uses',
    'uses:in':         'used by',
    'defines:out':     'defines',
    'defines:in':      'defined by',
    'relates_to:out':  'relates to',
    'relates_to:in':   'related from',
    'fixed_by:out':    'fixed by',
    'fixed_by:in':     'fixes',
    'references:out':  'references',
    'references:in':   'referenced by',
    'queries:out':     'queries',
    'queries:in':      'queried by',
    'writes_to:out':   'writes to',
    'writes_to:in':    'written by'
  };

  // Edge type → visible flag (controls which relationship groups render)
  var EDGE_DEFAULT_ON = { contains: true, phase_contains: true, task_contains: true,
                          calls: true, imports: true, inherits: true, implements: true,
                          uses: true, defines: true };
  var edgeVisible = {};
  Object.keys(EDGE_STYLE).forEach(function(t) { edgeVisible[t] = !!(EDGE_DEFAULT_ON[t]); });

  // Lazily-built adjacency index for JS-side expansion
  var _adjOut = null;  // {nodeId: {edgeType: [targetIds]}}
  var _adjIn  = null;  // {nodeId: {edgeType: [sourceIds]}}

  function _buildAdj() {
    if (_adjOut) return;
    _adjOut = {}; _adjIn = {};
    RAW.edges.forEach(function(e) {
      var s = e.source_id, t = e.target_id, et = e.edge_type;
      if (!s || !t || !et) return;
      if (!_adjOut[s]) _adjOut[s] = {};
      if (!_adjOut[s][et]) _adjOut[s][et] = [];
      _adjOut[s][et].push(t);
      if (!_adjIn[t]) _adjIn[t] = {};
      if (!_adjIn[t][et]) _adjIn[t][et] = [];
      _adjIn[t][et].push(s);
    });
  }

  // Collect the set of ancestor node IDs on the path from root down to targetId.
  function _collectAncestorIds(root, targetId) {
    var ancestors = {};
    function dfs(node, path) {
      if (node._real && node.id) {
        path.push(node.id);
        if (node.id === targetId) {
          path.forEach(function(id) { ancestors[id] = true; });
          path.pop();
          return true;
        }
      }
      var ch = node.children || [];
      for (var i = 0; i < ch.length; i++) {
        if (dfs(ch[i], path)) { if (node._real && node.id) path.pop(); return true; }
      }
      if (node._real && node.id) path.pop();
      return false;
    }
    dfs(root, []);
    return ancestors;
  }

  // Compute virtual children for a real node (JS-side, for on-demand expansion)
  function computeRelationshipGroups(nodeId) {
    _buildAdj();
    var groups = [];
    var groupOrder = [
      ['contains','out'],['phase_contains','out'],['task_contains','out'],
      ['calls','out'],['calls','in'],['imports','out'],['imports','in'],
      ['inherits','out'],['inherits','in'],['implements','out'],['implements','in'],
      ['uses','out'],['uses','in'],['defines','out'],['defines','in'],
      ['relates_to','out'],['relates_to','in'],['fixed_by','out'],['fixed_by','in'],
      ['references','out'],['references','in'],['queries','out'],['queries','in'],
      ['writes_to','out'],['writes_to','in']
    ];

    var ancestors = _graphTreeRoot ? _collectAncestorIds(_graphTreeRoot, nodeId) : {};

    groupOrder.forEach(function(pair) {
      var etype = pair[0], dir = pair[1];
      if (!edgeVisible[etype]) return;
      var adj = dir === 'out' ? _adjOut : _adjIn;
      var targets = (adj[nodeId] || {})[etype] || [];
      if (!targets.length) return;

      // Partition into normal targets and back-references (ancestors)
      var normalTargets = targets.filter(function(t) { return !ancestors[t]; });
      var backTargets = targets.filter(function(t) { return ancestors[t] && t !== nodeId; });

      // Sort by importance
      normalTargets = normalTargets.slice().sort(function(a, b) {
        return (NODE_MAP[b] ? NODE_MAP[b].importance : 0) - (NODE_MAP[a] ? NODE_MAP[a].importance : 0);
      });

      var label = GROUP_LABELS[etype + ':' + dir] || etype;
      var kids = normalTargets.map(function(tid) {
        var n = NODE_MAP[tid];
        if (!n) return null;
        return {
          id: tid, name: n.name, node_type: n.node_type,
          language: n.language || '',
          importance: n.importance || 0,
          _real: true, _hasMore: true
        };
      }).filter(Boolean);

      // Append back-reference leaf nodes for ancestor connections
      backTargets.forEach(function(tid) {
        var n = NODE_MAP[tid];
        if (!n) return;
        kids.push({
          id: tid, name: n.name + ' \u21a9',
          node_type: n.node_type, language: n.language || '',
          importance: n.importance || 0,
          _backRef: true, _refTargetId: tid
        });
      });

      if (kids.length) {
        groups.push({ name: '[' + label + ']', _virtual: true,
                      _edgeType: etype, _direction: dir, children: kids });
      }
    });

    return groups.map(styleTreeNode);
  }

  // Find a node by id in _graphTreeRoot (DFS)
  function _findNodeById(root, id) {
    if (!root) return null;
    if (root.id === id) return root;
    var ch = root.children || [];
    for (var i = 0; i < ch.length; i++) {
      var found = _findNodeById(ch[i], id);
      if (found) return found;
    }
    return null;
  }

  // Pre-populate _expandedIds for nodes within initial graphDepth.
  // Counts only real-node depth (virtual groups don't count toward depth).
  function _initExpandedState(node, realLevelsLeft) {
    if (!node) return;
    if (node._real) {
      _expandedIds[node.id] = true;
      if (realLevelsLeft <= 0) return;
      (node.children || []).forEach(function(c) { _initExpandedState(c, realLevelsLeft - 1); });
    } else {
      // virtual / truncated: pass through without counting depth
      (node.children || []).forEach(function(c) { _initExpandedState(c, realLevelsLeft); });
    }
  }

  // Apply ECharts visual styles. collapsed state is driven by _expandedIds.
  function styleTreeNode(node) {
    var out = {};
    Object.keys(node).forEach(function(k) { out[k] = node[k]; });

    if (node._virtual) {
      var es = edgeStyleFor(node._edgeType || 'contains');
      var col = es[0];
      out.symbol = 'emptyRect';
      out.symbolSize = [12, 12];
      out.collapsed = false; // virtual groups always auto-open with their parent
      out.itemStyle = { color: col + '18', borderColor: col,
                        borderWidth: 1, borderType: 'dashed' };
      out.lineStyle = { color: col, width: es[2] || 1,
                        type: es[1] ? 'dashed' : 'solid', opacity: 0.55 };
      out.label = { color: col, fontSize: 10, fontStyle: 'italic',
                    distance: 5, overflow: 'truncate' };
    } else if (node._real) {
      var color = nodeColor(node);
      var imp = node.importance || 0;
      var sz = Math.max(8, Math.min(40, 10 + imp * 30));
      var nt = node.node_type || 'other';
      var sym = nt === 'bug' ? 'diamond' :
                (nt === 'class' || nt === 'phase' || nt === 'struct' ||
                 nt === 'interface' || nt === 'enum') ? 'roundRect' : 'circle';
      out.symbol = sym;
      out.symbolSize = sz;
      out.collapsed = !_expandedIds[node.id]; // collapsed unless explicitly expanded
      out.itemStyle = { color: color + '28', borderColor: color,
                        borderWidth: 1.5 + imp * 2 };
      out.label = { color: '#e6edf3', fontSize: 12, distance: 7,
                    fontWeight: imp > 0.6 ? 'bold' : 'normal',
                    overflow: 'truncate' };
    } else if (node._backRef) {
      var refColor = nodeColor(node);
      out.symbol = 'triangle';
      out.symbolSize = 10;
      out.collapsed = true;
      out.itemStyle = { color: refColor + '30', borderColor: refColor,
                        borderWidth: 1, borderType: 'dashed' };
      out.lineStyle = { color: refColor, width: 1, type: 'dotted', opacity: 0.4 };
      out.label = { color: refColor, fontSize: 10, fontStyle: 'italic', distance: 5 };
    }

    if (node.children) {
      var filteredKids = node.children.filter(function(c) {
        if (!c._virtual) return true;
        return edgeVisible[c._edgeType !== undefined ? c._edgeType : 'contains'];
      });
      out.children = filteredKids.map(styleTreeNode);
    }
    return out;
  }

  // Pan the chart so the node with nodeId is centered in the viewport.
  // Runs after a short delay to let ECharts finish its layout animation.
  function _centerOnNode(chart, nodeId, delay) {
    setTimeout(function() {
      try {
        var model = chart.getModel();
        var series = model.getSeriesByIndex(0);
        if (!series) return;
        var data = series.getData();
        var idx = -1;
        data.each(function(i) {
          var raw = data.getRawDataItem(i);
          if (raw && raw.id === nodeId) idx = i;
        });
        if (idx < 0) return;
        var layout = data.getItemLayout(idx);
        if (!layout) return;
        var pixel = chart.convertToPixel({ seriesIndex: 0 }, [layout.x, layout.y]);
        if (!pixel) return;
        var cx = chart.getWidth() / 2;
        var cy = chart.getHeight() / 2;
        chart.dispatchAction({ type: 'roam', seriesIndex: 0, dx: cx - pixel[0], dy: cy - pixel[1] });
      } catch(e) { /* ECharts internal API may vary; silently skip */ }
    }, delay !== undefined ? delay : 120);
  }

  // Rebuild chart option and call setOption.
  // replace=true: resets zoom/pan (use after adding new nodes or layout change)
  // replace=false: merge mode, preserves user zoom/pan (use for toggle of existing nodes)
  function _applyGraphOption(chart, replace) {
    var styledRoot = styleTreeNode(_graphTreeRoot);
    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item', confine: true, enterable: false,
        formatter: function(params) {
          var d = params.data;
          if (!d) return '';
          if (d._virtual) {
            var cnt = (d.children || []).length;
            return '<span style="color:#8b949e;font-size:11px">'
              + (d.name || '') + ' &nbsp;<em>(' + cnt + ')</em></span>';
          }
          if (d._backRef) return '<span style="color:#8b949e;font-style:italic">\u21a9 ' + (d.name||'') + '</span>';
          var n = NODE_MAP[d.id];
          return n ? tipHtml(n) : (d.name || '');
        }
      },
      series: [{
        type: 'tree',
        data: [styledRoot],
        layout: _graphLayout,
        orient: _graphLayout === 'orthogonal' ? 'LR' : undefined,
        roam: true,
        expandAndCollapse: false, // we own all expand/collapse state via _expandedIds
        animationDuration: 350,
        animationDurationUpdate: 500,
        animationEasingUpdate: 'cubicOut',
        scaleLimit: { min: 0.08, max: 8 },
        lineStyle: { color: '#21262d', width: 1, curveness: 0.5 },
        emphasis: { focus: 'descendant', lineStyle: { width: 2, color: '#58a6ff44' } },
        blur: { itemStyle: { opacity: 0.25 }, lineStyle: { opacity: 0.1 }, label: { opacity: 0.2 } },
        labelLayout: { hideOverlap: true },
        leaves: {
          label: { position: _graphLayout === 'orthogonal' ? 'right' : 'bottom',
                   align: _graphLayout === 'orthogonal' ? 'left' : 'center',
                   fontSize: 11, color: '#8b949e', distance: 5 }
        }
      }]
    }, !!replace);
  }

  function _updateBackBtn() {
    var btn = document.getElementById('graph-back-btn');
    if (btn) btn.style.display = _navHistory.length ? 'block' : 'none';
  }

  function drawGraph() {
    var chart = getChart('graph');
    if (!chart) return;
    if (!RAW.virtual_tree) { chart.clear(); return; }

    // Clone tree and init expanded state on first draw
    if (!_graphTreeRoot) {
      _graphTreeRoot = JSON.parse(JSON.stringify(RAW.virtual_tree));
      _expandedIds = {};
      _navHistory = [];
      _initExpandedState(_graphTreeRoot, (state.graphDepth || 3) - 1);
      _graphFirstDraw = true;
      _updateBackBtn();
    }

    _applyGraphOption(chart, _graphFirstDraw);
    _graphFirstDraw = false;

    chart.off('click');
    chart.on('click', function(params) {
      var d = params.data;
      if (!d) return;

      // Back-reference node clicked — show ancestor detail and center on it
      if (d._backRef) {
        var refNode = NODE_MAP[d._refTargetId];
        if (refNode) showDetail(refNode);
        _centerOnNode(chart, d._refTargetId);
        return;
      }

      // Virtual group click: no action — real parent controls visibility
      if (d._virtual) return;

      // Real node clicked — show detail panel
      var n = NODE_MAP[d.id];
      if (n) showDetail(n);

      // Find in our source tree
      var treeNode = _findNodeById(_graphTreeRoot, d.id);
      if (!treeNode) return;

      var wasExpanded = !!_expandedIds[d.id];

      if (!wasExpanded) {
        // ── EXPAND ──────────────────────────────────────────────────────────
        // Push current state onto navigation history before expanding
        _navHistory.push({ focusId: d.id, expandedSnapshot: JSON.parse(JSON.stringify(_expandedIds)) });
        _updateBackBtn();

        _expandedIds[d.id] = true;

        // Add relationship groups on-demand if not yet computed
        var addedNew = false;
        if (treeNode._hasMore && !(treeNode.children && treeNode.children.length)) {
          treeNode.children = computeRelationshipGroups(d.id);
          treeNode._hasMore = false;
          addedNew = true;
        }

        if (!treeNode.children || !treeNode.children.length) {
          // True leaf: no children — skip redraw, just show detail panel
          _navHistory.pop(); // undo nav push since no expansion happened
          _updateBackBtn();
          return;
        }

        // Redraw: merge mode to preserve zoom/pan, then center on the clicked node
        _applyGraphOption(chart, false);
        _centerOnNode(chart, d.id);

      } else {
        // ── COLLAPSE ────────────────────────────────────────────────────────
        delete _expandedIds[d.id];
        _applyGraphOption(chart, false);
      }
    });

    // Wire back button (safe to re-wire on each drawGraph call)
    var backBtn = document.getElementById('graph-back-btn');
    if (backBtn && !backBtn._wired) {
      backBtn._wired = true;
      backBtn.addEventListener('click', function() {
        if (!_navHistory.length) return;
        var prev = _navHistory.pop();
        _expandedIds = prev.expandedSnapshot;
        _updateBackBtn();
        _applyGraphOption(chart, false);
        _centerOnNode(chart, prev.focusId, 200);
      });
    }
  }

  // ── Graph sidebar controls ─────────────────────────────────────────────────

  function buildEdgeToggles() {
    var div = document.getElementById('edge-toggles');
    if (!div) return;
    clearEl(div);
    Object.keys(EDGE_STYLE).forEach(function(type) {
      var es = edgeStyleFor(type);
      var id = 'et-' + type;
      var cb = mk('input', { type: 'checkbox', id: id });
      cb.checked = !!(edgeVisible[type]);
      cb.addEventListener('change', function() {
        edgeVisible[type] = cb.checked;
        if (state.tab === 'graph') drawGraph();
      });
      var swatch = mk('span', { class: 'edge-toggle-swatch', style: 'background:' + es[0] });
      var EDGE_DISPLAY = { phase_contains: 'phase contains', task_contains: 'task contains' };
      var label = EDGE_DISPLAY[type] || GROUP_LABELS[type + ':out'] || type.replace(/_/g, '\u00a0');
      var lbl = mk('label', { 'for': id }, [swatch, label]);
      div.appendChild(mk('div', { class: 'filter-row' }, [cb, lbl]));
    });
  }

  // ── Sidebar ────────────────────────────────────────────────────────────────

  function buildSidebar() {
    // Header stats
    var hs = document.getElementById('header-stats');
    var counts = {};
    RAW.nodes.forEach(function(n) { counts[n.category] = (counts[n.category] || 0) + 1; });
    Object.keys(counts).forEach(function(cat) {
      var span = document.createElement('span');
      span.style.color = catColor(cat);
      span.textContent = counts[cat] + ' ' + cat;
      hs.appendChild(span);
    });

    // Category checkboxes
    var catDiv = document.getElementById('cat-filters');
    ['codebase', 'bugs', 'tasks'].forEach(function(cat) {
      var cb = mk('input', { type: 'checkbox', id: 'cat-' + cat });
      cb.checked = true;
      cb.addEventListener('change', function() { state.cats[cat] = cb.checked; redraw(); });
      catDiv.appendChild(mk('div', { class: 'filter-row' }, [
        cb,
        mk('label', { 'for': 'cat-' + cat, style: 'color:' + catColor(cat) }, [cat])
      ]));
    });

    // Node type multi-select
    var allTypes = [];
    var seen = {};
    RAW.nodes.forEach(function(n) {
      if (!seen[n.node_type]) { seen[n.node_type] = 1; allTypes.push(n.node_type); }
    });
    allTypes.sort();
    var sel = document.getElementById('type-filter');
    allTypes.forEach(function(t) { sel.appendChild(mk('option', { value: t }, [t])); });
    sel.addEventListener('change', function() {
      state.types = {};
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].selected) state.types[sel.options[i].value] = true;
      }
      redraw();
    });

    // Language filter checkboxes
    var langDiv = document.getElementById('lang-filters');
    var allLangs = [];
    var seenLangs = {};
    RAW.nodes.forEach(function(n) {
      if (n.language && !seenLangs[n.language]) { seenLangs[n.language] = 1; allLangs.push(n.language); }
    });
    allLangs.sort();
    if (allLangs.length > 1) {
      allLangs.forEach(function(lang) {
        var cb = mk('input', { type: 'checkbox', id: 'lang-' + lang });
        cb.checked = true;
        cb.addEventListener('change', function() {
          if (cb.checked) { delete state.languages[lang]; }
          else { state.languages[lang] = false; }
          redraw();
        });
        langDiv.appendChild(mk('div', { class: 'filter-row' }, [
          cb,
          mk('label', { 'for': 'lang-' + lang }, [lang])
        ]));
      });
    } else {
      langDiv.appendChild(mk('div', { style: 'color:#888;font-size:11px;padding:2px 0' }, [(allLangs[0] || 'Single language') + ' only — index more languages to filter']));
    }

    // Importance slider
    var impSlider = document.getElementById('imp-slider');
    var impLo = document.getElementById('imp-lo');
    impSlider.addEventListener('input', function() {
      state.minImp = parseFloat(impSlider.value);
      impLo.textContent = state.minImp.toFixed(2);
      redraw();
    });

    // Status checkboxes
    var statusDiv = document.getElementById('status-filters');
    ['open', 'fixed', 'pending', 'in_progress', 'done'].forEach(function(st) {
      var cb = mk('input', { type: 'checkbox', id: 'st-' + st });
      cb.checked = true;
      cb.addEventListener('change', function() { state.statuses[st] = cb.checked; redraw(); });
      statusDiv.appendChild(mk('div', { class: 'filter-row' }, [
        cb,
        mk('label', { 'for': 'st-' + st }, [st])
      ]));
    });

    // Search
    var searchIn = document.getElementById('search-input');
    var searchTimer;
    searchIn.addEventListener('input', function() {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(function() { state.search = searchIn.value; redraw(); }, 200);
    });

    // Depth slider
    var depthSlider = document.getElementById('depth-slider');
    var depthVal = document.getElementById('depth-val');
    depthSlider.addEventListener('input', function() {
      state.depth = parseInt(depthSlider.value);
      depthVal.textContent = state.depth;
      if (state.tab === 'hierarchy') drawHierarchy();
      else if (state.tab === 'vectors') drawVectors();
    });

    // Close detail
    document.getElementById('close-detail').addEventListener('click', function() {
      state.selectedId = null;
      var ac = charts[state.tab];
      if (ac) ac.dispatchAction({ type: 'downplay', seriesIndex: 0 });
      document.getElementById('detail-panel').classList.remove('open');
      setTimeout(function() {
        var ac2 = charts[state.tab]; if (ac2) ac2.resize();
      }, 320);
    });

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        state.tab = btn.getAttribute('data-tab');
        document.querySelectorAll('.view-panel').forEach(function(p) { p.classList.remove('active'); });
        document.getElementById('view-' + state.tab).classList.add('active');
        var gc = document.getElementById('graph-controls');
        if (gc) gc.style.display = state.tab === 'graph' ? 'block' : 'none';
        // Resize chart once panel is visible, then draw
        setTimeout(function() {
          if (charts[state.tab]) charts[state.tab].resize();
          redraw();
        }, 20);
      });
    });

    // Label toggle (Graph tab — tree series supports label.show)
    var labelCb = document.getElementById('label-toggle');
    if (labelCb) {
      labelCb.addEventListener('change', function() {
        var chart = charts['graph'];
        if (chart) chart.setOption({ series: [{ label: { show: labelCb.checked } }] }, false);
      });
    }

    // Graph layout toggle (orthogonal / radial)
    var layoutSel = document.getElementById('graph-layout-select');
    if (layoutSel) {
      layoutSel.addEventListener('change', function() {
        _graphLayout = layoutSel.value;
        _graphFirstDraw = true; // force re-fit on layout change
        if (state.tab === 'graph') drawGraph();
      });
    }

    // Graph expand-depth slider — resets expanded state to new depth
    var gDepthSlider = document.getElementById('graph-depth-slider');
    var gDepthVal    = document.getElementById('graph-depth-val');
    if (gDepthSlider) {
      gDepthSlider.addEventListener('input', function() {
        state.graphDepth = parseInt(gDepthSlider.value);
        if (gDepthVal) gDepthVal.textContent = state.graphDepth;
        if (_graphTreeRoot) {
          _expandedIds = {};
          _initExpandedState(_graphTreeRoot, state.graphDepth - 1);
          _graphFirstDraw = true; // re-fit after depth change
        }
        if (state.tab === 'graph') drawGraph();
      });
    }

    // Edge toggles
    buildEdgeToggles();
  }

  // ── Cross-tab selection ────────────────────────────────────────────────────

  function applySelection() {
    if (!state.selectedId) return;
    var chart = charts[state.tab];
    if (!chart) return;
    var n = NODE_MAP[state.selectedId];
    if (!n) return;
    chart.dispatchAction({ type: 'highlight', seriesIndex: 0, name: n.name });
  }

  // ── Redraw ─────────────────────────────────────────────────────────────────

  function redraw() {
    if (state.tab === 'hierarchy') drawHierarchy();
    else if (state.tab === 'vectors') drawVectors();
    else if (state.tab === 'graph') drawGraph();
    applySelection();
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  buildSidebar();
  drawHierarchy();

  window.addEventListener('resize', function() {
    setTimeout(function() {
      Object.keys(charts).forEach(function(id) { if (charts[id]) charts[id].resize(); });
    }, 50);
  });
}

startApp();
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dashboard_html(
    storage: Storage,
    output_path: Optional[str] = None,
    project_root: str = ".",
) -> str:
    """Generate the interactive ECharts dashboard HTML.

    Args:
        storage: Open Storage instance.
        output_path: If provided, write the HTML to this path.
        project_root: Used to locate cached ECharts JS.

    Returns:
        HTML string.
    """
    data = _extract_data(storage)
    data_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    echarts_raw = _get_echarts(project_root)
    echarts_script = echarts_raw if echarts_raw else _echarts_cdn_loader()

    html = _build_html(data_json, echarts_script)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    return html
