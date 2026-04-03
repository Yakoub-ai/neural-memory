"""Interactive ECharts dashboard for neural memory knowledge graph.

Generates a single self-contained HTML file with:
- Sidebar: category/type/importance/status/search filters
- Tab 1: Hierarchy treemap (ECharts treemap)
- Tab 2: Vector space scatter (PCA-projected, ECharts scatter)
- Tab 3: Force-directed graph (ECharts graph/force)
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


def _graph_layout(nodes: list[dict], edges: list[dict]) -> dict[str, list[float]]:
    """Fruchterman-Reingold layout — numpy fast path, pure-Python fallback.

    Returns {node_id: [x, y]} scaled to fit node count without overlap.
    """
    if not nodes:
        return {}

    ids = [n["id"] for n in nodes]
    id_idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)

    # Canvas scales so a sqrt(n)×sqrt(n) grid of nodes fits with comfortable gaps.
    # We need area ∝ n (linear), so area_scale ∝ sqrt(n).
    # sqrt(n/4) gives ~2× more space per node than sqrt(n/15).
    area_scale = max(1.0, math.sqrt(n / 4.0))
    W, H = 1000.0 * area_scale, 800.0 * area_scale

    # Seed from PCA so semantically similar nodes start close
    pca = _pca_positions(nodes)

    try:
        import numpy as np

        pos = np.array(
            [[pca.get(nid, [0.0, 0.0])[0] * W * 0.38,
              pca.get(nid, [0.0, 0.0])[1] * H * 0.38] for nid in ids],
            dtype=np.float64,
        )

        # Build edge index (undirected)
        ep = []
        for e in edges:
            s, t = e.get("source_id"), e.get("target_id")
            if s in id_idx and t in id_idx and s != t:
                ep.append((id_idx[s], id_idx[t]))
        ep_arr = np.array(ep, dtype=np.int32) if ep else np.empty((0, 2), dtype=np.int32)

        # k: ideal FR inter-node distance. Increased multiplier (2.0) and floor (400)
        # so nodes spread aggressively during FR before overlap removal takes over.
        k = max(math.sqrt(W * H / max(n, 1)) * 2.0, 400.0)
        temp = W / 4.0
        cooling = temp / 61.0

        for _ in range(60):
            delta = pos[:, np.newaxis] - pos[np.newaxis, :]  # (n,n,2)
            dist = np.linalg.norm(delta, axis=2)             # (n,n)
            dist = np.where(dist < 0.5, 0.5, dist)
            np.fill_diagonal(dist, 1.0)

            rep = (k * k / dist)[:, :, np.newaxis] * delta / dist[:, :, np.newaxis]
            disp = rep.sum(axis=1)

            if len(ep_arr):
                i_idx, j_idx = ep_arr[:, 0], ep_arr[:, 1]
                dv = pos[i_idx] - pos[j_idx]
                d = np.linalg.norm(dv, axis=1, keepdims=True).clip(min=1.0)
                attr = dv / d * (d ** 2 / k)
                np.add.at(disp, i_idx, -attr)
                np.add.at(disp, j_idx,  attr)

            dlen = np.linalg.norm(disp, axis=1, keepdims=True).clip(min=1.0)
            pos += disp / dlen * np.minimum(dlen, temp)
            # Clip to ±W (was ±W/2) so FR itself can spread nodes across the full canvas
            pos[:, 0] = pos[:, 0].clip(-W, W)
            pos[:, 1] = pos[:, 1].clip(-H, H)
            temp = max(temp - cooling, 0.5)

        # Overlap removal: push apart nodes whose bounding boxes collide.
        #
        # min_sep is sized so a sqrt(n)×sqrt(n) grid fits in [-4W, 4W]:
        #   grid_width = (sqrt(n)-1) * min_sep ≤ 8*W
        #   → min_sep = 8*W / (sqrt(n)-1) ≈ 6*W / sqrt(n)  [for n≥4]
        #
        # This guarantees: after overlap removal, ECharts auto-fit maps nodes to
        # ≥2× their rendered pixel width apart at any node count.
        sqn = max(math.sqrt(n), 2.0)
        min_sep_x = max(200.0, 6.0 * W / sqn)
        min_sep_y = max(100.0, 6.0 * H / sqn)

        # Lower push factor (0.35 vs 0.5) and many more iterations to avoid
        # oscillation on dense graphs; break early when fully resolved.
        for _ in range(150):
            dx = pos[:, np.newaxis, 0] - pos[np.newaxis, :, 0]   # (n,n)
            dy = pos[:, np.newaxis, 1] - pos[np.newaxis, :, 1]
            ovlp_x = min_sep_x - np.abs(dx)
            ovlp_y = min_sep_y - np.abs(dy)
            mask = (ovlp_x > 0) & (ovlp_y > 0)
            np.fill_diagonal(mask, False)
            if not mask.any():
                break
            sx = np.where(dx >= 0, 1.0, -1.0)
            sy = np.where(dy >= 0, 1.0, -1.0)
            push_x = (mask * ovlp_x * sx * 0.35).sum(axis=1)
            push_y = (mask * ovlp_y * sy * 0.35).sum(axis=1)
            pos[:, 0] = (pos[:, 0] + push_x).clip(-W * 4.0, W * 4.0)
            pos[:, 1] = (pos[:, 1] + push_y).clip(-H * 4.0, H * 4.0)

        return {ids[i]: [round(float(pos[i, 0]), 1), round(float(pos[i, 1]), 1)]
                for i in range(n)}

    except Exception:
        pass

    # Pure-Python fallback: scale PCA positions proportionally
    fallback_scale = 460.0 * area_scale
    return {nid: [round(pca.get(nid, [0.0, 0.0])[0] * fallback_scale, 1),
                  round(pca.get(nid, [0.0, 0.0])[1] * fallback_scale * 0.78, 1)]
            for nid in ids}


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
    graph_pos = _graph_layout(nodes, edges)

    # Remove embedding from node data (large, not needed in JS)
    for n in nodes:
        n.pop("embedding", None)
        n["px"] = pca.get(n["id"], [0.0, 0.0])[0]
        n["py"] = pca.get(n["id"], [0.0, 0.0])[1]
        gp = graph_pos.get(n["id"], [0.0, 0.0])
        n["gx"] = gp[0]
        n["gy"] = gp[1]

    hierarchy = _build_hierarchy(nodes, edges)

    stats = storage.get_stats() if hasattr(storage, "get_stats") else {}

    return {
        "nodes": nodes,
        "edges": edges,
        "hierarchy": hierarchy,
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
      <div class="sb-head">Edge Types</div>
      <div id="edge-toggles"></div>
      <div class="sb-head" style="margin-top:10px">Node Spacing</div>
      <input type="range" id="air-slider" min="0" max="20" step="1" value="2">
      <div class="range-labels"><span>Tight</span><span>Spread</span></div>
      <div class="sb-head" style="margin-top:6px">Graph Options</div>
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
    <div id="view-graph" class="view-panel">
      <div id="chart-graph" class="chart-container"></div>
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
    module:             ['#58a6ff', 112, 42],
    class:              ['#d2a8ff', 122, 44],
    function:           ['#79c0ff',  92, 38],
    method:             ['#79c0ff',  92, 36],
    project_overview:   ['#f0e68c', 132, 44],
    directory_overview: ['#8b949e', 112, 40],
    config:             ['#e3b341',  92, 36],
    export:             ['#56d364',  92, 34],
    type_def:           ['#bc8cff',  96, 36],
    other:              ['#8b949e',  86, 34],
    bug:                ['#f85149', 106, 40],
    phase:              ['#3fb950', 122, 44],
    task:               ['#3fb950', 106, 38],
    subtask:            ['#56d364',  92, 34],
    database:           ['#d2a8ff', 132, 46],
    table:              ['#b28cf5', 112, 40],
    column:             ['#9c6fe0',  92, 34]
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
  function nodeW(n) { var s = nodeStyleFor(n.node_type); return s[1] + (n.importance || 0) * 20; }
  function nodeH(n) { var s = nodeStyleFor(n.node_type); return s[2] + (n.importance || 0) * 8; }
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
    minImp: 0,
    statuses: { open: true, fixed: true, pending: true, in_progress: true, done: true, '': true },
    search: '',
    depth: 3,
    selectedId: null
  };

  var edgeVisible = {};
  Object.keys(EDGE_STYLE).forEach(function(t) { edgeVisible[t] = true; });

  function visibleNodes() {
    var q = state.search.toLowerCase();
    return RAW.nodes.filter(function(n) {
      if (!state.cats[n.category]) return false;
      var typeKeys = Object.keys(state.types);
      if (typeKeys.length > 0 && !state.types[n.node_type]) return false;
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
      var MAX_CONN = 25;
      var shown = connected.slice(0, MAX_CONN);
      var rest = connected.length - MAX_CONN;
      var connBlock = mk('div', {class:'d-field'});
      connBlock.appendChild(mk('div', {class:'d-label'}, ['Neurals to (' + connected.length + ')']));
      shown.forEach(function(c) {
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
        connBlock.appendChild(link);
      });
      if (rest > 0) {
        connBlock.appendChild(mk('div', {style:'color:#8b949e;font-size:11px;margin-top:4px'}, ['+' + rest + ' more']));
      }
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

  // ── Force-directed graph ───────────────────────────────────────────────────

  var _graphNodes = [];  // snapshot for 2-hop focus
  var _graphZoom = 1;    // cumulative zoom factor for label scaling

  function buildGraphNodes(nodes, zoom, airScale) {
    zoom = zoom || 1;
    airScale = airScale || 1;
    // Scale nodes down for large graphs so they don't overlap after ECharts auto-fit.
    // ECharts fits the bounding box to viewport, so screen_sep ∝ 1/sqrt(n).
    // Scaling node size by the same factor keeps separation > node size at any n.
    var sizeScale = Math.max(0.15, Math.min(1.0, Math.sqrt(30 / Math.max(1, nodes.length))));
    return nodes.map(function(n) {
      var color = nodeColor(n);
      var catIdx = CATEGORY_LIST.indexOf(n.node_type);
      var w = Math.round(nodeW(n) * sizeScale);
      var h = Math.round(nodeH(n) * sizeScale);
      var maxChars = Math.max(3, Math.floor((w - 10) / 10));
      var label = n.name.length > maxChars ? n.name.slice(0, maxChars - 1) + '\u2026' : n.name;
      return {
        id: n.id,
        name: n.name,
        x: (n.gx || 0) * airScale,
        y: (n.gy || 0) * airScale,
        fixed: true,
        symbolSize: [w, h],
        symbol: n.node_type === 'bug' ? 'diamond' : 'roundRect',
        itemStyle: {
          color: color + '22',
          borderColor: color,
          borderWidth: Math.max(1, (1.5 + (n.importance || 0) * 2) * sizeScale),
          opacity: 1
        },
        emphasis: {
          itemStyle: { color: color + '55', borderColor: color, borderWidth: 3 },
          label: { show: true }
        },
        label: {
          show: true,
          formatter: label,
          fontSize: Math.min(52, Math.max(10, Math.round(16 * zoom * sizeScale))),
          color: '#e6edf3',
          position: 'inside',
          overflow: 'truncate',
          width: w - 10
        },
        category: catIdx >= 0 ? catIdx : 0,
        value: n.importance || 0,
        _id: n.id
      };
    });
  }

  function buildGraphLinks(nodes, edges) {
    var visSet = {};
    nodes.forEach(function(n) { visSet[n.id] = true; });
    return edges
      .filter(function(e) {
        return visSet[e.source_id] && visSet[e.target_id] && edgeVisible[e.edge_type];
      })
      .map(function(e) {
        var es = edgeStyleFor(e.edge_type);
        return {
          source: e.source_id,
          target: e.target_id,
          lineStyle: {
            color: es[0],
            type: es[1] ? 'dashed' : 'solid',
            width: es[2],
            curveness: 0.25,
            opacity: 0.7
          },
          symbol: ['none', es[3] ? 'arrow' : 'none'],
          symbolSize: [8, 8],
          _edgeType: e.edge_type
        };
      });
  }

  var _airScale = 1 + 2 * 0.2;  // default matches air-slider value="2" (tighter default; layout is already spread)

  function drawGraph() {
    var chart = getChart('graph');
    if (!chart) return;
    _graphZoom = 1;

    var nodes = visibleNodes();
    _graphNodes = nodes;

    if (!nodes.length) { chart.clear(); return; }

    var allEdges = visibleEdges((function() {
      var s = {}; nodes.forEach(function(n) { s[n.id] = true; }); return s;
    })());

    var gNodes = buildGraphNodes(nodes, 1, _airScale);
    var links = buildGraphLinks(nodes, allEdges);
    var categories = CATEGORY_LIST.map(function(type) {
      return { name: type, itemStyle: { color: NODE_STYLE[type][0] } };
    });

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        confine: true,
        enterable: false,
        formatter: function(params) {
          if (params.dataType === 'edge') {
            return '<span style="color:#8b949e;font-size:10px">' +
              (params.data._edgeType || 'edge').replace(/_/g, ' ') + '</span>';
          }
          var n = NODE_MAP[params.data._id || params.data.id];
          return n ? tipHtml(n) : params.name;
        }
      },
      legend: [{ show: false, data: categories.map(function(c) { return c.name; }) }],
      series: [{
        type: 'graph',
        layout: 'none',
        roam: true,
        draggable: true,
        animation: false,
        // zoom:1 = ECharts auto-fits the full position bounding box to the viewport.
        // This guarantees all nodes are visible on load; users zoom in for detail.
        zoom: 1.0,
        emphasis: { scale: false, focus: 'adjacency', blurScope: 'global' },
        blur: { itemStyle: { opacity: 0.05 }, lineStyle: { opacity: 0.02 }, label: { opacity: 0 } },
        categories: categories,
        data: gNodes,
        links: links,
        lineStyle: { opacity: 0.65 }
      }]
    }, true);

    // Build a dataIndex lookup for dispatchAction (gNodes is stable after init)
    var _nodeDataIndex = {};
    gNodes.forEach(function(d, idx) { _nodeDataIndex[d._id] = idx; });

    chart.off('click');
    chart.off('dblclick');
    chart.off('graphroam');

    var _focusedId = null;

    function _applyFocus(focusId) {
      var connSet = {};
      connSet[focusId] = true;
      allEdges.forEach(function(e) {
        if (e.source_id === focusId) connSet[e.target_id] = true;
        if (e.target_id === focusId) connSet[e.source_id] = true;
      });
      var indices = [];
      gNodes.forEach(function(d, idx) { if (connSet[d._id]) indices.push(idx); });
      chart.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex: indices });
    }

    function _clearFocus() {
      _focusedId = null;
      chart.dispatchAction({ type: 'downplay', seriesIndex: 0 });
    }

    chart.on('click', function(params) {
      if (params.dataType !== 'node') return;
      var nid = params.data._id || params.data.id;
      var n = NODE_MAP[nid];
      if (n) showDetail(n);
      _focusedId = nid;
      _applyFocus(nid);
    });

    chart.getZr().on('click', function(evt) {
      if (!evt.target && _focusedId) _clearFocus();
    });

    chart.on('dblclick', function(params) {
      if (params.dataType !== 'node') return;
      focusSubgraph(params.data._id || params.data.id, nodes, allEdges);
    });

    chart.getZr().on('dblclick', function(evt) {
      if (!evt.target) drawGraph();
    });

    // Zoom: update only fontSize, no node rebuild, no layout change
    var _zoomTimer = null;
    chart.on('graphroam', function(params) {
      if (!params.zoom) return;
      _graphZoom = Math.max(0.3, Math.min(6, _graphZoom * params.zoom));
      clearTimeout(_zoomTimer);
      _zoomTimer = setTimeout(function() {
        var fs = Math.min(52, Math.max(14, Math.round(16 * _graphZoom)));
        chart.setOption({ series: [{ label: { fontSize: fs } }] }, false);
      }, 80);
    });
  }

  // 2-hop subgraph focus via dispatchAction — no data rebuild
  function focusSubgraph(nid, nodes, edges) {
    var chart = charts['graph'];
    if (!chart) return;

    var adj = {};
    edges.forEach(function(e) {
      if (!adj[e.source_id]) adj[e.source_id] = {};
      if (!adj[e.target_id]) adj[e.target_id] = {};
      adj[e.source_id][e.target_id] = true;
      adj[e.target_id][e.source_id] = true;
    });

    var inScope = {};
    inScope[nid] = true;
    Object.keys(adj[nid] || {}).forEach(function(id1) {
      inScope[id1] = true;
      Object.keys(adj[id1] || {}).forEach(function(id2) { inScope[id2] = true; });
    });

    var indices = [];
    _graphNodes.forEach(function(n, idx) { if (inScope[n.id]) indices.push(idx); });
    chart.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex: indices });
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
      cb.checked = edgeVisible[type];
      cb.addEventListener('change', function() {
        edgeVisible[type] = cb.checked;
        if (charts['graph']) drawGraph();
      });
      var swatch = mk('span', { class: 'edge-toggle-swatch', style: 'background:' + es[0] });
      var lbl = mk('label', { 'for': id }, [swatch, type.replace(/_/g, '\u00a0')]);
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

    // Label toggle
    var labelCb = document.getElementById('label-toggle');
    if (labelCb) {
      labelCb.addEventListener('change', function() {
        var chart = charts['graph'];
        if (chart) chart.setOption({ series: [{ label: { show: labelCb.checked } }] }, false);
      });
    }

    // Air (node spacing) slider — scales pre-computed positions, debounced, nodes only
    var airSlider = document.getElementById('air-slider');
    var _airDebounce = null;
    if (airSlider) {
      airSlider.addEventListener('input', function() {
        var chart = charts['graph'];
        if (!chart || state.tab !== 'graph') return;
        _airScale = 1 + parseInt(airSlider.value) * 0.2;
        clearTimeout(_airDebounce);
        _airDebounce = setTimeout(function() {
          // Only update node positions — links don't need rebuild for layout:'none'
          chart.setOption({ series: [{ data: buildGraphNodes(_graphNodes, _graphZoom, _airScale) }] }, false);
        }, 60);
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
