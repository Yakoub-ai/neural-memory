"""Interactive HTML visualizations for the neural memory graph.

Two views:
  - Hierarchy: treemap of module → class → function/method containment,
    sized by importance, colored by node type.
  - Vector space: 2D/3D scatter of PCA-projected composite embeddings,
    colored by node type (or file), sized by importance.

Both outputs are self-contained HTML files (Plotly.js inlined, ~3 MB each).

Requires: numpy + plotly  (viz optional extra).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .storage import Storage

# Node-type → display color
_TYPE_COLORS = {
    "module":   "#4C72B0",
    "class":    "#DD8452",
    "function": "#55A868",
    "method":   "#C44E52",
    "config":   "#8172B3",
    "export":   "#937860",
    "type_def": "#DA8BC3",
    "other":    "#8C8C8C",
}

_TOTAL_DIMS = 118  # must match embeddings._TOTAL_DIMS


def _viz_available() -> bool:
    try:
        import plotly  # noqa: F401
        import numpy   # noqa: F401
        return True
    except ImportError:
        return False


# ── Hierarchy View ─────────────────────────────────────────────────────────────

def _build_hierarchy_data(storage: "Storage") -> tuple[list, list, list, list, list, list]:
    """Walk CONTAINS edges to build treemap lists.

    Returns (ids, labels, parents, values, colors, hovers).
    """
    from .models import EdgeType

    nodes = {n.id: n for n in storage.get_all_nodes()}
    if not nodes:
        return [], [], [], [], [], []

    # Build parent map from CONTAINS edges
    child_to_parent: dict[str, str] = {}
    for node in nodes.values():
        for edge in storage.get_edges_from(node.id):
            if edge.edge_type == EdgeType.CONTAINS:
                child_to_parent[edge.target_id] = edge.source_id

    ids, labels, parents, values, colors, hovers = [], [], [], [], [], []

    for node_id, node in nodes.items():
        ids.append(node_id)
        labels.append(node.name)
        parents.append(child_to_parent.get(node_id, ""))
        # Size: use importance, but give a floor so all nodes are visible
        values.append(max(node.importance, 0.05))
        colors.append(_TYPE_COLORS.get(node.node_type.value, "#8C8C8C"))
        hover = (
            f"<b>{node.name}</b><br>"
            f"Type: {node.node_type.value}<br>"
            f"File: {node.file_path}:{node.line_start}<br>"
            f"Importance: {node.importance:.2f}<br>"
            f"{node.summary_short[:120] + '…' if len(node.summary_short) > 120 else node.summary_short}"
        )
        hovers.append(hover)

    return ids, labels, parents, values, colors, hovers


def generate_hierarchy_html(storage: "Storage", output_path: str) -> str:
    """Generate a treemap HTML file from CONTAINS edges.

    Returns the output path on success, or an error message.
    """
    if not _viz_available():
        return "Error: 'plotly' and 'numpy' are required. Install with: pip install neural-memory[viz]"

    import plotly.graph_objects as go

    ids, labels, parents, values, colors, hovers = _build_hierarchy_data(storage)
    if not ids:
        return "Error: No nodes indexed. Run `/neural-index` first."

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        textinfo="label",
        maxdepth=4,
    ))

    # Build legend annotation
    legend_items = "  ".join(
        f'<span style="color:{color}">■</span> {ntype}'
        for ntype, color in _TYPE_COLORS.items()
    )

    fig.update_layout(
        title=dict(
            text="Neural Memory — Codebase Hierarchy",
            font=dict(size=18),
        ),
        margin=dict(t=60, l=10, r=10, b=40),
        annotations=[dict(
            text=legend_items,
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.02,
            xanchor="center",
            font=dict(size=11),
        )],
    )

    out = Path(output_path)
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    return str(out)


# ── Vector Space View ──────────────────────────────────────────────────────────

def _pca_project(matrix, dims: int = 2):
    """Project a float32 matrix [n, d] to [n, dims] via PCA (numpy SVD)."""
    import numpy as np

    centered = matrix - matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:dims].T  # [n, dims]


def generate_vector_space_html(
    storage: "Storage",
    output_path: str,
    dimensions: int = 2,
    color_by: str = "node_type",  # "node_type" or "file"
) -> str:
    """Generate a PCA scatter plot of node embeddings.

    Returns the output path on success, or an error message.
    """
    if not _viz_available():
        return "Error: 'plotly' and 'numpy' are required. Install with: pip install neural-memory[viz]"

    import numpy as np
    import plotly.graph_objects as go
    from .embeddings import _unpack

    dimensions = max(2, min(dimensions, 3))

    all_embeddings = storage.get_all_embeddings()
    if not all_embeddings:
        return "Error: No embeddings found. Run `/neural-index` with the vectors extra installed first."

    nodes = {n.id: n for n in storage.get_all_nodes()}
    # Keep only nodes that have embeddings
    common_ids = [nid for nid in all_embeddings if nid in nodes]
    if len(common_ids) < 3:
        return "Error: Need at least 3 embedded nodes to visualize."

    matrix = np.stack([_unpack(all_embeddings[nid], _TOTAL_DIMS) for nid in common_ids])
    projected = _pca_project(matrix, dims=dimensions)  # [n, 2] or [n, 3]

    # Build per-point display data
    point_nodes = [nodes[nid] for nid in common_ids]
    names = [n.name for n in point_nodes]
    types = [n.node_type.value for n in point_nodes]
    files = [n.file_path for n in point_nodes]
    importances = [max(n.importance, 0.05) for n in point_nodes]
    summaries = [
        n.summary_short[:100] + "…" if len(n.summary_short) > 100 else n.summary_short
        for n in point_nodes
    ]
    hovers = [
        f"<b>{name}</b><br>Type: {t}<br>File: {f}<br>Importance: {imp:.2f}<br>{s}"
        for name, t, f, imp, s in zip(names, types, files, importances, summaries)
    ]

    # Marker sizes scaled by importance
    marker_sizes = [6 + imp * 18 for imp in importances]

    # Group by color dimension
    if color_by == "file":
        groups = sorted(set(files))
        group_key = files
    else:
        groups = sorted(_TYPE_COLORS.keys())
        group_key = types

    color_palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        "#CCB974", "#64B5CD", "#1F77B4", "#FF7F0E",
    ]

    traces = []
    for i, group in enumerate(groups):
        mask = [j for j, g in enumerate(group_key) if g == group]
        if not mask:
            continue

        color = _TYPE_COLORS.get(group, color_palette[i % len(color_palette)])

        if dimensions == 3:
            trace = go.Scatter3d(
                x=projected[mask, 0].tolist(),
                y=projected[mask, 1].tolist(),
                z=projected[mask, 2].tolist(),
                mode="markers",
                name=group,
                marker=dict(
                    size=[marker_sizes[j] / 4 for j in mask],
                    color=color,
                    opacity=0.8,
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[hovers[j] for j in mask],
            )
        else:
            trace = go.Scatter(
                x=projected[mask, 0].tolist(),
                y=projected[mask, 1].tolist(),
                mode="markers+text",
                name=group,
                marker=dict(
                    size=[marker_sizes[j] for j in mask],
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                text=[names[j] if importances[j] > 0.4 else "" for j in mask],
                textposition="top center",
                textfont=dict(size=9),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[hovers[j] for j in mask],
            )
        traces.append(trace)

    fig = go.Figure(data=traces)

    axis_style = dict(showgrid=True, zeroline=False, showticklabels=False)
    if dimensions == 3:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="PC1", **axis_style),
                yaxis=dict(title="PC2", **axis_style),
                zaxis=dict(title="PC3", **axis_style),
            ),
        )
    else:
        fig.update_layout(
            xaxis=dict(title="PC1", **axis_style),
            yaxis=dict(title="PC2", **axis_style),
        )

    n_nodes = len(common_ids)
    fig.update_layout(
        title=dict(
            text=f"Neural Memory — Vector Space ({n_nodes} nodes, PCA {dimensions}D)",
            font=dict(size=18),
        ),
        legend=dict(title=dict(text=color_by.replace("_", " ").title())),
        margin=dict(t=60, l=40, r=20, b=40),
        hovermode="closest",
    )

    out = Path(output_path)
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    return str(out)
