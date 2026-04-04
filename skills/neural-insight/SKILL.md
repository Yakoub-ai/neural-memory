---
name: neural-insight
description: Generate comprehensive technical documentation by synthesizing all accumulated insights from the neural memory knowledge graph. Also use to save a new insight with neural_add_insight.
---

# Neural Memory — Insight Bank

The insight bank accumulates technical knowledge about the project — implementation choices, architecture decisions, and patterns discovered during development. Use `/neural-insight` to synthesize everything into structured documentation.

## Generate full technical documentation

Use the `neural-doc-writer` agent (bundled with this plugin) to synthesize all accumulated insights:

```
Use the neural-doc-writer agent to generate documentation
```

Or call the MCP tool directly:

```json
Tool: neural_generate_docs
{
  "project_root": "."
}
```

Output is returned as markdown **and** written to `.neural-memory/technical-docs.md`.

## Bundled agents

This plugin includes three agents installed alongside the MCP server and skills:

| Agent | Purpose |
|-------|---------|
| `neural-memory:neural-explorer` | Codebase exploration via semantic graph search |
| `neural-memory:neural-insight-collector` | Captures insights from conversations into the bank |
| `neural-memory:neural-doc-writer` | Synthesizes all insights into technical documentation |

These agents have the neural-memory MCP tools pre-configured — no extra setup needed.

## Save an insight

```json
Tool: neural_add_insight
{
  "content": "The bump script atomically updates 4 files to keep versions in sync...",
  "topic": "versioning",
  "related_files": ["scripts/bump_version.py"]
}
```

Insights are deduplicated — re-saving the same insight updates it rather than creating a duplicate.

## Browse insights

```json
Tool: neural_list_insights
{
  "topic": "storage"
}
```

Omit `topic` to list all insights grouped by topic.

## Parameters — neural_add_insight

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `content` | Yes | str | The insight text (min 10 chars) |
| `topic` | Yes | str | Topic area, e.g. `storage`, `hooks`, `embeddings`, `cli` |
| `related_files` | No | list[str] | File paths to link via RELATES_TO edges |
| `project_root` | No | str | Default: `"."` |

## Parameters — neural_generate_docs / neural_list_insights

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `project_root` | No | str | Default: `"."` |
| `topic` | No | str | (`list_insights` only) Filter by topic |

## Notes

- Insights accumulate without a lifecycle — no status or archiving
- The more insights saved during development sessions, the richer the generated documentation
- `session_context.md` includes an insight bank summary (topic counts) automatically
- Use `neural_query` to search insights alongside code nodes
