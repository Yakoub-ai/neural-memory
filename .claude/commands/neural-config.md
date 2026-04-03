# Neural Memory — Configuration

View and modify neural memory settings.

## Available actions

- **view**: Show current configuration as JSON
- **set_mode**: Change indexing mode
  - `ast_only` — Fast, local-only, no API cost. Uses AST heuristics for summaries.
  - `api_only` — Rich AI summaries via Claude. Sends code to API.
  - `both` — Heuristic base + API enrichment for important nodes (default).
- **add_exclude**: Add a glob pattern to exclude files (e.g., `**/tests/**`)
- **add_redaction_pattern**: Add a custom regex for secret detection
- **set_staleness_threshold**: How many commits behind before the agent warns you (default: 5)

## Data residency
If your code must stay local, use `set_mode` with `ast_only`. No code will leave your machine.

Run the `neural_config` tool with your chosen action and value.
