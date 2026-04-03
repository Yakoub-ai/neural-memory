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

## How to call

**Via MCP tool** (neural-memory configured as MCP server in Claude Code):
```json
Tool: neural_config
{ "action": "view" }
```
```json
{ "action": "set_mode", "value": "ast_only" }
```
```json
{ "action": "add_exclude", "value": "**/tests/**" }
```

**Via Python** (working directly in the project):
```python
import asyncio
from neural_memory.server import neural_config, ConfigInput

# View config
asyncio.run(neural_config(ConfigInput(action="view")))

# Set indexing mode
asyncio.run(neural_config(ConfigInput(action="set_mode", value="ast_only")))

# Add an exclusion pattern
asyncio.run(neural_config(ConfigInput(action="add_exclude", value="**/tests/**")))

# Set staleness threshold
asyncio.run(neural_config(ConfigInput(action="set_staleness_threshold", value="10")))
```

## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_root` | str | `"."` | Project root directory |
| `action` | str | `"view"` | One of: `view`, `set_mode`, `add_exclude`, `add_redaction_pattern`, `set_staleness_threshold` |
| `value` | str | None | Value for the action (required for all actions except `view`) |

## Data residency
If your code must stay local, use `set_mode` with `ast_only`. No code will leave your machine.
