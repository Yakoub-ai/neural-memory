# Neural Memory — Deep Inspect

Deep-dive into a specific code element — see its full context in the knowledge graph.

## What you get
- **Full summary**: Detailed explanation of purpose, logic, and interface
- **Parent**: Which module or class contains this
- **Callers**: Who calls this function (upstream)
- **Callees**: What this function calls (downstream)
- **Siblings**: Other functions/methods at the same level
- **Children**: Contained elements (methods in a class, etc.)
- **Call chains**: Trace execution paths up and down the graph
- **Source code**: The actual implementation (optional)

## Usage
Run the `neural_inspect` tool. You can identify the target by:
- **node_id**: Exact ID from query results (most precise)
- **node_name**: Name search (finds best match)

Options:
- `show_code=true` — include raw source code
- `trace_calls=true` — show upstream/downstream call chains

This is where the neural graph shines — understand any function without reading the whole codebase.
