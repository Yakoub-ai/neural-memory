# Neural Memory — Status Check

Check the health and freshness of the neural memory index.

## What this checks
- Whether neural memory has been initialized
- How many commits behind the index is
- Which files have changed since last index
- Total nodes, edges, and files in the graph
- Current index mode and configuration

## Agent behavior
The neural agent runs this check automatically and will suggest:
- `/neural-index` if no index exists
- `/neural-update` if the index is stale (default: 5+ commits behind)

Run the `neural_status` tool to see the full report.
