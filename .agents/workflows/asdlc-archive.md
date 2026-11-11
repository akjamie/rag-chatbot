---
name: asdlc-archive
description: "Stage 7: Sync spec deltas to living specs, archive change directory, and refresh codebase graph"
---

# /asdlc-archive - Sync & Archive Verified Change

Archive a verified change after implementation PR merge.

## Usage

```bash
asdlc archive <CHANGE_ID>
```

## Behavior

1. Merges spec deltas into canonical `openspec/specs/` living specifications.
2. Moves change directory to `openspec/changes/archive/<slug>/`.
3. Rebuilds Graphify AST knowledge graph.
4. Refreshes `AGENTS.md` system status.
