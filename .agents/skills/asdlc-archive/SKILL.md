---
name: asdlc-archive
description: "Explicitly close and archive a verified change (Stage 7): confirm post-merge review, move change to archive, refresh code knowledge graph, and update AGENTS.md system status."
---

# /asdlc:archive - Deliberate Human Archive Gate

This skill executes **Stage 7 (Archive)** of the Agentic SDLC. Archiving is an explicit post-merge milestone — never automatically triggered during active implementation or unmerged PRs — marking that post-merge testing and trunk integration are complete.

---

## Why Archive is an Explicit Post-Merge Gate

1. `asdlc verify` proves that code matches the approved spec before merging.
2. An open implementation PR may receive comments, review rounds, or rework. Running archive inside an unmerged PR would corrupt state.
3. Therefore, **archiving happens strictly post-merge**. Root `README.md` is never mutated during archiving (eliminating merge conflict hotspots), and `AGENTS.md` is only updated for constitutional governance changes or when explicitly requested via `--agents`.

---

## Workflow Sequence

### Step 1: Human Confirmation Check
Confirm with the user:
> "Have the PR merge, post-merge verification, and staging testing completed for change `<CHANGE_ID>`?"

Wait for human confirmation before proceeding.

### Step 2: Capture Durable Lessons Learned
Before archiving, check if this change introduced durable governance knowledge or constitutional contracts (`.sdlc/policy.yaml`, `openspec/config.yaml`, `governance/**`). If so, `AGENTS.md` will be refreshed automatically, or run `asdlc archive <CHANGE_ID> --agents`.

### Step 3: Execute `asdlc archive`
For a single merged change:
```powershell
asdlc archive <CHANGE_ID>
```

Or for post-merge batch compaction across all merged changes:
```powershell
asdlc archive --merged
```

The CLI:
1. Moves `openspec/changes/<SLUG>/` into `openspec/changes/archive/<SLUG>/`.
2. Syncs living specs into `openspec/specs/` without mutating root `README.md`.
3. Rebuilds the AST knowledge graph (`graphify-out/`) against the new HEAD commit.
4. Updates `AGENTS.md` only if the change modified constitutional governance paths or if `--agents` was passed.

### Step 4: Confirm Archive Completion
Summarize the result to the user:
> "✅ Change `<CHANGE_ID>` successfully archived!
> - The change artifacts are archived under `openspec/changes/archive/<SLUG>/`.
> - Living specs have synced into `openspec/specs/`.
> - The code knowledge graph has been refreshed to reflect the updated codebase."
