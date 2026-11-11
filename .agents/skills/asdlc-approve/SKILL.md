---
name: asdlc-approve
description: "Execute Stage 4 of Agentic SDLC: approve spec intent, lock specSha, and transition change to READY_FOR_IMPL across any enterprise Git workflow."
---

# /asdlc:approve - Spec Approval & Implementation Unlocking

This skill executes **Stage 4 (Approve)** of the Agentic SDLC. It transitions a change from approved intent into the implementation stage (`READY_FOR_IMPL`), locking the spec to a path-limited approved commit SHA (`specSha`).

---

## Core Concept: The Spec SHA Freeze Gate

`asdlc approve` is **branch-agnostic**. It does not enforce a rigid branching dogma, nor does it require selecting artificial "modes". 

In enterprise environments (whether using `feature -> staging -> main`, trunk-based, or multi-level git flow), **Approve** represents a single Git-anchored tamper-detection gate:

```text
[Stage 2: Grilling / Proposal] 
       ↓
[Stage 3: Lint & Spec PR] ── asdlc lint <ID> / asdlc pr <ID> [target]
       ↓
[Stage 4: APPROVE GATE]  ── asdlc approve <ID>
       │                   Locks: specSha = sha(proposal.md + specs/**)
       │                   Sets:  status = ready_for_impl
       │                   Writes: review.md (verdict: APPROVE)
       ↓
[Stage 5: TDD Implementation] ── asdlc apply <ID>
       ↓
[Stage 6: CI Verification] ── asdlc verify <ID>
```

Whenever human stakeholders or Tech Leads agree on the spec:
1. `asdlc approve <CHANGE_ID>` calculates the exact Git commit SHA that touched `proposal.md` and `specs/**`.
2. Stamped into `.openspec.yaml` as `specSha: <sha>` and `status: ready_for_impl`.
3. Stamped into `review.md` with `## Verdict: APPROVE`.
4. Git commit automatically created for the approval stamp.

From this moment on, **Stage 5 (`asdlc apply`) is unlocked**, and **Stage 6 (`asdlc verify`) will fail if anyone or any AI tampers with the approved spec files during implementation**.

---

## Enterprise Workflows Supported

### 1. Unified Feature Branch Flow (Most Popular)
- Developer/AI authors spec on `feat/xxx` (or `spec/xxx`).
- Open Spec PR for early feedback: `asdlc pr <ID> staging` (or `main`, matching your pipeline).
- Tech Lead / Product Owner reviews:
  - Run `asdlc approve <ID>` directly on the branch.
- Developer/AI runs `asdlc apply <ID>` and writes code + tests on the same branch.
- Final PR merges code + verified spec together into the target branch.

### 2. Spec-First Staged PR Flow (Multi-tier Enterprise Pipelines)
- Spec is submitted as an isolated Spec-only PR into `staging` or `main`.
- When the Spec PR merges, the target-branch merge commit SHA is recognized by `asdlc apply` and `asdlc verify`, or stamped via `asdlc approve <ID>`.
- The implementation is then performed on an implementation branch branching from the approved merge commit.

### 3. Local / In-Place Fast Loop (Agile Squads)
- Tech Lead and developers collaborate locally or pair on a branch.
- Tech Lead reviews locally and executes `asdlc approve <ID>`.
- Unblocks `asdlc apply <ID>` immediately.

---

## Command Reference

### Opening the Spec PR
```powershell
asdlc pr <CHANGE_ID> [targetBranch]
```
- `targetBranch`: The base branch this PR will merge into (`staging`, `main`, `develop`, or parent feature branch). If omitted, defaults to the repository's upstream or default branch.
- Automatically lints the spec before pushing (fails fast if blocking findings).
- Automatically extracts the PR Title and Body from `proposal.md`.
- Automatically opens PR via `gh`/`glab` CLI, GitHub REST API, or 1-click web comparison link.

### Approving the Spec
```powershell
asdlc approve <CHANGE_ID>
```
Executes:
1. Identifies the path-limited `approvedSha` of the spec.
2. Updates `.openspec.yaml`: `specSha: <approvedSha>`, `status: ready_for_impl`.
3. Creates or updates `review.md` with `APPROVE` verdict and approval audit record.
4. Creates a Git commit with the approval stamp.

Expected output:
```text
✔ <SLUG>: APPROVED specSha=<shortSha> status=ready_for_impl
✔ <SLUG>: stamped review.md (verdict=APPROVE, specSha=<shortSha>)
✔ committed approval stamp
```

---

## Next Steps After Approval

1. Inspect `openspec/changes/<CHANGE_ID>/tasks.md` and confirm strict TDD task pairing:
   ```markdown
   ## 1. Implementation
   - [ ] 1.1 [TEST] Add failing test for AC1 in `test/auth.test.js`
   - [ ] 1.2 [IMPL] Implement authentication logic
   ```
2. Prompt developer / AI:
   > "✅ Spec successfully approved!
   > - `specSha` locked at `<shortSha>`.
   > - Status stamped: `READY_FOR_IMPL`.
   > Run `/asdlc:apply <CHANGE_ID>` to begin the task-level TDD implementation loop."
