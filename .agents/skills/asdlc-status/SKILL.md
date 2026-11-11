---
name: asdlc-status
description: "Show change lifecycle progress, active artifacts, next actionable command, or an active portfolio dashboard of all open changes in the repository."
---

# /asdlc:status - Lifecycle Navigation & Portfolio Dashboard

This skill provides navigation and visibility across the 7 stages of Agentic SDLC. It orientates developers and agents on where a change stands and what exact command to execute next.

---

## Workflow Sequence

### Case A: Inspect a Specific Change (`/asdlc:status <CHANGE_ID>`)
Run:

```powershell
asdlc status <CHANGE_ID>
```

Parse and display:
1. **Current Stage**: (e.g. Stage 2: Spec Authoring, Stage 4: Approved on Main, Stage 5: Ready for Impl).
2. **Artifact Checklist**:
   - `proposal.md`: present / missing
   - `specs/**`: count and files
   - `explore.md`: grounded status
   - `clarifications.md`: settled / open
   - `review.md`: verdict (`APPROVE` / `NEEDS_REVISION`)
   - `tasks.md`: checked / unchecked count
3. **Next Recommended Command**: Exact command to unblock or advance the change (e.g. `asdlc lint <ID>`, `asdlc approve <ID>`, or `asdlc verify <ID>`).

---

### Case B: Portfolio Overview (`/asdlc:status`)
If no `<CHANGE_ID>` is supplied, inspect `openspec/changes/`:

Render a dashboard of all active changes:

```markdown
### 📊 Agentic SDLC Portfolio Dashboard

| Change ID | Stage | Flow State | Next Action |
|---|---|---|---|
| `jwt-authentication` | Stage 5 (Apply) | READY_FOR_IMPL | Run `/asdlc:apply jwt-authentication` |
| `rate-limiting` | Stage 2 (New) | EXPLORING | Complete `clarifications.md` |
| `cache-layer` | Stage 3 (Lint) | LINT_FAIL | Fix BDD scenarios with `/asdlc:lint` |
```

Offer to drill down into any change or start a new one with `/asdlc:new "<requirement>"`.
