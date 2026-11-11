---
name: asdlc-new
description: "Stage 1/2: Start a new spec-driven change: ground in codebase graph, run grill-me, author proposal and specs"
---

# /asdlc-new - Start a New Spec-Driven Change

Start a new spec-driven change in the repository using the Agentic SDLC workflow.

## Usage

```bash
asdlc new "<requirement>" [--batch-grill] [--no-grill]
```

## Workflow Sequence

1. **Scaffold Change & Ground in Graph**:
   - Derives a clean kebab-case change slug from intent or ticket ID.
   - Refreshes the Graphify AST knowledge graph and generates `explore.md`.
   - Seeds `clarifications.md` and `proposal.md`.
2. **High-Signal Grilling Interview (Stage 2)**:
   - Settle architectural decisions and trade-offs in `clarifications.md`.
   - Stop questioning as soon as the author reaches consensus or forks hit zero.
3. **Specification Delta Authoring**:
   - Author capability spec deltas under `specs/<capability>/spec.md` with RFC-2119 BDD requirements.
   - If UI or external APIs are involved, author `design.md`.
4. **Pre-flight Lint**:
   - Run `asdlc lint <CHANGE_ID>` to verify that all BDD rules (R0–R8) pass with 0 blockers.
