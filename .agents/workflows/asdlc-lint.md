---
name: asdlc-lint
description: "Stage 3: Deterministic spec lint gate (R0-R8 BDD rules, scenario coverage, and test mappings)"
---

# /asdlc-lint - Deterministic Spec Quality Gate

Verify that the active specification meets enterprise BDD, traceability, and review standards before creating a Pull Request.

## Usage

```bash
asdlc lint <CHANGE_ID> [--json]
```

## Checks Performed

1. **Artifact Completeness**: `proposal.md`, `specs/**/*.md`, and `design.md` (for UI/external API changes).
2. **Deterministic Rules (R0–R8)**:
   - `R0-no-requirements`: Explicit requirement headers.
   - `R1-purpose`: Capability purpose length >= 50 characters.
   - `R2-normative`: RFC-2119 `SHALL` / `MUST` invariant statements.
   - `R3-scenario`: Concrete observable BDD scenarios.
   - `R4-when / R4-then / R4-given`: Bold Gherkin clauses.
   - `R5-ac & R5-scenario-coverage`: Traceable checkboxes covering all scenarios (`count(ACs) >= count(Scenarios)`).
   - `R6-ac-test`: Every AC maps to a test file (`(test: <path>)`).
   - `R7-test-strategy`: Multi-tiered test pyramid section.
   - `R8-removed-reason`: Explicit rationale for deprecations / removals.

See `docs/spec-format.md` for worked pass/fail examples and 1-minute remedies.
