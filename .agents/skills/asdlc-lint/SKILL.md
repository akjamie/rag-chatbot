---
name: asdlc-lint
description: "Verify spec quality against deterministic BDD, Acceptance Criteria, and review rules (R0-R8). Iteratively diagnose and fix blocking findings before opening the spec PR."
---

# /asdlc:lint - Deterministic Spec Quality Gate

This skill executes **Stage 3 (Lint)** of the Agentic SDLC. It ensures that the change artifacts (`proposal.md`, `specs/**/*.md`, `review.md`) meet enterprise BDD, traceability, and review standards before creating a Pull Request.

---

## What this skill checks

The linter evaluates spec text with pure deterministic rules (no LLM jitter):
- **`R0-no-requirements`**: Must contain at least one `### Requirement:`.
- **`R1-purpose`**: Capability `## Purpose` must be at least 50 characters.
- **`R2-normative`**: Every requirement must contain a normative statement using `SHALL` or `MUST`.
- **`R3-scenario`**: Every requirement must have at least one `#### Scenario:`.
- **`R4-when / R4-then / R4-given`**: Every scenario must follow BDD structure with **WHEN**, **THEN**, and **GIVEN**.
- **`R5-ac`**: Every requirement must have an Acceptance Criteria block (`- [ ] AC1: ...`).
- **`R5-scenario-coverage`**: Every scenario under a requirement must have corresponding acceptance criteria (`scenarios.length <= acs.length`), preventing happy-path-only AC shortcuts when edge cases / alternative flows are specified in scenarios.
- **`R6-ac-test`**: Every acceptance criterion must map to an explicit test file path (`(test: <path>)`).
- **`R7-test-strategy`**: Must contain a non-empty `## Test Strategy` section.
- **`R8-removed-reason`**: Any requirement under `## REMOVED Requirements` must document a `**Reason**`.
- **`design.md` Quality & Enforcement**: Mandatory whenever proposal or specs touch UI/frontend or external/3rd-party APIs. Fails if missing when UI or external APIs are present, or if it contains empty template placeholders.
- **Visual Acceptance Criteria (UI specs)**: UI requirements must include visual ACs covering responsive breakpoints (mobile, desktop) and interaction states (loading, empty, error), mapped to test files (`(test: <path>)`).
- **Review Status**: If `review.md` is present, it must have `## Verdict: APPROVE` and zero blocking findings. (On pre-merge branches, `review.md` is pending human PR review and is automatically generated upon `asdlc approve`).

---

## Workflow Sequence

### Step 1: Execute `asdlc lint` with JSON Output
Run:

```powershell
asdlc lint <CHANGE_ID> --json
```

Parse the JSON output:
- `pass`: Boolean indicating overall pass or failure.
- `results`: Detailed breakdown per rule and artifact file.

### Step 2: Handle Blocking Findings (If Failed)
If `pass` is false:
1. Group findings by file (`specs/<capability>/spec.md` or `review.md`).
2. For each blocking finding:
   - Identify the exact section and requirement name.
   - Explain the rule violation in plain language.
   - Propose the exact markdown fix (e.g. adding `**WHEN** / **THEN**`, converting prose to normative `SHALL`, or adding `(test: path/to/test)`).
3. Apply the edits directly to the spec files.
4. Re-run `asdlc lint <CHANGE_ID> --json` to verify that blockers reach zero.

### Step 3: Lint Complete (Self-Review Passed)
When `pass` is true (`0 blockers`):
Report to the author that the spec is structurally valid, fully traceable, and passes all BDD/acceptance criteria checks:

> "Spec lint passed with 0 blocking issues. The spec is ready for PR. When you are ready to submit it for review, use `asdlc pr <CHANGE_ID> <targetBranch>`."

