---
name: asdlc-verify
description: "Stage 6: Verify implementation against all 12 enterprise quality gates"
---

# /asdlc-verify - Enterprise Quality Gates Audit

Audit the implementation diff against all 12 deterministic quality gates before merging to main.

## Usage

```bash
asdlc verify <CHANGE_ID> [--repo <name>] [--json]
```

## The 12 Quality Gates

1. `level-integrity`: Validates declared risk level matches change topology.
2. `pr-type`: Classifies PR type (normal vs. spec vs. impl).
3. `artifacts`: Ensures proposal, tasks, and spec deltas are present.
4. `approved-spec`: Verifies spec PR was approved on protected branch.
5. `spec-tamper`: Enforces that locked spec files were not modified during implementation.
6. `flow-state`: Validates `READY_FOR_IMPL` transition.
7. `risk-escape`: Detects unannounced API/DDL escalations.
8. `human-gate`: Validates mandatory Tech Lead approval for L2/L3 changes.
9. `tasks-complete`: Verifies all tasks in `tasks.md` are checked off.
10. `test-presence`: Enforces that source code changes are paired with test changes.
11. `tdd-evidence`: Verifies red-green-refactor execution log.
12. `BDD + acceptance criteria`: Deterministic R0–R8 quality lint clean.
