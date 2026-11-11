---
name: asdlc-apply
description: "Stage 5: Guarded task-level TDD implementation loop"
---

# /asdlc-apply - Guarded Implementation Launcher

Launch task-level Test-Driven Development (TDD) loop from `tasks.md`.

## Usage

```bash
asdlc apply <CHANGE_ID> [--auto-pr] [--target-branch <branch>]
```

## Behavior

1. **Approval Guard**: Enforces that the spec has been approved and `approvedSha` is locked (or L0/L1 single-PR mode).
2. **Task-Level TDD Loop**:
   - RED: Author failing unit/integration tests mapped to requirement ACs.
   - GREEN: Implement minimal production code to pass the tests.
   - REFACTOR: Clean code structure without altering behavior.
   - EVIDENCE: Record passing test evidence.
3. **Anti-Hallucination & Anti-Drift Guard**: Prevents modifying locked specification paths during code implementation.
