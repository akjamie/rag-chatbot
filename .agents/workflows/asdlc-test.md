---
name: asdlc-test
description: "Stage 5.5: Execute real test runners mapped to Acceptance Criteria annotations"
---

# /asdlc-test - Run Mapped Acceptance Criteria Tests

Execute project test suites referenced by spec acceptance criteria lines `(test: <path>)`.

## Usage

```bash
asdlc test <CHANGE_ID> [--json]
```

## Behavior

1. Parses all `(test: <path>)` annotations across the change's capability specs.
2. Resolves test runners (e.g. Node `--test`, Jest, Pytest, Go test, Cargo test).
3. Executes target test files and reports individual test execution status.
