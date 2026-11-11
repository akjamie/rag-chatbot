---
name: asdlc-verify
description: "Execute Stage 6 verification gate: validate path-limited approved SHA, artifacts, tasks complete, test-presence, tdd-evidence, BDD linting, external tests, and independent AI verification."
---

# /asdlc:verify - Implementation Verification Gate

This skill executes **Stage 6 (Verify)** of the Agentic SDLC. It runs the full suite of deterministic and AI gates against the implementation, providing structured failure diagnostics and clear transition steps.

---

## What this skill verifies

`asdlc verify` executes 12 coordinated checks:
1. **`pr-type`**: Discriminates PR type based strictly on paths.
2. **`artifacts`**: Checks presence of `proposal.md`, `specs/**`, `review.md`, `explore.md`, `tasks.md`.
3. **`approved-spec`**: Verifies that the spec commit is merged into target branch and matches declared `specSha`.
4. **`spec-tamper`**: Enforces that locked spec files (`proposal.md`, `specs/**`) are 100% unmodified in the Impl PR diff.
5. **`flow-state`**: Verifies status is `READY_FOR_IMPL` (stamped by `asdlc approve`).
5. **`tasks-complete`**: Confirms zero unchecked task boxes (`- [ ]`) in `tasks.md`.
6. **`test-presence`**: Enforces that any production code diff is accompanied by test diffs.
7. **`tdd-evidence`**: Confirms that completed tasks record verification commands and test execution outputs.
8. **`BDD + AC`**: Confirms all Acceptance Criteria test targets exist on disk and pass linting.
9. **`openspec-validate`**: Strict schema validation via OpenSpec CLI.
10. **`schema-managed`**: Validates BU-managed schema version pin and checks for forbidden local overrides.
11. **`external`**: Executes repository test suites and security scans in a scrubbed environment.
12. **`AI verification`**: Executes independent referee verifier against the full implementation diff.

---

## Workflow Sequence

### Step 1: Run `asdlc verify`
Execute:

```powershell
asdlc verify <CHANGE_ID>
```

### Step 2: Handle Verification Failures (If Failed)
If any check outputs `[FAIL]`:
- **`approved-spec`**: The spec is not yet merged on the protected branch. Guide the user: merge the spec PR, then run `asdlc approve <CHANGE_ID>`.
- **`test-presence`**: Source code was changed with zero test files. Guide the user: write corresponding unit/integration tests following TDD.
- **`tdd-evidence`**: Tasks were checked without test run output. Guide the user: paste test execution results under tasks in `tasks.md`.
- **`tasks-complete`**: Unchecked tasks remain. Guide the user to finish implementing remaining tasks.
- **`BDD + AC`**: An acceptance criterion points to a test file that does not exist. Create the missing test file.
- **`external` (unit tests)**: Run the failing test command, debug the failure, fix the code, and re-run.

### Step 3: Next Actions (Depending on Lifecycle Stage)
When all checks output `[PASS]` and verify succeeds:

- **Scenario A: Pre-PR Local Implementation Verification (Stage 6)**
  If running on a feature branch before opening the Implementation PR:
  Inform the developer:
  > "🎉 Local verification passed! All 12 gates satisfied.
  > When you are ready to submit this implementation for CI AI review and team approval, run:
  > `asdlc pr <CHANGE_ID> <targetBranch>` (or trigger `/asdlc:pr`).
  > *(The agent will NOT automatically open a PR without your explicit command).* "

- **Scenario B: Post-Merge Trunk Verification (Stage 6 -> Stage 7)**
  If running after the Implementation PR has been approved and merged into the protected target branch (`main`):
  Inform the developer:
  > "🎉 Post-merge verification passed! All 12 gates verified on the target branch.
  > To close out and archive this change into the master specification catalog, run:
  > `asdlc archive <CHANGE_ID>` (or trigger `/asdlc:archive`)."
