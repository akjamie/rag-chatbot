---
name: asdlc-apply
description: "Execute Stage 5 of Agentic SDLC: guard verification followed by strict task-level TDD loop (Red -> Green -> Refactor -> Evidence -> Converge) from tasks.md. Defaults to local-only implementation; auto-PR is permitted only with explicit delegation via --auto-pr."
---

# /asdlc:apply - Task-Level TDD Implementation Engine

This skill executes **Stage 5 (Apply)** of the Agentic SDLC. It enforces strict task-level Test-Driven Development (TDD) and the Converge protocol.

---

## The TDD Principle

> **No production code is written without a prior failing test.**
> Marking a task complete without captured test evidence is a verification blocker.

---

## Workflow Sequence

### Step 1: Run the Apply Guard (Pre-Implementation Security Check)
Before editing any implementation or test code, run the apply guard:

```powershell
asdlc apply <CHANGE_ID> [--auto-pr] [--target-branch <ref>]
```

`asdlc apply` deterministically verifies Git-anchored preconditions:
1. **Target Branch Approved Commit (`approvedSha`)**: Confirms that a commit touching the locked paths (`proposal.md` and `specs/**`) is physically merged into the target branch (`main` or `origin/main`).
2. **Change State Machine (`ready_for_impl`)**: Confirms `.openspec.yaml` has been transitioned to `status: ready_for_impl` (via `asdlc approve <ID>`).
3. **Spec Fingerprint Integrity (`specSha`)**: Verifies that the declared `specSha` in `.openspec.yaml` exactly matches the `approvedSha` on the protected branch.
4. **Working Tree Tamper Defense**: Ensures that locked spec files (`proposal.md`, `specs/**`) have no uncommitted local modifications.
5. **PR Authority Delegation (`--auto-pr`)**: By default, `apply` operates in **local-only mode** (`autoPr: false`). Supplying `--auto-pr` explicitly delegates authority to automatically create an Implementation PR after verification passes (persisted to `.openspec.yaml`).

- If `asdlc apply` fails: **STOP immediately**. Follow the provided hint (e.g. merge Spec PR or run `asdlc approve`).
- If `asdlc apply` succeeds: Proceed to Step 2.

> [!NOTE]
> **Governance Level & Single PR Flow (L0 / L1)**:
> For changes evaluated as `level: L0` or `level: L1`, prior merging of a Spec PR to the protected branch is waived. `asdlc apply` immediately unblocks local TDD implementation, allowing source code and spec delta to be submitted together in a Single PR.


### Step 2: Read `tasks.md` and Ground Context
1. Open and read `openspec/changes/<CHANGE_ID>/tasks.md`.
2. Inspect the spec requirements in `openspec/changes/<CHANGE_ID>/specs/**/*.md` to understand the acceptance criteria (`AC<n>`) and required test mappings (`(test: <path>)`).
3. Identify the next unchecked task (`- [ ]`).

### Step 3: The Task-Level TDD Loop (Per Task)

For every task in `tasks.md`, execute this exact 5-phase loop:

#### Phase A: RED (Write the Failing Test)
1. Write or expand the unit/integration test in the target test file.
2. Run the repository test command (e.g., `npm test`, `pytest`, `go test`).
3. **Assert Failure:** Verify that the test fails for the *expected* reason (e.g., function not found, assertion mismatched).
4. If the test passes before implementation: the test is invalid or redundant. Fix the test until it accurately tests the unwritten feature.

#### Phase B: GREEN (Implement the Minimum Code)
1. Implement the minimal code required to satisfy the failing test.
2. Do not write anticipatory code or unrequested abstractions.
3. Run the test command again.
4. **Assert Pass:** Verify that the test now passes cleanly (`Exit Code: 0`).

#### Phase C: REFACTOR & CONVERGE
1. Clean up implementation, adhere to repo standards, remove dead code.
2. Re-run all existing tests to ensure no regressions.
3. **Converge Protocol:** If reality disagrees with the spec during implementation:
   - **Type A (Implementation error):** Write reproducing test, append fix task, remain in PR.
   - **Type B (Missing test):** Add missing test, remain in PR.
   - **Type C (Spec gap/conflict):** **STOP**. Never silently edit locked spec files (`proposal.md`, `specs/**`). Report `SPEC_GAP` to the user and approver.

#### Phase D: Record Evidence & Check Box
1. Under the completed task in `tasks.md`, record the test run command and output:
   ```markdown
   - [x] 1.1 [TEST] Add failing test for AC1 (`test/feature.test.js`)
     - verify: `npm test` -> PASS (test/feature.test.js: 3 passed)
   - [x] 1.2 [IMPL] Implement minimal code for AC1
     - evidence: `npm test` -> PASS (all 35 passed)
   ```
2. Mark the checkbox `- [x]`.

---

### Special Guidance: Web & Mobile Applications (UI Design Tasks)

> [!IMPORTANT]
> **Is UI Design required or nice-to-have for Web/Mobile apps?**
> **Verdict: It is ESSENTIAL (Required in `tasks.md`), not a nice-to-have.**

When a change introduces or modifies user-facing interfaces (web, mobile, or desktop):
1. **Design Foundation Must Precede Logic**:
   Never jump directly into business logic or raw HTML/JSX. Skipping UI design creates unstyled, disjointed MVPs that violate usability and require costly refactoring.
2. **Decompose UI Work into Explicit `tasks.md` Tasks**:
   For any change touching frontend views, `tasks.md` should structure tasks into three progressive tiers:
   - **Tier 1 · [UI/DESIGN] Visual & Token Scaffolding**:
     Establish or update design tokens (colors, typography, spacing, CSS variables, dark/light theme), responsive layout containers, and navigation hierarchy.
   - **Tier 2 · [UI/COMPONENTS] Interactive Component States**:
     Build reusable components with all mandatory states: **Empty**, **Loading**, **Error**, and **Success/Data-bound**. Validate via component render tests or visual snapshots.
   - **Tier 3 · [LOGIC/INTEGRATION] State & API Wiring**:
     Wire state stores, API clients, forms, and business logic with unit & E2E integration tests fulfilling the spec's ACs.

---

### Step 4: Run Local Verification Check
Once all tasks in `tasks.md` are checked and evidenced, run `asdlc verify <CHANGE_ID>` locally:

```powershell
asdlc verify <CHANGE_ID>
```

Verify that all checks pass:
- `approved-spec`
- `flow-state`
- `tasks-complete`
- `test-presence` (diff contains tests alongside source code)
- `tdd-evidence` (tasks have valid verification statements/evidence)
- `BDD + acceptance criteria`
- `openspec-validate`

If any check fails, resolve the issue locally within the TDD loop and re-run `asdlc verify <CHANGE_ID>`.

---

### Step 5: Implementation Complete & Submission Protocol (Dual-Mode Authority)

Once all tasks in `tasks.md` are marked complete with captured evidence and local `asdlc verify <CHANGE_ID>` passes (all 12 verification gates exit 0), determine the submission path based on explicit authority:

#### Path A: Default Local Mode (Safe Default, `autoPr: false` or unspecified)
> [!CAUTION]
> **Stage Boundary Guard — No Automatic PR Creation by Default**:
> `asdlc-apply` is strictly the **local implementation engine** (Stage 5).
> **NEVER automatically push commits to origin or open a Pull Request during `apply` by default.**
> Pushing code to a remote repository and opening a PR is a public, collaborative action with team-wide visibility, git ref mutation, and CI compute costs.
>
> In default mode, an agent executing `apply` must NEVER run `asdlc pr` or `git push` on its own.

1. Summarize completed tasks and verification results for the developer.
2. Inform the developer that the implementation is verified locally and ready:
   > "🎉 Implementation complete! All tasks in `tasks.md` are implemented with TDD evidence, and all local verification checks passed.
   > When you are ready to open the Implementation PR, run:
   > `asdlc pr <CHANGE_ID> <targetBranch>` (or trigger `/asdlc:pr`)."

#### Path B: Explicit Delegation Mode (`--auto-pr` or explicit intent)
> [!TIP]
> **Authorized Agent Delegation Channel**:
> When the user has explicitly authorized automatic PR creation (via `asdlc apply <CHANGE_ID> --auto-pr`, `autoPr: true` in `.openspec.yaml`, or explicit user prompt instructions):
>
> 1. Confirm that `asdlc verify <CHANGE_ID>` exited 0 with all verification checks passing cleanly.
> 2. The agent is authorized to automatically open the Implementation PR via:
>    ```powershell
>    asdlc pr <CHANGE_ID> <targetBranch>
>    ```
> 3. Provide the resulting PR link to the user and announce readiness for AI / Human Implementation Review.
