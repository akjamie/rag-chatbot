---
name: asdlc-pr
description: "Open a Spec or Implementation PR: pre-flight checks (lint vs verify), smart staging, auto-extract title/body, and attach labels (asdlc:spec-review vs asdlc:impl-review) for independent review agents."
---

# /asdlc:pr - Automated Pull Request Engine (Spec & Implementation)

This skill opens a **Spec PR** (Stage 3) or an **Implementation PR** (Stage 6.5) with deterministic pre-flight quality gates, smart staging guards, and distinct labels routing to independent review agent sessions.

---

## Command

```powershell
asdlc pr <CHANGE_ID> <targetBranch>
```

- `<targetBranch>`: Target branch to merge into (e.g. `main`, `develop`, `release/*`). Required.
- Zero manual flags: `asdlc pr` automatically detects whether to open a Spec PR or an Implementation PR.

---

## Dual-PR Lifecycle & Pre-flight Gates

| PR Type | Phase & Moment | Pre-Flight Guard | Auto Labels | Review Automation Trigger |
| :--- | :--- | :--- | :--- | :--- |
| **Spec PR** | Stage 3 (After drafting proposal & specs) | **`asdlc lint <ID>`**<br>(0 blocking BDD/AC issues) | `asdlc:spec-review`<br>`spec` | Runs `asdlc lint` + routes to **Independent Spec AI Review** (`asdlc-spec-review`) |
| **Impl PR** | Stage 6.5 (After TDD apply & tasks complete) | **`asdlc verify <ID>`**<br>(12 deterministic gates pass) | `asdlc:impl-review`<br>`impl` | Runs `asdlc verify` + routes to **Independent Impl AI Review** (`asdlc-impl-review`) |

---

## Execution Pipeline

1. **Auto-Detection (Spec PR vs Implementation PR)**:
   `asdlc pr` inspects the state machine deterministically without requiring manual flags:
   - **Detected as Implementation PR (`impl`) if ANY of these conditions match**:
     - `.openspec.yaml` has `status: ready_for_impl` or a locked `specSha`.
     - Spec intent commit is already merged on `<targetBranch>`.
     - Branch name starts with `impl/`.
     - Source code changes exist outside `openspec/changes/<slug>/` and tasks in `tasks.md` are marked complete.
   - **Detected as Spec PR (`spec`) otherwise**:
     - Spec artifacts (`proposal.md`, `specs/**`) are in drafting/review stage before approval.

2. **Pre-flight Quality Gate**:
   - **Spec PR**: runs `asdlc lint <ID>`. Fails fast if any blocking BDD or acceptance criteria findings exist.
   - **Impl PR**: runs `asdlc verify <ID>`. Fails fast if any of the 12 verification checks fail.

3. **Title & Body Extraction**:
   - **Spec PR**: derives clean title (`spec(<slug>): <title>`) and embeds proposal intent.
   - **Impl PR**: derives title (`impl(<slug>): <title>`) and embeds completed tasks, TDD evidence, and spec intent.

4. **Smart Staging & Commit**:
   - Instead of a blind `git add -A`, `asdlc pr` applies a deterministic safety filter:
     - Automatically skips untracked sensitive credentials (`.env*`, `*.pem`, `*.key`, `*.p12`, secrets).
     - Automatically skips crash dumps and logs (`*.hprof`, `*.dmp`, `core.*`, `*.log`).
     - Automatically skips oversized untracked binaries (`> 50MB`).
     - Emits visible `[WARN]` diagnostics without breaking unattended automation.
     - Safely stages legitimate tracked edits and untracked code/spec files, commits with standard prefix, and pushes branch to origin.

5. **Creation Strategy & Self-Healing Label Guarantee**:
   - **JIT Label Provisioning**: Before creating or editing the PR, `asdlc pr` provisions required governance labels on remote (`asdlc:spec-review` + `spec` or `asdlc:impl-review` + `impl`).
   - **Tier 1 (CLI)**: Executes `gh pr create` with both labels (`--label asdlc:spec-review,spec`). If a PR for the branch already exists, detects it and runs `gh pr edit --add-label` to ensure labels are attached.
   - **Tier 2 (Direct REST API)**: Pre-creates labels via GitHub API and attaches them to the PR issue.
   - **Tier 3 (Browser Fallback)**: Generates a 1-click comparison URL pre-filled with branch and label query parameters.
