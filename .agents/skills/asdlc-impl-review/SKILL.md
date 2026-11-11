---
name: asdlc-impl-review
description: "Independent AI Implementation Review (Stage 6.5): evaluate an Impl PR for spec conformance, code quality, test fidelity, and production readiness before human lead approval. Incorporates Anthropic code review principles (anti-thrashing convergence, 3-round cap, repo-level REVIEW.md contracts, introduced-vs-preexisting defect attribution, and risk tiering). Accepts a GitHub PR URL or local CHANGE_ID."
aliases:
  - impl-review
  - code-review-asdlc
---

# /asdlc:impl-review — Independent Implementation Review Agent

This skill executes **independent AI implementation review**, executed by an independent reviewer agent (either triggered on PR or invoked locally on a change ID).
It evaluates the code against approved specifications, test evidence, and production safety.

It applies the **Player ≠ Referee** principle: the reviewing agent MUST NOT be the agent that authored the implementation.

### Core Philosophy: Protecting Human Time & Attention

Drawing from Anthropic's engineering practices for scaling AI code reviews, this reviewer enforces:
1. **Convergence Protocol**: Review loops must converge, not thrash. Capped at **3 rounds** max; re-reviews inspect **incremental diffs only**.
2. **Defect Attribution**: Clearly separates defects introduced by this PR (`[Blocker: Introduced]`) from pre-existing legacy issues (`[Tech Debt: Pre-existing]`). Pre-existing flaws **never block** the PR.
3. **Repository Contract (`REVIEW.md`)**: Respects repository-defined domain rules, failure modes, and ignore paths from the trusted base branch over generic AI opinions.
4. **Mandatory Acceptance Test Verification (`asdlc test`)**: Reviewer mandates that `asdlc test <CHANGE_ID>` has executed and all mapped AC assertions pass before issuing approval.
5. **Risk-Based Tiering**: Scrutiny scales with the change's risk profile (Tier 1 Docs/Style, Tier 2 Standard, Tier 3 Critical/Auth/Billing). Highest-risk file sets the PR tier.
6. **Class-of-Bug Grouping**: Defect patterns are grouped into singular findings rather than fragmented into line-by-line spam.
7. **Structural & Actionable Findings**: Every finding provides explicit Location, Defect, Impact, and Action Required for instant clarity to both AI agents and human leads.

> [!IMPORTANT]
> **Scope**: Reviews *implementation conformance and code quality* — spec adherence, test rigor, architecture, and production safety.
> Requires `asdlc test <CHANGE_ID>` and `asdlc verify <CHANGE_ID>` to have passed before this review can issue an APPROVE verdict.
> Does NOT review spec intent — that is `asdlc-spec-review`.

---

## Invocation & Triggers

This skill can be invoked:
1. **Locally by an independent agent**: `/asdlc:impl-review <CHANGE_ID>`
2. **Automatically on PRs** carrying the `asdlc:impl-review` label via PR automation workflows:

```yaml
# GitHub Actions / PR Automation example
- name: AI Implementation Review
  if: contains(github.event.pull_request.labels.*.name, 'asdlc:impl-review')
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    PR_NUMBER: ${{ github.event.pull_request.number }}
    REPO: ${{ github.repository }}
  run: |
    # Independent Review Agent session executes: /asdlc:impl-review $PR_NUMBER $REPO
```

---

## Workflow Sequence

### Step 1: Acquire PR Context & Discover Repo Contract

Run the following commands to collect PR metadata, history, and repo review contracts:

```bash
# 1. Fetch PR metadata and previous review comments
gh pr view $PR_NUMBER --repo $REPO --json title,body,baseRefName,headRefName,headRefOid,comments,files

# 2. Check for repository-level REVIEW.md contract from trusted base branch
# Checked in order: .asdlc/REVIEW.md -> openspec/REVIEW.md -> REVIEW.md
# IMPORTANT: Always read from origin/<baseRefName>, NEVER from PR HEAD,
# to prevent self-review bypasses (e.g., PR injecting path exclusions or rewriting failure definitions).
git show origin/<baseRefName>:.asdlc/REVIEW.md 2>/dev/null || \
git show origin/<baseRefName>:openspec/REVIEW.md 2>/dev/null || \
git show origin/<baseRefName>:REVIEW.md 2>/dev/null || true
```

Extract from the PR body, comments, and repo contract:
- `CHANGE_ID` from the `**Change ID**:` field or branch name (`asdlc/impl/<CHANGE_ID>`)
- `specSha` from `.openspec.yaml` (read from base branch or PR files)
- `headRefOid`: Current commit SHA under review
- `reviewRound`: Count previous `## 🤖 AI Implementation Review` comments on this PR:
  - If 0 previous comments → **Round 1 (Initial Review)**
  - If 1 previous comment → **Round 2 (Re-review)**
  - If 2 previous comments → **Round 3 (Final Automated Round)**
  - If 3+ previous comments → **ESCALATED TO HUMAN** (Anti-thrashing limit reached: do NOT post another review comment; set CI status to failure pointing to previous escalation).
- `lastReviewedSha`: Extracted from the `> **Commit SHA**:` line of the most recent AI review comment.

#### Repository Review Contract (`REVIEW.md`)
If a `REVIEW.md` exists on the base branch, parse and enforce its declared specifications:
- **What failure looks like**: Domain-specific definitions of defect (e.g., "for billing: currency precision loss; for router: unhandled 404 fallback").
- **Path Exclusions / Ignores**: Exclude auto-generated files, lockfiles, or fixture dumps from review scrutiny.
- **Non-Negotiable Invariants**: Explicit architectural rules that supersede generic style preferences.

#### Risk-Based Tiering
Classify the PR into one of three risk tiers based on files changed (**highest-risk file sets the PR tier**):
- **Tier 1 (Low Risk)**: Documentation, comments, copy, purely additive test fixtures, styling/CSS.
  *Protocol*: Single-pass lightweight check for factual accuracy, typos, and broken links. Skip the AC Conformance Matrix unless specific ACs were defined in the spec.
- **Tier 2 (Standard Risk)**: Standard application features, domain services, UI state, API endpoints.
  *Protocol*: Full 5-dimension review, AC Conformance Matrix, test fidelity check.
- **Tier 3 (High Risk)**: Auth/security, payment/billing, cryptography, DB migrations, core CI/governance engine, or **any PR modifying `REVIEW.md` or governance files**.
  *Protocol*: Rigorous scrutiny, failure-mode analysis, strict zero-tolerance for untested edge cases, mandatory human lead sign-off. Flag governance modification explicitly in the review body.

#### Read Approved Spec Ground Truth
```bash
# Read the approved spec from the base branch
git show origin/<baseRefName>:openspec/changes/<CHANGE_ID>/specs/<capability>/spec.md
git show origin/<baseRefName>:openspec/changes/<CHANGE_ID>/tasks.md

# Execute Acceptance Tests in CI (Player != Referee gate)
asdlc test <CHANGE_ID> --json
```

> [!NOTE]
> Read the **spec first**, then the implementation diff. The spec is the ground truth; the code is the candidate.
> Enforce **Player ≠ Referee**: CI executes `asdlc test <CHANGE_ID> --json` independently rather than trusting claims in `tasks.md`.

---

### Step 2: Determine Diff Scope & Anti-Thrashing Protocol

To prevent infinite review-author thrashing and fatigue:

#### A. Diff-Only Re-Review (Rounds 2 & 3)
- **Round 1**: Inspect the full PR diff:
  ```bash
  gh pr diff $PR_NUMBER --repo $REPO
  ```
- **Round 2+**: Inspect **ONLY** the incremental diff since the last reviewed commit:
  ```bash
  # Ensure lastReviewedSha is available in shallow CI clones
  git cat-file -e $lastReviewedSha^{commit} 2>/dev/null || \
    git fetch origin $lastReviewedSha --depth=1 2>/dev/null || \
    git fetch origin $baseRefName --depth=50 2>/dev/null || true

  # Check for force-push / history rewrite:
  # If lastReviewedSha is not an ancestor of headRefOid, reset to fresh Round 1 full diff
  if ! git merge-base --is-ancestor $lastReviewedSha $headRefOid 2>/dev/null; then
    echo "Force-push or rebase detected: commit history was rewritten. Resetting to full-diff review."
    gh pr diff $PR_NUMBER --repo $REPO
  else
    git diff $lastReviewedSha...$headRefOid
  fi
  ```
  *Never re-review unchanged baseline code or lines previously deemed acceptable.*

#### B. Thread Resolution & Accounting
On re-reviews (Rounds 2 & 3), audit each previously reported Blocker and Major finding using stable IDs (`R1-#1`, `R1-#2`, `R2-#1`):
- **✅ Resolved**: Explicitly verified as fixed in the incremental diff.
- **⚠️ Disputed / Justified**: Author provided clear architectural justification or documented trade-off in PR comments.
- **❌ Unresolved**: Finding remains unaddressed without justification.

#### C. Class-of-Bug Grouping
Do not post 10 separate line comments for the same recurring pattern (e.g., missing timeout on HTTP requests or unhandled nulls).
Roll them up into a single finding citing all affected locations:
> *"Class of Bug: Missing timeoutMs on outbound fetch calls in 4 handlers (`api/users.ts:42`, `api/billing.ts:88`, ...)"*

#### D. No-Repeat Suppression
Do not re-surface previously posted findings unless the author explicitly claimed to fix them in the current commit and failed.

#### E. 3-Round Convergence Cap
If Round 3 completes with unresolved blockers:
- Set verdict to `⚠️ ESCALATED_TO_HUMAN`.
- Do not trigger further automated loops.
- Provide a concise summary of the impasse for the human lead.

---

### Step 3: Apply the Five Review Dimensions

Evaluate the diff against all five dimensions, informed by `REVIEW.md` and spec ground truth. Mark each `PASS`, `CONCERN`, or `FAIL`.

#### D1 — Spec Conformance (AC-by-AC)
- **Acceptance Test Gate**: Did `asdlc test <CHANGE_ID>` pass with exit code 0? If any mapped test failed or is missing, mark as **🔴 Blocker: Introduced**.
- For every `ACn:` in `specs/**/*.md`:
  - Does the implementation satisfy the exact acceptance criterion as written?
  - Is the behavior exercised in the mapped `(test: <path>)` test?
  - Are there any requirements in the spec with **no implementation coverage**?
  - Are there any implemented behaviors with **no corresponding AC** (scope creep)?
- Flag scope creep as Major: it bypasses the governance gate and introduces untested surface area.

#### D2 — Code Quality & Architecture
- **REVIEW.md Compliance**: Adheres to all repository-declared architectural invariants.
- **Deep modules**: Does each new module expose a simple interface hiding a complex interior? Flag shallow wrappers that leak implementation detail.
- **Single Responsibility**: Does each function / class do one thing? Flag functions mixing IO and business logic in the same body.
- **Error propagation**: Are errors surfaced explicitly (typed error objects or Result types), not swallowed silently?
- **Dead code**: Flag any commented-out or unreachable production code.
- **Naming**: Are names precise and consistent with the spec's domain language (enum values, field names, error codes defined in the spec)?

#### D3 — Test Fidelity & TDD Evidence
- Are tests **behavior-driven**, not implementation-driven? (Assert observable outputs for given inputs, not call order or internal state)
- Are mocks **minimal and scoped**? A mock replacing an entire subsystem and always returning a happy-path fixture is a vacuous test.
- Does each test exercise the **failure path** as well as the happy path?
- Does `tasks.md` contain real test runner output for every `[x]` task (not just "✅ done")?
- Are edge cases from the spec each exercised by a distinct test case?

#### D4 — Production Readiness & Performance
- **NFRs met**: If the spec declares a latency SLA, is there a test or benchmark that fails when the SLA is violated?
- **Resource leaks**: Are HTTP connections, file handles, and streams properly closed?
- **Retry / timeout**: Are upstream HTTP calls wrapped with timeouts to prevent hanging?
- **Observability**: Are structured log statements present at entry and exit of key operations? Are PII fields excluded from logs?

#### D5 — Security & Secret Hygiene
- Are there any hard-coded credentials, API keys, or tokens in the diff?
- Is user-supplied input sanitized before being passed to external API query parameters or shell commands?
- Are error messages free of internal stack traces or system paths that would leak to end users?
- Is any new HTTP endpoint protected from excessively large or malformed payloads?

---

### Step 4: Formulate Structural Findings with Defect Attribution

Every finding must follow a structured schema with explicit defect attribution so that findings are unambiguously actionable for both AI agents and human engineers:

- **🔴 [Blocker: Introduced]** — Broken AC, test cheating/vacuous mock, regression, hardcoded secret, or resource leak **introduced by this PR**. **Blocks merge.**
- **🟡 [Major: Introduced]** — Material defect introduced by this PR (scope creep, missing error handling, unhandled edge case). Requires resolution or author justification.
- **⚪ [Tech Debt: Pre-existing]** — Existing legacy codebase flaw or code smell observed in touched files but **not introduced by this PR**. **DOES NOT block merge.** Documented separately for backlog tracking.
- **🟢 [Advisory]** — Non-blocking suggestions for readability, naming clarity, or optional polish.

---

### Step 5: Post PR Comment

Compose the review markdown and post it as a PR comment using the GitHub API:

```bash
gh pr comment $PR_NUMBER --repo $REPO --body "$(cat <<'EOF'
<review-markdown>
EOF
)"
```

Format the comment body strictly as:

~~~markdown
## 🤖 AI Implementation Review — `<CHANGE_ID>`

> **Reviewer**: AI Independent Implementation Reviewer (`asdlc-impl-review`)
> **Commit SHA**: `<headRefOid>`
> **Base spec SHA**: `<specSha>`
> **Risk Tier**: Tier 1 (Low) | Tier 2 (Standard) | Tier 3 (High)
> **Review Round**: Round 1 of 3 | Round 2 of 3 | Round 3 of 3 (Final)
> **Verdict**: ✅ APPROVE | 🔴 REQUEST_CHANGES | ⚠️ ESCALATED_TO_HUMAN

### Dimension Scores

| Dimension | Rating | Key Finding |
| :--- | :---: | :--- |
| D1 Spec Conformance | ✅ PASS / ⚠️ CONCERN / ❌ FAIL | <one-line summary> |
| D2 Code Quality & Architecture | ✅ / ⚠️ / ❌ | <one-line summary> |
| D3 Test Fidelity & TDD Evidence | ✅ / ⚠️ / ❌ | <one-line summary> |
| D4 Production Readiness | ✅ / ⚠️ / ❌ | <one-line summary> |
| D5 Security & Secret Hygiene | ✅ / ⚠️ / ❌ | <one-line summary> |

### AC Conformance Matrix
<!-- For Tier 1 changes with no spec ACs, replace table with: "*Tier 1 change — AC Conformance Matrix omitted.*" -->

| AC | Spec Assertion | Implemented In | Test File | Status |
| :--- | :--- | :--- | :--- | :--- |
| AC1 | <assertion text> | src/<module>.ts | test/<feature>.test.ts | ✅ PASS |

<!-- On Rounds 2 and 3, include Thread Resolution Accounting -->
### 🔄 Re-Review Accounting (Round <round>)
| Finding ID | Previous Finding Summary | Status | Resolution / Author Rationale |
| :---: | :--- | :---: | :--- |
| `R1-#1` | Missing 5000ms timeout in fetch | ✅ Resolved | Added `AbortSignal.timeout(5000)` in `src/http.ts:24` |
| `R1-#2` | Scope creep helper `parseAmount` | ⚠️ Justified | Author rationale: Shared utility required for currency normalization |

### 🔴 Blocking Findings (Introduced)
<!-- Omit section if none. Use structural format for AI/Human clarity -->
- **ID**: `R<round>-#<num>`
  - **Category**: `Spec Conformance` | `Code Quality` | `Test Fidelity` | `Production Readiness` | `Security`
  - **Rule / AC**: `<ACn or REVIEW.md invariant or Dimension D1-D5>`
  - **Location**: `<file>:<line-range>`
  - **Defect**: `<concise statement of what is broken>`
  - **Impact**: `<why this breaks correctness, security, or production>`
  - **Action Required**: `<exact condition or code fix required to resolve>`

### 🟡 Major Findings (Introduced)
<!-- Omit section if none -->
- **ID**: `R<round>-#<num>`
  - **Category**: `<category>`
  - **Location**: `<file>:<line-range>`
  - **Defect**: `<description>`
  - **Impact**: `<impact>`
  - **Action Required**: `<action required>`

### ⚪ Tech Debt: Pre-existing Codebase Observations
<!-- Non-blocking; omit section if none -->
- **ID**: `DEBT-#<num>`
  - **Location**: `<file>:<line-range>`
  - **Defect**: `<description of pre-existing legacy issue, untouched by PR>`
  - **Attribution**: Pre-existing code untouched by this PR diff.
  - **Status**: Non-blocking. Logged for tech debt backlog.

### 🟢 Advisory Notes
<!-- Omit section if none -->
1. **[Advisory]** <description>

---
*Human lead: review the findings above, then merge when satisfied. After merge, run `asdlc archive <CHANGE_ID>`.*
~~~

---

### Step 6: Set CI Status

After posting the comment, set the commit status check:

- **If APPROVE**: set CI check status to `success` (green). The PR is unblocked for lead approval and merge.
- **If REQUEST_CHANGES**: set CI check status to `failure` (red). The PR is blocked until blockers are resolved.
- **If ESCALATED_TO_HUMAN**: set CI check status to `failure` (red) with description noting human arbitration is required.

```bash
gh api repos/$REPO/statuses/$SHA \
  -f state="success|failure" \
  -f context="asdlc/impl-ai-review" \
  -f description="AI Impl Review: APPROVE | REQUEST_CHANGES | ESCALATED_TO_HUMAN" \
  -f target_url="$PR_URL"
```
