---
name: asdlc-spec-review
description: "Independent AI Spec Review (Stage 3): evaluate a spec PR or local change for completeness, ambiguity, contradiction, testability, and security before human TL approval. Accepts a GitHub PR URL or local CHANGE_ID."
aliases:
  - spec-review
---

# /asdlc:spec-review — Independent Spec Review Agent

This skill executes **independent AI spec review**, executed by an independent reviewer agent (either triggered on PR or invoked locally on a change ID).
It evaluates the specification and provides clear review findings so the human Tech Lead can make informed approval decisions.

It applies the **Player ≠ Referee** principle: the reviewing agent MUST NOT be the agent that authored the spec.

> [!IMPORTANT]
> **Scope**: Reviews *intent* — BDD quality, logical consistency, architectural soundness, AC testability.
> Does NOT re-run `asdlc lint` (deterministic structural checks are already gated by `asdlc pr`).
> Does NOT review implementation code — that is `asdlc-impl-review`.
> Output can be posted as a PR comment or reported locally.

---

## Invocation & Triggers

This skill can be invoked:
1. **Locally by an independent agent**: `/asdlc:spec-review <CHANGE_ID>`
2. **Automatically on PRs** carrying the `asdlc:spec-review` label via PR automation workflows:

```yaml
# GitHub Actions / PR Automation example
- name: AI Spec Review
  if: contains(github.event.pull_request.labels.*.name, 'asdlc:spec-review')
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    PR_NUMBER: ${{ github.event.pull_request.number }}
    REPO: ${{ github.repository }}
  run: |
    # Independent Review Agent session executes: /asdlc:spec-review $PR_NUMBER $REPO
```

---

## Workflow Sequence

### Step 1: Acquire PR Context

Run the following commands to collect everything needed for review:

```bash
# List changed files
gh pr view $PR_NUMBER --repo $REPO --json title,body,baseRefName,headRefName,files

# Read the full spec diff
gh pr diff $PR_NUMBER --repo $REPO
```

Extract from the PR body:
- `CHANGE_ID` from the `**Change ID**:` field
- The proposal summary from the PR body itself

> [!NOTE]
> Read **all** changed files in the diff before forming any finding. Partial context produces false positives.

---

### Step 2: Apply the Six Review Dimensions

Evaluate every requirement in the diff against all six dimensions. Mark each `PASS`, `CONCERN`, or `FAIL`.

#### D1 — Completeness
- Does the spec cover all user journeys stated in the proposal `## Why`?
- Are nil/empty inputs, upstream failures, and degradation paths handled?
- Is every `## MODIFIED` or `## REMOVED` requirement accompanied by a `**Reason**`?

#### D2 — Ambiguity & Contradiction
- Does each requirement use normative `SHALL` / `MUST`, not passive voice?
- Are there contradictory ACs or scenarios within the spec? (e.g., one AC implies HTTP 200 graceful fallback, another implies HTTP 4xx hard error for the same trigger)
- Are threshold values, enum members, and schema field names precisely specified?

#### D3 — Cross-Repo Consistency
- Does this change alter a shared API contract, event schema, or RPC interface?
- If yes: is a System Design Amendment referenced in `proposal.md ## System Design refs`?

#### D4 — Contract Impact
- Are breaking changes explicitly flagged `**BREAKING**` in `proposal.md ## What Changes`?
- Are backward-compatibility commitments documented?

#### D5 — Testability
- Does every `### Requirement:` have ≥ 1 `- [ ] ACn:` with a concrete `(test: <path>)` mapping?
- Is each AC deterministically observable? ("must feel responsive" is not testable; "must respond in <2s (p99)" is)
- Are integration and contract test targets distinct from unit test targets where the scope demands it?

#### D6 — Security & Privacy
- Is authentication / authorization addressed for any new API or CLI endpoint?
- Are secrets, tokens, and credentials never logged or hard-coded?
- Is user-supplied input validated before use?
- If the change stores user data, is PII lifecycle (retention, deletion) documented?

#### D7 — Technical Design & UI/API Soundness
- **Design Ground Truth**: When UI is involved, does `design.md` cite an authoritative ground truth reference (mockup, Figma URL, design system URL, or existing brand CSS)? Flag as **Major** if tokens appear hallucinated without reference or justification.
- **Visual Testability**: When UI is involved, does the spec contain Visual Acceptance Criteria covering responsive breakpoints (mobile, tablet, desktop) and interaction states (loading, empty, error, active), mapped to test files (e.g. Playwright / visual snapshot tests)? Flag as **Blocker** if UI specs lack visual ACs or test mappings.
- **Unverified Assumptions**: Are there any `[UNVERIFIED: <reason>]` tags present in `clarifications.md`, `design.md`, or the specs? D7 MUST extract every `[UNVERIFIED]` tag. If an AC depends on an unverified endpoint, quota, or rate limit without a verified fallback, flag as **Major** and require explicit human TL sign-off.
- **API & Client Lifecycle**: When 3rd-party/external APIs are integrated, does the technical design enforce client reuse (singleton / connection pooling, never recreate per request), in-memory TTL caching for geocoding/lookup endpoints, rate-limit awareness, and degraded fallbacks? Flag as **Blocker** if clients are instantiated per request or lack caching/retry policies.
- **Interface & DTO Contracts**: Are request/response schemas, DTOs, and domain models rigorously typed (e.g. TypeScript interfaces) with validation boundaries?

---

### Step 3: Build the Requirement → AC → Test Traceability Matrix

For each `### Requirement:` found in the diff, produce one row covering all its ACs and their test file mappings.

---

### Step 4: Formulate Findings

Classify each finding:

- **🔴 Blocker** — Contradictory ACs, untestable assertion, missing normative language, missing visual ACs for UI, missing singleton/caching for external APIs, security risk. Blocks `asdlc approve`.
- **🟡 Major** — Ambiguous scenario, unhandled edge case, unverified assumption on external API (`[UNVERIFIED]`), missing design ground truth reference. Should be resolved or acknowledged before approval.
- **🟢 Advisory** — Clarity or style improvements. Does not block.

---

### Step 5: Post PR Comment

Compose the review as markdown and post it as a PR comment using the GitHub API:

```bash
gh pr comment $PR_NUMBER --repo $REPO --body "$(cat <<'EOF'
<review-markdown>
EOF
)"
```

Format the comment body strictly as:

~~~markdown
## 🤖 AI Spec Review — `<CHANGE_ID>`

> **Reviewer**: AI Independent Spec Reviewer (`asdlc-spec-review`)
> **Verdict**: ✅ APPROVE | 🔴 NEEDS_REVISION

### Dimension Scores

| Dimension | Rating | Key Finding |
| :--- | :---: | :--- |
| D1 Completeness | ✅ PASS / ⚠️ CONCERN / ❌ FAIL | <one-line summary> |
| D2 Ambiguity & Contradiction | ✅ / ⚠️ / ❌ | <one-line summary> |
| D3 Cross-repo consistency | ✅ / ⚠️ / ❌ | <one-line summary> |
| D4 Contract impact | ✅ / ⚠️ / ❌ | <one-line summary> |
| D5 Testability | ✅ / ⚠️ / ❌ | <one-line summary> |
| D6 Security & Privacy | ✅ / ⚠️ / ❌ | <one-line summary> |
| D7 Technical Design & UI/API Soundness | ✅ / ⚠️ / ❌ | <one-line summary> |

### Requirement → AC → Test Traceability

| Requirement | ACs | Target Test | Status |
| :--- | :--- | :--- | :--- |
| <Requirement name> | AC1, AC2 | test/<feature>.test.ts | ✅ VERIFIED |

### ⚠️ Unverified Assumptions Requiring TL Acknowledgment
<!-- List any [UNVERIFIED: ...] tags found across clarifications.md, design.md, and specs. Omit section if none. -->
- `[UNVERIFIED: <reason>]` — <context / affected requirement>

### 🔴 Blocking Findings
<!-- Omit section if none -->
1. **[Blocker]** `<Requirement: Name>` — AC<n>: <description of contradiction or missing assertion>

### 🟡 Major Findings
<!-- Omit section if none -->
1. **[Major]** <description>

### 🟢 Advisory Notes
<!-- Omit section if none -->

---
*Human TL: review the findings above, then run `asdlc approve <CHANGE_ID>` when satisfied.*
~~~

---

### Step 6: Set CI Status

After posting the comment:

- **If APPROVE**: set CI check status to `success` (green). The PR is unblocked for TL approval.
- **If NEEDS_REVISION**: set CI check status to `failure` (red). The PR is blocked until blockers are resolved.

```bash
# Using gh CLI to set a commit status
gh api repos/$REPO/statuses/$SHA \
  -f state="success|failure" \
  -f context="asdlc/spec-ai-review" \
  -f description="AI Spec Review: APPROVE|NEEDS_REVISION" \
  -f target_url="$PR_URL"
```
