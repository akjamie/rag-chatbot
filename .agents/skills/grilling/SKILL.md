---
name: grilling
description: "High-signal design-tree interview for Stage 2 of Agentic SDLC. Supports conversational grill-me (default, 1 decision at a time with anti-fatigue guardrails) and batch-grill-me (invoked via --batch-grill or /batch-grill-me) to settle clarifications.md before spec authoring."
aliases:
  - grill-me
  - batch-grill-me
---

# Design-Tree Grilling (`grilling` / `grill-me` / `batch-grill-me`)

This skill drives the **Grilling phase** (Stage 2) of the Agentic SDLC. It operates between `asdlc new "<requirement>"` and spec authoring (`proposal.md` / `specs/**`).

It supports two distinct execution styles depending on developer preference and CLI invocation:
1. **Conversational `grill-me` (Default)**: Fast, iterative pair-programming dialogue. Presents **1 material decision at a time**, leading with a concrete codebase-grounded recommendation. Halts immediately when open forks hit zero or when the author fast-forwards.
2. **`batch-grill-me` (Batch Mode, invoked via `--batch-grill` or `/batch-grill-me`)**: Presents the active decision frontier all at once in structured, numbered batches with trade-offs and recommendations.

---

## 1. Core Protocol: The Decision Tree & The Frontier

1. **Map the Design Tree**:
   Every requirement branches into architectural decisions:
   - Seams & Scope (data storage, protocols, domain models).
   - Compatibility (breaking change vs deprecation).
   - Failure & Resilience (fallback vs error propagation).

2. **Work the Frontier**:
   The **frontier** is every decision whose prerequisites are *already settled* — the questions you can address *now* without guessing at answers to dependent questions.

3. **No Tree-Flattening (CRITICAL RULE)**:
   > **A question whose answer depends on another open question belongs to a *later* round, not this one.**
   Never ask child questions whose parent decision is still unsettled.

4. **Early Exit / Fast-Forward Protocol**:
   The author is always sovereign. If the author says:
   - *"Proceed with your defaults"*
   - *"LGTM / Looks good"*
   - *"I trust your recommendations"*
   The agent must **immediately halt questioning**, resolve any remaining open decisions using its stated engineering recommendations, write them into `clarifications.md`, and proceed directly to drafting `proposal.md` and `specs/**`.

---

## 2. Fact-Finding vs. Human Decisions

1. **Finding facts is your job, never the author's.**
   When a frontier question needs a fact from the environment (filesystem, codebase AST, existing specs, `explore.md`, tools), look it up yourself — don't ask the author for anything you could look up yourself.

2. **Non-blocking fact exploration (Granular Facts).**
   When finding a fact requires deeper investigation (a sub-agent or targeted research):
   - **Do not block on it.** A running exploration is an unsettled prerequisite, so only the questions downstream of it wait for the result.
   - **Ask the rest of the frontier now.** All independent frontier branches are asked immediately without waiting for the fact exploration to complete.
   - Once the fact is established, unblock the downstream branch and include it in the next round.

3. **External search failure degradation (Offline/Sandbox Safety).**
   If an external doc or web search fails (network down, rate limited, sandbox timeout):
   - Emit an advisory: `[WARN] External search failed (<reason>); proceeding with local codebase context only.`
   - Record unverified assumptions in `clarifications.md` under `## Unverified External Assumptions`.
   - **NEVER block or fail the interview**. Proceed with best-effort engineering judgment.

4. **Decisions belong to the author.**
   Architectural trade-offs, scope boundaries, migration strategies, and error policies belong to the human author. Put each to them and wait.

---

## 3. Question Formulation & Anti-Patterns

> [!CAUTION]
> **Strict Anti-Pattern Enforcement**:

1. **NO Pseudo-Questions / NO "Rec: yes"**:
   - Every question on the frontier MUST offer $\ge 2$ distinct, mutually exclusive choices: **Option A vs Option B [vs Option C]**.
   - **NEVER write a statement ending with `→ Rec: yes` or `→ Rec: agree`**.
   - If you catch yourself writing `Rec: yes`, it is a red flag indicating:
     - *Case 1*: You flattened a child question whose parent is still open. **Fix**: Delete it from this round; ask it in a later round once the parent is settled.
     - *Case 2*: You stated an obvious implementation detail with no architectural trade-off. **Fix**: Do not ask the user; treat it as an engineering fact and implement it in the spec.

2. **No Over-Fragmentation of Implementation Details**:
   - Group closely related technical facets into a single coherent architectural decision rather than five micro-questions.
   - Ask about system seams, contracts, and migration paths — not variable names or code trivia.

3. **Small-Fix / Patch Heuristic**:
   - For a small bugfix, typo, or minor patch, the entire frontier is **0 to 1 question maximum** (confirming root cause and regression test target). Do not subject minor fixes to architectural grilling.

---

## 4. Pre-Flight Self-Check (Run Before Every Batch)

Before emitting a batch round to the author, verify these 3 conditions:
- [ ] **Dependency Check**: Does any question in this round depend on the answer to another question in this same round? (If YES: cut the dependent question; save it for the next round).
- [ ] **Real Options Check**: Does every question have distinct, mutually exclusive options (Option A vs Option B)? Are there any `Rec: yes` or rhetorical statements? (If YES: rewrite as real options or drop).
- [ ] **Fact Check**: Did I ask the author anything that could be looked up in the codebase or `explore.md`? (If YES: look it up; don't ask).

---

## 5. Presentation Formats

### 5.1 Conversational `grill-me` Format (Default)
Present **1 focused decision at a time** with a concrete recommended default:

```markdown
### 🎯 Architectural Decision: <Decision Title>

<1-2 sentences explaining the architectural fork and the trade-offs>

- **Option A**: <Description of Option A>
- **Option B**: <Description of Option B>

➡️ **Recommendation**: **Option A** — <Fact-backed rationale grounded in codebase/explore.md>

*Reply "Agree" to proceed with Option A, specify an alternative (e.g. "Option B"), or type "proceed with all defaults" to fast-forward.*
```

### 5.2 Batch Presentation Format (`--batch-grill` or `/batch-grill-me`)
When `--batch-grill` is passed, present all independent active frontier decisions in a single structured round:

```markdown
### 🎯 Batch Decision Frontier (Round <N>)

Requirement: **"<requirement>"**
Active Frontier: <N> independent decisions ready for resolution.

---

❓ **Q1** — **<Decision Title>**:
<Concise explanation of the architectural fork and trade-offs>
- **Option A**: <Description of Option A>
- **Option B**: <Description of Option B>

➡️ **Recommendation**: **Option A** — <Fact-backed rationale grounded in codebase/explore.md>

---

❓ **Q2** — **<Decision Title>**:
<Concise explanation of the architectural fork and trade-offs>
- **Option A**: <Description of Option A>
- **Option B**: <Description of Option B>

➡️ **Recommendation**: **Option B** — <Fact-backed rationale>

---

💡 *To proceed efficiently, reply **"Approve all recommendations"**, specify adjustments (e.g. `Q1: B, Q2: accept`), or type "proceed with defaults".*
```

---

## 6. Termination Gate & ASDLC Artifact Sync

The session is done when the **frontier is empty**:
1. Every branch of the design tree has been visited.
2. Nothing is left silently assumed.
3. **Do not act until the author confirms you have reached a shared understanding** (e.g. by approving recommendations or answering all questions).
4. Record all settled decisions in `openspec/changes/<SLUG>/clarifications.md`:
   ```markdown
   ## Settled decisions

   - **<Decision 1 Title>**: <Chosen Option>
     - *Rationale*: <Why this was chosen>
   - **<Decision 2 Title>**: <Chosen Option>
     - *Rationale*: <Why this was chosen>
   ```
5. Only after `clarifications.md` is settled may you transition to authoring `proposal.md` and `specs/**`.
