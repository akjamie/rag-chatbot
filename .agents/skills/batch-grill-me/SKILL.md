---
name: batch-grill-me
description: "A relentless design-tree interview that evaluates the active frontier and asks questions in structured batches with recommended defaults and non-blocking fact discovery. Invoked via asdlc new --batch-grill or /batch-grill-me."
aliases:
  - grilling
  - grill-me
---

# Batch Grilling — Relentless Design-Tree Interview (`batch-grill-me`)

This skill drives the **Grilling phase** (Stage 2) of the Agentic SDLC when invoked in batch mode (via `asdlc new "<requirement>" --batch-grill` or `/batch-grill-me`). For interactive 1-by-1 dialogue, use `/grill-me`.

Interview the author relentlessly until you reach a shared understanding. Map the requirement as a **design tree**: every decision branches into the decisions that hang off it.

---

## 1. Core Protocol: The Decision Tree & The Frontier

1. **Work the tree in rounds.**
   The **frontier** is every decision whose prerequisites are *already settled* — the questions you can ask *now* without guessing at answers you haven't heard yet. Ask the whole frontier in one round: number each question and give your recommended answer. Then wait for the author's answers before the next round.

2. **No Tree-Flattening (CRITICAL RULE).**
   > **A question whose answer depends on another question still open in this round belongs to a *later* round, not this one.**

   - Never ask child or grandchild questions whose parent decision is still open in the current round.
   - If Question C1 asks *"Should we support internationalization (en + zh)?"*, you **CANNOT** ask *"Which fields should be translated?"* or *"What format should the message catalog use?"* in the same round! Those depend on C1 being answered.
   - Asking child questions prematurely forces you to guess at parent answers, which leads directly to pseudo-questions and fake choices. Hold them for later rounds.

3. **Dynamic Tree Reshaping.**
   Each round the author answers reshapes the tree — settled decisions push the frontier outward and unblock questions that depended on them. Recompute the frontier and ask the next round.

4. **Early Exit / Fast-Forward Protocol**:
   If the author says:
   - *"Approve all recommendations"* / *"Proceed with defaults"* / *"LGTM"*
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

## 5. Batch Presentation Format

Format each round of the interview strictly as follows:

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
3. Record all settled decisions in `openspec/changes/<SLUG>/clarifications.md`:
   ```markdown
   ## Settled decisions

   - **<Decision 1 Title>**: <Chosen Option>
     - *Rationale*: <Why this was chosen>
   - **<Decision 2 Title>**: <Chosen Option>
     - *Rationale*: <Why this was chosen>
   ```
4. Only after `clarifications.md` is settled may you transition to authoring `proposal.md` and `specs/**`.
