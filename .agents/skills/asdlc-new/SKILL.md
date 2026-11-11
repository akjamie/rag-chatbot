---
name: asdlc-new
description: "Start a new spec-driven change: derives change slug from natural language requirement, grounds in knowledge graph (explore.md), runs batch-grill-me interview, and drafts initial proposal, technical design (design.md), and spec deltas."
---

# /asdlc:new - Start Feature & Batch-Grill Author

This skill executes **Stage 2 (New)** of the Agentic SDLC. It transforms a natural language requirement into an approved intent change package (`proposal.md`, `specs/**`, and `design.md`).

---

## Workflow Sequence

### Step 0: Input Grounding & Ticket Resolution (Anti-Hallucination Guard)
When the user invokes `/asdlc:new` or provides a task, check if it references a ticket ID or link (e.g. Jira key `PROJ-123`, Jira URL, GitHub Issue `#42`, or internal company tracker link):

1. **Attempt Tool/Skill Resolution**:
   - If you have an applicable skill, MCP tool, or CLI available (e.g. Jira skill, GitHub MCP, `gh issue view`), use it to retrieve the ticket title, description, and acceptance criteria.
2. **Handle Inaccessible / Unresolvable Tickets (Mandatory Pause)**:
   - If the ticket link or ID cannot be read or accessed (e.g. no integration tool installed, unauthorized/VPN required, 404 not found, or network error):
     - 🛑 **PAUSE IMMEDIATELY**: Do NOT proceed to scaffold or write specs.
     - ❌ **DO NOT GUESS OR INVENT REQUIREMENTS**: Hallucinating specifications without verified ground truth is strictly prohibited.
     - ❓ **ASK THE USER DIRECTLY**: Halt and prompt the user:
       > *"I see you referenced ticket/link `<ID/URL>`, but I'm unable to access your issue tracker directly. To ensure the specifications are trustworthy and accurate, could you please paste the ticket title, description, and any acceptance criteria here?"*
3. **Proceed with Grounded Context**:
   - Once the user supplies the ticket details or confirms the requirement in plain language, proceed to Step 1 using that grounded requirement text. (If ticket acceptance criteria were retrieved or pasted, carry them forward into `clarifications.md` in Step 3).

### Step 1: Scaffold Change & Refresh Graph
When the grounded requirement is established:

```powershell
asdlc new "<requirement>" [--batch-grill] [--no-grill]
```

The CLI:
1. Derives a clean canonical slug (e.g. `jwt-authentication-payment-api` or content-hashed `change-<id>`).
2. Creates `openspec/changes/<SLUG>/`.
3. Runs Graphify queries using the extracted requirement terms to discover real codebase AST nodes and call chains.
4. Generates `explore.md` and seeds `clarifications.md` configured for the chosen grilling mode:
   - **Default (conversational)**: Runs high-signal, recommendation-led `grill-me` (1 material decision at a time).
   - **`--batch-grill`**: Evaluates the entire design tree and presents the active frontier in structured batches.
   - **`--no-grill`**: Bypasses interactive grilling for self-evident fixes and pre-aligned tasks.
5. **Agent-Native Semantic Grounding**:
   - As an AI agent in an active reasoning session, evaluate the true architectural and security intent of the requirement.
   - If the requirement touches APIs, schemas, or destructive operations, record your agent evaluation into `openspec/changes/<SLUG>/.agent-assess.json` (e.g. `{"classification":"api_contract","rationale":"..."}`). The governance engine reads this alongside Graphify AST hubs to enforce accurate risk tiers without developer gaming.
6. **Technical Design Auto-Detection**: If the requirement touches UI/UX, frontend, APIs, services, databases, or architecture (or if `--design` is passed), seeds `design.md` with enterprise-grade technical design scaffolding.

> [!IMPORTANT]
> **NO PAUSE / CONTINUOUS EXECUTION**: Do NOT stop after `asdlc new` to ask the author "Should I start grilling?".
> You MUST proceed immediately to Step 2 and Step 3 in the same turn!
> Only after `clarifications.md` has all questions settled may you draft `proposal.md`, `design.md`, and `specs/**`.

### Step 2: Ground Yourself in Codebase & External Context (Strict Order)
Follow this strict discovery sequence:
1. **Local Graph & Workspace First (Primary)**:
   - Read `openspec/changes/<SLUG>/explore.md`.
   - Note the **Architectural Hubs (god nodes)** that this change touches.
   - Review existing functions, types, and exported symbols. Never guess: ground internal assumptions in real paths from `explore.md`.
2. **Policy Guard & Targeted External Search (Enabled by Default)**:
   - **Enterprise Egress Policy Guard**: Check `.sdlc/policy.yaml` before executing any external web or API doc search.
     - **Default is ENABLED (`allow`)**: External web search is enabled by default. If `web_search` is omitted, unspecified, or set to `allow` in `.sdlc/policy.yaml`, search is permitted and encouraged for grounding 3rd-party APIs, vendor SDKs, and specs.
     - **Strict Egress Deny**: Search is blocked **ONLY if `.sdlc/policy.yaml` explicitly sets `web_search: deny`** (for air-gapped / offline enterprise environments).
       In that case, emit:
       `[POLICY] External search disabled by .sdlc/policy.yaml (web_search: deny); proceeding with local context only.`
       Tag any external API assumptions in `clarifications.md` and `design.md` with `[UNVERIFIED: policy_denied]`.
   - **Authoritative Fact Lookup** (when search is allowed / default):
     If the requirement references 3rd-party APIs, vendor SDKs, protocols, or external library specs (e.g. Amap, Weather APIs, Maps, Payment Gateways):
     Execute a targeted web/doc search to look up authoritative endpoints, rate limits, request/response formats, error codes, and caching best practices.
   - **Non-blocking failure tolerance**: If external search fails (offline, rate limited, sandbox network restriction, DNS failure, timeout), **DO NOT BLOCK**.
     Output:
     `[WARN] External search failed or offline: <reason>. Continuing with defined steps.`
     Tag any unverified external API assumptions, endpoints, schemas, or rate limits in `clarifications.md` and `design.md` with `[UNVERIFIED: <reason>]`.
     These will be explicitly surfaced on the PR review by CI/D7 for mandatory human confirmation before merge.

### Step 3: Run the Grilling Phase (Mode-Aware)
Before authoring any spec markdown, settle all open architectural questions into `clarifications.md`:

1. **If Mode is `interactive` (Default `grill-me`)**:
   - Run conversational, recommendation-led grilling (skill: `grilling`).
   - **Anti-Fatigue Guardrails**:
     - *Codebase-first*: Never ask questions that can be answered by reading the code or AST.
     - *Material forks only*: Ask only about real architectural trade-offs (e.g. storage options, error policies, backwards compatibility); decide implementation micro-details yourself.
     - *Lead with recommendations*: Frame each question with a concrete recommended default and rationale (e.g. "Recommend Option A because... Agree, or prefer Option B?").
     - *Early Exit*: Immediately halt questioning if the author says "proceed", "use defaults", or when open forks hit zero.

2. **If Mode is `batch` (`--batch-grill` passed)**:
   - Run `batch-grill-me` protocol: evaluate the active decision frontier and present all independent trade-offs in one structured numbered batch with recommended options.
   - Author approves all or adjusts specific choices.

3. **If Mode is `none` (`--no-grill` passed)**:
   - Ground assumptions in AST / codebase facts, record context-inferred decisions in `clarifications.md`, leave open questions as none, and proceed directly to Step 4.

4. Record every agreed answer in `openspec/changes/<SLUG>/clarifications.md` under `## Settled decisions`.

### Step 4: Author Approved Intent Artifacts
Once `## Open questions` in `clarifications.md` is empty, author the change documents:

1. **`proposal.md`**:
   - Sections: `## Why`, `## What Changes`, `## Capabilities`, `## Impact`, `## Out of Scope`, `## Dependencies & Assumptions`.
   - 1-2 pages maximum. Focus on **WHY**, not code implementation details.

2. **`design.md` (Technical Design Document - Mandatory for UI & Complex Systems)**:
   - When a change touches UI/frontend, external APIs, databases, or services, author `design.md` to define **HOW**:
     - **Architecture & Component Interaction**:
       - Embed clean, self-contained Mermaid flowcharts, sequence diagrams, or state transitions directly in `design.md`. Keep diagrams focused on component boundaries, responsibilities, and request/response lifecycles.
     - **Interface Contracts & Schemas (`api-design` recommended)**:
       - Follow the `api-design` skill for RESTful conventions (plural nouns, kebab-case, hierarchical nesting, zero verbs in URLs).
       - Specify exact TypeScript interfaces or JSON schemas for request/response payloads.
       - Standardize on structured error envelopes (Stripe-style `{ error: { code, message, details } }` or RFC 7807 problem details `{ type, title, status, detail, instance }`). Never return `200 OK` with an error body.
       - Define explicit pagination strategy: cursor-based (`cursor`, `limit`, `has_more`) or offset-based.
     - **External API Integration & Resilience**:
       - Endpoints, authentication, rate limits (`X-RateLimit-*`), client-level in-memory/TTL caching (never recreate clients per request).
       - Mutating endpoints must specify `Idempotency-Key` header with deduplication cache.
       - Error taxonomy, fallback degradation, and exponential backoff retry with jitter.
     - **Security & Operational NFRs (Optional / When Applicable)**: Document authentication, boundary validation, secret lifecycles, and performance budgets/quotas when the architecture demands them.
     - **UI/UX Design Specifications (`frontend-design` & Vercel Guidelines)**:
       - **Anti-AI Slop Mandate**: Zero tolerance for cliché glowing cards, arbitrary gradient text highlights, uniform pill buttons, or monotonous card grids.
       - **Design Direction & Subject Grounding**: Real subject matter, vernacular, and domain tone (e.g., high-density analytical dashboard vs consumer app). Link to mockup, brand tokens, or benchmark URL.
       - **Concrete Palettes & Typography**: 4–6 deliberate Hex color tokens (`--color-primary`, `--color-background`, `--color-surface`, etc.) and distinct font pairings (heading + body) with WCAG AA/AAA contrast.
       - **Hero Treatment & Layout**: Opening hero element, responsive breakpoints (Mobile <= 640px, Tablet 641-1024px, Desktop > 1024px).
       - **5 Mandatory States**: Default, Hover/Focus (150-200ms ease, `:focus-visible`), Loading Skeleton (never blank screens), Empty State with prompt, Error Banner with retry.

3. **`specs/<capability>/spec.md`**:
   - `## Purpose`: Clear capability statement (50+ chars).
   - `## ADDED Requirements`:
     - Every requirement: `### Requirement: <name>` using **SHALL** or **MUST**.
     - Every requirement has `#### Scenario: <name>` with **GIVEN / WHEN / THEN**.
     - Every requirement has an `**Acceptance criteria**` block:
       `- [ ] AC1: <concrete testable assertion> (test: <path/to/test>)`
     - **UI Testability**: For any UI capability, include visual Acceptance Criteria for responsive breakpoints and key interaction states (empty, loading, error), mapped to component/E2E test targets (e.g. Playwright / snapshot tests).
     - `## Constraints & NFRs`: Verifiable performance/security statements.
     - `## Out of Scope` and `## Test Strategy`.

4. **Review Provenance (Player $\ne$ Referee)**:
   - The authoring agent MUST NOT approve its own spec.
   - Do NOT stamp `review.md` with self-approval.
   - Independent AI review (`asdlc:spec-review`) is executed by an independent review agent (locally or via PR automation).
   - Formal approval is given by the human lead via PR merge or `asdlc approve <SLUG>`.

### Step 5: Transition to Lint
Once artifacts are written, guide the developer:
> "Artifacts drafted! Now run `/asdlc:lint <SLUG>` to verify BDD scenarios, technical design, and acceptance criteria before opening the spec PR."
