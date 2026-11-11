---
name: asdlc-init
description: "Initialize an enterprise repository for Agentic SDLC: setup OpenSpec, Graphify code AST graph, governance policies, and bundled skills."
---

# /asdlc:init - Repository Initialization & Tooling Setup

This skill executes **Stage 1 (Init)** of the Agentic SDLC. It transitions an unmanaged or standard codebase into an enterprise-governed agentic repository.

---

## What this skill does

1. **Detects Environment**: Inspects the repository type (Node, Python, Go, Java, Rust) and active agent toolchains (`.agents`, `.claude`, `.opencode`, Cursor, etc.).
2. **Zero-Intrusion Integration**: Bundles the OpenSpec engine directly inside `asdlc`—zero extraneous `package.json` or `node_modules` are introduced into non-Node repositories.
3. **Executes `asdlc init`**: Runs the CLI initialization command to configure:
   - Zero-intrusion bundled OpenSpec runtime with stateless schema validation
   - Graphify code knowledge graph engine
   - Enterprise managed schema (`agentic-repo-change`)
   - Platform governance policy (`.sdlc/policy.yaml`)
   - Bundled IDE skills (`grilling`, `asdlc-apply`, and `/asdlc:*` suite)
   - Context routing layer (`AGENTS.md`)
4. **Validates Health**: Automatically runs `asdlc doctor` to ensure zero configuration drifts.

---

## Workflow Sequence

### Step 1: Execute Initialization
Run in the repository root:

```powershell
asdlc init
```

Parse output and verify:
- `installed openspec` (zero-intrusion bundled runtime)
- `installed managed schema v1`
- `wrote governance config (openspec/config.yaml)`
- `installed bundled skills -> .agents/skills/`
- `built graphify code graph -> graphify-out/`
- `refreshed AGENTS.md`

### Step 2: Run Environment Health Check
Execute:

```powershell
asdlc doctor
```

Verify that all checks are `[PASS]`:
- Node version >= 20.19
- OpenSpec CLI present
- Managed schema installed centrally (or bundled stateless)
- `.sdlc/policy.yaml` present
- `AGENTS.md` routing valid

### Step 3: Next Steps Orientation
Inform the developer:
> "Repository successfully initialized with Agentic SDLC governance! To start your first feature, run:
> `/asdlc:new \"<your feature requirement>\"`"
