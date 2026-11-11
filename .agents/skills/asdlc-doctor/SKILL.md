---
name: asdlc-doctor
description: "Diagnose repository health, toolchain integrity, and governance drift across Node, OpenSpec, Graphify AST, schemas, and policy. Provide automated remediation."
---

# /asdlc:doctor - Repository Health & Governance Diagnostics

This skill executes health, environment, and configuration integrity checks for the Agentic SDLC toolchain. It provides structured diagnostics and automated remediation recipes when platform drift occurs.

---

## What this skill checks

`asdlc doctor` verifies environmental and governance invariants:

| Check | Requirement | Failure Impact | Remediation |
|---|---|---|---|
| **`node >= 20.19`** | Node.js version >= 20.19.0 | CLI sub-processes fail | Upgrade Node via `nvm` or system installer |
| **`OpenSpec CLI`** | Pinned or bundled `openspec` runtime available | Schema validation fails | Run `asdlc init` |
| **`Graphify`** | Graphify CLI installed | Cannot build code AST graph | Run `uv tool install graphifyy` or `pip install graphifyy` |
| **`OpenSpec project`** | `openspec/` directory exists | Cannot store changes | Run `asdlc init` |
| **`managed schema`** | `agentic-repo-change` v1 installed centrally or bundled | Cannot validate changes | Run `asdlc init` |
| **`schema override`** | No committed overrides in `openspec/schemas/` | Forbidden local fork | Delete tracked local schema file |
| **`repo config schema`**| `openspec/config.yaml` declares `schema: agentic-repo-change` | Schema mismatch | Run `asdlc init --force` |
| **`.sdlc/policy.yaml`** | Committed platform policy exists | Approval & gate policy unset | Run `asdlc init` |
| **`AGENTS.md routing`** | `AGENTS.md` contains routing block | Agent lacks navigation rules | Run `asdlc init` |
| **`knowledge graph`** | `graphify-out/` is fresh vs `HEAD` | Graph queries stale | Run `asdlc init` or `graphify .` |
| **`graphify-out ignored`**| `graphify-out` present in `.gitignore` | Cache committed to repo | Add `graphify-out` to `.gitignore` |

---

## Workflow Sequence

### Step 1: Run Diagnostics
Run in the repository root:

```powershell
asdlc doctor
```

Inspect output summary:
- If all checks show `[PASS]`: Report that the repository is completely healthy and aligned with platform standards.
- If any check shows `[FAIL]` or `[WARN]`: Proceed to Step 2.

### Step 2: Automated Remediation Playbooks

#### Scenario A: Missing Governance or OpenSpec Project
If `OpenSpec project`, `managed schema`, or `.sdlc/policy.yaml` failed:
```powershell
asdlc init --force
```
This restores all core templates, policy files, and central schema pins.

#### Scenario B: Forbidden Schema Override
If `schema override` failed (e.g. `openspec/schemas/agentic-repo-change/schema.yaml COMMITTED (forbidden)`):
Platform governance forbids individual repository overrides of the enterprise schema.
Run:
```bash
git rm -f openspec/schemas/agentic-repo-change/schema.yaml
git commit -m "chore(governance): remove local schema override"
```

#### Scenario C: Stale Git Hooks Configuration
If legacy hooks path remains configured:
```bash
asdlc doctor --fix
```
This automatically clears legacy `core.hooksPath` settings.

#### Scenario D: Stale Knowledge Graph
If `knowledge graph` is stale or missing:
Run:
```powershell
asdlc init
```
Or directly refresh the AST graph:
```bash
graphify .
```

### Step 3: Re-verify Health
Re-run `asdlc doctor` to confirm `doctor: ok (0 failed, 0 warnings)` before continuing work.
