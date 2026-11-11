---
name: asdlc-upgrade
description: "Upgrade the Agentic SDLC toolchain: asdlc CLI, OpenSpec engine, Graphify knowledge graph, managed schemas, and post-upgrade health check."
---

# /asdlc:upgrade - Self-Maintenance & Toolchain Upgrader

This skill upgrades all components of the Agentic SDLC toolchain.

---

## Command

```powershell
asdlc upgrade
```

### Options

- `--no-asdlc`: Skip upgrading the `asdlc` CLI.
- `--no-openspec`: Skip upgrading OpenSpec.
- `--no-graphify`: Skip upgrading Graphify.
- `--no-schema`: Skip refreshing the BU-managed schema.
- `--no-doctor`: Skip the post-upgrade health check.

---

## What It Does

1. **CLI Upgrade**: Updates `@akjamie/agentic-sdlc` to the latest release via `npm install -g @akjamie/agentic-sdlc@latest`.
2. **OpenSpec Upgrade**: Keeps OpenSpec CLI up-to-date (`npm install -g openspec@latest`).
3. **Graphify Upgrade**: Updates the AST knowledge graph tool (`uv tool install --upgrade graphifyy` or `pip install --upgrade graphifyy`).
4. **Schema Refresh**: Refreshes centrally managed schemas and templates.
5. **Doctor Check**: Runs an automatic post-upgrade diagnostic report.
