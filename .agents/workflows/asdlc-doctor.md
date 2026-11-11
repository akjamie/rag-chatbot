---
name: asdlc-doctor
description: "Diagnose repository health, toolchain integrity, and per-tool skill/command status"
---

# /asdlc-doctor - Diagnostic Health Check

Verify environment, toolchain health, git hooks, managed schema integrity, and IDE agent tool readiness.

## Usage

```bash
asdlc doctor [--fix] [--global]
```

## Checks Performed

1. Node version compatibility (`>= 20.19.0`).
2. Pinned OpenSpec version availability.
3. Graphify AST toolchain availability.
4. Managed schema installation and absence of committed overrides.
5. Repository configuration and `.githooks` integration.
6. `AGENTS.md` routing table and system status freshness.
7. Per-tool skill and slash command parity across detected IDE tools (Claude, OpenCode, Pi, Antigravity, GitHub Copilot, Cursor, etc.).
