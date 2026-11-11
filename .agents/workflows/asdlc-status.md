---
name: asdlc-status
description: "Show lifecycle state, active artifacts, next actionable command, or portfolio overview"
---

# /asdlc-status - Change & Portfolio Status Overview

Inspect the active change state or get an overview of all in-flight changes across the repository.

## Usage

```bash
# Portfolio overview (lists all active changes in flight)
asdlc status [--json]

# Single change status
asdlc status <CHANGE_ID> [--json]
```

## Output

- In portfolio mode: outputs compact single-line table with `CHANGE`, `LEVEL`, `STAGE`, and `NEXT COMMAND` (strictly excludes archived changes).
- In single-change mode: outputs completed artifacts, stage progression, and next exact command.
