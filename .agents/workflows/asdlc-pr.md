---
name: asdlc-pr
description: "Stage 3.5: Open a Spec or Implementation PR with automated title, body, and governance labels"
---

# /asdlc-pr - Open Spec or Implementation Pull Request

Push your branch and automatically create a Pull Request with governance labels on GitHub or GitLab.

## Usage

```bash
asdlc pr <CHANGE_ID> <target-branch>
```

## Behavior

1. **Pre-flight Gate Audit**:
   - For Spec PRs: runs `asdlc lint` (must pass with 0 blockers).
   - For Implementation PRs: runs `asdlc verify` (must pass all quality gates).
2. **Git Push**: Automatically pushes `spec/<CHANGE_ID>` or implementation branch to origin.
3. **PR Metadata Extraction**: Automatically populates Title and Description from `proposal.md`.
4. **Governance Labels**:
   - Spec PR -> `asdlc:spec-review` (triggers automated CI AI Spec Review).
   - Impl PR -> `asdlc:impl-review` (triggers automated CI AI Impl Review).
