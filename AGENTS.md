# AGENTS.md

Instructions for coding agents working in this repository.

## Code

- Do not add code comments unless the user explicitly asks for them.
- Prefer minimal, focused diffs. Do not drive-by refactor unrelated files.
- Do not commit secrets, machine-local junk, or `.local/` model weights.

## Delivery (Issue + PR + CI)

Canonical loop:

```text
Issue open → branch from main → PR into main (Fixes #N) → CI
  ├─ red  → fix & push (Issue stays open)
  └─ green → merge to main → Issue auto-closes
```

Hard rules:

1. **Base branch is always `main`.** Open feature PRs into `main` only unless a maintainer explicitly names another base.
2. **Issue first** for intentional code/doc/process changes. Reuse an open Issue when one already tracks the work; otherwise create one.
3. **Close Issue on merge only.** PR body includes `Fixes #N` or `Closes #N`. Never close the Issue when the PR is merely opened, while CI is red, or before merge.
4. **CI is the merge gate.** Required check: workflow **CI**, job **`gate`** (`.github/workflows/ci.yml`). Wait for green; fix and push on red. Do not merge red. CI must not auto-close Issues.
5. **One primary Issue per PR** when possible. Extra Issues: link without extra closing keywords unless intentional.
6. **Branch prefix:** `codex/` by default (e.g. `codex/topic-slug`).
7. **Do not commit / push / open PRs / file Issues** unless the user asks to deliver, ship, push, open a PR, bootstrap delivery, or equivalent.
8. **No merge permission:** still open PR + comment on the Issue with links; leave Issue open; hand off to a maintainer.
9. **User overrides** win for that turn only (skip Issue, direct push to main, ignore red CI). State the override in the PR/Issue comment or handoff.

Human process twin: [CONTRIBUTING.md](CONTRIBUTING.md).  
PR shape: [.github/pull_request_template.md](.github/pull_request_template.md).

### Verify (project)

```bash
make all
make test
```

Optional Python gates when touching Python bindings:

```bash
make test-python-bitexact
```

### Gaps

- Branch protection / required checks must be enabled by a repo admin on `main` (require job `gate` from workflow CI). Docs alone cannot enforce GitHub branch protection.
