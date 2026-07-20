# Contributing to MemX

Thanks for helping. Intentional changes use an **Issue → PR → CI → merge** loop so work is reviewable and Issues only close when the fix is actually on `main`.

## Delivery loop

```text
Issue open
    → branch from main (codex/<topic>)
    → PR into main with Fixes #N / Closes #N
    → CI green (workflow CI, job gate)
    → merge to main
    → Issue auto-closes
```

1. **Open or reuse a GitHub Issue** describing the problem, scope, and acceptance criteria.  
   https://github.com/shiaho777/memx/issues
2. **Branch from up-to-date `main`.** Prefer `codex/<short-topic>`.
3. **Open a PR into `main` only.** Use [.github/pull_request_template.md](.github/pull_request_template.md).
4. **Link the Issue for auto-close on merge:** put `Fixes #N` or `Closes #N` in the PR body. Do not close the Issue when the PR opens or while CI is red.
5. **CI is the merge gate.** Required check name: **CI / gate** (workflow file [`.github/workflows/ci.yml`](.github/workflows/ci.yml), job id `gate`). Do not merge red checks. CI does not close Issues; merge does (via `Fixes` / `Closes`).
6. **One primary Issue per PR** when possible.

Agents: see the Delivery section in [AGENTS.md](AGENTS.md).

## Local checks

```bash
make all
make test
```

FullHost LLM harness and capsule vessel demos are optional and need local model weights under `MEMX_MODEL_PATH` (not committed).

## What not to commit

- Secrets, keystores, credentials  
- `.local/` models and other large binary caches  
- IDE / OS junk (`.DS_Store`, etc.)

## Exemptions

- Fully automated maintenance bots may omit Issues if a maintainer documents the exemption.  
- Tiny doc-only or emergency hotfixes may go direct to `main` only with an explicit maintainer override recorded in the commit/PR notes.

## License

By contributing, you agree your contributions are under the repository license (MIT).
