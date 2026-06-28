# RootSolvers.jl Agent Guide

## Ecosystem Guidelines

Please refer to the shared CliMA agent index for ecosystem-wide rules regarding architecture, performance, code quality, infrastructure, and workflows:

- [docs/dev-guides/AGENTS.md](docs/dev-guides/AGENTS.md) — Shared CliMA agent guidelines.

> Shared guides live at `docs/dev-guides/` and are vendored from the canonical source:
> <https://github.com/CliMA/DeveloperGuides>. Edit shared guides there, not here. They are
> synced automatically each month by `.github/workflows/update_dev_guides.yml`.

## Before You Act: Agent Autonomy

Before making changes that are externally visible or consequential (`git push`, version bumps, CI config changes, public API renames), check [docs/dev-guides/workflow/agent_autonomy.md](docs/dev-guides/workflow/agent_autonomy.md). The boundaries listed there require explicit user approval.

## Repo-Specific Guidelines

RootSolvers.jl is a small, focused package providing numerical methods for finding roots of scalar nonlinear equations. It is GPU-compatible and supports broadcasting over arrays and custom field types.

### Architecture

- **Single-module library**: all code lives in `src/RootSolvers.jl` — the method, solution, and tolerance types plus the public `find_zero` interface and the internal solvers.
- **Pure, stateless functions**: solvers take a method (or a method *type*), a solution type, and a tolerance as arguments and return a results struct; they do not mutate global state.
- **GPU compatibility**: hot loops use `ifelse` rather than branches to avoid divergence; [`CompactSolution`](docs/src/API.md) is the GPU-friendly output, while `VerboseSolution` (which stores iteration history) is CPU-only. Broadcasting is supported by passing the method *type* (e.g. `SecantMethod`) so the same method applies across an array of initial guesses.

### Source layout

| Path | Purpose |
|------|---------|
| `src/RootSolvers.jl` | The entire package: method/solution/tolerance types and the `find_zero` solvers |
| `test/runtests.jl` | Main test suite (CPU; pass `CuArray` to also run GPU tests) |
| `test/test_helper.jl` | Test utilities and problem generators |
| `test/test_printing.jl` | Solution-printing/formatting tests |
| `docs/` | Documentation source (`docs/src/`) and shared dev-guides (`docs/dev-guides/`) |

## Local norms

- For package tests, prefer `Pkg.test()` over manually `include`ing `test/runtests.jl`, so test-only dependencies load through the package test path. GPU tests run via `julia --project=. test/runtests.jl CuArray`.
- Match existing style: explicit names, narrow imports, comments that explain *why*.
- Docstrings follow [docs/dev-guides/code-quality/documentation_policy.md](docs/dev-guides/code-quality/documentation_policy.md): an indented signature line, single-`#` section headings in plural standard form (`# Arguments`, `# Returns`, `# Fields`, `# Examples`, `# Notes`), and `[`name`](@ref)` cross-references for every type/function mentioned.
- Run `julia -e 'using JuliaFormatter; format(".")'` before committing code (config in `.JuliaFormatter.toml`, margin 92).

## Self-correction

- If the source layout table above is discovered to be stale, update it.
- If the user gives a correction about how work should be done in this repo, add it to `Local norms` or another clearly labeled persistent section in this file so future sessions inherit it.
