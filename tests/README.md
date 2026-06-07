# Regression harness (WebGL → WebGPU migration)

This is the **numeric oracle** for the migration described in
[`../WEBGPU_MIGRATION_PLAN.md`](../WEBGPU_MIGRATION_PLAN.md). It folds a fixed
set of example patterns under the **current WebGL solver** and snapshots the
resulting node positions. Every later change — the three.js upgrade and the
WGSL compute solver — is then validated by re-running this harness and checking
that positions still match the snapshots within tolerance.

## How it works

`fold-regression.spec.mjs` drives the real app through Playwright + a headless
Chromium (software WebGL via ANGLE/SwiftShader, so no GPU is required):

1. Load a pattern via the `?model=` URL param (e.g. `?model=Bases/squareBase.svg`).
2. Pause the continuously-running animation loop (`globals.threeView.pauseSimulation()`)
   so the only solver iterations are the ones we issue — this makes the run
   deterministic.
3. `globals.model.reset()` → flat/zero state.
4. Set `globals.creasePercent` + `globals.shouldChangeCreasePercent = true`.
5. Run a **fixed** number of `globals.model.step(globals.numSteps)` iterations.
6. Read `globals.model.getPositionsArray()` + the global-error readout.

The snapshot (positions rounded to 1e-6, plus bbox/centroid/error metadata) is
stored in `tests/golden/<key>.json`.

## Commands

```bash
npm install                     # install Vite + Playwright (already in package.json)
npx playwright install chromium # download the browser (needs network access; see note)

npm run regression:update       # create/refresh golden snapshots (UPDATE_SNAPSHOTS=1)
npm run test:regression         # compare current behavior against goldens
```

Useful environment variables:

| Var | Default | Purpose |
| --- | --- | --- |
| `UPDATE_SNAPSHOTS` | — | When set, write goldens instead of comparing. |
| `POS_TOL` | `2e-4` | Absolute per-coordinate tolerance (geometry is ~unit scale). |
| `VITE_PORT` | `5179` | Dev-server port used by the harness. |

The test matrix (models, fold %, iteration counts) lives at the top of
`fold-regression.spec.mjs` in `CASES` — extend it as needed. Tolerance is tight
because software WebGL is deterministic run-to-run; **loosen `POS_TOL` for
cross-backend comparisons** (WebGPU vs WebGL) where small `f32`-vs-`mediump`
and reduction-order differences are expected (see plan §4.6).

## ⚠️ Generating the baseline

The golden snapshots are intentionally **not committed yet**: they must be
generated in an environment where a browser is available, because they define
the baseline the migration is measured against.

> **Note on the Claude Code web sandbox:** this container's network policy
> blocks the Playwright browser CDN (`cdn.playwright.dev`) and Ubuntu only
> offers Chromium as a `snap` (no `snapd` in the container), so the headless
> browser could not be installed *here*. The harness itself is verified
> (`npx playwright test --list` discovers all cases, and the Vite dev server
> serves the app unchanged). Run `npm run regression:update` once on a machine
> or CI runner with browser access to produce `tests/golden/*.json`, then
> commit those files.

A `SessionStart` hook or CI step that runs `npx playwright install chromium`
(where the CDN is reachable) is the recommended way to keep this runnable.
