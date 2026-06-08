import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// WebGPU compute solver — Verlet integration validation (Phase 2).
//
// There is no committed Verlet golden (the oracle is Euler), so this validates
// the WGSL Verlet kernels directly against the *legacy WebGL Verlet solver*
// run in the same page: load a model, switch globals.integrationType to
// 'verlet', fold it with the legacy solver to get a reference, then fold it
// with the WebGPU solver and compare node positions.
//
// Stable folds only — the chaotic crane diverges between any two f32 solvers
// (see webgpu-solver.spec.mjs) and is validated there via the centroid
// invariant. Skips when no WebGPU adapter is available.
// ---------------------------------------------------------------------------

// Verlet accumulates a little more cross-implementation f32 drift than Euler
// over thousands of steps (the velocity is derived from a position difference),
// so the tolerance is a touch looser than the Euler spec's 1e-3 — still tight
// enough to catch a real kernel error (the complex waterbomb matches to ~1e-4).
const POS_TOL = Number(process.env.WEBGPU_POS_TOL) || 5e-3;

const CASES = [
  { model: 'SimpleFolds/simpleVertex.svg', fold: 1.0, iters: 40 },
  { model: 'Bases/squareBase.svg', fold: 1.0, iters: 40 },
  { model: 'Tessellations/huffmanWaterbomb.svg', fold: 1.0, iters: 30 },
];

// Runs in the browser: fold with the legacy Verlet solver, then the WebGPU
// Verlet solver, and return both position arrays.
async function driveVerlet({ fold, iters }) {
  const g = window.globals;
  if (!g || typeof g.initWebGPUSolver !== 'function') return { ok: false, reason: 'initWebGPUSolver missing' };

  const solver = g.initWebGPUSolver(g);
  const ready = await solver.init();
  if (!ready) return { ok: false, skip: true, reason: 'no WebGPU adapter' };

  if (g.threeView && g.threeView.pauseSimulation) g.threeView.pauseSimulation();
  g.simulationRunning = false;
  g.integrationType = 'verlet';
  const numSteps = g.numSteps;

  // Legacy Verlet reference.
  g.model.reset();
  g.creasePercent = fold;
  g.shouldChangeCreasePercent = true;
  for (let i = 0; i < iters; i++) g.model.step(numSteps);
  const legacy = Array.from(g.model.getPositionsArray());

  // WebGPU Verlet.
  solver.syncNodesAndEdges();
  solver.reset();
  g.creasePercent = fold;
  for (let i = 0; i < iters; i++) solver.solve(numSteps);
  await solver.readback();
  const webgpu = Array.from(g.model.getPositionsArray());

  return { ok: true, numNodes: legacy.length / 3, legacy, webgpu };
}

for (const c of CASES) {
  test(`webgpu verlet ${c.model} @ ${Math.round(c.fold * 100)}% (${c.iters} iters)`, async ({ page }) => {
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

    await page.goto(`/index.html?model=${encodeURIComponent(c.model)}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => {
        const g = window.globals;
        return !!(g && g.model && typeof g.model.getPositionsArray === 'function'
          && g.model.getPositionsArray() && g.model.getPositionsArray().length > 0);
      },
      undefined,
      { timeout: 60_000, polling: 250 }
    ).catch((e) => { throw new Error(`model "${c.model}" never produced geometry. console=${JSON.stringify(consoleErrors)}\n${e}`); });

    const result = await page.evaluate(driveVerlet, { fold: c.fold, iters: c.iters });
    if (result.skip) { test.skip(true, result.reason); return; }
    expect(result.ok, `${result.reason} | console=${JSON.stringify(consoleErrors)}`).toBe(true);

    let maxAbs = 0, sum = 0;
    for (let i = 0; i < result.legacy.length; i++) {
      const d = Math.abs(result.legacy[i] - result.webgpu[i]);
      sum += d; if (d > maxAbs) maxAbs = d;
    }
    console.log(`[verlet compare] ${c.model}  maxAbsDiff=${maxAbs.toExponential(3)} meanAbsDiff=${(sum / result.legacy.length).toExponential(3)} (tol=${POS_TOL.toExponential(3)})`);
    expect(Number.isFinite(result.webgpu[0]), `WebGPU Verlet produced NaN | console=${JSON.stringify(consoleErrors)}`).toBe(true);
    expect(maxAbs, `WebGPU Verlet vs legacy Verlet drift for ${c.model}: ${maxAbs.toExponential(3)}`).toBeLessThanOrEqual(POS_TOL);
  });
}
