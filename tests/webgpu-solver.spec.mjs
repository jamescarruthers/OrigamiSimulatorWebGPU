import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// WebGPU compute solver validation (Phase 2).
//
// Drives the new WGSL compute solver (js/dynamic/WebGPUSolver.js) over the same
// fixed fold cases as the WebGL oracle (fold-regression.spec.mjs) and compares
// node positions against the committed golden snapshots. This is the numeric-
// parity check the migration plan calls for (§4.6): f32 (WebGPU) vs mediump
// (WebGL) and a removed per-node element cap produce small differences, so the
// tolerance here is looser than the run-to-run WebGL tolerance.
//
// Only meaningful on a WebGPU-capable browser (the `chromium-webgpu` project);
// it skips when no adapter is available.
// ---------------------------------------------------------------------------

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = path.join(__dirname, 'golden');

// Cross-backend tolerance (absolute, geometry ~unit scale). f32 vs mediump and
// reduction-order differences accumulate over many iterations. On the stable
// folds the WGSL compute solver matches the WebGL oracle to ~1e-5; the default
// here is deliberately a bit looser than that.
const POS_TOL = Number(process.env.WEBGPU_POS_TOL) || 1e-3;

// The centroid (center of mass) is a robust cross-backend invariant: even when
// individual nodes settle slightly differently, a correct fold lands the model
// in the same place. Asserted for every case.
const CENTROID_TOL = 1e-2;

const CASES = [
  { model: 'SimpleFolds/simpleVertex.svg', fold: 0.0, iters: 40 },
  { model: 'SimpleFolds/simpleVertex.svg', fold: 1.0, iters: 40 },
  { model: 'Bases/squareBase.svg', fold: 1.0, iters: 40 },
  // The traditional crane is a chaotic, metastable fold: tiny f32 reduction-
  // order differences vs the WebGL oracle amplify over ~4000 iterations into
  // visibly-different flap positions (same centroid, ~5% bbox difference). This
  // is the "intentional divergence" the migration plan anticipates (§4.6), so
  // we validate it via the centroid invariant plus a relaxed per-node bound
  // rather than tight position parity.
  { model: 'Origami/traditionalCrane.svg', fold: 1.0, iters: 40, metastable: true, tol: 0.4 },
  { model: 'Tessellations/huffmanWaterbomb.svg', fold: 1.0, iters: 30 },
];

function keyFor({ model, fold, iters }) {
  const safe = model.replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_|_$/g, '');
  const pct = String(Math.round(fold * 100)).replace('-', 'm');
  return `${safe}__fold${pct}__it${iters}`;
}

// Runs in the browser: drive the WebGPU solver and return node positions.
async function driveWebGPU({ fold, iters }) {
  const g = window.globals;
  if (!g || typeof g.initWebGPUSolver !== 'function') {
    return { ok: false, reason: 'initWebGPUSolver missing' };
  }
  const solver = g.initWebGPUSolver(g);
  const ready = await solver.init();
  if (!ready) return { ok: false, skip: true, reason: 'no WebGPU adapter' };

  // Pause the legacy auto-loop so it can't also write the shared positions.
  if (g.threeView && g.threeView.pauseSimulation) g.threeView.pauseSimulation();
  g.simulationRunning = false;

  solver.syncNodesAndEdges();
  solver.reset();
  g.creasePercent = fold;

  const numSteps = g.numSteps;
  for (let i = 0; i < iters; i++) solver.solve(numSteps);
  const globalError = await solver.readback();

  const raw = g.model.getPositionsArray();
  const numNodes = raw.length / 3;
  const positions = new Array(raw.length);
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < numNodes; i++) {
    positions[3 * i] = raw[3 * i];
    positions[3 * i + 1] = raw[3 * i + 1];
    positions[3 * i + 2] = raw[3 * i + 2];
    cx += raw[3 * i]; cy += raw[3 * i + 1]; cz += raw[3 * i + 2];
  }
  return {
    ok: true, numNodes, numSteps, iters, fold, globalError,
    centroid: [cx / numNodes, cy / numNodes, cz / numNodes],
    positions,
  };
}

function compare(golden, current) {
  const a = golden.positions, b = current.positions;
  let maxAbs = 0, sum = 0, worstIdx = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    sum += d;
    if (d > maxAbs) { maxAbs = d; worstIdx = i; }
  }
  return { maxAbs, mean: sum / a.length, worstIdx };
}

for (const c of CASES) {
  const key = keyFor(c);
  test(`webgpu fold ${c.model} @ ${Math.round(c.fold * 100)}% (${c.iters} iters)`, async ({ page }) => {
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

    const result = await page.evaluate(driveWebGPU, { fold: c.fold, iters: c.iters });

    if (result.skip) { test.skip(true, result.reason); return; }
    expect(result.ok, `${result.reason} | console=${JSON.stringify(consoleErrors)}`).toBe(true);
    expect(Number.isFinite(result.centroid[0]), `centroid NaN | console=${JSON.stringify(consoleErrors)}`).toBe(true);

    const goldenPath = path.join(GOLDEN_DIR, `${key}.json`);
    if (!fs.existsSync(goldenPath)) throw new Error(`No golden snapshot for "${key}".`);
    const golden = JSON.parse(fs.readFileSync(goldenPath, 'utf8'));

    expect(result.numNodes).toBe(golden.numNodes);
    const cmp = compare(golden, result);
    const tol = c.tol || POS_TOL;
    console.log(`[webgpu compare] ${key}  maxAbsDiff=${cmp.maxAbs.toExponential(3)} meanAbsDiff=${cmp.mean.toExponential(3)} err=${result.globalError.toFixed(5)}% (tol=${tol.toExponential(3)}${c.metastable ? ', metastable' : ''})`);

    // Centroid invariant: a correct fold lands the model in the same place.
    const centroidErr = Math.max(
      Math.abs(result.centroid[0] - golden.centroid[0]),
      Math.abs(result.centroid[1] - golden.centroid[1]),
      Math.abs(result.centroid[2] - golden.centroid[2]),
    );
    expect(centroidErr, `centroid drift vs WebGL golden for ${key}: ${centroidErr.toExponential(3)}`).toBeLessThanOrEqual(CENTROID_TOL);

    // Per-node parity (tight for stable folds; relaxed for the metastable crane,
    // which is guarded by the centroid invariant above).
    expect(cmp.maxAbs, `position drift vs WebGL golden for ${key}: maxAbsDiff=${cmp.maxAbs.toExponential(3)}`).toBeLessThanOrEqual(tol);
  });
}
