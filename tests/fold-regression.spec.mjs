import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Fold regression harness  (WebGL -> WebGPU migration, Phase 0)
//
// For each (model, foldPercent) case we:
//   1. load the example pattern via the ?model= URL param,
//   2. pause the continuously-running animation loop (for determinism),
//   3. reset the solver to the flat/zero state,
//   4. set the target fold percent,
//   5. run a FIXED number of solver iterations,
//   6. snapshot the resulting node positions + summary stats.
//
// The snapshot is written to tests/golden/<key>.json the first time (or when
// UPDATE_SNAPSHOTS=1). On subsequent runs the live result is compared against
// the stored golden within a tight tolerance. This is the numeric oracle that
// every later migration step is validated against.
// ---------------------------------------------------------------------------

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = path.join(__dirname, 'golden');
const UPDATE = !!process.env.UPDATE_SNAPSHOTS;

// Absolute tolerance for position comparison. Geometry is centered and scaled
// to ~unit size, so coordinates are O(1). Software-rendered WebGL is
// deterministic run-to-run on the same machine, so this is deliberately tight;
// it can be loosened for cross-backend (WebGPU vs WebGL) comparisons later.
const POS_TOL = Number(process.env.POS_TOL) || 2e-4;

const CASES = [
  { model: 'SimpleFolds/simpleVertex.svg', fold: 0.0, iters: 40 },
  { model: 'SimpleFolds/simpleVertex.svg', fold: 1.0, iters: 40 },
  { model: 'Bases/squareBase.svg', fold: 1.0, iters: 40 },
  { model: 'Origami/traditionalCrane.svg', fold: 1.0, iters: 40 },
  { model: 'Tessellations/huffmanWaterbomb.svg', fold: 1.0, iters: 30 },
];

function keyFor({ model, fold, iters }) {
  const safe = model.replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_|_$/g, '');
  const pct = String(Math.round(fold * 100)).replace('-', 'm');
  return `${safe}__fold${pct}__it${iters}`;
}

// Runs entirely inside the browser. Must be self-contained (serializable args
// in, serializable result out).
function driveFold({ fold, iters }) {
  const g = window.globals;
  if (!g || !g.model || typeof g.model.getPositionsArray !== 'function') {
    return { ok: false, reason: 'globals/model not ready' };
  }
  const getPos = () => g.model.getPositionsArray();
  if (!getPos() || getPos().length === 0) {
    return { ok: false, reason: 'no positions (model not loaded?)' };
  }

  // Stop the auto-stepping render loop so our step count is the only source of
  // solver iterations.
  if (g.threeView && g.threeView.pauseSimulation) g.threeView.pauseSimulation();
  g.simulationRunning = false;

  // Known starting state, then fold toward the target.
  g.model.reset();
  g.creasePercent = fold;
  g.shouldChangeCreasePercent = true;

  const numSteps = g.numSteps;
  for (let i = 0; i < iters; i++) g.model.step(numSteps);

  const raw = getPos();
  const numNodes = raw.length / 3;
  const positions = new Array(raw.length);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < numNodes; i++) {
    const x = raw[3 * i], y = raw[3 * i + 1], z = raw[3 * i + 2];
    positions[3 * i] = Math.round(x * 1e6) / 1e6;
    positions[3 * i + 1] = Math.round(y * 1e6) / 1e6;
    positions[3 * i + 2] = Math.round(z * 1e6) / 1e6;
    if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
    if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
    cx += x; cy += y; cz += z;
  }
  const errEl = document.getElementById('globalError');
  return {
    ok: true,
    numNodes,
    numSteps,
    iters,
    integrationType: g.integrationType,
    fold,
    globalErrorText: errEl ? errEl.textContent.trim() : null,
    bbox: {
      min: [minX, minY, minZ].map((v) => +v.toFixed(6)),
      max: [maxX, maxY, maxZ].map((v) => +v.toFixed(6)),
    },
    centroid: [cx / numNodes, cy / numNodes, cz / numNodes].map((v) => +v.toFixed(6)),
    positions,
  };
}

function compare(golden, current) {
  if (golden.numNodes !== current.numNodes) {
    return { pass: false, msg: `node count changed: ${golden.numNodes} -> ${current.numNodes}` };
  }
  const a = golden.positions, b = current.positions;
  let maxAbs = 0, sum = 0;
  let worstIdx = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    sum += d;
    if (d > maxAbs) { maxAbs = d; worstIdx = i; }
  }
  const mean = sum / a.length;
  return {
    pass: maxAbs <= POS_TOL,
    maxAbs,
    mean,
    worstIdx,
    msg: `maxAbsDiff=${maxAbs.toExponential(3)} meanAbsDiff=${mean.toExponential(3)} (tol=${POS_TOL.toExponential(3)})`,
  };
}

test.beforeAll(() => {
  if (!fs.existsSync(GOLDEN_DIR)) fs.mkdirSync(GOLDEN_DIR, { recursive: true });
});

for (const c of CASES) {
  const key = keyFor(c);
  test(`fold ${c.model} @ ${Math.round(c.fold * 100)}% (${c.iters} iters)`, async ({ page }) => {
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

    await page.goto(`/index.html?model=${encodeURIComponent(c.model)}`, {
      waitUntil: 'domcontentloaded',
    });

    // Bail out clearly if the browser can't do float textures.
    const unsupported = await page
      .locator('#noSupportModal.in, #coverImg')
      .count()
      .catch(() => 0);
    expect(
      unsupported,
      'browser reported WebGL float textures unsupported (notSupported() fired)'
    ).toBe(0);

    // Wait until the pattern has loaded and produced geometry.
    await page.waitForFunction(
      () => {
        const g = window.globals;
        return !!(g && g.model && typeof g.model.getPositionsArray === 'function'
          && g.model.getPositionsArray() && g.model.getPositionsArray().length > 0);
      },
      undefined,
      { timeout: 60_000, polling: 250 }
    ).catch((e) => {
      throw new Error(`model "${c.model}" never produced geometry. console=${JSON.stringify(consoleErrors)}\n${e}`);
    });

    const result = await page.evaluate(driveFold, { fold: c.fold, iters: c.iters });
    expect(result.ok, `${result.reason} | console=${JSON.stringify(consoleErrors)}`).toBe(true);
    expect(result.numNodes).toBeGreaterThan(0);
    // Folded states must not be NaN/degenerate.
    expect(Number.isFinite(result.centroid[0])).toBe(true);

    const goldenPath = path.join(GOLDEN_DIR, `${key}.json`);
    const meta = {
      model: c.model,
      fold: c.fold,
      iters: c.iters,
      numSteps: result.numSteps,
      integrationType: result.integrationType,
      numNodes: result.numNodes,
      globalErrorText: result.globalErrorText,
      bbox: result.bbox,
      centroid: result.centroid,
    };

    if (UPDATE) {
      fs.writeFileSync(
        goldenPath,
        JSON.stringify({ ...meta, positions: result.positions }, null, 0) + '\n'
      );
      console.log(`[golden updated] ${key}  nodes=${result.numNodes}  err=${result.globalErrorText}`);
      test.info().annotations.push({ type: 'golden', description: `updated ${key}` });
      return;
    }

    // A missing golden is a FAILURE, not an auto-created baseline. Otherwise
    // `npm run test:regression` would pass without asserting anything, and CI
    // could silently establish a new baseline. Baselines must be generated
    // explicitly and committed.
    if (!fs.existsSync(goldenPath)) {
      throw new Error(
        `No golden snapshot for "${key}". Generate baselines first with ` +
        '`npm run regression:update` (UPDATE_SNAPSHOTS=1), then commit tests/golden/*.json.'
      );
    }

    const golden = JSON.parse(fs.readFileSync(goldenPath, 'utf8'));
    const cmp = compare(golden, { numNodes: result.numNodes, positions: result.positions });
    console.log(`[compare] ${key}  ${cmp.msg}`);
    expect(cmp.pass, `position drift for ${key}: ${cmp.msg}`).toBe(true);
  });
}
