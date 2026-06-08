import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Compute-path interactions (Phase 2 follow-up).
//
// On ?solver=webgpu the WGSL compute solver now handles the interaction flags
// the WebGL solver does (drag / external forces / fixed / material), re-uploading
// the affected packed-data region or the lastPosition buffer in solve(). This
// drives those interactions on the live WebGPU path and checks they take effect
// without error. Skips when no WebGPU adapter is available.
// ---------------------------------------------------------------------------

const W = 900, H = 700;
test.use({ viewport: { width: W, height: H } });

const READY = () => {
  const g = window.globals;
  return !!(g && g.model && typeof g.model.getPositionsArray === 'function'
    && g.model.getPositionsArray() && g.model.getPositionsArray().length > 0);
};

test('grab / material / forces on the WebGPU compute solver', async ({ page }) => {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/CERT|decode|OTS/.test(m.text())) errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto('/index.html?solver=webgpu&model=Origami/traditionalCrane.svg', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(READY, undefined, { timeout: 60_000, polling: 250 });

  const onWebGPU = await page.evaluate(() => !!(window.globals.useWebGPUSolver && window.globals.webgpuSolver));
  test.skip(!onWebGPU, 'no WebGPU adapter (WebGL fallback)');

  await page.evaluate(() => { const g = window.globals; g.userInteractionEnabled = true; g.creasePercent = 0.9; g.shouldChangeCreasePercent = true; });
  await page.waitForTimeout(3000);

  // ---- GRAB: drag a node a long way and confirm a node follows the cursor.
  let grabDelta = 0;
  for (const [dx, dy] of [[0, 0], [0, -60], [40, -20], [-40, -20], [0, 50], [60, 10], [-60, 10]]) {
    const cx = W / 2 + dx, cy = H / 2 + dy;
    const before = await page.evaluate(() => Array.from(window.globals.model.getPositionsArray()));
    await page.mouse.move(cx, cy); await page.waitForTimeout(50);
    await page.mouse.down();
    await page.mouse.move(cx + 120, cy - 90, { steps: 8 });
    await page.mouse.move(cx + 185, cy - 40, { steps: 8 });
    await page.mouse.up(); await page.waitForTimeout(150);
    const after = await page.evaluate(() => Array.from(window.globals.model.getPositionsArray()));
    let md = 0;
    for (let i = 0; i < before.length; i += 3) {
      const d = Math.hypot(after[i] - before[i], after[i + 1] - before[i + 1], after[i + 2] - before[i + 2]);
      if (d > md) md = d;
    }
    if (md > grabDelta) grabDelta = md;
    if (md > 0.1) break;
  }
  expect(grabDelta, 'grabbing a node on the WebGPU solver moves it').toBeGreaterThan(0.1);

  // ---- MATERIAL: a large stiffness change must be picked up by the solver and
  // shift the settled shape, without error.
  const bboxY = async () => page.evaluate(() => {
    const p = window.globals.model.getPositionsArray(); let mn = Infinity, mx = -Infinity;
    for (let i = 1; i < p.length; i += 3) { if (p[i] < mn) mn = p[i]; if (p[i] > mx) mx = p[i]; }
    return mx - mn;
  });
  const beforeY = await bboxY();
  await page.evaluate(() => { const g = window.globals; g.axialStiffness = 0.5; g.materialHasChanged = true; });
  await page.waitForTimeout(2500);
  const afterY = await bboxY();
  expect(Number.isFinite(afterY) && afterY > 0, 'model still valid after material change').toBe(true);
  expect(Math.abs(afterY - beforeY), 'stiffness change altered the settled shape').toBeGreaterThan(0.005);

  // ---- FORCES: toggling the external-force flag is handled without error.
  await page.evaluate(() => { window.globals.forceHasChanged = true; });
  await page.waitForTimeout(1000);
  const posOk = await page.evaluate(() => {
    const p = window.globals.model.getPositionsArray();
    return p.length > 0 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);
  });
  expect(posOk, 'model still valid after force flag').toBe(true);

  expect(errors, `errors during interactions: ${JSON.stringify(errors)}`).toEqual([]);
});
