import { test, expect } from '@playwright/test';

// Smoke test for the solver benchmark (globals.benchmark). It doesn't assert
// performance numbers (those are meaningless under software rendering and vary
// by machine) — only that the benchmark runs, isolates + restores the solver,
// and reports positive throughput for each available backend without error.

test.use({ viewport: { width: 640, height: 520 } });

test('solver benchmark runs and reports throughput', async ({ page }) => {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/CERT|decode|OTS/.test(m.text())) errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto('/index.html?solver=webgpu&model=SimpleFolds/simpleVertex.svg', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => {
      const g = window.globals;
      return !!(g && g.model && typeof g.model.getPositionsArray === 'function'
        && g.model.getPositionsArray() && g.model.getPositionsArray().length > 0 && typeof g.benchmark === 'function');
    },
    undefined,
    { timeout: 60_000, polling: 250 }
  );
  await page.evaluate(() => { const g = window.globals; g.creasePercent = 0.9; g.shouldChangeCreasePercent = true; });
  await page.waitForTimeout(1500);

  const res = await page.evaluate(async () => await window.globals.benchmark(700));

  expect(res, 'benchmark returned a result').toBeTruthy();
  expect(res.numNodes).toBeGreaterThan(0);
  // WebGL solver is always available.
  expect(res.webgl.solvesPerSec, 'WebGL throughput > 0').toBeGreaterThan(0);
  expect(Number.isFinite(res.webgl.stepsPerSec)).toBe(true);
  // WebGPU is present on the webgpu project, absent (null) on the WebGL2 one.
  if (res.webgpu) {
    expect(res.webgpu.solvesPerSec, 'WebGPU throughput > 0').toBeGreaterThan(0);
    expect(res.speedup, 'speedup computed').toBeGreaterThan(0);
  }

  // The benchmark must leave the app running again (it stops the loop while measuring).
  const stillRunning = await page.evaluate(() => window.globals.simulationRunning);
  expect(stillRunning, 'simulation resumed after benchmark').toBe(true);

  expect(errors, `errors during benchmark: ${JSON.stringify(errors)}`).toEqual([]);
});
