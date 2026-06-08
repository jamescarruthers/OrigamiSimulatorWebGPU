import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Node grab/drag interaction (regression test for the three r184 upgrade).
//
// camera.getWorldDirection() requires a target vector in r184 (r87 returned a
// new one); without it, dragging a node threw "Cannot read properties of
// undefined (reading 'set')" in 3dUI.getIntersectionWithObjectPlane. This drives
// a real grab+drag over the rendered mesh and asserts it runs without error and
// actually moves a node.
//
// Uses the default (WebGL) solver so node positions are CPU-current and the
// raycast hits the folded mesh.
// ---------------------------------------------------------------------------

const W = 900, H = 700;

test.use({ viewport: { width: W, height: H } });

test('grab and drag a node without error', async ({ page }) => {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/CERT|decode|OTS/.test(m.text())) errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

  await page.goto('/index.html?model=Origami/traditionalCrane.svg', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => {
      const g = window.globals;
      return !!(g && g.model && typeof g.model.getPositionsArray === 'function'
        && g.model.getPositionsArray() && g.model.getPositionsArray().length > 0);
    },
    undefined,
    { timeout: 60_000, polling: 250 }
  );

  // Fold to a clear 3D state, enable interaction, then pause the solver so the
  // dragged position + nodePositionHasChanged flag aren't consumed mid-test.
  await page.evaluate(() => { const g = window.globals; g.userInteractionEnabled = true; g.creasePercent = 0.9; g.shouldChangeCreasePercent = true; });
  await page.waitForTimeout(3000);
  await page.evaluate(() => { window.globals.threeView.pauseSimulation(); window.globals.simulationRunning = false; });

  // Hover to highlight a node, then press-drag it. Try a few points around the
  // centre so we reliably land on the mesh.
  let dragged = false;
  const candidates = [[0, 0], [0, -60], [40, -20], [-40, -20], [0, 50], [60, 10], [-60, 10], [20, -80]];
  for (const [dx, dy] of candidates) {
    const cx = W / 2 + dx, cy = H / 2 + dy;
    await page.evaluate(() => { window.globals.nodePositionHasChanged = false; });
    await page.mouse.move(cx, cy);
    await page.waitForTimeout(40);
    await page.mouse.down();
    await page.mouse.move(cx + 30, cy - 30, { steps: 5 });
    await page.mouse.move(cx + 55, cy - 15, { steps: 5 });
    await page.mouse.up();
    await page.waitForTimeout(40);
    dragged = await page.evaluate(() => window.globals.nodePositionHasChanged === true);
    if (dragged) break;
  }

  expect(errors, `console/page errors during grab: ${JSON.stringify(errors)}`).toEqual([]);
  expect(dragged, 'a node was grabbed and moved (drag branch ran getIntersectionWithObjectPlane)').toBe(true);
});
