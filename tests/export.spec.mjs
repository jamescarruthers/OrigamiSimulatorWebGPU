import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Export validation (Phase 3).
//
// STL/OBJ/FOLD export was rewritten for three r184 (THREE.Geometry/Face3 were
// removed) and made async so it can map the GPU position buffer on the WebGPU
// path. This drives each export through the real UI handler, intercepts the
// FileSaver `saveAs` blob, and checks the output is structurally valid.
//
// Loaded with ?solver=webgpu: the chromium-webgpu project exercises the WebGPU
// async getPositions() path; chromium-swiftshader falls back to the WebGL solver
// (so it also covers the default export path).
// ---------------------------------------------------------------------------

const MODELS = [
  'SimpleFolds/simpleVertex.svg',
  'Origami/traditionalCrane.svg',
];

// Override saveAs, click the export button, and return the captured blob.
async function captureExport(page, buttonId) {
  return await page.evaluate(async (id) => {
    return await new Promise((resolve, reject) => {
      const orig = window.saveAs;
      const timer = setTimeout(() => { window.saveAs = orig; reject(new Error('saveAs not called within timeout')); }, 20000);
      window.saveAs = async (blob, filename) => {
        try {
          const buf = await blob.arrayBuffer();
          clearTimeout(timer); window.saveAs = orig;
          resolve({ filename, byteLength: buf.byteLength, bytes: Array.from(new Uint8Array(buf.slice(0, 88))), text: new TextDecoder().decode(buf) });
        } catch (e) { reject(e); }
      };
      window.$(id).click();
    });
  }, buttonId);
}

for (const model of MODELS) {
  test(`export STL/OBJ/FOLD for ${model}`, async ({ page }) => {
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error' && !/CERT|decode|OTS/.test(m.text())) consoleErrors.push(m.text()); });
    page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

    await page.goto(`/index.html?solver=webgpu&model=${encodeURIComponent(model)}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => {
        const g = window.globals;
        return !!(g && g.model && typeof g.model.getPositionsArray === 'function'
          && g.model.getPositionsArray() && g.model.getPositionsArray().length > 0);
      },
      undefined,
      { timeout: 60_000, polling: 250 }
    );
    // Fold it and let the solver run a bit so we export a non-flat state.
    await page.evaluate(() => { const g = window.globals; g.creasePercent = 0.8; g.shouldChangeCreasePercent = true; });
    await page.waitForTimeout(2500);

    const counts = await page.evaluate(() => {
      const g = window.globals;
      const geo = g.model.getGeometry();
      return { numNodes: g.model.getPositionsArray().length / 3, numTris: geo.index ? geo.index.array.length / 3 : 0 };
    });
    expect(counts.numNodes).toBeGreaterThan(0);
    expect(counts.numTris).toBeGreaterThan(0);

    // ---- STL (binary): 80-byte header + uint32 triangle count + 50 bytes/tri
    const stl = await captureExport(page, '#doSTLsave');
    expect(stl.filename).toMatch(/\.stl$/);
    const dv = new DataView(new Uint8Array(stl.bytes).buffer);
    const triCount = dv.getUint32(80, true);
    expect(triCount, 'STL triangle count = numFaces (doublesidedSTL=false)').toBe(counts.numTris);
    expect(stl.byteLength, 'STL byte length = 84 + 50*tris').toBe(84 + 50 * triCount);

    // ---- OBJ
    const obj = await captureExport(page, '#doOBJsave');
    expect(obj.filename).toMatch(/\.obj$/);
    const vLines = (obj.text.match(/^v /gm) || []).length;
    const fLines = (obj.text.match(/^f /gm) || []).length;
    expect(vLines, 'OBJ vertex count = numNodes').toBe(counts.numNodes);
    expect(fLines).toBeGreaterThan(0);

    // ---- FOLD
    const fold = await captureExport(page, '#doFOLDsave');
    expect(fold.filename).toMatch(/\.fold$/);
    const foldJson = JSON.parse(fold.text);
    expect(foldJson.vertices_coords.length, 'FOLD vertices = numNodes').toBe(counts.numNodes);
    expect(foldJson.faces_vertices.length).toBeGreaterThan(0);
    expect(Number.isFinite(foldJson.vertices_coords[0][0])).toBe(true);
    // The exported model must be the *folded* (3D) state, which proves the
    // export captured current positions (incl. the WebGPU GPU readback) rather
    // than a stale flat array: the vertical (y) extent should be non-trivial.
    let minY = Infinity, maxY = -Infinity;
    for (const v of foldJson.vertices_coords) { if (v[1] < minY) minY = v[1]; if (v[1] > maxY) maxY = v[1]; }
    expect(maxY - minY, 'exported model is folded (non-zero y extent)').toBeGreaterThan(0.02);

    expect(consoleErrors, `console errors during export: ${JSON.stringify(consoleErrors)}`).toEqual([]);
  });
}
