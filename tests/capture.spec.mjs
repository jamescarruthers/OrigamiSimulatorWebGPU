import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Screen capture verification (Phase 3).
//
// Capture grabs the renderer's canvas: PNG via canvas.toBlob(), GIF/WebM via
// CCapture(domElement). On the WebGPURenderer these could break (the render is
// async and the canvas might not preserve content), so this verifies a real
// capture produces a non-blank frame. On the chromium-webgpu project the canvas
// is the WebGPU backend; on chromium-swiftshader it's the WebGL2 fallback.
// ---------------------------------------------------------------------------

test.use({ viewport: { width: 640, height: 520 } });

async function loadAndFold(page) {
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
  await page.evaluate(() => { const g = window.globals; g.creasePercent = 0.9; g.shouldChangeCreasePercent = true; });
  await page.waitForTimeout(3000);
}

test('PNG capture produces a non-blank image', async ({ page }) => {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/CERT|decode|OTS/.test(m.text())) errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

  await loadAndFold(page);

  const png = await page.evaluate(async () => {
    return await new Promise((resolve, reject) => {
      const orig = window.saveAs;
      const timer = setTimeout(() => { window.saveAs = orig; reject(new Error('saveAs not called')); }, 15000);
      window.saveAs = (blob) => {
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          clearTimeout(timer); window.saveAs = orig;
          const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
          const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0);
          const d = ctx.getImageData(0, 0, img.width, img.height).data;
          let lumMin = 255, lumMax = 0, nonWhite = 0;
          for (let i = 0; i < d.length; i += 4) {
            const lum = (d[i] + d[i+1] + d[i+2]) / 3;
            if (lum < lumMin) lumMin = lum;
            if (lum > lumMax) lumMax = lum;
            if (lum < 250) nonWhite++;
          }
          resolve({ w: img.width, h: img.height, size: blob.size, lumMin, lumMax, nonWhitePct: 100 * nonWhite / (d.length / 4) });
        };
        img.onerror = () => { clearTimeout(timer); window.saveAs = orig; reject(new Error('PNG decode failed')); };
        img.src = url;
      };
      window.$('#doPNGCapture').click();
    });
  });

  expect(png.w).toBeGreaterThan(0);
  expect(png.size).toBeGreaterThan(1000);
  // The rendered crane must actually be in the image: a real luminance range and
  // a meaningful chunk of non-white (model) pixels — i.e. not a blank canvas.
  expect(png.lumMax - png.lumMin, 'PNG has luminance range (not blank)').toBeGreaterThan(50);
  expect(png.nonWhitePct, 'PNG has model pixels').toBeGreaterThan(2);
  expect(errors, `errors during PNG capture: ${JSON.stringify(errors)}`).toEqual([]);
});

test('CCapture (WebM) records frames from the canvas without error', async ({ page }) => {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/CERT|decode|OTS/.test(m.text())) errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

  await loadAndFold(page);

  // Start a WebM recording; the render loop calls capturer.capture(domElement)
  // each frame. Let frames accumulate, then abandon (don't trigger the heavy
  // encode/download) — we're verifying the per-frame canvas grab works.
  const frames = await page.evaluate(async () => {
    window.$('#doScreenRecord').click();
    await new Promise((r) => setTimeout(r, 2000));
    const n = window.globals.capturerFrames;
    window.globals.capturer = null;   // abandon the recording
    window.globals.shouldAnimateFoldPercent = false;
    return n;
  });

  expect(frames, 'CCapture captured canvas frames').toBeGreaterThan(0);
  expect(errors, `errors during WebM capture: ${JSON.stringify(errors)}`).toEqual([]);
});
