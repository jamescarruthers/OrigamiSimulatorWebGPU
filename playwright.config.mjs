import { defineConfig, devices } from '@playwright/test';

// Regression harness for the WebGL -> WebGPU migration (Phase 0).
//
// This launches the existing app via the Vite dev server and drives it with a
// headless Chromium that renders WebGL through ANGLE/SwiftShader (software), so
// it runs on machines without a GPU (CI, containers). The harness folds a fixed
// set of example patterns to fixed fold percentages for a fixed number of solve
// iterations, then snapshots the resulting node positions. Those snapshots are
// the "oracle" every later phase (three.js upgrade, WGSL compute solver) is
// validated against.
const PORT = Number(process.env.VITE_PORT) || 5179;

export default defineConfig({
  testDir: './tests',
  // The solver is iterative and a little slow under software rendering.
  timeout: 180_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    headless: true,
  },
  projects: [
    {
      name: 'chromium-swiftshader',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            // Force software GL so WebGL (incl. OES_texture_float) works
            // without a physical GPU.
            '--use-gl=angle',
            '--use-angle=swiftshader',
            '--enable-unsafe-swiftshader',
            '--ignore-gpu-blocklist',
            '--enable-webgl',
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
