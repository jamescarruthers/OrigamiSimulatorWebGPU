import { defineConfig, devices, chromium } from '@playwright/test';
import path from 'node:path';

// Regression harness for the WebGL -> WebGPU migration.
//
// This launches the existing app via the Vite dev server and drives it with a
// headless Chromium. The harness folds a fixed set of example patterns to fixed
// fold percentages for a fixed number of solve iterations, then snapshots the
// resulting node positions. Those snapshots are the "oracle" every later phase
// (three.js upgrade, WGSL compute solver) is validated against.
//
// Two projects:
//   - chromium-swiftshader: software WebGL (ANGLE/SwiftShader). Exercises the
//     legacy WebGL1 GPGPU solver + WebGL2 render fallback. Runs anywhere.
//   - chromium-webgpu: software *WebGPU* via SwiftShader's Vulkan ICD (Dawn).
//     This is what makes the Phase 2 WGSL compute solver testable without a
//     physical GPU. See README for the flag/ICD details.
const PORT = Number(process.env.VITE_PORT) || 5179;

// Software-GL flags (needed for WebGL1 OES_texture_float and the WebGL2 render
// fallback) — shared by both projects.
const SWIFTSHADER_GL_ARGS = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
  '--enable-webgl',
];

// SwiftShader's Vulkan ICD ships next to the Playwright Chromium binary; Dawn
// uses it as a software WebGPU adapter when pointed at it via VK_ICD_FILENAMES.
const CHROME_DIR = path.dirname(chromium.executablePath());
const SWIFTSHADER_VK_ICD = path.join(CHROME_DIR, 'vk_swiftshader_icd.json');

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
        launchOptions: { args: SWIFTSHADER_GL_ARGS },
      },
    },
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            ...SWIFTSHADER_GL_ARGS,
            // Enable WebGPU on a software Vulkan (SwiftShader) backend.
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--disable-gpu-sandbox',
          ],
          env: {
            ...process.env,
            VK_ICD_FILENAMES: SWIFTSHADER_VK_ICD,
            VK_DRIVER_FILES: SWIFTSHADER_VK_ICD,
            LD_LIBRARY_PATH: CHROME_DIR + (process.env.LD_LIBRARY_PATH ? ':' + process.env.LD_LIBRARY_PATH : ''),
          },
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
