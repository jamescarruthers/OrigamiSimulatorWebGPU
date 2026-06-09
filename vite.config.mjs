import { defineConfig } from 'vite';
import { cpSync } from 'node:fs';

// Vite config for the WebGL -> WebGPU app.
//
// Dev: serves the existing app (classic <script> tags + globals + inline GLSL)
// plus the ES-module entry (js/main.js, which imports three from npm).
//
// Build (`vite build`): bundles the module graph (three + the app) and the CSS,
// and emits HTML-referenced images. Two things still need copying verbatim,
// because they're not part of the module graph:
//   - the classic `dependencies/*.js` libraries loaded via <script> tags, plus
//     their fonts and the gif/CCapture web workers,
//   - the runtime-fetched example models under `assets/` (loaded by URL at run
//     time, so Vite can't see them).
// A `base: './'` build makes every emitted path relative, so the output works
// when served from a domain root, a project-pages subpath, or a custom domain
// without further configuration.
const PORT = Number(process.env.VITE_PORT) || 5179;

function copyStaticRuntimeFiles() {
  return {
    name: 'copy-static-runtime-files',
    apply: 'build',
    closeBundle() {
      for (const dir of ['dependencies', 'assets', 'fonts']) {
        cpSync(dir, `dist/${dir}`, { recursive: true });
      }
    },
  };
}

export default defineConfig(({ command }) => ({
  root: '.',
  base: command === 'build' ? './' : '/',
  // Disable Vite's dependency discovery scan. The app calls a global
  // `window.require('fold' | 'cdt2d' | 'svgpath')` — vendored Browserify bundles
  // loaded as classic <script>s, NOT npm packages. With discovery off, three
  // (the one real npm dependency) is resolved on demand.
  optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [copyStaticRuntimeFiles()],
  server: {
    port: PORT,
    strictPort: true,
    host: '127.0.0.1',
  },
  preview: {
    port: PORT,
    strictPort: true,
    host: '127.0.0.1',
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
  },
}));
