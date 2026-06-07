import { defineConfig } from 'vite';

// Phase 0 of the WebGL -> WebGPU migration.
//
// The goal of this config is intentionally minimal: serve the EXISTING app
// (plain <script> tags, globals, inline GLSL) unchanged through Vite's dev
// server so we get a module-capable toolchain in place WITHOUT changing any
// runtime behavior. The legacy dependencies in `dependencies/` and the global
// `js/*.js` files are loaded as classic scripts from `index.html`; Vite serves
// them statically in dev and does not try to bundle them.
//
// NOTE: `vite build` (full Rollup bundling) is intentionally NOT relied upon
// yet. It will be wired up incrementally in later phases as the global scripts
// are converted to ES modules (see WEBGPU_MIGRATION_PLAN.md, Phase 0/1).
const PORT = Number(process.env.VITE_PORT) || 5179;

export default defineConfig({
  root: '.',
  // Don't let Vite's dep optimizer scan/transform the vendored libraries.
  optimizeDeps: { disabled: true },
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
    // Placeholder — full bundling is enabled once scripts become ES modules.
    outDir: 'dist',
    emptyOutDir: true,
  },
});
