/**
 * Solver benchmark (Phase 4). Compares throughput of the WebGL GPGPU solver vs
 * the WGSL compute solver on the currently-loaded model.
 *
 * Metric: `solve(numSteps)` calls completed per second, with the render loop
 * stopped and GPU work synced each batch. One such call is exactly the physics
 * work the app does per displayed frame, so this is the *uncapped* potential
 * frame rate of each solver (the on-screen FPS is min(this, monitor refresh)).
 *
 * Usage:
 *   - append `?benchmark` to the URL (auto-runs after the model loads), or
 *   - call `await globals.benchmark()` in the console.
 *
 * NOTE: run it on real hardware. Under software rendering (SwiftShader) both
 * backends are CPU-emulated, so the numbers don't reflect a real GPU.
 */
export function initBenchmark(globals) {

  // WebGPU queues commands; await real completion. WebGL's solve() blocks on a
  // readPixels each call, so it's already synchronous.
  async function syncGPU(solver) {
    if (solver.isWebGPUSolver && solver.finish) await solver.finish();
  }

  async function measure(solver, durationMs) {
    const numSteps = globals.numSteps;
    // WebGPU: queue a few solves between syncs to amortise the await round-trip;
    // WebGL is synchronous so a batch of 1 is correct.
    const batch = solver.isWebGPUSolver ? 8 : 1;

    // Warm up (also folds the model a little so we measure a representative state).
    for (let i = 0; i < 12; i++) solver.solve(numSteps);
    await syncGPU(solver);

    let count = 0;
    const start = performance.now();
    while (performance.now() - start < durationMs) {
      for (let b = 0; b < batch; b++) solver.solve(numSteps);
      await syncGPU(solver);
      count += batch;
    }
    const elapsed = (performance.now() - start) / 1000;
    return {
      solvesPerSec: +(count / elapsed).toFixed(1),       // uncapped potential FPS
      stepsPerSec: Math.round((count * numSteps) / elapsed),
      solves: count,
      elapsedSec: +elapsed.toFixed(2),
    };
  }

  // The WebGPU solver may not be the active one; create + init it on demand.
  // Returns null if no WebGPU adapter (renderer fell back to WebGL2).
  async function getWebGPUSolver() {
    // Reuse the live solver if present, else create one. init() is idempotent
    // (returns true only once a WebGPU device is held), so always await it —
    // globals.webgpuSolver can exist with a *failed* init on the WebGL2 fallback.
    const s = globals.webgpuSolver
      || (typeof globals.initWebGPUSolver === 'function' ? globals.initWebGPUSolver(globals) : null);
    if (!s) return null;
    const ok = await s.init();
    if (!ok) return null;
    globals.webgpuSolver = s;
    return s;
  }

  async function benchmark(durationMs) {
    durationMs = durationMs || 2500;
    const positions = globals.model && globals.model.getPositionsArray();
    if (!positions || positions.length === 0) {
      console.warn('[benchmark] no model loaded');
      return null;
    }

    const renderer = globals.threeView.renderer;
    const rendererBackend = renderer && renderer.backend
      ? (renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl2') : 'unknown';

    const results = {
      model: globals.filename || globals.url || 'unknown',
      numNodes: positions.length / 3,
      numSteps: globals.numSteps,
      integration: globals.integrationType,
      rendererBackend,
      durationMs,
    };

    // Isolate the solver from rendering/vsync.
    globals.threeView.stopAnimation();
    const wasRunning = globals.simulationRunning;
    globals.simulationRunning = false;

    const legacy = globals.dynamicSolver;
    try {
      legacy.syncNodesAndEdges();
      legacy.reset();
      results.webgl = await measure(legacy, durationMs);

      const wgpu = await getWebGPUSolver();
      if (wgpu) {
        wgpu.syncNodesAndEdges();
        wgpu.reset();
        results.webgpu = await measure(wgpu, durationMs);
        results.speedup = +(results.webgpu.solvesPerSec / results.webgl.solvesPerSec).toFixed(2);
      } else {
        results.webgpu = null;
        results.note = 'WebGPU solver unavailable (renderer is on the WebGL2 fallback).';
      }
    } finally {
      // Resume: the active solver is left in a folded state by the benchmark and
      // simply continues. (When ?solver=webgpu the WebGPU solver drives the
      // geometry; otherwise the WebGL solver does.)
      globals.simulationRunning = wasRunning;
      globals.threeView.startAnimation();
    }

    const fmt = (r) => r ? `${r.solvesPerSec} solves/s (${r.stepsPerSec.toLocaleString()} steps/s)` : 'n/a';
    console.log(
      `[benchmark] ${results.model}  (${results.numNodes} nodes, ${results.numSteps} steps/solve, renderer=${rendererBackend})\n` +
      `  WebGL : ${fmt(results.webgl)}\n` +
      `  WebGPU: ${fmt(results.webgpu)}` +
      (results.speedup ? `\n  speedup (WebGPU/WebGL): ${results.speedup}x` : (results.note ? `\n  ${results.note}` : '')) +
      `\n  (uncapped solver throughput; on-screen FPS is capped at the monitor refresh rate)`
    );
    return results;
  }

  // Minimal on-page overlay so ?benchmark shows results without the console.
  function showOverlay(results) {
    if (!results) return;
    const row = (label, r) => `<tr><td style="padding:2px 12px 2px 0">${label}</td>` +
      `<td style="text-align:right">${r ? r.solvesPerSec.toFixed(1) : '—'}</td>` +
      `<td style="text-align:right;padding-left:12px">${r ? r.stepsPerSec.toLocaleString() : '—'}</td></tr>`;
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;top:70px;left:50%;transform:translateX(-50%);z-index:99999;' +
      'background:rgba(20,20,24,.92);color:#eee;font:13px/1.5 monospace;padding:14px 18px;border-radius:8px;' +
      'box-shadow:0 4px 20px rgba(0,0,0,.4);max-width:90vw';
    el.innerHTML =
      `<div style="font-weight:bold;margin-bottom:6px">Solver benchmark — ${results.model} (${results.numNodes} nodes)</div>` +
      `<table style="border-collapse:collapse"><tr style="opacity:.6"><td></td>` +
      `<td style="text-align:right">solves/s</td><td style="text-align:right;padding-left:12px">steps/s</td></tr>` +
      row('WebGL', results.webgl) + row('WebGPU', results.webgpu) + `</table>` +
      (results.speedup ? `<div style="margin-top:6px;color:#7fd">WebGPU is ${results.speedup}× the WebGL throughput</div>` : '') +
      (results.note ? `<div style="margin-top:6px;color:#fc8">${results.note}</div>` : '') +
      `<div style="margin-top:8px;opacity:.6">renderer: ${results.rendererBackend} · ${results.numSteps} steps/solve · uncapped (on-screen FPS is vsync-capped)</div>` +
      `<div style="margin-top:4px;opacity:.6;font-style:italic">Software rendering underestimates a real GPU — run on hardware for true numbers.</div>`;
    el.addEventListener('click', () => el.remove());
    document.body.appendChild(el);
  }

  globals.benchmark = benchmark;
  globals.showBenchmarkOverlay = showOverlay;
}
