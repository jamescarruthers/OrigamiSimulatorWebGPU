# WebGL → WebGPU Migration: Evaluation & Plan

> Status: planning document. No runtime behaviour has been changed yet.
> Scope target: the GPGPU physics solver. Rendering migration is treated as an
> optional later phase.

## 1. Executive summary

Origami Simulator runs its fold-solver as **general-purpose GPU computation
(GPGPU) built on WebGL 1**. There is no real "graphics" being done by the
solver — it abuses fragment shaders and floating-point textures as a parallel
array processor. A second, *separate* use of WebGL is the **three.js renderer**
that draws the folded mesh to the screen.

Converting "to WebGPU" therefore means two largely independent efforts:

| Subsystem | Today | WebGPU target | Priority |
|-----------|-------|---------------|----------|
| **Physics solver** (`js/dynamic/*`, GLSL shaders in `index.html`) | WebGL1 GPGPU: float textures + fragment shaders, ping-pong FBOs, float-to-byte readback hack | WGSL **compute** shaders + storage buffers, direct float readback | **Primary** |
| **Renderer** | three.js r87 `WebGLRenderer` | three.js `WebGPURenderer` (r150+) | Optional / later |
| Static & rigid solvers | Pure CPU (`numeric.js`) | unchanged | n/a |

**Recommendation:** do the solver in WebGPU first (Phases 1–4 below) while
keeping the existing three.js `WebGLRenderer` for display. This delivers the
core benefit (a modern, faster, more accurate compute path that drops the
float-packing hack and `mediump` precision) with contained risk. Upgrading
three.js to the `WebGPURenderer` is a much larger, mostly orthogonal dependency
upgrade and should be a separate, optional phase.

This is a **rewrite of the compute layer, not a line-by-line port** — the
programming models differ enough that a clean reimplementation behind the same
internal interface is lower-risk than a mechanical translation.

---

## 2. How the current solver works

### 2.1 The GPGPU framework

Two files implement a tiny compute engine on top of WebGL1:

- **`js/dynamic/GLBoilerplate.js`** — compile/link shader programs, bind a
  full-screen quad (`a_position` = `[-1,-1, 1,-1, -1,1, 1,1]`), and create
  `RGBA`/`FLOAT` textures (`OES_texture_float`).
- **`js/dynamic/GPUMath.js`** — the engine object:
  - `createProgram` — compiles a vertex+fragment pair, caches uniforms.
  - `initTextureFromData` / `initFrameBufferForTexture` — data arrays live in
    `RGBA32F` textures; each writable texture gets a framebuffer (render
    target).
  - `setUniformForProgram`, `setSize` (viewport), `setProgram`.
  - **`step(program, inputTextures[], outputTexture)`** — the core "kernel
    launch": bind output FBO, bind input textures to texture units, draw the
    full-screen quad. The fragment shader runs once per output texel.
  - **`swapTextures` / `swap3Textures`** — ping-pong (double/triple buffering).
  - **`readPixels`** — read back `UNSIGNED_BYTE` only (WebGL1 cannot read
    float render targets directly).

### 2.2 Data model

All simulation state is stored in square `RGBA32F` textures. A 1-D logical
index `i` is mapped to a texel via:

```
texel = ( mod(i, dim) + 0.5 , floor(i / dim) + 0.5 ) / dim
```

Texture dimensions are powers of two chosen by `calcTextureSize(n)` so that
`dim*dim >= n` (with padding cells that the shaders guard against). There are
several domains, each with its own texture size:

| Domain | Texture dim var | Per-element payload (RGBA) |
|--------|-----------------|----------------------------|
| Nodes | `textureDim` | position / velocity / mass / meta / forces |
| Beams (edges) | `textureDimEdges` | `beamMeta = [k, d, length, otherNodeIndex]` |
| Creases | `textureDimCreases` | `theta`, `creaseMeta`, `creaseGeo`, `creaseVectors`, `creaseMeta2` |
| Faces | `textureDimFaces` | `normals`, `faceVertexIndices`, `nominalTriangles` |
| Node→crease | `textureDimNodeCreases` | `nodeCreaseMeta = [creaseIndex, nodeNum]` |
| Node→face | `textureDimNodeFaces` | `nodeFaceMeta = [faceIndex, a, b, c]` |

Each node gathers its neighbours (beams/creases/faces) by walking these
flattened adjacency arrays using `meta`/`meta2` offset+count records.

### 2.3 The kernels (GLSL fragment shaders in `index.html`)

| Shader id | Domain | Role |
|-----------|--------|------|
| `vertexShader` | — | shared pass-through quad vertex shader |
| `normalCalc` | faces | recompute per-face normals from current positions |
| `thetaCalc` | creases | dihedral fold angle per crease (uses `atan`, normals, crease vector) |
| `updateCreaseGeo` | creases | moment arms `h1,h2` and projection coefficients |
| `velocityCalc` + `positionCalc` | nodes | **Euler** integration: sum beam/crease/face forces → velocity → position |
| `positionCalcVerlet` + `velocityCalcVerlet` | nodes | **Verlet** integration (default) |
| `packToBytes` | nodes | encode `float32` → 4×`uint8` so WebGL1 can read it back |
| `zeroTexture`, `zeroThetaTexture`, `centerTexture`, `copyTexture` | various | utility/reset ops |

### 2.4 The solve loop

`dynamicSolver.js` orchestrates everything:

- `syncNodesAndEdges()` builds typed arrays from the model, uploads textures,
  creates programs.
- `solveStep()` runs one timestep — the pipeline per step is:
  1. `normalCalc` (size = faces)
  2. `thetaCalc` (size = creases)
  3. `updateCreaseGeo` (size = creases)
  4. Verlet: `positionCalcVerlet` then `velocityCalcVerlet` (size = nodes);
     or Euler: `velocityCalc` then `positionCalc`
  5. `swapTextures` to ping-pong `theta`/`velocity`/`position`
- `solve(numSteps)` runs N steps then `render()`.
- **`render()`** runs `packToBytes` on `u_lastPosition`, `readPixels` as bytes,
  reinterprets the `Uint8Array` buffer as `Float32Array`, adds positions to
  `originalPosition`, and writes them into the three.js `BufferGeometry`
  position (and optionally strain-colour) attributes. This readback also
  produces the per-node error shown in the UI.

### 2.5 Rendering

`js/threeView.js` + `js/model.js` use three.js **r87** `WebGLRenderer`.
The solver feeds the renderer **through the CPU**: positions are read back each
frame and copied into `geometry.attributes.position`. The renderer and the
solver do **not** currently share GPU memory.

---

## 3. Why move to WebGPU? (evaluation)

**Wins**
- **Real compute shaders.** Kernels become `@compute` WGSL over storage
  buffers — the natural model for this workload, instead of disguising compute
  as fragment shading.
- **Direct float readback.** `mapAsync` returns `Float32Array` directly, so the
  entire `packToBytes` float-encoding shader and the byte→float reinterpret can
  be **deleted**. Simpler and faster.
- **Full f32 precision.** Current shaders are `precision mediump float`, which
  on some GPUs (especially mobile) is fp16 for computation. WebGPU compute is
  f32 throughout → better numerical stability of the solver.
- **Cleaner indexing.** WGSL has real integers; the `mod/floor`/`+0.5` 1-D↔2-D
  texel-address arithmetic can be replaced with plain buffer indexing.
- **Path to zero-copy rendering.** With three.js `WebGPURenderer`, the solver's
  position buffer could be shared with the render geometry, removing the
  per-frame CPU readback entirely (a significant future speedup).
- Modern, actively-developed API; WebGL is in maintenance.

**Costs / risks**
- **Async initialization.** `navigator.gpu.requestAdapter()` / `requestDevice()`
  are `async`. Today `initGPUMath()` is synchronous and called inline during
  startup. This is the single biggest structural change — startup/model-load
  must become promise-aware.
- **Browser support & fallback.** WebGPU is in stable Chrome/Edge, Safari 18+,
  and rolling out in Firefox, but is not universal. Must feature-detect
  `navigator.gpu` and keep a graceful "not supported" path (the existing
  `#noSupportModal` can be reused). Consider keeping the WebGL solver as a
  runtime fallback (see Phase 5).
- **Full rewrite of the compute layer**, plus translating ~10 GLSL kernels to
  WGSL and re-validating the physics for numerical parity.
- **Tooling:** WGSL has no `#define`; constants and small helpers must be
  inlined or templated in JS.

---

## 4. Target WebGPU architecture

### 4.1 Data layout — use storage buffers, not textures

Recommended: replace the square float textures with **flat storage buffers** of
`array<vec4<f32>>` (or `vec4<u32>` where indices are stored). Each kernel is a
1-D dispatch; invocation `gid.x` maps **directly** to element `i`, with an early
`if (i >= count) return;` to skip padding. This removes all 2-D texel-address
math and is the idiomatic compute layout.

> Alternative (lower-diff) layout: keep the 2-D texture model using
> `texture_storage_2d<rgba32float>`. This mirrors the current code most closely
> but carries the texel-addressing baggage forward. Prefer storage buffers
> unless a problem forces otherwise.

### 4.2 Component mapping

| WebGL concept | WebGPU replacement |
|---------------|--------------------|
| `gl` context on `#gpuMathCanvas` | `GPUDevice` from `await adapter.requestDevice()` (no canvas needed for compute) |
| `OES_texture_float` | native `f32` storage buffers — no extension |
| `RGBA32F` texture (data) | `GPUBuffer` (STORAGE), `array<vec4<f32>>` |
| Framebuffer / render target | the output `GPUBuffer` bound as read-write storage |
| Uniforms (`uniform1f`, …) | uniform `GPUBuffer` + `@group/@binding`, written via `queue.writeBuffer` |
| Fragment shader kernel | `@compute @workgroup_size(64)` WGSL entry point |
| `gl_FragCoord` → texel | `@builtin(global_invocation_id)` → element index |
| `gpuMath.step(...)` (draw quad) | `pass.setPipeline` + `setBindGroup` + `dispatchWorkgroups(ceil(n/64))` |
| `swapTextures` | swap `GPUBuffer` references (and bind groups) |
| `packToBytes` + `readPixels` | copy storage→`MAP_READ` buffer, `await buffer.mapAsync(READ)`, read `Float32Array` — **delete `packToBytes`** |
| `bool u_calcFaceStrain` | uniform `u32` flag (WGSL uniforms can't be `bool`) |

### 4.3 New / replaced files

```
js/dynamic/
  GPUMathWebGPU.js   (new) — device init, buffer & uniform management,
                            createKernel(), dispatch(), readback(); async
  wgsl/              (new) — one WGSL file per kernel (or template strings):
      normalCalc.wgsl, thetaCalc.wgsl, updateCreaseGeo.wgsl,
      positionCalc.wgsl, velocityCalc.wgsl,
      positionCalcVerlet.wgsl, velocityCalcVerlet.wgsl,
      center.wgsl, copy.wgsl, zero.wgsl, zeroTheta.wgsl
  dynamicSolver.js   (edited) — same public API
                     (syncNodesAndEdges/solve/render/reset) but async-aware
                     and calling the new engine; packToBytes path removed
```

`GLBoilerplate.js` and `GPUMath.js` are retired (or kept only for the optional
WebGL fallback). The `<script id="...Shader">` blocks in `index.html` are
removed once WGSL replaces them.

### 4.4 Async startup

- `initDynamicSolver` becomes `async` (or returns a ready-promise).
  `globals.gpuMath` is only valid after the device resolves.
- The animation loop (`threeView._loop` → `model.step`) must not call `solve()`
  until the device + buffers are ready. Add a `globals.solverReady` flag (or
  gate `model.step()` on a promise) so frames before init just render.
- Model load (`model.sync` → `syncSolver` → `syncNodesAndEdges`) must `await`
  buffer (re)allocation when geometry changes.

---

## 5. Kernel translation notes (GLSL → WGSL)

General rules:
- Each fragment `main()` → a `@compute` function; the first line becomes
  `let i = gid.x; if (i >= uCount) { return; }`.
- `texture2D(tex, scaledCoord)` reads → `buffer[index]` reads. The
  `getFromArray(index1D, dim, tex)` helper collapses to `buf[u32(index1D)]`.
- `gl_FragColor = v` → `outBuf[i] = v`.
- `vec3/vec4/mat*` → `vec3<f32>` etc. `mix`, `cross`, `dot`, `normalize`,
  `length`, `acos`, `atan(y,x)`→`atan2(y,x)`, `mod`→`x - y*floor(x/y)` (or
  integer `%`).
- The `for (j=0;j<100;...) if (j>=count) break;` neighbour loops port directly;
  with buffer indexing the `100` cap can stay (safe) or become dynamic.
- `#define TWO_PI ...` (in `thetaCalc`) → a `const TWO_PI: f32 = ...;`.
- `mediump` qualifiers are dropped (all f32).

Kernel-specific:
- **`normalCalc`** — straightforward cross-product/normalize per face.
- **`thetaCalc`** — dihedral angle; watch `atan`→`atan2` and the ±2π wrap logic;
  preserves `normalIndex` passthrough in `.zw`.
- **`updateCreaseGeo`** — moment arms; preserve the "disable crease" sentinel
  (`-1`).
- **`velocityCalc` / `positionCalcVerlet`** — the big ones: beam, crease, and
  face force accumulation. Port the three neighbour loops faithfully; keep the
  `nodeError` accumulation that feeds the UI/strain colour. `u_calcFaceStrain`
  becomes a `u32` uniform.
- **utility kernels** (`zero`, `zeroTheta`, `center`, `copy`) — trivial.
- **`packToBytes`** — **removed entirely**; replaced by direct buffer readback.

---

## 6. Rendering: keep WebGL for now

The solver migration does **not** require touching the renderer. `render()`
already hands positions to three.js via the CPU, so a WebGPU solver that
produces the same `Float32Array` plugs straight in.

**Optional Phase 6 — three.js `WebGPURenderer`:** this is a large, separate
undertaking. three.js r87 (2017) predates `WebGPURenderer` by years; adopting
it means upgrading to a modern release (r150+/r160+), which brings breaking
changes the codebase relies on, e.g.:
- `BufferGeometry.addAttribute` → `setAttribute`; legacy `Geometry` removed.
- `TrackballControls`, `OBJExporter`, `SVGLoader`, etc. move to ES-module
  `examples/jsm` imports (the project currently loads global `<script>` files).
- Material/`flatShading`/`vertexColors` API changes.
- WebGL/WebVR helpers (`WebVR.js`, `VRController.js`) are already deprecated.

The payoff (beyond "all WebGPU") is **zero-copy**: share the solver's position
`GPUBuffer` with the render geometry and skip the per-frame readback. Worth it
eventually, but it should not block the compute migration.

---

## 7. Validation strategy

1. **Numerical parity harness.** Run WebGL and WebGPU solvers on the same
   imported model with identical `dt`/steps; compare node positions and the
   reported global error. Define a tolerance (expect WebGPU to be *more*
   accurate due to f32 vs `mediump`).
2. **Per-kernel diffing.** During bring-up, read back each intermediate buffer
   (`normals`, `theta`, `creaseGeo`, `velocity`) and compare against the WebGL
   texture (decoded) for the same inputs.
3. **Visual regression.** Fold the bundled example patterns (e.g. the crane)
   across the full −100%…100% slider range and confirm identical folding and
   strain colouring.
4. **Stability/perf.** Confirm the solver stays stable at the computed `dt`,
   and measure steps/sec vs the WebGL build on a large pattern.

---

## 8. Phased delivery plan

| Phase | Deliverable | Notes |
|-------|-------------|-------|
| **0. Spike** | Minimal `GPUMathWebGPU` proof: init device, one storage buffer, one trivial compute kernel, readback `Float32Array`. | De-risks async init + readback. |
| **1. Engine** | `GPUMathWebGPU.js` with buffer/uniform/kernel/dispatch/readback + ping-pong. Behind a `?webgpu` flag. | Mirrors the `GPUMath` API surface so `dynamicSolver` changes stay small. |
| **2. Data upload** | Port `initTypedArrays`/`updateX` uploads to storage buffers; async `syncNodesAndEdges`. | No physics yet. |
| **3. Kernels** | Translate all kernels to WGSL; wire `solveStep`. Validate per-kernel (§7.2). | Bulk of the work. |
| **4. Readback & integrate** | Replace `packToBytes`/`readPixels` with `mapAsync`; full parity test (§7). Gate animation loop on `solverReady`. | Solver fully on WebGPU. |
| **5. Fallback & cleanup** | Feature-detect `navigator.gpu`; keep WebGL path as fallback **or** show `#noSupportModal`. Remove dead GLSL/`packToBytes`. | Decide fallback policy (see open question). |
| **6. (Optional) WebGPURenderer** | Upgrade three.js; move rendering to WebGPU; share position buffer (zero-copy). | Large, independent; schedule separately. |

A reasonable first PR = Phases 0–1 (engine + spike) so the async/readback model
is proven before the kernel rewrite.

---

## 9. Open questions for the maintainer

1. **Fallback policy.** When WebGPU is unavailable, should we (a) keep the
   WebGL1 solver as a runtime fallback (more code to maintain, widest support),
   or (b) drop WebGL and show the existing "not supported" modal (simpler,
   modern-only)?
2. **Renderer scope.** Is upgrading three.js to `WebGPURenderer` (Phase 6) in
   scope for this effort, or should we keep the r87 `WebGLRenderer` and migrate
   only the compute solver?
3. **Data layout.** Approve the move to flat **storage buffers** (recommended)
   over keeping the 2-D **storage-texture** layout?
4. **Module strategy.** Keep the current global `<script>` loading, or take this
   opportunity to move `js/dynamic/*` to ES modules (cleaner for WGSL imports
   and a future three.js upgrade)?
