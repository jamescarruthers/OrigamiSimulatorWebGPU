// Phase 1 (WebGL -> WebGPU migration): publish modern three.js as a global.
//
// The app and several vendored helper scripts (the custom `SVGLoader`, the
// `TrackballControls` from `dependencies/`) were written against three r87
// loaded as a classic <script> that exposed a global `THREE`. Rather than
// rewrite every `THREE.*` reference at once, we re-publish the modern (r184)
// module namespace as a single *mutable* global object so that:
//   - app code keeps using `THREE.Vector3`, `THREE.Mesh`, ... unchanged, and
//   - the vendored helpers can still attach `THREE.SVGLoader` /
//     `THREE.TrackballControls` onto it.
//
// An ES module namespace object is sealed (its properties can't be extended),
// so we copy it into a plain object the helpers can write to. The copied
// values are the same class references three uses internally, so `instanceof`
// checks and the renderer continue to work.
//
// We import from `three/webgpu`, which bundles the full three core *and* the
// WebGPURenderer (+ node system). WebGPURenderer runs on WebGPU when available
// and automatically falls back to WebGL2 otherwise. Importing the core only
// from here (never also from 'three') keeps a single three instance so
// `instanceof` checks hold.
import * as THREE from 'three/webgpu';

const ThreeGlobal = Object.assign({}, THREE);
window.THREE = ThreeGlobal;

export default ThreeGlobal;
