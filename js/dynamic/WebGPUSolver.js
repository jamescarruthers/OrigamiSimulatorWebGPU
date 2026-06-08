/**
 * WebGPU compute solver (Phase 2 of the WebGL -> WebGPU migration).
 *
 * Re-implements the GPGPU origami solver (originally hand-written WebGL1
 * fragment passes over RGBA-float textures; see the GLSL in index.html and
 * js/dynamic/dynamicSolver.js) as WGSL compute kernels over storage buffers.
 *
 * This module is deliberately self-contained: it builds its own packed data
 * from globals.model and never touches the legacy solver, so it can be enabled
 * behind a flag without regressing the WebGL path. The data-prep logic mirrors
 * dynamicSolver.initTypedArrays exactly, but stores tight linear arrays (length
 * = element count, not padded to a power-of-two texture) since compute kernels
 * index by global_invocation_id with a bounds check, and the stored indirection
 * indices are already linear (so the texture mod/floor address math is gone).
 *
 * Numerics match the legacy solver: displacement-from-original storage, the
 * same beam/crease/face force accumulation, Euler integration. Differences from
 * WebGL1: f32 throughout (vs mediump), dynamic loop bounds (the 100-element-
 * per-node cap is gone), and direct f32 readback via mapAsync (the
 * encode_float/packToBytes hack is deleted).
 *
 * All read-only per-element arrays are concatenated into one `staticData`
 * storage buffer (with per-array offsets in the uniform) because the software
 * WebGPU adapter caps maxStorageBuffersPerShaderStage at 10 and the force
 * kernel would otherwise need 16 separate storage buffers.
 *
 * Only the Euler integration path is implemented so far (the harness default);
 * `syncNodesAndEdges()` throws for verlet until that kernel is ported.
 */

const WORKGROUP_SIZE = 64;

// Params uniform shared by every kernel. All f32 so the std140-ish layout is a
// flat run of vec4s. Offsets are element (vec4) indices into `staticData`.
const PARAMS_WGSL = /* wgsl */ `
struct Params {
  counts : vec4<f32>,   // (numNodes, numFaces, numCreases, numEdges)
  sim    : vec4<f32>,   // (creasePercent, dt, axialStiffness, faceStiffness)
  flags  : vec4<f32>,   // (calcFaceStrain, _, _, _)
  off0   : vec4<f32>,   // (originalPosition, externalForces, mass, meta)
  off1   : vec4<f32>,   // (meta2, beamMeta, creaseMeta, creaseMeta2)
  off2   : vec4<f32>,   // (creaseVectors, nodeCreaseMeta, nodeFaceMeta, faceVertexIndices)
  off3   : vec4<f32>,   // (nominalTriangles, _, _, _)
};
`;

// Static-data accessor + getPos helper (lastPosition is bound separately).
const STATIC_WGSL = /* wgsl */ `
fn sOriginalPosition(i : u32) -> vec4<f32> { return staticData[u32(params.off0.x) + i]; }
fn sExternalForces(i : u32)   -> vec4<f32> { return staticData[u32(params.off0.y) + i]; }
fn sMass(i : u32)             -> vec4<f32> { return staticData[u32(params.off0.z) + i]; }
fn sMeta(i : u32)             -> vec4<f32> { return staticData[u32(params.off0.w) + i]; }
fn sMeta2(i : u32)            -> vec4<f32> { return staticData[u32(params.off1.x) + i]; }
fn sBeamMeta(i : u32)         -> vec4<f32> { return staticData[u32(params.off1.y) + i]; }
fn sCreaseMeta(i : u32)       -> vec4<f32> { return staticData[u32(params.off1.z) + i]; }
fn sCreaseMeta2(i : u32)      -> vec4<f32> { return staticData[u32(params.off1.w) + i]; }
fn sCreaseVectors(i : u32)    -> vec4<f32> { return staticData[u32(params.off2.x) + i]; }
fn sNodeCreaseMeta(i : u32)   -> vec4<f32> { return staticData[u32(params.off2.y) + i]; }
fn sNodeFaceMeta(i : u32)     -> vec4<f32> { return staticData[u32(params.off2.z) + i]; }
fn sFaceVertexIndices(i : u32)-> vec4<f32> { return staticData[u32(params.off2.w) + i]; }
fn sNominalTriangles(i : u32) -> vec4<f32> { return staticData[u32(params.off3.x) + i]; }
fn getPos(i : u32) -> vec3<f32> { return lastPosition[i].xyz + sOriginalPosition(i).xyz; }
`;

const NORMAL_CALC_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<storage, read>       staticData   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       lastPosition : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> normals      : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params       : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.y)) { return; }   // numFaces
  let idx = sFaceVertexIndices(i);
  let a = getPos(u32(idx.x));
  let b = getPos(u32(idx.y));
  let c = getPos(u32(idx.z));
  normals[i] = vec4<f32>(normalize(cross(b - a, c - a)), 0.0);
}`;

const THETA_CALC_WGSL = /* wgsl */ `
${PARAMS_WGSL}
const TWO_PI : f32 = 6.283185307179586;
@group(0) @binding(0) var<storage, read>       staticData   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       lastPosition : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       normals      : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       lastTheta    : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> theta        : array<vec4<f32>>;
@group(0) @binding(5) var<uniform>             params       : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.z)) { return; }   // numCreases
  let lt = lastTheta[i];
  if (lt.z < 0.0) { theta[i] = vec4<f32>(lt.x, 0.0, -1.0, -1.0); return; }
  let normal1 = normals[u32(lt.z)].xyz;
  let normal2 = normals[u32(lt.w)].xyz;
  let dotNormals = clamp(dot(normal1, normal2), -1.0, 1.0);
  let cv = sCreaseVectors(i);
  let node0 = getPos(u32(cv.x));
  let node1 = getPos(u32(cv.y));
  let creaseVector = normalize(node1 - node0);
  let x = dotNormals;
  let y = dot(cross(normal1, creaseVector), normal2);
  let th0 = atan2(y, x);
  var diff = th0 - lt.x;
  if (diff < -5.0) { diff = diff + TWO_PI; } else if (diff > 5.0) { diff = diff - TWO_PI; }
  let th = lt.x + diff;
  theta[i] = vec4<f32>(th, diff, lt.z, lt.w);
}`;

const UPDATE_CREASE_GEO_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<storage, read>       staticData   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       lastPosition : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> creaseGeo    : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params       : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.z)) { return; }   // numCreases
  let cm = sCreaseMeta2(i);
  let node1 = getPos(u32(cm.x));
  let node2 = getPos(u32(cm.y));
  let node3 = getPos(u32(cm.z));
  let node4 = getPos(u32(cm.w));
  let tol = 0.000001;
  var creaseVector = node4 - node3;
  let creaseLength = length(creaseVector);
  if (abs(creaseLength) < tol) { creaseGeo[i] = vec4<f32>(-1.0); return; }
  creaseVector = creaseVector / creaseLength;
  let vector1 = node1 - node3;
  let vector2 = node2 - node3;
  let proj1Length = dot(creaseVector, vector1);
  let proj2Length = dot(creaseVector, vector2);
  let dist1 = sqrt(abs(dot(vector1, vector1) - proj1Length * proj1Length));
  let dist2 = sqrt(abs(dot(vector2, vector2) - proj2Length * proj2Length));
  if (dist1 < tol || dist2 < tol) { creaseGeo[i] = vec4<f32>(-1.0); return; }
  creaseGeo[i] = vec4<f32>(dist1, dist2, proj1Length / creaseLength, proj2Length / creaseLength);
}`;

const VELOCITY_CALC_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<storage, read>       staticData   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       lastPosition : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       lastVelocity : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       normals      : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       theta        : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       creaseGeo    : array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> velocity     : array<vec4<f32>>;
@group(0) @binding(7) var<uniform>             params       : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.x)) { return; }   // numNodes
  let creasePercent = params.sim.x;
  let dt = params.sim.y;
  let faceStiffness = params.sim.w;
  let calcFaceStrain = params.flags.x > 0.5;

  let massI = sMass(i).xy;
  if (massI.y == 1.0) { velocity[i] = vec4<f32>(0.0); return; }   // fixed

  var force = sExternalForces(i).xyz;
  let lastPositionI = lastPosition[i].xyz;
  let lastVelocityI = lastVelocity[i].xyz;
  let originalPositionI = sOriginalPosition(i).xyz;
  let metaI = sMeta(i);
  let meta2I = sMeta2(i).xy;
  var nodeError = 0.0;

  // beams
  let numBeams = u32(metaI.y);
  for (var j = 0u; j < numBeams; j = j + 1u) {
    let bm = sBeamMeta(u32(metaI.x) + j);       // (k, d, length, otherNodeIndex)
    let nIdx = u32(bm.w);
    let nominalDist = sOriginalPosition(nIdx).xyz - originalPositionI;
    var deltaP = lastPosition[nIdx].xyz - lastPositionI + nominalDist;
    let deltaPLength = length(deltaP);
    deltaP = deltaP - deltaP * (bm.z / deltaPLength);
    if (!calcFaceStrain) { nodeError = nodeError + abs(deltaPLength / length(nominalDist) - 1.0); }
    let deltaV = lastVelocity[nIdx].xyz - lastVelocityI;
    force = force + deltaP * bm.x + deltaV * bm.y;
  }
  if (!calcFaceStrain) { nodeError = nodeError / metaI.y; }

  // creases
  let numCreases = u32(metaI.w);
  for (var j = 0u; j < numCreases; j = j + 1u) {
    let ncm = sNodeCreaseMeta(u32(metaI.z) + j); // (creaseIndex, nodeNum, _, _)
    let cIdx = u32(ncm.x);
    let thetas = theta[cIdx];                    // (theta, w, n1Index, n2Index)
    let cMeta = sCreaseMeta(cIdx).xyz;           // (k, d, targetTheta)
    let cGeo = creaseGeo[cIdx];                  // (h1, h2, coef1, coef2)
    if (cGeo.x < 0.0) { continue; }
    let targetTheta = cMeta.z * creasePercent;
    let angForce = cMeta.x * (targetTheta - thetas.x);
    let nodeNum = ncm.y;
    if (nodeNum > 2.0) {                         // node lies on the crease (reaction)
      let normal1 = normals[u32(thetas.z)].xyz;
      let normal2 = normals[u32(thetas.w)].xyz;
      var coef1 = cGeo.z;
      var coef2 = cGeo.w;
      if (nodeNum == 3.0) { coef1 = 1.0 - coef1; coef2 = 1.0 - coef2; }
      force = force - angForce * (coef1 / cGeo.x * normal1 + coef2 / cGeo.y * normal2);
    } else {
      var normalIndex = u32(thetas.z);
      var momentArm = cGeo.x;
      if (nodeNum == 2.0) { normalIndex = u32(thetas.w); momentArm = cGeo.y; }
      force = force + angForce / momentArm * normals[normalIndex].xyz;
    }
  }

  // faces
  let numFaces = u32(meta2I.y);
  for (var j = 0u; j < numFaces; j = j + 1u) {
    let fm = sNodeFaceMeta(u32(meta2I.x) + j);   // (faceIndex, a, b, c)
    let nominalAngles = sNominalTriangles(u32(fm.x)).xyz;
    var faceIndex = 0u;
    if (fm.z < 0.0) { faceIndex = 1u; }
    if (fm.w < 0.0) { faceIndex = 2u; }
    let selfPos = lastPositionI + originalPositionI;
    var a : vec3<f32>; var b : vec3<f32>; var c : vec3<f32>;
    if (faceIndex == 0u) { a = selfPos; } else { a = getPos(u32(fm.y)); }
    if (faceIndex == 1u) { b = selfPos; } else { b = getPos(u32(fm.z)); }
    if (faceIndex == 2u) { c = selfPos; } else { c = getPos(u32(fm.w)); }
    var ab = b - a;
    var ac = c - a;
    var bc = c - b;
    let lengthAB = length(ab);
    let lengthAC = length(ac);
    let lengthBC = length(bc);
    let tol = 0.0000001;
    if (abs(lengthAB) < tol || abs(lengthBC) < tol || abs(lengthAC) < tol) { continue; }
    ab = ab / lengthAB;
    ac = ac / lengthAC;
    bc = bc / lengthBC;
    let angles = vec3<f32>(acos(dot(ab, ac)), acos(-1.0 * dot(ab, bc)), acos(dot(ac, bc)));
    var anglesDiff = (nominalAngles - angles) * faceStiffness;
    let normal = normals[u32(fm.x)].xyz;
    if (faceIndex == 0u) {
      let normalCrossAC = cross(normal, ac) / lengthAC;
      let normalCrossAB = cross(normal, ab) / lengthAB;
      force = force - anglesDiff.x * (normalCrossAC - normalCrossAB);
      if (calcFaceStrain) { nodeError = nodeError + abs((nominalAngles.x - angles.x) / nominalAngles.x); }
      force = force - anglesDiff.y * normalCrossAB;
      force = force + anglesDiff.z * normalCrossAC;
    } else if (faceIndex == 1u) {
      let normalCrossAB = cross(normal, ab) / lengthAB;
      let normalCrossBC = cross(normal, bc) / lengthBC;
      force = force - anglesDiff.x * normalCrossAB;
      force = force + anglesDiff.y * (normalCrossAB + normalCrossBC);
      if (calcFaceStrain) { nodeError = nodeError + abs((nominalAngles.y - angles.y) / nominalAngles.y); }
      force = force - anglesDiff.z * normalCrossBC;
    } else {
      let normalCrossAC = cross(normal, ac) / lengthAC;
      let normalCrossBC = cross(normal, bc) / lengthBC;
      force = force + anglesDiff.x * normalCrossAC;
      force = force - anglesDiff.y * normalCrossBC;
      force = force + anglesDiff.z * (normalCrossBC - normalCrossAC);
      if (calcFaceStrain) { nodeError = nodeError + abs((nominalAngles.z - angles.z) / nominalAngles.z); }
    }
  }
  if (calcFaceStrain) { nodeError = nodeError / meta2I.y; }

  velocity[i] = vec4<f32>(force * dt / massI.x + lastVelocityI, nodeError);
}`;

const POSITION_CALC_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<storage, read>       staticData   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       velocity     : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       lastPosition : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> position     : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params       : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.x)) { return; }   // numNodes
  let lastPositionI = lastPosition[i].xyz;
  if (sMass(i).y == 1.0) { position[i] = vec4<f32>(lastPositionI, 0.0); return; }
  let v = velocity[i];
  position[i] = vec4<f32>(v.xyz * params.sim.y + lastPositionI, v.w);  // dt = sim.y, carry error
}`;

// Verlet position integrator. The force-accumulation block is identical to
// velocityCalc's (beams + creases + faces); kept inline here rather than shared
// to avoid disturbing the validated Euler kernel. Integration differs:
//   nextPosition = force*dt^2/mass + 2*lastPosition - lastLastPosition.
const POSITION_CALC_VERLET_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<storage, read>       staticData       : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       lastPosition     : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       lastVelocity     : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       lastLastPosition : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       normals          : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       theta            : array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       creaseGeo        : array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> position         : array<vec4<f32>>;
@group(0) @binding(8) var<uniform>             params           : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.x)) { return; }   // numNodes
  let creasePercent = params.sim.x;
  let dt = params.sim.y;
  let faceStiffness = params.sim.w;
  let calcFaceStrain = params.flags.x > 0.5;

  let massI = sMass(i).xy;
  if (massI.y == 1.0) { position[i] = vec4<f32>(lastPosition[i].xyz, 0.0); return; }   // fixed

  var force = sExternalForces(i).xyz;
  let lastPositionI = lastPosition[i].xyz;
  let lastVelocityI = lastVelocity[i].xyz;
  let originalPositionI = sOriginalPosition(i).xyz;
  let metaI = sMeta(i);
  let meta2I = sMeta2(i).xy;
  var nodeError = 0.0;

  // beams
  let numBeams = u32(metaI.y);
  for (var j = 0u; j < numBeams; j = j + 1u) {
    let bm = sBeamMeta(u32(metaI.x) + j);
    let nIdx = u32(bm.w);
    let nominalDist = sOriginalPosition(nIdx).xyz - originalPositionI;
    var deltaP = lastPosition[nIdx].xyz - lastPositionI + nominalDist;
    let deltaPLength = length(deltaP);
    deltaP = deltaP - deltaP * (bm.z / deltaPLength);
    if (!calcFaceStrain) { nodeError = nodeError + abs(deltaPLength / length(nominalDist) - 1.0); }
    let deltaV = lastVelocity[nIdx].xyz - lastVelocityI;
    force = force + deltaP * bm.x + deltaV * bm.y;
  }
  if (!calcFaceStrain) { nodeError = nodeError / metaI.y; }

  // creases
  let numCreases = u32(metaI.w);
  for (var j = 0u; j < numCreases; j = j + 1u) {
    let ncm = sNodeCreaseMeta(u32(metaI.z) + j);
    let cIdx = u32(ncm.x);
    let thetas = theta[cIdx];
    let cMeta = sCreaseMeta(cIdx).xyz;
    let cGeo = creaseGeo[cIdx];
    if (cGeo.x < 0.0) { continue; }
    let targetTheta = cMeta.z * creasePercent;
    let angForce = cMeta.x * (targetTheta - thetas.x);
    let nodeNum = ncm.y;
    if (nodeNum > 2.0) {
      let normal1 = normals[u32(thetas.z)].xyz;
      let normal2 = normals[u32(thetas.w)].xyz;
      var coef1 = cGeo.z;
      var coef2 = cGeo.w;
      if (nodeNum == 3.0) { coef1 = 1.0 - coef1; coef2 = 1.0 - coef2; }
      force = force - angForce * (coef1 / cGeo.x * normal1 + coef2 / cGeo.y * normal2);
    } else {
      var normalIndex = u32(thetas.z);
      var momentArm = cGeo.x;
      if (nodeNum == 2.0) { normalIndex = u32(thetas.w); momentArm = cGeo.y; }
      force = force + angForce / momentArm * normals[normalIndex].xyz;
    }
  }

  // faces
  let numFaces = u32(meta2I.y);
  for (var j = 0u; j < numFaces; j = j + 1u) {
    let fm = sNodeFaceMeta(u32(meta2I.x) + j);
    let nominalAngles = sNominalTriangles(u32(fm.x)).xyz;
    var faceIndex = 0u;
    if (fm.z < 0.0) { faceIndex = 1u; }
    if (fm.w < 0.0) { faceIndex = 2u; }
    let selfPos = lastPositionI + originalPositionI;
    var a : vec3<f32>; var b : vec3<f32>; var c : vec3<f32>;
    if (faceIndex == 0u) { a = selfPos; } else { a = getPos(u32(fm.y)); }
    if (faceIndex == 1u) { b = selfPos; } else { b = getPos(u32(fm.z)); }
    if (faceIndex == 2u) { c = selfPos; } else { c = getPos(u32(fm.w)); }
    var ab = b - a;
    var ac = c - a;
    var bc = c - b;
    let lengthAB = length(ab);
    let lengthAC = length(ac);
    let lengthBC = length(bc);
    let tol = 0.0000001;
    if (abs(lengthAB) < tol || abs(lengthBC) < tol || abs(lengthAC) < tol) { continue; }
    ab = ab / lengthAB;
    ac = ac / lengthAC;
    bc = bc / lengthBC;
    let angles = vec3<f32>(acos(dot(ab, ac)), acos(-1.0 * dot(ab, bc)), acos(dot(ac, bc)));
    var anglesDiff = (nominalAngles - angles) * faceStiffness;
    let normal = normals[u32(fm.x)].xyz;
    if (faceIndex == 0u) {
      let normalCrossAC = cross(normal, ac) / lengthAC;
      let normalCrossAB = cross(normal, ab) / lengthAB;
      force = force - anglesDiff.x * (normalCrossAC - normalCrossAB);
      if (calcFaceStrain) { nodeError = nodeError + abs((nominalAngles.x - angles.x) / nominalAngles.x); }
      force = force - anglesDiff.y * normalCrossAB;
      force = force + anglesDiff.z * normalCrossAC;
    } else if (faceIndex == 1u) {
      let normalCrossAB = cross(normal, ab) / lengthAB;
      let normalCrossBC = cross(normal, bc) / lengthBC;
      force = force - anglesDiff.x * normalCrossAB;
      force = force + anglesDiff.y * (normalCrossAB + normalCrossBC);
      if (calcFaceStrain) { nodeError = nodeError + abs((nominalAngles.y - angles.y) / nominalAngles.y); }
      force = force - anglesDiff.z * normalCrossBC;
    } else {
      let normalCrossAC = cross(normal, ac) / lengthAC;
      let normalCrossBC = cross(normal, bc) / lengthBC;
      force = force + anglesDiff.x * normalCrossAC;
      force = force - anglesDiff.y * normalCrossBC;
      force = force + anglesDiff.z * (normalCrossBC - normalCrossAC);
      if (calcFaceStrain) { nodeError = nodeError + abs((nominalAngles.z - angles.z) / nominalAngles.z); }
    }
  }
  if (calcFaceStrain) { nodeError = nodeError / meta2I.y; }

  let nextPosition = force * dt * dt / massI.x + 2.0 * lastPositionI - lastLastPosition[i].xyz;
  position[i] = vec4<f32>(nextPosition, nodeError);
}`;

const VELOCITY_CALC_VERLET_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<storage, read>       staticData   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       position     : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       lastPosition : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocity     : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params       : Params;
${STATIC_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.counts.x)) { return; }   // numNodes
  if (sMass(i).y == 1.0) { velocity[i] = vec4<f32>(0.0); return; }
  velocity[i] = vec4<f32>((position[i].xyz - lastPosition[i].xyz) / params.sim.y, 0.0);
}`;

// ---- module -----------------------------------------------------------

export function initWebGPUSolver(globals) {

  let device = null;
  let nodes, edges, faces, creases;
  let positions, colors;

  // packed per-element data (tight, vec4-per-element)
  const arr = {};
  let staticData;             // concatenated read-only data
  const off = {};             // element offsets into staticData
  let thetaInitArr;
  let counts;                 // element counts
  let dt = 0;

  const buf = {};
  let paramsBuf = null;
  let readBuf = null;
  let lastPos, curPos, lastVel, curVel, lastTheta, curTheta;
  let lastLastPos;   // verlet only (position two steps ago)
  const pipe = {};

  function vec4Array(numElements) { return new Float32Array(Math.max(numElements, 1) * 4); }

  function buildPackedData() {
    nodes = globals.model.getNodes();
    edges = globals.model.getEdges();
    faces = globals.model.getFaces();
    creases = globals.model.getCreases();
    positions = globals.model.getPositionsArray();
    colors = globals.model.getColorsArray();

    const numNodes = nodes.length;
    const numFaces = faces.length;
    const numCreases = creases.length;

    const nodeFaces = [];
    let numNodeFaces = 0;
    for (let i = 0; i < numNodes; i++) {
      nodeFaces.push([]);
      for (let j = 0; j < faces.length; j++) {
        if (faces[j].indexOf(i) >= 0) { nodeFaces[i].push(j); numNodeFaces++; }
      }
    }
    let numEdges = 0;
    for (let i = 0; i < numNodes; i++) numEdges += nodes[i].numBeams();
    let numNodeCreases = 0;
    for (let i = 0; i < numNodes; i++) numNodeCreases += nodes[i].numCreases();
    numNodeCreases += numCreases * 2;

    counts = { numNodes, numFaces, numCreases, numEdges, numNodeCreases, numNodeFaces };

    arr.originalPosition = vec4Array(numNodes);
    arr.externalForces = vec4Array(numNodes);
    arr.mass = vec4Array(numNodes);
    arr.meta = vec4Array(numNodes);
    arr.meta2 = vec4Array(numNodes);
    arr.beamMeta = vec4Array(numEdges);
    arr.creaseMeta = vec4Array(numCreases);
    arr.creaseMeta2 = vec4Array(numCreases);
    arr.creaseVectors = vec4Array(numCreases);
    arr.nodeCreaseMeta = vec4Array(numNodeCreases);
    arr.nodeFaceMeta = vec4Array(numNodeFaces);
    arr.faceVertexIndices = vec4Array(numFaces);
    arr.nominalTriangles = vec4Array(numFaces);
    thetaInitArr = vec4Array(numCreases);

    for (let i = 0; i < faces.length; i++) {
      const face = faces[i];
      arr.faceVertexIndices[4 * i] = face[0];
      arr.faceVertexIndices[4 * i + 1] = face[1];
      arr.faceVertexIndices[4 * i + 2] = face[2];
      const a = nodes[face[0]].getOriginalPosition();
      const b = nodes[face[1]].getOriginalPosition();
      const c = nodes[face[2]].getOriginalPosition();
      const ab = b.clone().sub(a).normalize();
      const ac = c.clone().sub(a).normalize();
      const bc = c.clone().sub(b).normalize();
      arr.nominalTriangles[4 * i] = Math.acos(ab.dot(ac));
      arr.nominalTriangles[4 * i + 1] = Math.acos(-1 * ab.dot(bc));
      arr.nominalTriangles[4 * i + 2] = Math.acos(ac.dot(bc));
    }

    for (let i = 0; i < numCreases; i++) {
      thetaInitArr[i * 4 + 2] = creases[i].getNormal1Index();
      thetaInitArr[i * 4 + 3] = creases[i].getNormal2Index();
    }

    let index = 0;
    for (let i = 0; i < numNodes; i++) {
      arr.meta2[4 * i] = index;
      const num = nodeFaces[i].length;
      arr.meta2[4 * i + 1] = num;
      for (let j = 0; j < num; j++) {
        const _index = (index + j) * 4;
        const face = faces[nodeFaces[i][j]];
        arr.nodeFaceMeta[_index] = nodeFaces[i][j];
        arr.nodeFaceMeta[_index + 1] = face[0] === i ? -1 : face[0];
        arr.nodeFaceMeta[_index + 2] = face[1] === i ? -1 : face[1];
        arr.nodeFaceMeta[_index + 3] = face[2] === i ? -1 : face[2];
      }
      index += num;
    }

    index = 0;
    for (let i = 0; i < numNodes; i++) {
      arr.mass[4 * i] = nodes[i].getSimMass();
      arr.mass[4 * i + 1] = nodes[i].isFixed() ? 1 : 0;
      arr.meta[i * 4 + 2] = index;
      const nodeCreases = nodes[i].creases;
      const nodeInvCreases = nodes[i].invCreases;
      arr.meta[i * 4 + 3] = nodeCreases.length + nodeInvCreases.length;
      for (let j = 0; j < nodeCreases.length; j++) {
        arr.nodeCreaseMeta[index * 4] = nodeCreases[j].getIndex();
        arr.nodeCreaseMeta[index * 4 + 1] = nodeCreases[j].getNodeIndex(nodes[i]);
        index++;
      }
      for (let j = 0; j < nodeInvCreases.length; j++) {
        arr.nodeCreaseMeta[index * 4] = nodeInvCreases[j].getIndex();
        arr.nodeCreaseMeta[index * 4 + 1] = nodeInvCreases[j].getNodeIndex(nodes[i]);
        index++;
      }
    }

    index = 0;
    for (let i = 0; i < numNodes; i++) {
      arr.meta[4 * i] = index;
      arr.meta[4 * i + 1] = nodes[i].numBeams();
      for (let j = 0; j < nodes[i].beams.length; j++) {
        const beam = nodes[i].beams[j];
        arr.beamMeta[4 * index] = beam.getK();
        arr.beamMeta[4 * index + 1] = beam.getD();
        arr.beamMeta[4 * index + 2] = beam.getLength();
        arr.beamMeta[4 * index + 3] = beam.getOtherNode(nodes[i]).getIndex();
        index += 1;
      }
    }

    for (let i = 0; i < numNodes; i++) {
      const op = nodes[i].getOriginalPosition();
      arr.originalPosition[4 * i] = op.x;
      arr.originalPosition[4 * i + 1] = op.y;
      arr.originalPosition[4 * i + 2] = op.z;
      const ef = nodes[i].getExternalForce();
      arr.externalForces[4 * i] = ef.x;
      arr.externalForces[4 * i + 1] = ef.y;
      arr.externalForces[4 * i + 2] = ef.z;
    }

    for (let i = 0; i < numCreases; i++) {
      const crease = creases[i];
      arr.creaseMeta[i * 4] = crease.getK();
      arr.creaseMeta[i * 4 + 2] = crease.getTargetTheta();
      arr.creaseMeta2[i * 4] = crease.node1.getIndex();
      arr.creaseMeta2[i * 4 + 1] = crease.node2.getIndex();
      arr.creaseMeta2[i * 4 + 2] = crease.edge.nodes[0].getIndex();
      arr.creaseMeta2[i * 4 + 3] = crease.edge.nodes[1].getIndex();
      arr.creaseVectors[i * 4] = crease.edge.nodes[0].getIndex();
      arr.creaseVectors[i * 4 + 1] = crease.edge.nodes[1].getIndex();
    }

    // concatenate static arrays (in declared order) and record vec4 offsets
    const order = ['originalPosition', 'externalForces', 'mass', 'meta', 'meta2',
      'beamMeta', 'creaseMeta', 'creaseMeta2', 'creaseVectors', 'nodeCreaseMeta',
      'nodeFaceMeta', 'faceVertexIndices', 'nominalTriangles'];
    let total = 0;
    for (const name of order) { off[name] = total / 4; total += arr[name].length; }
    staticData = new Float32Array(total);
    let cursor = 0;
    for (const name of order) { staticData.set(arr[name], cursor); cursor += arr[name].length; }

    dt = calcDt();
  }

  function calcDt() {
    let maxFreqNat = 0;
    for (let i = 0; i < edges.length; i++) {
      const f = edges[i].getNaturalFrequency();
      if (f > maxFreqNat) maxFreqNat = f;
    }
    return (1 / (2 * Math.PI * maxFreqNat)) * 0.9;
  }

  function storageBuffer(data) {
    const b = device.createBuffer({
      size: Math.max(data.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(b, 0, data);
    return b;
  }

  function createBuffers() {
    const n = counts.numNodes, f = counts.numFaces, c = counts.numCreases;
    buf.staticData = storageBuffer(staticData);
    buf.normals = storageBuffer(vec4Array(f));
    buf.creaseGeo = storageBuffer(vec4Array(c));
    buf.posA = storageBuffer(vec4Array(n));
    buf.posB = storageBuffer(vec4Array(n));
    buf.posC = storageBuffer(vec4Array(n));   // verlet needs a third position buffer
    buf.velA = storageBuffer(vec4Array(n));
    buf.velB = storageBuffer(vec4Array(n));
    buf.thetaA = storageBuffer(thetaInitArr);
    buf.thetaB = storageBuffer(thetaInitArr);

    readBuf = device.createBuffer({
      size: Math.max(n * 16, 16),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    paramsBuf = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeParams();
    resetPingPong();
  }

  function writeParams() {
    const p = new Float32Array(28);   // 7 vec4
    p[0] = counts.numNodes; p[1] = counts.numFaces; p[2] = counts.numCreases; p[3] = counts.numEdges;
    p[4] = globals.creasePercent; p[5] = dt; p[6] = globals.axialStiffness; p[7] = globals.faceStiffness;
    p[8] = globals.calcFaceStrain ? 1 : 0;
    p[12] = off.originalPosition; p[13] = off.externalForces; p[14] = off.mass; p[15] = off.meta;
    p[16] = off.meta2; p[17] = off.beamMeta; p[18] = off.creaseMeta; p[19] = off.creaseMeta2;
    p[20] = off.creaseVectors; p[21] = off.nodeCreaseMeta; p[22] = off.nodeFaceMeta; p[23] = off.faceVertexIndices;
    p[24] = off.nominalTriangles;
    device.queue.writeBuffer(paramsBuf, 0, p);
  }

  function resetPingPong() {
    lastPos = buf.posA; curPos = buf.posB; lastLastPos = buf.posC;
    lastVel = buf.velA; curVel = buf.velB;
    lastTheta = buf.thetaA; curTheta = buf.thetaB;
  }

  function makePipeline(code) {
    const module = device.createShaderModule({ code });
    return device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }

  function createPipelines() {
    pipe.normalCalc = makePipeline(NORMAL_CALC_WGSL);
    pipe.thetaCalc = makePipeline(THETA_CALC_WGSL);
    pipe.updateCreaseGeo = makePipeline(UPDATE_CREASE_GEO_WGSL);
    pipe.velocityCalc = makePipeline(VELOCITY_CALC_WGSL);
    pipe.positionCalc = makePipeline(POSITION_CALC_WGSL);
    pipe.positionCalcVerlet = makePipeline(POSITION_CALC_VERLET_WGSL);
    pipe.velocityCalcVerlet = makePipeline(VELOCITY_CALC_VERLET_WGSL);
  }

  function dispatch(encoder, pipeline, buffers, count) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    }));
    pass.dispatchWorkgroups(Math.ceil(Math.max(count, 1) / WORKGROUP_SIZE));
    pass.end();
  }

  function encodeStep(encoder) {
    const sd = buf.staticData;
    dispatch(encoder, pipe.normalCalc, [sd, lastPos, buf.normals, paramsBuf], counts.numFaces);
    dispatch(encoder, pipe.thetaCalc, [sd, lastPos, buf.normals, lastTheta, curTheta, paramsBuf], counts.numCreases);
    dispatch(encoder, pipe.updateCreaseGeo, [sd, lastPos, buf.creaseGeo, paramsBuf], counts.numCreases);
    dispatch(encoder, pipe.velocityCalc, [sd, lastPos, lastVel, buf.normals, curTheta, buf.creaseGeo, curVel, paramsBuf], counts.numNodes);
    dispatch(encoder, pipe.positionCalc, [sd, curVel, lastPos, curPos, paramsBuf], counts.numNodes);
    let t;
    t = lastPos; lastPos = curPos; curPos = t;
    t = lastVel; lastVel = curVel; curVel = t;
    t = lastTheta; lastTheta = curTheta; curTheta = t;
  }

  function encodeStepVerlet(encoder) {
    const sd = buf.staticData;
    dispatch(encoder, pipe.normalCalc, [sd, lastPos, buf.normals, paramsBuf], counts.numFaces);
    dispatch(encoder, pipe.thetaCalc, [sd, lastPos, buf.normals, lastTheta, curTheta, paramsBuf], counts.numCreases);
    dispatch(encoder, pipe.updateCreaseGeo, [sd, lastPos, buf.creaseGeo, paramsBuf], counts.numCreases);
    dispatch(encoder, pipe.positionCalcVerlet,
      [sd, lastPos, lastVel, lastLastPos, buf.normals, curTheta, buf.creaseGeo, curPos, paramsBuf], counts.numNodes);
    dispatch(encoder, pipe.velocityCalcVerlet, [sd, curPos, lastPos, curVel, paramsBuf], counts.numNodes);
    // Verlet swaps (mirror dynamicSolver.solveStep): rotate the three position
    // buffers so lastPos<-curPos and lastLastPos<-(old lastPos), plus the usual
    // velocity/theta swaps.
    let t;
    t = lastPos; lastPos = lastLastPos; lastLastPos = t;   // swap(lastPos, lastLastPos)
    t = curPos; curPos = lastPos; lastPos = t;             // swap(curPos, lastPos)
    t = lastVel; lastVel = curVel; curVel = t;
    t = lastTheta; lastTheta = curTheta; curTheta = t;
  }

  // ---- public interface ----

  async function init() {
    if (device) return true;
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (e) => {
      console.error('[WebGPUSolver] uncaptured error: ' + e.error.message);
    });
    return true;
  }

  function syncNodesAndEdges() {
    if (!device) throw new Error('WebGPUSolver.init() must be awaited first');
    buildPackedData();
    createBuffers();
    createPipelines();
  }

  function reset() {
    device.queue.writeBuffer(buf.posA, 0, vec4Array(counts.numNodes));
    device.queue.writeBuffer(buf.posB, 0, vec4Array(counts.numNodes));
    device.queue.writeBuffer(buf.posC, 0, vec4Array(counts.numNodes));
    device.queue.writeBuffer(buf.velA, 0, vec4Array(counts.numNodes));
    device.queue.writeBuffer(buf.velB, 0, vec4Array(counts.numNodes));
    device.queue.writeBuffer(buf.thetaA, 0, thetaInitArr);
    device.queue.writeBuffer(buf.thetaB, 0, thetaInitArr);
    resetPingPong();
  }

  function solve(numSteps) {
    if (numSteps === undefined) numSteps = globals.numSteps;
    writeParams();
    const verlet = globals.integrationType === 'verlet';
    const encoder = device.createCommandEncoder();
    for (let s = 0; s < numSteps; s++) {
      if (verlet) encodeStepVerlet(encoder); else encodeStep(encoder);
    }
    device.queue.submit([encoder.finish()]);
  }

  let readbackInFlight = false;
  let lastGlobalError = 0;
  async function readback() {
    // Self-throttle: the staging buffer can only be mapped once at a time, so a
    // readback issued while a previous one is still mapping is skipped (the live
    // loop keeps advancing the sim on the GPU regardless).
    if (readbackInFlight) return lastGlobalError;
    readbackInFlight = true;
    const n = counts.numNodes;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(lastPos, 0, readBuf, 0, n * 16);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ, 0, n * 16);
    const data = new Float32Array(readBuf.getMappedRange(0, n * 16).slice(0));
    readBuf.unmap();

    let globalError = 0;
    const shouldUpdateColors = globals.colorMode === 'axialStrain';
    for (let i = 0; i < n; i++) {
      const k = i * 4;
      globalError += data[k + 3] * 100;
      positions[3 * i] = data[k] + nodes[i]._originalPosition.x;
      positions[3 * i + 1] = data[k + 1] + nodes[i]._originalPosition.y;
      positions[3 * i + 2] = data[k + 2] + nodes[i]._originalPosition.z;
      if (shouldUpdateColors) {
        let e = data[k + 3] * 100;
        if (e > globals.strainClip) e = globals.strainClip;
        const color = new THREE.Color();
        color.setHSL((1 - e / globals.strainClip) * 0.7, 1, 0.5);
        colors[3 * i] = color.r; colors[3 * i + 1] = color.g; colors[3 * i + 2] = color.b;
      }
    }
    const avg = globalError / n;
    const el = document.getElementById('globalError');
    if (el) el.innerHTML = avg.toFixed(7) + ' %';
    lastGlobalError = avg;
    readbackInFlight = false;
    return avg;
  }

  // Debug aid: read back N vec4 elements from a named internal buffer.
  async function debugRead(name, count) {
    const map = {
      normals: buf.normals, creaseGeo: buf.creaseGeo, staticData: buf.staticData,
      lastPos: lastPos, lastVel: lastVel, lastTheta: lastTheta, curTheta: curTheta,
    };
    const b = map[name];
    const size = Math.max(count, 1) * 16;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(b, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = Array.from(new Float32Array(staging.getMappedRange().slice(0)));
    staging.unmap();
    staging.destroy();
    return out;
  }

  return { init, syncNodesAndEdges, reset, solve, readback, debugRead, getCounts: () => counts, isWebGPUSolver: true };
}
