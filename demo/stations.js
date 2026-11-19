// stations.js — the two layers (u, s) as fields of glowing instanced neurons, repeated over the NL layers of the
// unrolled chain. Folded, only the live pair is visible (the s station at the front of the ring, u at the back).
//   solid sphere : colour / brightness = the unit's value in [-1, 1], displaced out of the panel by value
//   TEAL ring    : the ghost twin at the PREDICTED value (û(s) for u, ŝ(u) for s)
//   ROSE segment : the gap between the two = the prediction error; it glows with log |error|
// Neuron placement inside a layer is a layout choice (the network is permutation-invariant inside a layer):
// s units are grouped into a 4 x 4 arrangement of blocks by the board cell their readout weights V favour most,
// u units by the s block they feed most strongly through W2.
import * as THREE from 'three';
import { CHAIN_GLSL, chainUniforms, NL, PANEL_Y } from './chain.js';
import { panelFrameJS } from './chain.js';
import { COLORS, FONTS, FOG_PARS_F, sharedUniforms, makeLabel } from './world.js';
import { vodeInfo } from './sim.js';

const PITCH = 0.33;
const BLOCK = 1.6;            // 4 units x 0.33 + a 0.28 gap between the 4 x 4 blocks
const RADIUS = 0.105;
const DISP = 0.32;            // displacement (world units) of a unit at value +/-1 (kept small: legible from afar)

function greedyBalanced(score, n, groups, cap) {
  // score: Float32Array(n*groups); returns group per unit and the per-group ordered member list
  const pairs = new Array(n * groups);
  for (let i = 0; i < n; i++) for (let p = 0; p < groups; p++) pairs[i * groups + p] = i * groups + p;
  pairs.sort((a, b) => score[b] - score[a]);
  const grp = new Int16Array(n).fill(-1);
  const fill = new Int16Array(groups);
  const members = Array.from({ length: groups }, () => []);
  for (const k of pairs) {
    const i = (k / groups) | 0, p = k % groups;
    if (grp[i] >= 0 || fill[p] >= cap) continue;
    grp[i] = p; fill[p]++; members[p].push(i);
  }
  for (let i = 0; i < n; i++) if (grp[i] < 0) {           // safety net (cap * groups >= n always here)
    let p = 0; while (fill[p] >= cap) p++;
    grp[i] = p; fill[p]++; members[p].push(i);
  }
  return { grp, members };
}

/** layout: for each unit i -> panel coords (x, y) and block id; computed from the real weights when available */
export function computeLayouts(sim) {
  const H = sim.dims.H, P = 16, cap = Math.ceil(H / P);
  const W = sim.weights();
  const fallback = () => {
    const grp = new Int16Array(H), members = Array.from({ length: P }, () => []);
    for (let i = 0; i < H; i++) { const p = Math.min(P - 1, (i / cap) | 0); grp[i] = p; members[p].push(i); }
    return { grp, members };
  };
  let sAssign = fallback(), uAssign = fallback();
  try {
    if (W.V && W.V.shape[0] === H && W.V.shape[1] === P * 4) {
      const sc = new Float32Array(H * P), V = W.V.data, K = W.V.shape[1];
      for (let i = 0; i < H; i++) for (let p = 0; p < P; p++) {
        let s = 0; for (let d = 0; d < 4; d++) s += Math.abs(V[i * K + p * 4 + d]);
        sc[i * P + p] = s;
      }
      sAssign = greedyBalanced(sc, H, P, cap);
    }
    if (W.W2 && W.W2.shape[0] === H && W.W2.shape[1] === H) {
      const sc = new Float32Array(H * P), M = W.W2.data;
      for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) sc[i * P + sAssign.grp[j]] += Math.abs(M[i * H + j]);
      uAssign = greedyBalanced(sc, H, P, cap);
    }
  } catch (e) { console.warn('[stations] layout from weights failed, using index order', e); }
  const toGrid = (assign) => {
    const grid = new Float32Array(H * 2), block = new Float32Array(H);
    for (let p = 0; p < P; p++) {
      const br = p >> 2, bc = p & 3;
      assign.members[p].forEach((i, q) => {
        const qr = (q >> 2) & 3, qc = q & 3;
        grid[i * 2] = (bc - 1.5) * BLOCK + (qc - 1.5) * PITCH;
        grid[i * 2 + 1] = (1.5 - br) * BLOCK + (1.5 - qr) * PITCH;
        block[i] = p;
      });
    }
    return { grid, block };
  };
  return { s: toGrid(sAssign), u: toGrid(uAssign), fromWeights: !!(W.V && W.W2) };
}

export function createStations(world, sim) {
  const H = sim.dims.H;
  const count = NL * H;
  const layouts = computeLayouts(sim);
  const vodeOf = (l) => (l % 2 === 0 ? 's' : 'u');

  // ---------------------------------------------------------------- shared per-instance attributes
  const aLayer = new Float32Array(count), aGrid = new Float32Array(count * 2), aBlock = new Float32Array(count);
  for (let l = 0; l < NL; l++) {
    const lay = layouts[vodeOf(l)];
    for (let i = 0; i < H; i++) {
      const k = l * H + i;
      aLayer[k] = l;
      aGrid[k * 2] = lay.grid[i * 2]; aGrid[k * 2 + 1] = lay.grid[i * 2 + 1];
      aBlock[k] = lay.block[i];
    }
  }
  const attrLayer = new THREE.InstancedBufferAttribute(aLayer, 1);
  const attrGrid = new THREE.InstancedBufferAttribute(aGrid, 2);
  const attrBlock = new THREE.InstancedBufferAttribute(aBlock, 1);
  const attrVal = new THREE.InstancedBufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
  const attrPred = new THREE.InstancedBufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);

  const local = {
    uSize: { value: RADIUS },
    uDisp: { value: DISP },
    uLiveS: { value: 0 },
    uLiveU: { value: 1 },
    uHiBlock: { value: -1 },
    uShowPred: { value: 1 },
    uShowErr: { value: 1 },
    uPos: { value: new THREE.Color(COLORS.ink).multiplyScalar(0.85) },
    uNeg: { value: new THREE.Color(COLORS.steel) },
    uZero: { value: new THREE.Color('#1A2333') },
    uTeal: { value: new THREE.Color(COLORS.teal) },
    uRose: { value: new THREE.Color(COLORS.rose) },
    uGreen: { value: new THREE.Color(COLORS.green) },
    uMuted: { value: new THREE.Color(COLORS.muted) },
    uViolet: { value: new THREE.Color(COLORS.violet) },
    uEq: { value: 0 },
  };
  const U = { ...chainUniforms, ...sharedUniforms, ...local };
  const group = new THREE.Group();
  group.name = 'stations';

  const LIVE_GLSL = /* glsl */ `
    uniform float uLiveS, uLiveU;
    float isLive(float l) { return (abs(l - uLiveS) < 0.5 || abs(l - uLiveU) < 0.5) ? 1.0 : 0.0; }
  `;

  // ---------------------------------------------------------------- solids
  const sphere = new THREE.IcosahedronGeometry(1, world.state.mobile ? 0 : 1);
  const solidGeo = new THREE.InstancedBufferGeometry();
  solidGeo.index = sphere.index;
  solidGeo.setAttribute('position', sphere.getAttribute('position'));
  solidGeo.setAttribute('normal', sphere.getAttribute('normal'));
  solidGeo.setAttribute('aLayer', attrLayer);
  solidGeo.setAttribute('aGrid', attrGrid);
  solidGeo.setAttribute('aBlock', attrBlock);
  solidGeo.setAttribute('aVal', attrVal);
  solidGeo.instanceCount = count;
  const solidMat = new THREE.ShaderMaterial({
    defines: { NL }, uniforms: U, transparent: true,
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      uniform float uSize, uDisp, uHiBlock;
      uniform vec3 uPos, uNeg, uZero;
      attribute float aLayer; attribute vec2 aGrid; attribute float aBlock; attribute float aVal;
      varying vec3 vColor; varying float vAlpha; varying vec3 vN; varying float vFogDepth;
      void main() {
        vec3 A, N;
        float v = clamp(aVal, -1.0, 1.0);
        vec3 c = panelPoint(aLayer, aGrid, v * uDisp, A, N);
        int li = int(aLayer + 0.5);
        float alpha = uLayerAlpha[li];
        float lit = uLayerLit[li];
        float m = abs(v);
        float r = uSize * (0.62 + 0.38 * m) * step(0.002, alpha);
        vec4 mv = modelViewMatrix * vec4(c + position * r, 1.0);
        gl_Position = projectionMatrix * mv;
        vFogDepth = -mv.z;
        vN = normalize(normalMatrix * normal);
        vec3 base = mix(uZero, v < 0.0 ? uNeg : uPos, pow(m, 0.8));
        float hi = (uHiBlock >= 0.0 && abs(aBlock - uHiBlock) < 0.5) ? 1.0 : 0.0;
        vColor = base * (1.0 + 0.8 * lit) + hi * vec3(0.20, 0.16, 0.34);
        vAlpha = alpha;
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vColor; varying float vAlpha; varying vec3 vN;
      ${FOG_PARS_F}
      void main() {
        if (vAlpha < 0.002) discard;
        vec3 n = normalize(vN);
        float lam = 0.5 + 0.5 * max(dot(n, normalize(vec3(0.35, 0.8, 0.5))), 0.0);
        float rim = pow(1.0 - max(n.z, 0.0), 2.5);
        vec3 col = vColor * lam + vColor * rim * 0.3;
        col = mix(col, uFogColor, fogFactor());
        gl_FragColor = vec4(col, vAlpha);
        #include <colorspace_fragment>
      }`,
  });
  const solids = new THREE.Mesh(solidGeo, solidMat);
  solids.frustumCulled = false;
  solids.renderOrder = 3;
  group.add(solids);

  // ---------------------------------------------------------------- ghosts (TEAL rings at the predicted value)
  const pointGeo = (() => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    g.setAttribute('aLayer', attrLayer);
    g.setAttribute('aGrid', attrGrid);
    g.setAttribute('aVal', attrVal);
    g.setAttribute('aPred', attrPred);
    g.instanceCount = count;
    return g;
  })();
  const ghostMat = new THREE.ShaderMaterial({
    defines: { NL }, uniforms: U, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      ${LIVE_GLSL}
      uniform float uSize, uDisp, uProjScale, uShowPred;
      attribute float aLayer; attribute vec2 aGrid; attribute float aVal; attribute float aPred;
      varying float vA; varying float vFogDepth;
      void main() {
        vec3 A, N;
        float pv = clamp(aPred, -1.0, 1.0);
        vec3 c = panelPoint(aLayer, aGrid, pv * uDisp, A, N) + position;
        vec4 mv = modelViewMatrix * vec4(c, 1.0);
        gl_Position = projectionMatrix * mv;
        vFogDepth = -mv.z;
        float near = 1.0 - smoothstep(24.0, 44.0, -mv.z);      // level of detail: ghosts appear as you fly closer
        float a = uLayerAlpha[int(aLayer + 0.5)] * isLive(aLayer) * uShowPred * near;
        vA = a;
        gl_PointSize = a > 0.002 ? uSize * 2.8 * uProjScale / max(0.5, -mv.z) : 0.0;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uTeal;
      varying float vA;
      ${FOG_PARS_F}
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float ring = smoothstep(0.70, 0.78, d) * (1.0 - smoothstep(0.88, 0.98, d));
        float a = ring * vA * 0.9 * (1.0 - fogFactor());
        if (a < 0.004) discard;
        gl_FragColor = vec4(uTeal * a * 1.25, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const ghosts = new THREE.Points(pointGeo, ghostMat);
  ghosts.frustumCulled = false;
  ghosts.renderOrder = 4;
  group.add(ghosts);

  // ---------------------------------------------------------------- error gaps (ROSE segments + halo)
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true);
  const gapGeo = new THREE.InstancedBufferGeometry();
  gapGeo.index = cyl.index;
  gapGeo.setAttribute('position', cyl.getAttribute('position'));
  gapGeo.setAttribute('aLayer', attrLayer);
  gapGeo.setAttribute('aGrid', attrGrid);
  gapGeo.setAttribute('aVal', attrVal);
  gapGeo.setAttribute('aPred', attrPred);
  gapGeo.instanceCount = count;
  const ERR_GLSL = /* glsl */ `
    float errGlow(float e) { return clamp((log(max(abs(e), 1e-9)) / log(10.0) + 3.6) / 3.4, 0.0, 1.0); }
  `;
  const gapMat = new THREE.ShaderMaterial({
    defines: { NL }, uniforms: U, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      ${LIVE_GLSL}
      ${ERR_GLSL}
      uniform float uDisp, uShowErr;
      attribute float aLayer; attribute vec2 aGrid; attribute float aVal; attribute float aPred;
      varying float vA; varying float vFogDepth;
      void main() {
        vec3 A, N;
        float v = clamp(aVal, -1.0, 1.0), pv = clamp(aPred, -1.0, 1.0);
        vec3 p0 = panelPoint(aLayer, aGrid, v * uDisp, A, N);
        vec3 p1 = p0 + N * (pv - v) * uDisp;
        float g = errGlow(aVal - aPred);
        float near = 1.0 - smoothstep(24.0, 44.0, -(modelViewMatrix * vec4(p0, 1.0)).z);
        float a = uLayerAlpha[int(aLayer + 0.5)] * isLive(aLayer) * uShowErr * g * near;
        float rad = (0.012 + 0.025 * g) * step(0.002, a);
        vec3 p = mix(p0, p1, position.y + 0.5) + (A * position.x + vec3(0.0, 1.0, 0.0) * position.z) * rad;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        vFogDepth = -mv.z;
        vA = a;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uRose;
      varying float vA;
      ${FOG_PARS_F}
      void main() {
        float a = vA * (1.0 - fogFactor());
        if (a < 0.004) discard;
        gl_FragColor = vec4(uRose * a * 2.2, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const gaps = new THREE.Mesh(gapGeo, gapMat);
  gaps.frustumCulled = false;
  gaps.renderOrder = 4;
  group.add(gaps);

  const haloMat = new THREE.ShaderMaterial({
    defines: { NL }, uniforms: U, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      ${LIVE_GLSL}
      ${ERR_GLSL}
      uniform float uSize, uDisp, uProjScale, uShowErr;
      attribute float aLayer; attribute vec2 aGrid; attribute float aVal; attribute float aPred;
      varying float vA; varying float vFogDepth;
      void main() {
        vec3 A, N;
        float v = clamp(aVal, -1.0, 1.0), pv = clamp(aPred, -1.0, 1.0);
        vec3 c = panelPoint(aLayer, aGrid, 0.5 * (v + pv) * uDisp, A, N) + position;
        vec4 mv = modelViewMatrix * vec4(c, 1.0);
        gl_Position = projectionMatrix * mv;
        vFogDepth = -mv.z;
        float g = errGlow(aVal - aPred);
        float near = 1.0 - smoothstep(24.0, 44.0, -mv.z);
        float a = uLayerAlpha[int(aLayer + 0.5)] * isLive(aLayer) * uShowErr * g * (0.35 + 0.65 * near);
        vA = a;
        gl_PointSize = a > 0.01 ? uSize * (2.2 + 4.0 * g) * uProjScale / max(0.5, -mv.z) : 0.0;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uRose;
      varying float vA;
      ${FOG_PARS_F}
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float a = exp(-d * d * 4.0) * vA * 0.24 * (1.0 - fogFactor());
        if (a < 0.003) discard;
        gl_FragColor = vec4(uRose * a, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const halos = new THREE.Points(pointGeo, haloMat);
  halos.frustumCulled = false;
  halos.renderOrder = 4;
  group.add(halos);

  // ---------------------------------------------------------------- panel plates (zero plane + 4x4 block grid)
  const plane = new THREE.PlaneGeometry(1, 1);
  const plateGeo = new THREE.InstancedBufferGeometry();
  plateGeo.index = plane.index;
  plateGeo.setAttribute('position', plane.getAttribute('position'));
  plateGeo.setAttribute('uv', plane.getAttribute('uv'));
  const plateLayer = new Float32Array(NL);
  for (let l = 0; l < NL; l++) plateLayer[l] = l;
  plateGeo.setAttribute('aLayer', new THREE.InstancedBufferAttribute(plateLayer, 1));
  plateGeo.instanceCount = NL;
  const plateMat = new THREE.ShaderMaterial({
    defines: { NL }, uniforms: { ...U, uBlock: { value: BLOCK } }, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      ${LIVE_GLSL}
      uniform float uBlock;
      attribute float aLayer;
      varying vec2 vG; varying float vA; varying float vLive; varying float vS; varying float vFogDepth; varying float vLit;
      void main() {
        vec3 A, N;
        vec2 g = position.xy * (4.0 * uBlock + 0.24);
        vec3 c = panelPoint(aLayer, g, -0.004, A, N);
        vec4 mv = modelViewMatrix * vec4(c, 1.0);
        gl_Position = projectionMatrix * mv;
        vFogDepth = -mv.z;
        vG = g;
        int li = int(aLayer + 0.5);
        vA = uLayerAlpha[li];
        vLit = uLayerLit[li];
        vLive = isLive(aLayer);
        vS = 1.0 - layerParity(aLayer);
      }`,
    fragmentShader: /* glsl */ `
      uniform float uBlock, uEq;
      uniform vec3 uMuted, uGreen, uViolet;
      varying vec2 vG; varying float vA; varying float vLive; varying float vS; varying float vLit;
      ${FOG_PARS_F}
      void main() {
        if (vA < 0.002) discard;
        vec2 q = vG / uBlock + 2.0;
        vec2 f = fract(q);
        float inside = step(0.0, q.x) * step(q.x, 4.0) * step(0.0, q.y) * step(q.y, 4.0);
        float e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
        float grid = (1.0 - smoothstep(0.0, 0.035, e)) * inside;
        float outer = 2.0 * uBlock + 0.12;
        float bd = max(abs(vG.x), abs(vG.y));
        float border = smoothstep(outer - 0.05, outer - 0.01, bd) * (1.0 - smoothstep(outer + 0.06, outer + 0.1, bd));
        vec3 edge = mix(uMuted, uGreen, uEq * vLive);
        vec3 col = vec3(0.025, 0.035, 0.055) * inside + edge * (grid * 0.22 + border * (0.55 + 0.6 * vLit + 0.6 * uEq * vLive));
        float a = (0.55 * inside + border) * vA;
        col = mix(col, uFogColor, fogFactor());
        gl_FragColor = vec4(col, a);
        #include <colorspace_fragment>
      }`,
  });
  const plates = new THREE.Mesh(plateGeo, plateMat);
  plates.frustumCulled = false;
  plates.renderOrder = 2;
  group.add(plates);

  // ---------------------------------------------------------------- labels and hit planes
  const labels = [];
  const hits = [];
  const hitGeo = new THREE.PlaneGeometry(4 * BLOCK + 0.4, 4 * BLOCK + 0.4);
  const hitMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  for (let l = 0; l < NL; l++) {
    const lab = makeLabel([{ text: vodeOf(l), font: 'display', size: 30, weight: 600 }], { worldH: 0.9 });
    lab.sprite.visible = false;
    group.add(lab.sprite);
    labels.push({ lab, key: '' });
    const m = new THREE.Mesh(hitGeo, hitMat);
    m.visible = false;
    m.userData = { kind: 'layer', layer: l };
    group.add(m);
    hits.push(m);
  }

  // ---------------------------------------------------------------- per-frame update
  const info = { liveS: 0, liveU: 1, k: new Int32Array(NL), computed: new Uint8Array(NL), lenH: 1 };
  let lastVer = -1, lastLayerVer = -1, lastFold = -1, labelT = 0, eqT = 0;
  const lit = chainUniforms.uLayerLit.value;
  const alpha = chainUniforms.uLayerAlpha.value;
  const zeros = new Float32Array(H);

  function mapLayers() {
    const Ls = sim.layers || [];
    const lenH = Math.max(1, Ls.length);
    const k0 = Ls.length ? Ls[0].k : 0;
    info.lenH = lenH;
    info.liveS = 2 * (lenH - 1);
    info.liveU = Math.max(1, info.liveS - 1);
    for (let l = 0; l < NL; l++) {
      info.k[l] = l % 2 === 0 ? k0 + l / 2 : k0 + (l + 1) / 2;
      info.computed[l] = l <= info.liveS ? 1 : 0;
    }
    info.computed[info.liveU] = 1;
  }
  function layerValues(l) {
    const V = sim.view;
    if (l === info.liveS) return V.z.s;
    if (l === info.liveU) return V.z.u;
    const Ls = sim.layers || [];
    if (l % 2 === 0) { const e = Ls[l / 2]; return e ? e.s : zeros; }
    const e = Ls[(l + 1) / 2];
    return e ? e.u : zeros;
  }

  function update(dt) {
    const f = chainUniforms.uFold.value;
    if (sim.layerVersion !== lastLayerVer) {
      const grew = lastLayerVer >= 0;
      lastLayerVer = sim.layerVersion;
      mapLayers();
      if (grew) { lit[info.liveS] = 1; lit[info.liveU] = Math.max(lit[info.liveU], 0.85); }
      lastVer = -1;
    }
    for (let l = 0; l < NL; l++) lit[l] = Math.max(0, lit[l] - dt / 0.8);
    // layer visibility: folded -> only the live pair; unfolded -> every computed layer, future layers faint
    const unf = Math.min(1, Math.max(0, (f - 0.02) / 0.3));
    for (let l = 0; l < NL; l++) {
      const live = l === info.liveS || l === info.liveU;
      alpha[l] = live ? 1 : info.computed[l] ? unf : 0.16 * unf;
    }
    local.uLiveS.value = info.liveS;
    local.uLiveU.value = info.liveU;
    // green rim when the free phase has reached equilibrium
    eqT += ((sim.phase === 'free' && sim.converged ? 1 : 0) - eqT) * Math.min(1, dt * 3);
    local.uEq.value = eqT;

    if (sim.version !== lastVer) {
      lastVer = sim.version;
      const val = attrVal.array, pred = attrPred.array;
      for (let l = 0; l < NL; l++) {
        const vals = layerValues(l);
        val.set(vals.length === H ? vals : zeros, l * H);
        if (l === info.liveS) pred.set(sim.view.pred.s, l * H);
        else if (l === info.liveU) pred.set(sim.view.pred.u, l * H);
        else pred.set(vals.length === H ? vals : zeros, l * H);
      }
      attrVal.needsUpdate = true;
      attrPred.needsUpdate = true;
    }

    // labels + hit planes follow the chain
    labelT -= dt;
    const relabel = labelT <= 0 || Math.abs(f - lastFold) > 0.002;
    if (labelT <= 0) labelT = 0.25;
    if (relabel) {
      lastFold = f;
      for (let l = 0; l < NL; l++) {
        const L = labels[l];
        const fr = panelFrameJS(l);
        const top = fr.C.clone().add(new THREE.Vector3(0, 2 * BLOCK + 0.95, 0));
        L.lab.sprite.position.copy(top);
        L.lab.sprite.material.opacity = alpha[l] * (info.computed[l] ? 1 : 0.6);
        L.lab.sprite.visible = alpha[l] > 0.02;
        const v = vodeOf(l);
        const key = f < 0.5 ? `st:${v}` : `ly:${v}:${info.computed[l] ? info.k[l] : '-'}`;
        if (key !== L.key) {
          L.key = key;
          if (f < 0.5) {
            const vi = vodeInfo(v);
            L.lab.set([
              { text: v, font: 'display', size: 34, weight: 600, color: COLORS.ink },
              { text: `${vi.shapeTxt}`, font: 'mono', size: 13, color: COLORS.muted },
            ]);
            L.lab.sprite.scale.multiplyScalar(1.6);
          } else {
            L.lab.set([{ text: info.computed[l] ? `${v}${subscript(info.k[l])}` : `${v}…`, font: 'display', size: 30, weight: 600, color: info.computed[l] ? COLORS.ink : COLORS.muted }]);
          }
        }
        const m = hits[l];
        m.position.copy(fr.C);
        const basis = new THREE.Matrix4().makeBasis(fr.A, new THREE.Vector3(0, 1, 0), fr.N);
        m.quaternion.setFromRotationMatrix(basis);
        m.updateMatrixWorld(true);
      }
    }
  }

  function subscript(n) {
    const d = '₀₁₂₃₄₅₆₇₈₉';
    return String(n).split('').map((c) => d[+c] || c).join('');
  }

  return {
    group, layouts, info,
    hitMeshes: () => hits.filter((m, l) => alpha[l] > 0.3 && (info.computed[l] || l === info.liveU)),
    update,
    setShowPredictions(on) { local.uShowPred.value = on ? 1 : 0; },
    setShowErrors(on) { local.uShowErr.value = on ? 1 : 0; },
    setHighlightBlock(p) { local.uHiBlock.value = p == null ? -1 : p; },
    layerInfo(l) {
      return { vode: vodeOf(l), k: info.k[l], live: l === info.liveS || l === info.liveU, computed: !!info.computed[l] };
    },
    liveLayers: () => ({ s: info.liveS, u: info.liveU }),
    layerValues,
    panelCentre(l) { return panelFrameJS(l).C.clone().add(new THREE.Vector3(0, 0, 0)); },
    PANEL_Y,
  };
}
