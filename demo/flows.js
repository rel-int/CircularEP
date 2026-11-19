// flows.js — everything that travels between layers.
//  * synapse bundles: the strongest |W1| (s -> u) and |W2| (u -> s) entries as faint AMBER filaments along the ring arcs
//    (or between consecutive layers of the unrolled network);
//  * TEAL particles ("green" in the house palette's loose sense) travel FORWARD along them: predictions, i.e. the
//    signal a synapse is currently carrying, |w x presynaptic activity|. Their per-synapse pattern (which wires
//    light up) stays relative to the current spread, but their overall brightness (uGain, see update()) is driven
//    by the residual |sweep(z) - z| / |z| -- the per-step CHANGE, log-mapped against a fixed reference, floored so
//    the wiring stays faintly visible -- because activity itself never goes to zero, but how much it is still
//    moving does, and that is what should fade as the free phase settles;
//  * ROSE particles run BACKWARD: prediction errors, |w x postsynaptic error x tanh'| -- the term every unit's
//    gradient receives from the layer it predicts (gradient descent on E is the only free-phase dynamics, and the
//    nudged phase is gradient descent on E + beta C, so this is always "on"). Their overall brightness (uGain) is
//    driven by the true energy E, log-mapped against a fixed reference with a floor near the solver's own E < 1e-6
//    convergence test, so they start bright, fade to almost dark at equilibrium, and brighten again when a nudge
//    reopens the errors;
//  * the input pedestal (x: 16 cells x 5 one-hot = 80 inputs) and the VIOLET beam x_inj = x Win + bin into s
//    (into every s layer of the unrolled network);
//  * the readout synapses V from s to the board (TEAL forward, read-only, tapping EVERY unrolled s copy dimly and
//    the live copy at full brightness -- unrolled input and output both attach to s), and, while nudging, a VIOLET
//    pull -beta dC/ds from the label back into s (every s copy, unrolled, full brightness, same as x_inj);
//  * the EP signal: synapses glow AMBER in proportion to |ΔW| of the one-sided EP estimate.
import * as THREE from 'three';
import { CHAIN_GLSL, chainUniforms, NL, RING_R } from './chain.js';
import { COLORS, FONTS, FOG_PARS_F, sharedUniforms, makeLabel, PED_POS, PED_POS_UNF, anchors } from './world.js';

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const hash = (n) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const logmap = (x, lo = 4, span = 3.2) => Math.min(1, Math.max(0, (Math.log10(Math.max(x, 1e-12)) + lo) / span));

function topK(mat, k) {
  const { data, shape } = mat;
  const n = data.length;
  const absv = new Float32Array(n);
  for (let i = 0; i < n; i++) absv[i] = Math.abs(data[i]);
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => absv[b] - absv[a]);
  const m = Math.min(k, n), cols = shape[1];
  const src = new Int32Array(m), dst = new Int32Array(m), w = new Float32Array(m);
  for (let j = 0; j < m; j++) { const id = idx[j]; src[j] = Math.floor(id / cols); dst[j] = id % cols; w[j] = data[id]; }
  return { n: m, src, dst, w, wmax: absv[idx[0]] || 1 };
}

const POINT_FRAG = (colorUniform, gain = '1.0') => /* glsl */ `
  uniform vec3 ${colorUniform};
  varying float vA;
  ${FOG_PARS_F}
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = exp(-d * d * 3.2) * vA * (1.0 - fogFactor());
    if (a < 0.003) discard;
    gl_FragColor = vec4(${colorUniform} * a * ${gain}, 1.0);
    #include <colorspace_fragment>
  }`;

export function createFlows(world, sim, stations, board) {
  const mobile = world.state.mobile;
  const reduced = world.state.reduced;
  const H = sim.dims.H;
  const W = sim.weights();
  const K = mobile ? 160 : 300;
  const PPF = 2;
  const NG = NL - 1;
  const layouts = stations.layouts;
  const group = new THREE.Group();
  group.name = 'flows';
  world.scene.add(group);

  const sel = { W1: W.W1 ? topK(W.W1, K) : null, W2: W.W2 ? topK(W.W2, K) : null };
  const KV = mobile ? 96 : 160;
  const selV = W.V ? topK(W.V, KV) : null;

  const local = {
    uPinch: { value: 0.86 }, uSpread: { value: 0.55 }, uAlpha: { value: 0.16 }, uEpMix: { value: 0 },
    uLiveW1: { value: 0 }, uLiveW2: { value: 1 }, uLitGain: { value: 1.2 }, uSpeed: { value: 0.12 },
    uAmber: { value: new THREE.Color(COLORS.amber) }, uTeal: { value: new THREE.Color(COLORS.teal) },
    uRose: { value: new THREE.Color(COLORS.rose) }, uViolet: { value: new THREE.Color(COLORS.violet) },
    uSize: { value: 0.11 }, uLiveS: { value: 0 },
  };
  const U = { ...chainUniforms, ...sharedUniforms, ...local };

  // ---------------------------------------------------------------- synapse filaments (instanced line segments)
  const nFil = NG * K;
  const fGap = new Float32Array(nFil), fSrc = new Float32Array(nFil * 2), fDst = new Float32Array(nFil * 2);
  const fW = new Float32Array(nFil), fRnd = new Float32Array(nFil), fEp = new Float32Array(nFil);
  for (let g = 0; g < NG; g++) {
    const isW1 = g % 2 === 0;
    const S = isW1 ? sel.W1 : sel.W2;
    const sl = isW1 ? layouts.s : layouts.u, dl = isW1 ? layouts.u : layouts.s;
    for (let q = 0; q < K; q++) {
      const k = g * K + q;
      fGap[k] = g;
      fRnd[k] = hash(q + (isW1 ? 0 : 7919));
      if (!S || q >= S.n) continue;
      const i = S.src[q], j = S.dst[q];
      fSrc[2 * k] = sl.grid[2 * i]; fSrc[2 * k + 1] = sl.grid[2 * i + 1];
      fDst[2 * k] = dl.grid[2 * j]; fDst[2 * k + 1] = dl.grid[2 * j + 1];
      fW[k] = Math.abs(S.w[q]) / S.wmax;
    }
  }
  const attrEp = new THREE.InstancedBufferAttribute(fEp, 1).setUsage(THREE.DynamicDrawUsage);
  const SEG = mobile ? 14 : 22;
  const segBase = (n) => {
    const a = new Float32Array(n * 2 * 3);
    for (let s = 0; s < n; s++) { a[s * 6] = s / n; a[s * 6 + 3] = (s + 1) / n; }
    return a;
  };
  const filGeo = new THREE.InstancedBufferGeometry();
  filGeo.setAttribute('position', new THREE.BufferAttribute(segBase(SEG), 3));
  filGeo.setAttribute('aGap', new THREE.InstancedBufferAttribute(fGap, 1));
  filGeo.setAttribute('aSrc', new THREE.InstancedBufferAttribute(fSrc, 2));
  filGeo.setAttribute('aDst', new THREE.InstancedBufferAttribute(fDst, 2));
  filGeo.setAttribute('aW', new THREE.InstancedBufferAttribute(fW, 1));
  filGeo.setAttribute('aRnd', new THREE.InstancedBufferAttribute(fRnd, 1));
  filGeo.setAttribute('aEp', attrEp);
  filGeo.instanceCount = nFil;
  const filMat = new THREE.ShaderMaterial({
    defines: { NL }, uniforms: U, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      uniform float uPinch, uSpread, uAlpha, uEpMix, uLitGain, uLiveW1, uLiveW2;
      attribute float aGap; attribute vec2 aSrc; attribute vec2 aDst; attribute float aW; attribute float aRnd; attribute float aEp;
      varying float vA; varying float vEp; varying float vT; varying float vFogDepth;
      void main() {
        float t = position.x;
        vec3 p = synapsePoint(aGap, aSrc, aDst, t, aRnd, uPinch, uSpread);
        int g0 = int(aGap + 0.5);
        float liveGap = (abs(aGap - uLiveW1) < 0.5 || abs(aGap - uLiveW2) < 0.5) ? 1.0 : 0.0;
        float vis = liveGap > 0.5 ? 1.0 : min(uLayerAlpha[g0], uLayerAlpha[g0 + 1]);
        float lit = uLayerLit[g0 + 1];
        vA = vis * uAlpha * (0.15 + 0.85 * aW) * (1.0 + uLitGain * lit) * (1.0 - 0.65 * uEpMix) * step(0.0001, aW);
        vEp = vis * aEp * uEpMix * step(0.0001, aW);
        vT = t;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vFogDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uAmber;
      varying float vA; varying float vEp; varying float vT;
      ${FOG_PARS_F}
      void main() {
        float a = vA * (0.55 + 0.45 * sin(3.14159265 * vT)) + vEp * 1.6;
        a *= 1.0 - fogFactor();
        if (a < 0.002) discard;
        gl_FragColor = vec4(uAmber * a, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const filaments = new THREE.LineSegments(filGeo, filMat);
  filaments.frustumCulled = false;
  filaments.renderOrder = 2;
  group.add(filaments);

  // ---------------------------------------------------------------- particles along the synapses (forward TEAL, backward ROSE)
  const nPar = nFil * PPF;
  function makeSynapseParticles(colorName, dir, onlyLive, size = 0.11, gain = '1.5') {
    const pGap = new Float32Array(nPar), pSrc = new Float32Array(nPar * 2), pDst = new Float32Array(nPar * 2);
    const pRnd = new Float32Array(nPar), pT0 = new Float32Array(nPar), pSeed = new Float32Array(nPar);
    for (let k = 0; k < nFil; k++) for (let r = 0; r < PPF; r++) {
      const j = k * PPF + r;
      pGap[j] = fGap[k];
      pSrc[2 * j] = fSrc[2 * k]; pSrc[2 * j + 1] = fSrc[2 * k + 1];
      pDst[2 * j] = fDst[2 * k]; pDst[2 * j + 1] = fDst[2 * k + 1];
      pRnd[j] = fRnd[k];
      pT0[j] = (r + hash(j * 3.1 + dir) * 0.8) / PPF;
      pSeed[j] = hash(j * 7.7 + 13 * dir);
    }
    const flux = new THREE.InstancedBufferAttribute(new Float32Array(nPar), 1).setUsage(THREE.DynamicDrawUsage);
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    g.setAttribute('aGap', new THREE.InstancedBufferAttribute(pGap, 1));
    g.setAttribute('aSrc', new THREE.InstancedBufferAttribute(pSrc, 2));
    g.setAttribute('aDst', new THREE.InstancedBufferAttribute(pDst, 2));
    g.setAttribute('aRnd', new THREE.InstancedBufferAttribute(pRnd, 1));
    g.setAttribute('aT0', new THREE.InstancedBufferAttribute(pT0, 1));
    g.setAttribute('aSeed', new THREE.InstancedBufferAttribute(pSeed, 1));
    g.setAttribute('aFlux', flux);
    g.instanceCount = nPar;
    const m = new THREE.ShaderMaterial({
      defines: { NL }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...U, uDir: { value: dir }, uOnlyLive: { value: onlyLive ? 1 : 0 }, uGain: { value: 1 }, uSize: { value: size } },
      vertexShader: /* glsl */ `
        ${CHAIN_GLSL}
        uniform float uPinch, uSpread, uSpeed, uDir, uSize, uProjScale, uTime, uLiveW1, uLiveW2, uOnlyLive, uGain;
        attribute float aGap; attribute vec2 aSrc; attribute vec2 aDst; attribute float aRnd; attribute float aT0; attribute float aSeed; attribute float aFlux;
        varying float vA; varying float vFogDepth;
        void main() {
          float t = fract(aT0 + uDir * uTime * uSpeed);
          vec3 p = synapsePoint(aGap, aSrc, aDst, t, aRnd, uPinch, uSpread) + position;
          int g0 = int(aGap + 0.5);
          float liveGap = (abs(aGap - uLiveW1) < 0.5 || abs(aGap - uLiveW2) < 0.5) ? 1.0 : 0.0;
          float vis = liveGap > 0.5 ? 1.0 : min(uLayerAlpha[g0], uLayerAlpha[g0 + 1]) * (1.0 - uOnlyLive);
          float fl = clamp(aFlux * uGain, 0.0, 1.0);
          float on = step(aSeed, 0.06 + 1.05 * fl);
          float fade = smoothstep(0.0, 0.07, t) * smoothstep(1.0, 0.93, t);
          vA = vis * on * fade * (0.3 + 0.7 * fl);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vFogDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
          gl_PointSize = vA > 0.003 ? uSize * (0.8 + 0.7 * fl) * uProjScale / max(0.5, -mv.z) : 0.0;
        }`,
      fragmentShader: POINT_FRAG(colorName, gain),
    });
    const pts = new THREE.Points(g, m);
    pts.frustumCulled = false;
    pts.renderOrder = 4;
    group.add(pts);
    return { pts, flux, mat: m };
  }
  const fwd = makeSynapseParticles('uTeal', 1, false);
  const back = makeSynapseParticles('uRose', -1, true, 0.17, '2.6');     // errors: larger, brighter (they ride the amber bundle)

  // ---------------------------------------------------------------- input pedestal: x (80 one-hot inputs) on a plinth
  const ped = new THREE.Group();
  ped.name = 'pedestal';
  group.add(ped);
  const plinth = new THREE.Mesh(new THREE.CylinderGeometry(2.1, 2.5, 1.3, 64), new THREE.MeshBasicMaterial({ color: '#0B111A' }));
  plinth.position.y = 0.65;
  ped.add(plinth);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(2.12, 0.035, 8, 128), new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.violet).multiplyScalar(1.6), fog: true }));
  rim.rotation.x = Math.PI / 2;
  rim.position.y = 1.31;
  ped.add(rim);
  const plateCanvas = document.createElement('canvas');
  plateCanvas.width = plateCanvas.height = 512;
  const plateTex = new THREE.CanvasTexture(plateCanvas);
  plateTex.colorSpace = THREE.SRGBColorSpace;
  plateTex.anisotropy = 4;
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(3.3, 3.3), new THREE.MeshBasicMaterial({ map: plateTex, transparent: true, depthWrite: false }));
  plate.rotation.x = -Math.PI / 2 + 0.42;
  plate.position.y = 1.45;
  plate.renderOrder = 6;
  plate.userData = { kind: 'pedestal' };
  ped.add(plate);
  const pedLabel = makeLabel([
    { text: 'input x', font: 'display', size: 26, weight: 600, color: COLORS.violet },
    { text: '16 cells × 5 one-hot = 80', font: 'mono', size: 13, color: COLORS.muted },
    { text: 'x_inj = x·Win + bin → ŝ', font: 'mono', size: 13, color: COLORS.violet },
  ], { worldH: 1.25 });
  pedLabel.sprite.position.set(4.1, 2.2, 0.6);         // to the side, out of the sightline from the front to s
  ped.add(pedLabel.sprite);

  function drawPlate() {
    const ctx = plateCanvas.getContext('2d');
    const S = 512, m = 18, cell = (S - 2 * m) / 4;
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = 'rgba(12,17,27,0.94)';
    roundRect(ctx, 4, 4, S - 8, S - 8, 30); ctx.fill();
    ctx.strokeStyle = 'rgba(167,139,250,0.55)'; ctx.lineWidth = 3; ctx.stroke();
    const puz = sim.puzzle || new Int8Array(16);
    for (let p = 0; p < 16; p++) {
      const r = p >> 2, c = p & 3;
      const x = m + c * cell + 5, y = m + r * cell + 5, w = cell - 10;
      const v = puz[p];
      ctx.fillStyle = v ? 'rgba(167,139,250,0.12)' : 'rgba(255,255,255,0.03)';
      roundRect(ctx, x, y, w, w, 12); ctx.fill();
      ctx.strokeStyle = v ? 'rgba(167,139,250,0.75)' : 'rgba(139,152,165,0.25)';
      ctx.lineWidth = 2; ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      if (v) { ctx.fillStyle = COLORS.violet; ctx.font = `600 58px ${FONTS.display}`; ctx.fillText(String(v), x + w / 2, y + w * 0.42); }
      else { ctx.fillStyle = 'rgba(139,152,165,0.45)'; ctx.font = `500 22px ${FONTS.mono}`; ctx.fillText('blank', x + w / 2, y + w * 0.42); }
      // the 5-way one-hot strip: slots 0 (blank), 1..4
      const sw = (w - 16) / 5;
      for (let s = 0; s < 5; s++) {
        const sx = x + 8 + s * sw, sy = y + w - 24;
        const on = s === v;
        ctx.fillStyle = on ? (v ? COLORS.violet : COLORS.muted) : 'rgba(139,152,165,0.18)';
        roundRect(ctx, sx + 2, sy, sw - 4, 14, 4); ctx.fill();
      }
    }
    plateTex.needsUpdate = true;
  }

  // ---------------------------------------------------------------- the VIOLET beam x_inj -> s (every s layer when unfolded)
  const NB = mobile ? 56 : 96, PPB = 2;
  const NSL = (NL + 1) / 2;
  const nBeam = NSL * NB * PPB;
  const bLayer = new Float32Array(nBeam), bRnd = new Float32Array(nBeam), bT0 = new Float32Array(nBeam), bSeed = new Float32Array(nBeam);
  for (let m = 0; m < NSL; m++) for (let q = 0; q < NB; q++) for (let r = 0; r < PPB; r++) {
    const j = (m * NB + q) * PPB + r;
    bLayer[j] = 2 * m;
    bRnd[j] = hash(j * 1.37 + 5);
    bT0[j] = (r + hash(j * 2.71) * 0.9) / PPB;
    bSeed[j] = hash(j * 9.13 + 1);
  }
  const bGrid = new THREE.InstancedBufferAttribute(new Float32Array(nBeam * 2), 2);
  const bGain = new THREE.InstancedBufferAttribute(new Float32Array(nBeam), 1);
  const beamGeo = new THREE.InstancedBufferGeometry();
  beamGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
  beamGeo.setAttribute('aLayer', new THREE.InstancedBufferAttribute(bLayer, 1));
  beamGeo.setAttribute('aGrid', bGrid);
  beamGeo.setAttribute('aGainA', bGain);
  beamGeo.setAttribute('aRnd', new THREE.InstancedBufferAttribute(bRnd, 1));
  beamGeo.setAttribute('aT0', new THREE.InstancedBufferAttribute(bT0, 1));
  beamGeo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(bSeed, 1));
  beamGeo.instanceCount = nBeam;
  const beamU = { uPed: { value: new THREE.Vector3() }, uK0Zero: { value: 1 }, uUnf: { value: 0 }, uBeamGain: { value: 1 } };
  const beamMat = new THREE.ShaderMaterial({
    defines: { NL }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { ...U, ...beamU },
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      uniform vec3 uPed;
      uniform float uSize, uProjScale, uTime, uK0Zero, uUnf, uBeamGain;
      attribute float aLayer; attribute vec2 aGrid; attribute float aGainA; attribute float aRnd; attribute float aT0; attribute float aSeed;
      varying float vA; varying float vFogDepth;
      void main() {
        vec3 A, N;
        vec3 dst = panelPoint(aLayer, aGrid, 0.0, A, N);
        float ang = 6.2831853 * aRnd;
        vec3 src = uPed + vec3(cos(ang), 0.0, sin(ang)) * 1.1 * fract(aRnd * 13.7);
        float t = fract(aT0 + uTime * 0.3);
        vec3 mid = mix(src, dst, 0.5) + vec3(0.0, 2.4, 0.0);
        vec3 p = mix(mix(src, mid, t), mix(mid, dst, t), t) + position;
        float vis = uLayerAlpha[int(aLayer + 0.5)];
        if (aLayer < 0.5 && uK0Zero > 0.5) vis *= (1.0 - uUnf);
        float on = step(aSeed, 0.1 + 0.9 * aGainA);
        float fade = smoothstep(0.0, 0.1, t) * smoothstep(1.0, 0.88, t);
        vA = vis * on * fade * (0.35 + 0.65 * aGainA) * uBeamGain;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vFogDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
        gl_PointSize = vA > 0.003 ? uSize * (0.8 + 0.6 * aGainA) * uProjScale / max(0.5, -mv.z) : 0.0;
      }`,
    fragmentShader: POINT_FRAG('uViolet', '1.6'),
  });
  const beam = new THREE.Points(beamGeo, beamMat);
  beam.frustumCulled = false;
  beam.renderOrder = 4;
  group.add(beam);

  // ---------------------------------------------------------------- readout synapses V: EVERY s copy -> board cells
  // The folded loop attaches both x_inj and the readout V to the SAME node s; unrolled, both therefore reach every
  // s copy. x_inj already does (the beam above). The readout now taps every s copy the same way: read-only (dim,
  // s -> y, no gradient back) in the free phase, and in the nudge phase the pull -beta dC/ds pushes into every copy
  // (violet), exactly like x_inj. Only the LIVE copy is at full brightness in the free phase; folded, only the live
  // copy's layer alpha is nonzero anyway (stations.js), so the ring view is unchanged.
  const KVN = KV * NSL;
  const vSrc = new Float32Array(KVN * 2), vDst = new Float32Array(KVN * 3), vW = new Float32Array(KVN), vRnd = new Float32Array(KVN), vLayer = new Float32Array(KVN);
  if (selV) for (let m = 0; m < NSL; m++) for (let q = 0; q < selV.n; q++) {
    const j = m * KV + q;
    const i = selV.src[q], k = selV.dst[q];
    vSrc[2 * j] = layouts.s.grid[2 * i]; vSrc[2 * j + 1] = layouts.s.grid[2 * i + 1];
    const bl = board.barLocal(k >> 2, k & 3);
    vDst[3 * j] = bl.x; vDst[3 * j + 1] = bl.y; vDst[3 * j + 2] = bl.z;
    vW[j] = Math.abs(selV.w[q]) / selV.wmax;
    vRnd[j] = hash(q * 3.3 + 17 + m * 101);
    vLayer[j] = 2 * m;
  }
  const boardU = { uBoardMat: { value: board.group.matrixWorld }, uVAlpha: { value: 0.075 }, uReadoutDim: { value: 0.16 } };
  const V_GLSL = /* glsl */ `
    uniform mat4 uBoardMat;
    uniform float uReadoutDim;
    vec3 vPath(float srcLayer, vec2 src, vec3 dstL, float t, float rnd) {
      vec3 A, N;
      vec3 p0 = panelPoint(srcLayer, src, 0.0, A, N);
      vec3 p2 = (uBoardMat * vec4(dstL, 1.0)).xyz;
      vec3 p1 = mix(p0, p2, 0.5) + vec3(0.0, 1.8, 0.0) + N * (rnd - 0.5) * 2.0;
      return mix(mix(p0, p1, t), mix(p1, p2, t), t);
    }
    // 1 on the live s copy (full brightness), uReadoutDim on every other computed copy (read-only taps)
    float vTapVis(float srcLayer) {
      float live = abs(srcLayer - uLiveS) < 0.5 ? 1.0 : 0.0;
      return uLayerAlpha[int(srcLayer + 0.5)] * mix(uReadoutDim, 1.0, live);
    }`;
  const vGeo = new THREE.InstancedBufferGeometry();
  vGeo.setAttribute('position', new THREE.BufferAttribute(segBase(SEG), 3));
  vGeo.setAttribute('aSrc', new THREE.InstancedBufferAttribute(vSrc, 2));
  vGeo.setAttribute('aDstL', new THREE.InstancedBufferAttribute(vDst, 3));
  vGeo.setAttribute('aW', new THREE.InstancedBufferAttribute(vW, 1));
  vGeo.setAttribute('aRnd', new THREE.InstancedBufferAttribute(vRnd, 1));
  vGeo.setAttribute('aLayer', new THREE.InstancedBufferAttribute(vLayer, 1));
  vGeo.instanceCount = selV ? selV.n * NSL : 0;
  const vMat = new THREE.ShaderMaterial({
    defines: { NL }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { ...U, ...boardU },
    vertexShader: /* glsl */ `
      ${CHAIN_GLSL}
      uniform float uLiveS, uVAlpha;
      ${V_GLSL}
      attribute vec2 aSrc; attribute vec3 aDstL; attribute float aW; attribute float aRnd; attribute float aLayer;
      varying float vA; varying float vFogDepth;
      void main() {
        vec3 p = vPath(aLayer, aSrc, aDstL, position.x, aRnd);
        vA = uVAlpha * (0.2 + 0.8 * aW) * vTapVis(aLayer);
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vFogDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uAmber;
      varying float vA;
      ${FOG_PARS_F}
      void main() {
        float a = vA * (1.0 - fogFactor());
        if (a < 0.002) discard;
        gl_FragColor = vec4(uAmber * a, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const vLines = new THREE.LineSegments(vGeo, vMat);
  vLines.frustumCulled = false;
  vLines.renderOrder = 2;
  group.add(vLines);

  function makeVParticles(colorName, dir, dimmed) {
    const n = KVN * 2;
    const src = new Float32Array(n * 2), dstL = new Float32Array(n * 3), rnd = new Float32Array(n), t0 = new Float32Array(n), seed = new Float32Array(n), layer = new Float32Array(n);
    for (let j = 0; j < KVN; j++) for (let r = 0; r < 2; r++) {
      const jj = j * 2 + r;
      src[2 * jj] = vSrc[2 * j]; src[2 * jj + 1] = vSrc[2 * j + 1];
      dstL[3 * jj] = vDst[3 * j]; dstL[3 * jj + 1] = vDst[3 * j + 1]; dstL[3 * jj + 2] = vDst[3 * j + 2];
      rnd[jj] = vRnd[j]; t0[jj] = (r + hash(jj * 4.1 + dir) * 0.8) / 2; seed[jj] = hash(jj * 6.3 + dir * 3);
      layer[jj] = vLayer[j];
    }
    const flux = new THREE.InstancedBufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage);
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    g.setAttribute('aSrc', new THREE.InstancedBufferAttribute(src, 2));
    g.setAttribute('aDstL', new THREE.InstancedBufferAttribute(dstL, 3));
    g.setAttribute('aRnd', new THREE.InstancedBufferAttribute(rnd, 1));
    g.setAttribute('aT0', new THREE.InstancedBufferAttribute(t0, 1));
    g.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seed, 1));
    g.setAttribute('aFlux', flux);
    g.setAttribute('aLayer', new THREE.InstancedBufferAttribute(layer, 1));
    g.instanceCount = selV ? selV.n * NSL * 2 : 0;
    const m = new THREE.ShaderMaterial({
      defines: { NL }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...U, ...boardU, uDir: { value: dir }, uDimmed: { value: dimmed ? 1 : 0 } },
      vertexShader: /* glsl */ `
        ${CHAIN_GLSL}
        uniform float uLiveS, uSize, uProjScale, uTime, uDir, uDimmed;
        ${V_GLSL}
        attribute vec2 aSrc; attribute vec3 aDstL; attribute float aRnd; attribute float aT0; attribute float aSeed; attribute float aFlux; attribute float aLayer;
        varying float vA; varying float vFogDepth;
        void main() {
          float t = fract(aT0 + uDir * uTime * 0.22);
          vec3 p = vPath(aLayer, aSrc, aDstL, t, aRnd) + position;
          float fl = clamp(aFlux, 0.0, 1.0);
          float on = step(aSeed, 0.05 + 1.05 * fl);
          float fade = smoothstep(0.0, 0.08, t) * smoothstep(1.0, 0.92, t);
          // forward/teal taps are read-only on non-live copies (dim); the nudge pull (uDimmed = 0) is full on every copy
          float vis = uDimmed > 0.5 ? vTapVis(aLayer) : uLayerAlpha[int(aLayer + 0.5)];
          vA = on * fade * (0.3 + 0.7 * fl) * vis;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vFogDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
          gl_PointSize = vA > 0.003 ? uSize * (0.75 + 0.6 * fl) * uProjScale / max(0.5, -mv.z) : 0.0;
        }`,
      fragmentShader: POINT_FRAG(colorName, '1.5'),
    });
    const pts = new THREE.Points(g, m);
    pts.frustumCulled = false;
    pts.renderOrder = 4;
    group.add(pts);
    return { pts, flux };
  }
  const vFwd = makeVParticles('uTeal', 1, true);     // read-only forward taps: dim on every non-live s copy
  const vNudge = makeVParticles('uViolet', -1, false); // nudge pull: full brightness into every s copy (like x_inj)

  // ---------------------------------------------------------------- dynamic inputs
  const fluxF = new Float32Array(nFil), backF = new Float32Array(nFil);
  const dU = new Float32Array(H), dS = new Float32Array(H);
  let labelOneHot = null;
  const opts = { synapses: true, particles: true, beam: true };
  let epMix = 0;

  function setPuzzle() {
    drawPlate();
    // beam targets: the s units with the strongest injection |x_inj|
    const inj = sim.injNorm || new Float32Array(H);
    const order = Array.from({ length: H }, (_, i) => i).sort((a, b) => inj[b] - inj[a]);
    const grid = bGrid.array, gain = bGain.array;
    for (let m = 0; m < NSL; m++) for (let q = 0; q < NB; q++) for (let r = 0; r < PPB; r++) {
      const j = (m * NB + q) * PPB + r;
      const i = order[q];
      grid[2 * j] = layouts.s.grid[2 * i]; grid[2 * j + 1] = layouts.s.grid[2 * i + 1];
      gain[j] = inj[i];
    }
    bGrid.needsUpdate = true;
    bGain.needsUpdate = true;
  }
  function setPhase() {
    labelOneHot = null;
    if (sim.phase === 'nudge+') {
      const lab = sim.labelFor();
      if (lab) { labelOneHot = new Float32Array(64); lab.forEach((d, p) => { labelOneHot[p * 4 + d] = 1; }); }
    }
  }
  function setEP(ep) {
    const arr = attrEp.array;
    arr.fill(0);
    if (!ep || !ep.mats) { attrEp.needsUpdate = true; return; }
    for (const [key, S] of [['W1', sel.W1], ['W2', sel.W2]]) {
      const M = ep.mats[key];
      if (!M || !S) continue;
      const cols = M.shape[1];
      let mx = 1e-20;
      const vals = new Float32Array(S.n);
      for (let q = 0; q < S.n; q++) { vals[q] = Math.abs(M.data[S.src[q] * cols + S.dst[q]]); mx = Math.max(mx, vals[q]); }
      for (let g = key === 'W1' ? 0 : 1; g < NG; g += 2) for (let q = 0; q < S.n; q++) arr[g * K + q] = Math.pow(vals[q] / mx, 0.7);
    }
    attrEp.needsUpdate = true;
  }

  const tmpV = new THREE.Vector3();
  function update(dt) {
    const f = chainUniforms.uFold.value;
    const info = stations.info;
    const liveW1 = info.liveU - 1, liveW2 = info.liveU;
    local.uLiveW1.value = liveW1;
    local.uLiveW2.value = liveW2;
    local.uLiveS.value = info.liveS;
    const bundle = 1 - smooth(0.3, 0.95, f);
    local.uPinch.value = 0.86 * bundle;
    local.uSpread.value = 0.55 * bundle;
    local.uSpeed.value = (reduced ? 2.2 : 5.0) / (Math.PI * RING_R * chainUniforms.uG.value);
    epMix += ((sim.phase === 'ep' ? 1 : 0) - epMix) * Math.min(1, dt * 2.5);
    local.uEpMix.value = epMix;

    filaments.visible = opts.synapses;
    vLines.visible = opts.synapses;
    fwd.pts.visible = opts.particles;
    vFwd.pts.visible = opts.particles;
    back.pts.visible = opts.particles;
    vNudge.pts.visible = opts.particles && !!labelOneHot;
    beam.visible = opts.beam;
    ped.visible = true;

    // pedestal position morphs with the fold
    ped.position.lerpVectors(PED_POS, PED_POS_UNF, smooth(0.1, 0.9, f));
    ped.updateMatrixWorld(true);
    anchors.ped.copy(ped.position);
    beamU.uPed.value.copy(ped.position).add(tmpV.set(0, 1.45, 0));
    beamU.uK0Zero.value = sim.layers && sim.layers.length && sim.layers[0].k === 0 ? 1 : 0;
    beamU.uUnf.value = smooth(0.02, 0.32, f);

    // forward flux per synapse: |w x presynaptic activity| / max|w|
    for (let g = 0; g < NG; g++) {
      const isW1 = g % 2 === 0;
      const S = isW1 ? sel.W1 : sel.W2;
      if (!S) continue;
      let pre = stations.layerValues(g);
      if (f < 0.5 && g === liveW1) pre = sim.view.z.s;     // folded: û is predicted from the CURRENT s
      if (!pre || pre.length !== H) continue;
      const base = g * K, inv = 1 / S.wmax;
      for (let q = 0; q < S.n; q++) fluxF[base + q] = Math.abs(S.w[q] * pre[S.src[q]]) * inv;
    }
    // backward (errors, rose): |w x error x tanh'| on the live arcs -- the shape (which synapses carry the most
    // error right now) is still relative to the current max, but the OVERALL brightness is not: it is a global
    // gain from the absolute energy E, log-mapped against a fixed reference (not the live max), so particles start
    // bright and fade toward dark as E falls across its ~13 decades (1e2 -> 1e-11), and brighten again on a nudge
    // (E jumps back up). uGain carries this envelope into the shader (see makeSynapseParticles: fl = aFlux * uGain).
    backF.fill(0);
    if (opts.particles) {
      const V = sim.view;
      for (let i = 0; i < H; i++) {
        const pu = V.pred.u[i], ps = V.pred.s[i];
        dU[i] = V.err.u[i] * (1 - pu * pu);
        dS[i] = V.err.s[i] * (1 - ps * ps);
      }
      const fill = (g, S, delta) => {
        if (!S || g < 0 || g >= NG) return;
        for (let q = 0; q < S.n; q++) backF[g * K + q] = Math.abs(S.w[q] * delta[S.dst[q]]);
      };
      fill(liveW1, liveW1 % 2 === 0 ? sel.W1 : sel.W2, liveW1 % 2 === 0 ? dU : dS);
      fill(liveW2, liveW2 % 2 === 0 ? sel.W1 : sel.W2, liveW2 % 2 === 0 ? dU : dS);
      let mx = 0;
      for (let k = 0; k < nFil; k++) mx = Math.max(mx, backF[k]);
      if (mx > 0) for (let k = 0; k < nFil; k++) backF[k] = backF[k] > 0 ? logmap(backF[k] / mx, 2.5, 2.5) : 0;
    }
    // global brightness envelopes: errGain from the true energy (floor 1e-6 = the solver's own convergence test,
    // so "converged" and "almost dark" coincide); sigGain (predictions/teal) from the residual |sweep(z)-z|/|z|,
    // the per-step relative CHANGE -- predictions themselves never vanish, but how much they are still updating
    // does, so the teal brightness is driven by |Delta| per step and goes to 0 at equilibrium. Small floors keep
    // the wiring faintly visible rather than fully invisible.
    const curE = (sim.to && Number.isFinite(sim.to.E)) ? sim.to.E : 0;
    const curRes = (sim.to && Number.isFinite(sim.to.residual)) ? sim.to.residual : 0;
    const errGain = Math.max(0.035, logmap(curE, 6, 8));
    const sigGain = Math.max(0.08, logmap(curRes, 6, 8));
    fwd.mat.uniforms.uGain.value = sigGain;
    back.mat.uniforms.uGain.value = errGain;
    const pf = fwd.flux.array, pb = back.flux.array;
    for (let k = 0; k < nFil; k++) for (let r = 0; r < PPF; r++) { pf[k * PPF + r] = fluxF[k]; pb[k * PPF + r] = backF[k]; }
    fwd.flux.needsUpdate = true;
    back.flux.needsUpdate = true;

    // readout synapses, EVERY s copy: forward |V x s| (read-only, live copy uses the live s; other copies use that
    // copy's own recorded s, when computed, else the same live value as a placeholder -- they are dim anyway);
    // nudge pull |beta V (p - y)| / 16, broadcast identically to every copy, like x_inj.
    if (selV) {
      const probs = sim.view.probs;
      const a = vFwd.flux.array, b = vNudge.flux.array;
      let nbMax = 0;
      const nb = new Float32Array(selV.n);
      for (let q = 0; q < selV.n; q++) {
        const k = selV.dst[q];
        nb[q] = labelOneHot ? Math.abs(selV.w[q] * (probs[k] - labelOneHot[k])) : 0;   // ∝ |β ∂C/∂logit · V|
        nbMax = Math.max(nbMax, nb[q]);
      }
      for (let m = 0; m < NSL; m++) {
        const sLayer = stations.layerValues(2 * m);
        const sArr = sLayer && sLayer.length === H ? sLayer : sim.view.z.s;
        for (let q = 0; q < selV.n; q++) {
          const j = m * KV + q, i = selV.src[q];
          const fw = Math.abs(selV.w[q] * sArr[i]) / selV.wmax;
          const nbv = nbMax > 0 ? logmap(nb[q] / nbMax, 2.5, 2.5) : 0;
          a[2 * j] = a[2 * j + 1] = fw;
          b[2 * j] = b[2 * j + 1] = nbv;
        }
      }
      vFwd.flux.needsUpdate = true;
      vNudge.flux.needsUpdate = true;
    }
  }

  setPuzzle();
  return {
    group, update, setPuzzle, setPhase, setEP,
    setOptions(o) { Object.assign(opts, o); },
    pickables: [plate],
    redraw: drawPlate,
    selections: sel,
  };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
