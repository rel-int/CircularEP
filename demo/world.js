// world.js — renderer, camera, controls, bloom, sky, stars, the morphing ring track, fold animation,
// camera presets / flights / ride, text labels and adaptive quality.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import {
  chainUniforms, CHAIN_GLSL, NL, RING_R, LAYER_GAP, PANEL_Y, setFold, getFold, chainFrameJS, panelFrameJS, layerSigma,
} from './chain.js';

export const COLORS = {
  teal: '#4DD0C1', rose: '#F26D85', amber: '#F2B441', violet: '#A78BFA', green: '#3FB950',
  ink: '#E6EDF3', muted: '#8B98A5', panel: '#161B22', bg: '#05070D', steel: '#6F8FCF',
};
export const FONTS = {
  display: '"Fraunces", "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
  body: '"Instrument Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
};
const FOG_HEX = '#070B14';
export const LABEL_LAYER = 1;                 // all text sprites live on this layer (toggled by the camera)
// Layout (folded): ring of radius 13 at the origin, s at the front (+z), u at the back; the readout board floats up
// and to the right of the ring, the energy well sits low on the right, the input pedestal in front of s. Seen from the
// overview camera none of them crosses a station, the W1/W2 bundles or the V readout synapses.
// Two compositions, chosen from the aspect ratio of the screen area the HUD leaves free (applyLayout in createWorld):
//   landscape: board up-right of the ring, well low on the right;
//   portrait : board high behind the ring (offset right so the V synapses miss u), well deep below the front.
export const LAYOUTS = {
  landscape: { board: new THREE.Vector3(21, 9.6, -4) },
  portrait: { board: new THREE.Vector3(8.5, 17.5, -10) },
};
export const BOARD_POS = LAYOUTS.landscape.board.clone();         // folded (mutable: set by applyLayout)
export const BOARD_POS_UNF = new THREE.Vector3(24, 17, -14);      // unfolded: high, behind the live end of the chain
export const BOARD_SCALE = 0.85;
export const PED_POS = new THREE.Vector3(0, 0, RING_R + 6.8);     // folded
export const PED_POS_UNF = new THREE.Vector3(0, -3.6, 11.5);
/** live anchor points (updated by board.js / flows.js every frame), read by the camera presets */
export const anchors = { board: BOARD_POS.clone(), ped: PED_POS.clone() };

/** uniforms shared by every custom material (same {value} objects everywhere) */
export const sharedUniforms = {
  uTime: { value: 0 },          // flow clock (slower under reduced motion)
  uProjScale: { value: 600 },   // pixels per world unit at distance 1 (for point sizes)
  uFogColor: { value: new THREE.Color(FOG_HEX) },
  uFogDensity: { value: 0.0088 },
};
export const FOG_PARS_F = /* glsl */ `
uniform vec3 uFogColor;
uniform float uFogDensity;
varying float vFogDepth;
float fogFactor() { float d = uFogDensity * vFogDepth; return 1.0 - exp(-d * d); }
`;
export { CHAIN_GLSL, chainUniforms, NL };

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// ------------------------------------------------------------------ text labels (canvas sprites)
const labelRegistry = new Set();
/**
 * lines: [{text, font: 'display'|'body'|'mono', size (px at 1x), color, weight}]
 * returns {sprite, set(lines), dispose()}; world height of the sprite = worldH
 */
export function makeLabel(lines, opts = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.anisotropy = 4;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, fog: false, opacity: opts.opacity ?? 1 });
  const sprite = new THREE.Sprite(mat);
  sprite.renderOrder = opts.renderOrder ?? 5;
  sprite.layers.set(LABEL_LAYER);
  const worldH = opts.worldH ?? 1;
  let current = lines;
  const draw = () => {
    const scale = 2;
    const pad = (opts.pad ?? 10) * scale;
    const metrics = current.map((l) => {
      const fam = FONTS[l.font || 'body'];
      const font = `${l.weight || 500} ${Math.round((l.size || 22) * scale)}px ${fam}`;
      ctx.font = font;
      const w = ctx.measureText(l.text).width;
      return { font, w, h: (l.size || 22) * scale * 1.28 };
    });
    const W = Math.ceil(Math.max(8, ...metrics.map((m) => m.w)) + pad * 2);
    const Hh = Math.ceil(metrics.reduce((s, m) => s + m.h, 0) + pad * 2);
    if (canvas.width !== W || canvas.height !== Hh) {
      canvas.width = W; canvas.height = Hh;
      tex.dispose();                    // WebGL2 texture storage is immutable: re-allocate at the new size
    }
    ctx.clearRect(0, 0, W, Hh);
    if (opts.bg) {
      ctx.fillStyle = opts.bg;
      const r = 14 * scale;
      ctx.beginPath();
      ctx.moveTo(r, 0); ctx.arcTo(W, 0, W, Hh, r); ctx.arcTo(W, Hh, 0, Hh, r); ctx.arcTo(0, Hh, 0, 0, r); ctx.arcTo(0, 0, W, 0, r);
      ctx.closePath(); ctx.fill();
    }
    let y = pad;
    current.forEach((l, i) => {
      const m = metrics[i];
      ctx.font = m.font;
      ctx.fillStyle = l.color || COLORS.ink;
      ctx.textBaseline = 'middle';
      ctx.textAlign = opts.align === 'left' ? 'left' : 'center';
      ctx.fillText(l.text, opts.align === 'left' ? pad : W / 2, y + m.h / 2);
      y += m.h;
    });
    tex.needsUpdate = true;
    sprite.scale.set((worldH * W) / Hh, worldH, 1);
  };
  draw();
  const api = {
    sprite,
    set(next) { current = next; draw(); },
    redraw: draw,
    dispose() { labelRegistry.delete(api); tex.dispose(); mat.dispose(); },
  };
  labelRegistry.add(api);
  return api;
}
export function redrawAllLabels() { for (const l of labelRegistry) l.redraw(); }

// ------------------------------------------------------------------ the world
export function createWorld(container, opts = {}) {
  const reduced = !!opts.reduced;
  const mobile = !!opts.mobile;

  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', alpha: false, stencil: false });
  const dprCap = mobile ? 1.5 : 2;
  let dpr = Math.min(window.devicePixelRatio || 1, dprCap);
  renderer.setPixelRatio(dpr);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(FOG_HEX, 1);
  renderer.domElement.id = 'gl';
  renderer.domElement.setAttribute('aria-label', 'Interactive 3-D view of the circular network');
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(FOG_HEX);
  scene.fog = new THREE.FogExp2(FOG_HEX, sharedUniforms.uFogDensity.value);

  const camera = new THREE.PerspectiveCamera(mobile ? 50 : 40, 1, 0.1, 1600);
  camera.position.set(0, 18, 40);
  camera.layers.enable(LABEL_LAYER);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 3;
  controls.maxDistance = 140;
  controls.maxPolarAngle = Math.PI * 0.86;
  controls.target.set(0, 3, 0);
  controls.autoRotateSpeed = 0.16;
  controls.zoomToCursor = false;

  // post-processing: render -> bloom -> output (tone mapping + sRGB); the same path with bloom off
  const rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: mobile ? 0 : 4 });
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.62, 0.45, 0.8);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // ---------------------------------------------------------------- sky + stars + dust
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(700, 48, 24),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { uTop: { value: new THREE.Color('#03050A') }, uHorizon: { value: new THREE.Color('#0B1322') }, uA: { value: new THREE.Color(COLORS.violet) }, uB: { value: new THREE.Color(COLORS.teal) } },
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uTop, uHorizon, uA, uB;
        varying vec3 vDir;
        float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
        float noise(vec3 x) {
          vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
          return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                     mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
        }
        void main() {
          vec3 d = normalize(vDir);
          float h = d.y;
          vec3 col = mix(uHorizon, uTop, smoothstep(-0.25, 0.75, h));
          float n = noise(d * 2.6) * 0.62 + noise(d * 6.1) * 0.28 + noise(d * 13.0) * 0.10;
          col += uA * smoothstep(0.55, 0.92, n) * 0.035 * (1.0 - abs(h));
          col += uB * smoothstep(0.62, 0.95, noise(d * 1.9 + 7.3)) * 0.025;
          gl_FragColor = vec4(col, 1.0);
          #include <colorspace_fragment>
        }`,
    }),
  );
  sky.renderOrder = -10;
  sky.frustumCulled = false;
  scene.add(sky);

  const stars = (() => {
    const n = mobile ? 1400 : 2600;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), size = new Float32Array(n), ph = new Float32Array(n);
    const tints = [new THREE.Color(COLORS.ink), new THREE.Color(COLORS.teal), new THREE.Color(COLORS.violet), new THREE.Color('#FFE9C7')];
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, r = 520 + Math.random() * 120;
      const s = Math.sqrt(1 - u * u);
      pos.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
      const t = Math.random() < 0.82 ? tints[0] : tints[1 + Math.floor(Math.random() * 3)];
      const b = 0.35 + Math.pow(Math.random(), 3) * 0.9;
      col.set([t.r * b, t.g * b, t.b * b], i * 3);
      size[i] = 0.7 + Math.pow(Math.random(), 4) * 2.4;
      ph[i] = Math.random() * 6.283;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aPhase', new THREE.BufferAttribute(ph, 1));
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
      uniforms: { uTime: sharedUniforms.uTime, uPx: { value: dpr }, uTwinkle: { value: reduced ? 0 : 1 } },
      vertexShader: /* glsl */ `
        attribute vec3 aColor; attribute float aSize; attribute float aPhase;
        uniform float uTime, uPx, uTwinkle;
        varying vec3 vC;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          float tw = 1.0 - uTwinkle * 0.35 * (0.5 + 0.5 * sin(uTime * 1.3 + aPhase * 3.0));
          vC = aColor * tw;
          gl_PointSize = aSize * uPx;
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vC;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(vC * a, 1.0);
          #include <colorspace_fragment>
        }`,
    });
    const p = new THREE.Points(g, m);
    p.frustumCulled = false;
    p.renderOrder = -9;
    return p;
  })();
  scene.add(stars);

  const dust = (() => {
    const n = mobile ? 260 : 520;
    const pos = new Float32Array(n * 3), ph = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = 4 + Math.random() * 30, th = Math.random() * Math.PI * 2;
      pos.set([r * Math.cos(th), -6 + Math.random() * 20, r * Math.sin(th)], i * 3);
      ph[i] = Math.random() * 100;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aPhase', new THREE.BufferAttribute(ph, 1));
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...sharedUniforms, uColor: { value: new THREE.Color(COLORS.muted) } },
      vertexShader: /* glsl */ `
        attribute float aPhase;
        uniform float uTime, uProjScale;
        varying float vFogDepth; varying float vA;
        void main() {
          vec3 p = position;
          p.x += sin(uTime * 0.05 + aPhase) * 1.2;
          p.y += sin(uTime * 0.07 + aPhase * 1.7) * 0.8;
          p.z += cos(uTime * 0.04 + aPhase * 0.6) * 1.2;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          vFogDepth = -mv.z;
          vA = 0.10 + 0.08 * sin(aPhase * 5.0);
          gl_PointSize = clamp(0.07 * uProjScale / max(1.0, -mv.z), 1.0, 6.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        varying float vA;
        ${FOG_PARS_F}
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.0, d) * vA * (1.0 - fogFactor());
          gl_FragColor = vec4(uColor * a, 1.0);
          #include <colorspace_fragment>
        }`,
    });
    const p = new THREE.Points(g, m);
    p.frustumCulled = false;
    return p;
  })();
  scene.add(dust);

  // ---------------------------------------------------------------- the track: a luminous ribbon that follows the chain
  const track = (() => {
    const SEG = 900;
    const n = (SEG + 1) * 2;
    const aU = new Float32Array(n), aSide = new Float32Array(n), pos = new Float32Array(n * 3);
    for (let i = 0; i <= SEG; i++) {
      aU[2 * i] = aU[2 * i + 1] = i / SEG;
      aSide[2 * i] = -1; aSide[2 * i + 1] = 1;
    }
    const idx = [];
    for (let i = 0; i < SEG; i++) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aU', new THREE.BufferAttribute(aU, 1));
    g.setAttribute('aSide', new THREE.BufferAttribute(aSide, 1));
    g.setIndex(idx);
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      defines: { NL },
      uniforms: {
        ...chainUniforms, ...sharedUniforms,
        uTrackA: chainUniforms.uTrackA, uTrackB: chainUniforms.uTrackB,
        uCore: { value: new THREE.Color('#9FB6D6') }, uEdge: { value: new THREE.Color(COLORS.muted) },
        uPulse: { value: new THREE.Color(COLORS.teal) }, uWidth: { value: 0.62 }, uGain: { value: 1 },
      },
      vertexShader: /* glsl */ `
        ${CHAIN_GLSL}
        uniform float uTrackA, uTrackB, uWidth;
        attribute float aU; attribute float aSide;
        varying float vSide; varying float vSig; varying float vU; varying float vFogDepth;
        void main() {
          float sig = mix(uTrackA, uTrackB, aU);
          vec3 T, U;
          vec3 p = chainFrame(sig, T, U);
          p += U * aSide * uWidth;
          p.y -= 0.02;
          vSide = aSide; vSig = sig; vU = aU;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vFogDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uCore, uEdge, uPulse;
        uniform float uTime, uR, uFold, uGain;
        varying float vSide; varying float vSig; varying float vU;
        ${FOG_PARS_F}
        void main() {
          float s = abs(vSide);
          float core = exp(-s * s * 160.0);
          float rails = exp(-pow((s - 0.93) * 22.0, 2.0)) * 0.55;
          float body = 0.05 * (1.0 - s);
          float tick = smoothstep(0.92, 1.0, fract(vSig / (3.14159265 * uR / 12.0))) * (1.0 - s) * 0.22;
          float pulse = pow(0.5 + 0.5 * sin(vSig * 0.55 - uTime * 2.2), 18.0) * core * 0.9;
          float ends = smoothstep(0.0, 0.05, vU) * smoothstep(1.0, 0.95, vU);
          ends = mix(1.0, ends, smoothstep(0.0, 0.3, uFold));
          vec3 col = uCore * (core * 1.25 + body) + uEdge * (rails + tick) + uPulse * pulse;
          col *= ends * uGain * (1.0 - fogFactor());
          gl_FragColor = vec4(col, 1.0);
          #include <colorspace_fragment>
        }`,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1;
    return mesh;
  })();
  scene.add(track);

  // ---------------------------------------------------------------- state
  const state = {
    reduced, mobile, dpr, dprCap,
    fold: 0, foldTarget: 0, foldAnim: false, foldListeners: new Set(),
    fly: null, ride: null, autoRotate: !reduced, idleUntil: performance.now() + 45000,   // first, hold the composed view
    insets: { left: 0, right: 0, bottom: 0, top: 0 },
    w: 1, h: 1,
    fps: 60, frames: 0, fpsT: 0, quality: { auto: true, bloom: true, note: '' }, qualityListeners: new Set(),
    lastInteraction: performance.now(),
  };
  controls.addEventListener('start', () => {
    state.idleUntil = performance.now() + 25000;
    if (state.fly) { state.fly = null; controls.enabled = true; }
  });
  renderer.domElement.addEventListener('pointerdown', () => { if (state.ride) stopRide(); });

  // ---------------------------------------------------------------- sizing
  function resize() {
    const w = Math.max(1, container.clientWidth), h = Math.max(1, container.clientHeight);
    state.w = w; state.h = h;
    renderer.setSize(w, h);
    composer.setPixelRatio(state.dpr);
    composer.setSize(w, h);
    applyViewOffset();
    camera.updateProjectionMatrix();
    updateProjScale();
    applyLayout();
  }
  // shift the projection centre into the part of the screen not covered by the HUD (panel, caption, bottom sheet)
  function viewGeom() {
    const w = state.w, h = state.h, I = state.insets;
    const L = Math.min(I.left || 0, w * 0.4), R = Math.min(I.right || 0, w * 0.45);
    const T = Math.min(I.top || 0, h * 0.4), B = Math.min(I.bottom || 0, h * 0.6);
    const cx = (L - R) / 2, cy = (T - B) / 2;          // offset of the uncovered centre from the screen centre (y down)
    return { cx, cy, uw: Math.max(40, w - L - R), uh: Math.max(40, h - T - B), fullW: w + 2 * Math.abs(cx), fullH: h + 2 * Math.abs(cy) };
  }
  function applyViewOffset() {
    const w = state.w, h = state.h;
    const g = viewGeom();
    if (Math.abs(g.cx) < 1 && Math.abs(g.cy) < 1) { camera.aspect = w / h; camera.clearViewOffset(); return; }
    camera.aspect = g.fullW / g.fullH;                  // aspect of the full virtual image keeps pixels square
    camera.setViewOffset(g.fullW, g.fullH, g.cx > 0 ? 0 : -2 * g.cx, g.cy > 0 ? 0 : -2 * g.cy, w, h);
  }
  function updateProjScale() {
    const fullH = camera.view && camera.view.enabled ? camera.view.fullHeight : state.h;
    const s = (fullH * state.dpr) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    sharedUniforms.uProjScale.value = s;
    stars.material.uniforms.uPx.value = state.dpr;
  }
  function setInsets(ins) {
    Object.assign(state.insets, ins);
    applyViewOffset();
    camera.updateProjectionMatrix();
    updateProjScale();
    applyLayout();
  }
  function applyLayout() {
    const g = viewGeom();
    const L = g.uw / g.uh < 1.05 ? LAYOUTS.portrait : LAYOUTS.landscape;
    state.layout = L === LAYOUTS.portrait ? 'portrait' : 'landscape';
    BOARD_POS.copy(L.board);
  }
  window.addEventListener('resize', resize);
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(container);
  resize();

  // ---------------------------------------------------------------- fold
  function setFoldValue(f, fromUser = true) {
    state.fold = Math.min(1, Math.max(0, f));
    if (fromUser) { state.foldTarget = state.fold; state.foldAnim = false; }
    setFold(state.fold);
    for (const fn of state.foldListeners) fn(state.fold);
  }
  function animateFoldTo(t) {
    state.foldTarget = Math.min(1, Math.max(0, t));
    const dist = Math.abs(state.foldTarget - state.fold);
    // wall-clock timed (not dt-driven): completes on time even when the frame rate is low
    state.foldAnim = dist > 1e-4 ? { from: state.fold, t0: performance.now(), dur: (state.reduced ? 1000 : 3400) * Math.max(0.25, dist) } : false;
  }
  function onFold(fn) { state.foldListeners.add(fn); return () => state.foldListeners.delete(fn); }

  // ---------------------------------------------------------------- camera presets and flights
  /** half-angle tangents of the uncovered part of the screen */
  function viewTans() {
    const t = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const g = viewGeom();
    return { th: Math.max(1e-3, (t * g.uw) / g.fullH), tv: Math.max(1e-3, (t * g.uh) / g.fullH) };
  }
  /** distance at which a sphere of `radius` fits the uncovered part of the screen */
  function fitDistance(radius) {
    const { th, tv } = viewTans();
    return radius / Math.sin(Math.atan(Math.min(tv, th)));
  }
  /**
   * Frame a set of world points from direction `dir` (target -> camera): re-centre the target on the points' screen
   * extent and pick the smallest distance at which every point is inside the uncovered screen area (5% margin).
   */
  function frameFit(points, dir, target0) {
    const f = dir.clone().normalize();
    const right = new THREE.Vector3().crossVectors(f.clone().negate(), new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, f.clone().negate()).normalize();
    const { th, tv } = viewTans();
    const target = target0.clone();
    const rel = new THREE.Vector3();
    let d = 30;
    for (let pass = 0; pass < 3; pass++) {
      d = 0;
      for (const P of points) {
        rel.subVectors(P, target);
        const x = rel.dot(right), y = rel.dot(up), z = rel.dot(f);
        d = Math.max(d, z + Math.max(Math.abs(x) / th, Math.abs(y) / tv));
      }
      d *= 1.05;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const P of points) {
        rel.subVectors(P, target);
        const depth = d - rel.dot(f);
        const sx = rel.dot(right) / depth, sy = rel.dot(up) / depth;
        x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
      }
      target.addScaledVector(right, 0.5 * (x0 + x1) * d).addScaledVector(up, 0.5 * (y0 + y1) * d);
    }
    return { pos: target.clone().addScaledVector(f, d), target };
  }
  const box = (c, hx, hy, hz, out) => {
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) out.push(new THREE.Vector3(c.x + sx * hx, c.y + sy * hy, c.z + sz * hz));
    return out;
  };
  function yawDir() {
    const d = camera.position.clone().sub(controls.target);
    d.y = 0;
    if (d.lengthSq() < 1e-6) d.set(0, 0, 1);
    return d.normalize();
  }
  function presetPose(name) {
    const unrolled = state.foldTarget > 0.5;
    const H = PANEL_Y + 3.2 + 1.6;                         // top of a panel incl. its label
    if (name === 'overview') {
      const pts = [];
      if (unrolled) {
        const L = ((NL - 1) * LAYER_GAP) / 2 + 3.4;
        box(new THREE.Vector3(0, H / 2, 0), L, H / 2, 3.4, pts);
        box(BOARD_POS_UNF, 3.6 * BOARD_SCALE * 1.15, 5.2 * BOARD_SCALE, 0.5, pts);
        box(PED_POS_UNF.clone().add(new THREE.Vector3(0, 1, 0)), 2.6, 1.2, 2.6, pts);
        // look along the chain (panels face along it) so the layers read as a stack receding into depth
        return frameFit(pts, new THREE.Vector3(0.62, 0.42, 0.66), new THREE.Vector3(0, 4, 0));
      }
      box(new THREE.Vector3(0, 0, 0), RING_R + 2.6, 0.2, RING_R + 2.6, pts);   // ring + bundles
      pts.push(new THREE.Vector3(0, H, RING_R), new THREE.Vector3(0, H, -RING_R), new THREE.Vector3(3.2, H, RING_R), new THREE.Vector3(-3.2, H, -RING_R));
      box(BOARD_POS.clone().add(new THREE.Vector3(0, 0.8, 0)), 4.0 * BOARD_SCALE, 5.4 * BOARD_SCALE, 0.4, pts);
      box(PED_POS.clone().add(new THREE.Vector3(1.2, 1.2, 0)), 3.6, 1.4, 2.6, pts);
      return frameFit(pts, new THREE.Vector3(0, 0.5, 1), new THREE.Vector3(3, 4, 0));
    }
    if (name === 'board') {
      const c = anchors.board.clone().add(new THREE.Vector3(0, 0.55 * BOARD_SCALE, 0));
      const pts = box(c, 4.7 * BOARD_SCALE, 5.6 * BOARD_SCALE, 0.2, []);
      pts.push(c.clone().add(new THREE.Vector3(0, 8.2 * BOARD_SCALE, 0)));   // headroom: keeps the title clear of the top-left buttons
      return frameFit(pts, yawDir().add(new THREE.Vector3(0, 0.12, 0)), c);
    }
    if (name === 'injection') {
      const fr = panelFrameJS(NL - 1);
      const target = fr.C.clone().add(new THREE.Vector3(0, -1.2, 2.5));
      const pts = box(fr.C.clone().add(new THREE.Vector3(0, 0.6, 0)), 3.5, 4.2, 0.5, []);   // panel + its label
      box(anchors.ped.clone().add(new THREE.Vector3(1.5, 1, 0)), 4, 1.5, 2.6, pts);
      return frameFit(pts, new THREE.Vector3(0.55, 0.45, 1), target);
    }
    return null;
  }
  function layerPose(l) {
    const fr = panelFrameJS(l);
    const Y = new THREE.Vector3(0, 1, 0);
    const pts = [];
    for (const a of [-3.5, 3.5]) for (const y of [-3.5, 4.6]) for (const n of [-0.5, 0.5]) {
      pts.push(fr.C.clone().addScaledVector(fr.A, a).addScaledVector(Y, y).addScaledVector(fr.N, n));   // panel + its label
    }
    const dir = fr.N.clone().add(new THREE.Vector3(0, 0.55, 0)).addScaledVector(fr.A, 0.18);  // raised: clears the pedestal
    return frameFit(pts, dir, fr.C);
  }
  function flyTo(pose, dur = 1.3) {
    if (!pose) return;
    stopRide();
    const d = state.reduced ? Math.min(0.35, dur) : dur;
    state.fly = { p0: camera.position.clone(), t0: controls.target.clone(), p1: pose.pos.clone(), t1: pose.target.clone(), start: performance.now(), dur: d };
    controls.enabled = false;
    state.idleUntil = performance.now() + 25000;
  }
  function preset(name) {
    if (name === 'ride') { startRide(); return; }
    flyTo(presetPose(name));
  }
  /** jump to a preset without a flight (used once the HUD has reported its size) */
  function snap(name) {
    const pose = presetPose(name);
    if (!pose) return;
    state.fly = null;
    camera.position.copy(pose.pos);
    controls.target.copy(pose.target);
    camera.lookAt(controls.target);
    controls.update();
  }
  function startRide() {
    state.fly = null;
    controls.enabled = false;
    controls.autoRotate = false;
    const unrolled = state.foldTarget > 0.5;
    state.ride = { sig: unrolled ? layerSigma(0) : -0.25 * Math.PI * RING_R, dir: 1 };
  }
  function stopRide() {
    if (!state.ride) return;
    state.ride = null;
    controls.enabled = true;
    state.idleUntil = performance.now() + 25000;
  }

  // ---------------------------------------------------------------- adaptive quality
  function setBloom(on, byUser = true) {
    bloom.enabled = !!on;
    state.quality.bloom = !!on;
    if (byUser) state.quality.auto = false;
    state.quality.note = byUser ? '' : state.quality.note;
    for (const fn of state.qualityListeners) fn(state.quality);
  }
  function monitorFps(dt) {
    state.frames++;
    state.fpsT += dt;
    if (state.fpsT < 2.0) return;
    state.fps = state.frames / state.fpsT;
    state.frames = 0; state.fpsT = 0;
    if (!state.quality.auto || document.hidden) return;
    if (state.fps < 38 && bloom.enabled) {
      bloom.enabled = false; state.quality.bloom = false;
      state.quality.note = `bloom switched off automatically (${state.fps.toFixed(0)} fps)`;
      for (const fn of state.qualityListeners) fn(state.quality);
    } else if (state.fps < 26 && state.dpr > 1) {
      state.dpr = 1;
      renderer.setPixelRatio(1);
      resize();
      state.quality.note = `resolution lowered automatically (${state.fps.toFixed(0)} fps)`;
      for (const fn of state.qualityListeners) fn(state.quality);
    }
  }

  // ---------------------------------------------------------------- per-frame update
  const tmpA = { P: new THREE.Vector3(), T: new THREE.Vector3(), U: new THREE.Vector3() };
  const tmpB = { P: new THREE.Vector3(), T: new THREE.Vector3(), U: new THREE.Vector3() };
  function update(dt) {
    // fold animation
    if (state.foldAnim) {
      const A = state.foldAnim;
      const k = Math.min(1, (performance.now() - A.t0) / A.dur);
      setFoldValue(A.from + (state.foldTarget - A.from) * easeInOut(k), false);
      if (k >= 1) state.foldAnim = false;
    }
    // camera: flight, ride or orbit
    if (state.fly) {
      const F = state.fly;
      F.t = Math.min(1, (performance.now() - F.start) / (1000 * F.dur));     // wall-clock timed
      const k = easeInOut(F.t);
      camera.position.lerpVectors(F.p0, F.p1, k);
      controls.target.lerpVectors(F.t0, F.t1, k);
      // lift the path a little so flights arc over the scene
      camera.position.y += Math.sin(Math.PI * k) * 0.12 * F.p0.distanceTo(F.p1);
      camera.lookAt(controls.target);
      if (F.t >= 1) { state.fly = null; controls.enabled = true; }
    } else if (state.ride) {
      const unrolled = state.fold > 0.5;
      const speed = (state.reduced ? 1.0 : 2.6) / Math.max(0.15, chainUniforms.uG.value);
      const R = state.ride;
      R.sig += R.dir * speed * dt;
      if (unrolled) {
        const lo = layerSigma(0) - 0.4 * Math.PI * RING_R, hi = 0.4 * Math.PI * RING_R;
        if (R.sig > hi) { R.sig = hi; R.dir = -1; }
        if (R.sig < lo) { R.sig = lo; R.dir = 1; }
      } else if (R.sig > 0) R.sig -= 2 * Math.PI * RING_R;
      const a = chainFrameJS(R.sig, tmpA);
      const b = chainFrameJS(R.sig + R.dir * 0.42 * Math.PI * RING_R * Math.min(1, 0.5 + chainUniforms.uG.value), tmpB);
      const pos = a.P.clone().addScaledVector(a.U, unrolled ? 9.5 : 6.2).add(new THREE.Vector3(0, unrolled ? 6.5 : 4.4, 0));
      const tgt = b.P.clone().add(new THREE.Vector3(0, PANEL_Y - 0.4, 0));
      const k = 1 - Math.exp(-dt * 3.0);
      camera.position.lerp(pos, k);
      controls.target.lerp(tgt, k);
      camera.lookAt(controls.target);
    } else {
      controls.autoRotate = state.autoRotate && !state.reduced && performance.now() > state.idleUntil;
      controls.update(dt);
    }
    // keep sky + stars at infinity
    sky.position.copy(camera.position);
    stars.position.copy(camera.position);
    sharedUniforms.uTime.value += dt * (state.reduced ? 0.35 : 1);
    monitorFps(dt);
  }
  function render() { composer.render(); }

  // ---------------------------------------------------------------- picking helpers
  const raycaster = new THREE.Raycaster();
  function pick(clientX, clientY, objects) {
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    return raycaster.intersectObjects(objects, false);
  }

  // initial pose
  const p0 = presetPose('overview');
  camera.position.copy(p0.pos);
  controls.target.copy(p0.target);
  controls.update();

  return {
    THREE, renderer, scene, camera, controls, composer, bloom, track, sky, stars, dust, state,
    update, render, resize, setInsets, pick,
    preset, snap, presetPose, layerPose, flyTo, startRide, stopRide, fitDistance, frameFit,
    setFoldValue, animateFoldTo, onFold, getFold: () => state.fold,
    setBloom, onQuality: (fn) => { state.qualityListeners.add(fn); return () => state.qualityListeners.delete(fn); },
    setAutoRotate(on) { state.autoRotate = !!on; if (!on) controls.autoRotate = false; },
    setLabels(on) { if (on) camera.layers.enable(LABEL_LAYER); else camera.layers.disable(LABEL_LAYER); },
    isRiding: () => !!state.ride,
  };
}
