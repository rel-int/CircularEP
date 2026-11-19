// chain.js — the FOLD / UNFOLD geometry shared by every chain-aware material (GLSL chunk + JS mirror).
//
// The unrolled network has NL = 2*KMAX + 1 layers: s_{k0}, u_{k0+1}, s_{k0+1}, ..., u_{k0+K}, s_{k0+K}.
// Layer l sits at arc length sigma_l = (l - (NL-1)) * pi * R along a curve that morphs with the fold value f:
//   f = 0  -> the curve is the ring of radius R (every s layer lands on the s station at the front, every u layer
//             on the u station at the back: the loop);
//   f = 1  -> the curve is a straight line along +x (the unrolled feedforward network), centred on the origin.
// In between, the circle "unwinds" (radius R/(1-f)^2, arc length preserved up to the spacing scale) and lifts into
// a gentle helix so the turns do not overlap. Panels (one per layer) are tangent to the ring when folded and turn
// to face along the chain when unfolded.
import * as THREE from 'three';
import { KMAX } from './sim.js';

export const NL = 2 * KMAX + 1;
export const RING_R = 13;
export const LAYER_GAP = 10;             // world distance between consecutive layers when unfolded
export const PANEL_Y = 3.6;              // panel centre height above the track
export const PANEL_W = 6.4;              // panel width/height (16 x 16 neurons in 4 x 4 blocks)
const PI = Math.PI;
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export const chainUniforms = {
  uR: { value: RING_R },
  uG: { value: 1 },
  uCoilR: { value: RING_R },
  uRise: { value: 0 },
  uShift: { value: new THREE.Vector3() },
  uTurn: { value: 0 },
  uSigMid: { value: -KMAX * PI * RING_R },
  uPanelY: { value: PANEL_Y },
  uFold: { value: 0 },
  uLayerAlpha: { value: new Float32Array(NL) },
  uLayerLit: { value: new Float32Array(NL) },
  uTrackA: { value: -2 * PI * RING_R },
  uTrackB: { value: 0 },
};

export const CHAIN_GLSL = /* glsl */ `
#ifndef PI
#define PI 3.141592653589793
#endif
uniform float uR;
uniform float uG;
uniform float uCoilR;
uniform float uRise;
uniform vec3 uShift;
uniform float uTurn;
uniform float uSigMid;
uniform float uPanelY;
uniform float uFold;
uniform float uLayerAlpha[NL];
uniform float uLayerLit[NL];
float layerParity(float l) { return mod(l + 0.5, 2.0) > 1.0 ? 1.0 : 0.0; }
float layerSigma(float l) { return (l - float(NL - 1)) * PI * uR; }
vec3 chainFrame(float sig, out vec3 T, out vec3 U) {
  float r = uCoilR;
  float phi = sig * uG / r;
  float sn = sin(phi), cs = cos(phi), hh = sin(0.5 * phi);
  vec3 p = vec3(r * sn, 0.0, uR - 2.0 * r * hh * hh);
  T = vec3(cs, 0.0, -sn);
  U = vec3(sn, 0.0, cs);
  p.y += uRise * (sig - uSigMid) * uG / (2.0 * PI * r);   // rise per actual winding: a spring pulled apart
  return p + uShift;
}
float panelAngle(float l) { return mix(layerParity(l) * PI, 0.5 * PI, uTurn); }
vec3 panelPoint(float l, vec2 g, float d, out vec3 A, out vec3 N) {
  vec3 T, U;
  vec3 C = chainFrame(layerSigma(l), T, U);
  float a = panelAngle(l);
  A = cos(a) * T + sin(a) * U;
  N = cross(A, vec3(0.0, 1.0, 0.0));
  return C + A * g.x + vec3(0.0, uPanelY + g.y, 0.0) + N * d;
}
// a point at fraction t of a synapse from (layer l, grid gs) to (layer l+1, grid gd), bundled by pinch/spread
vec3 synapsePoint(float l, vec2 gs, vec2 gd, float t, float rnd, float pinch, float spread) {
  vec3 T, U;
  vec3 C = chainFrame(mix(layerSigma(l), layerSigma(l + 1.0), t), T, U);
  float a = mix(panelAngle(l), panelAngle(l + 1.0), t);
  vec3 A = cos(a) * T + sin(a) * U;
  vec3 N = cross(A, vec3(0.0, 1.0, 0.0));
  float bow = sin(PI * t);
  vec2 g = mix(gs, gd, t) * (1.0 - pinch * bow);
  float ang = 6.2831853 * rnd;
  float rr = spread * bow * (0.3 + 0.7 * fract(rnd * 7.31));
  vec3 off = N * (cos(ang) * rr) + vec3(0.0, sin(ang) * rr * 0.7, 0.0);
  return C + A * g.x + vec3(0.0, uPanelY + g.y, 0.0) + off;
}
`;

const _state = { f: 0 };

/** set the fold value (0 = ring, 1 = unrolled) and update every derived uniform */
export function setFold(f) {
  f = Math.min(1, Math.max(0, f));
  _state.f = f;
  const U = chainUniforms;
  const R = RING_R;
  const e = smooth(0, 1, f);
  const gEnd = LAYER_GAP / (PI * R);
  U.uFold.value = f;
  U.uG.value = 1 + (gEnd - 1) * e;
  U.uCoilR.value = Math.min(1e6, R / Math.pow(Math.max(1e-4, 1 - f), 2));
  U.uRise.value = 6.5 * Math.sin(PI * f);
  const L = (NL - 1) * LAYER_GAP;
  U.uShift.value.set(e * L * 0.5, 0, -e * R);
  U.uTurn.value = smooth(0.2, 0.9, f);
  const span = (NL - 1) * PI * R;
  const ext = smooth(0, 0.35, f);
  U.uTrackA.value = -2 * PI * R - ext * (span - 2 * PI * R + 0.6 * PI * R);
  U.uTrackB.value = ext * 0.6 * PI * R;
}
export function getFold() { return _state.f; }
export function layerSigma(l) { return (l - (NL - 1)) * PI * RING_R; }
export function layerParity(l) { return l % 2 === 1 ? 1 : 0; }

/** JS mirror of chainFrame: {P, T, U} */
export function chainFrameJS(sig, out) {
  const U = chainUniforms, R = RING_R;
  const r = U.uCoilR.value;
  const phi = (sig * U.uG.value) / r;
  const sn = Math.sin(phi), cs = Math.cos(phi), hh = Math.sin(0.5 * phi);
  const o = out || { P: new THREE.Vector3(), T: new THREE.Vector3(), U: new THREE.Vector3() };
  o.P.set(r * sn, 0, R - 2 * r * hh * hh);
  o.P.y += (U.uRise.value * (sig - U.uSigMid.value) * U.uG.value) / (2 * PI * r);
  o.P.add(U.uShift.value);
  o.T.set(cs, 0, -sn);
  o.U.set(sn, 0, cs);
  return o;
}
/** JS mirror of panelPoint: returns {P (point), A (panel x-axis), N (panel normal), C (panel centre)} */
export function panelFrameJS(l, gx = 0, gy = 0, d = 0) {
  const fr = chainFrameJS(layerSigma(l));
  const a = layerParity(l) * PI + (0.5 * PI - layerParity(l) * PI) * chainUniforms.uTurn.value;
  const A = fr.T.clone().multiplyScalar(Math.cos(a)).addScaledVector(fr.U, Math.sin(a));
  const N = new THREE.Vector3().crossVectors(A, new THREE.Vector3(0, 1, 0));
  const C = fr.P.clone().add(new THREE.Vector3(0, PANEL_Y, 0));
  const P = C.clone().addScaledVector(A, gx).add(new THREE.Vector3(0, gy, 0)).addScaledVector(N, d);
  return { P, A, N, C };
}

setFold(0);
