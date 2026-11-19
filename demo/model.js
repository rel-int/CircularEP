/**
 * model.js -- inference engine of the Circular Predictive Coding 4×4 Sudoku demo.
 * JavaScript port of the JAX model in this repository (circular predictive coding trained with equilibrium propagation):
 * a trained two-layer network, i.e. a dense feedforward network for 4x4 Sudoku folded into a loop.
 *
 * Pure ES module: no DOM, no three.js, no imports; runs in browsers and in node >= 18. Batch size 1. States, weights
 * and outputs are row-major (C-order) Float32Array; hot loops accumulate in JS doubles / Float64Array scratch that is
 * preallocated once per Model (no allocation inside the hot loops).
 *
 * LAYOUT:  x (din = P (N+1) = 80): x[p (N+1) + puzzle[p]] = 1.   state z = {u: F32(H), s: F32(H)}, H = 256.
 *   flat state = concat(u, s) (u first, then s), D = 2 H = 512.
 *   theta: Win (din, H), bin (H), W1 (H, H), b1 (H), W2 (H, H), b2 (H).   psi: V (H, P N), c (P N).
 *
 * MATH (row-vector convention x @ W; act = meta.act = "tanh"; no normalisation anywhere):
 *   x_inj    = x Win + bin
 *   hat u(s) = act(s W1 + b1)                 hat s(u) = act(u W2 + b2 + x_inj)
 *   sweep:   u <- hat u(s);  s <- hat s(u)     lap(s) = hat s(hat u(s))
 *   E(z)     = 1/2 (|u - hat u(s)|^2 + |s - hat s(u)|^2)
 *   logits   = s V + c viewed (P, N);   C(s, y) = mean_p [logsumexp(logits[p, :]) - logits[p, y_p]]
 */

// =====================================================================================================================
// constants
// =====================================================================================================================
export const VERSION = '2.0.1';
export const VODES = Object.freeze(['u', 's']);
export const THETA_NAMES = Object.freeze(['Win', 'bin', 'W1', 'b1', 'W2', 'b2']);
export const PSI_NAMES = Object.freeze(['V', 'c']);
export const SOLVER_KINDS = Object.freeze(['sweep', 'damped', 'gd', 'anderson', 'nudge']);
/** Fallback solver settings: free-phase sweeps, trace length, beta, nudge steps, GD step size / momentum, Anderson
 *  memory m and ridge lam, damping a. Overridden by values in the model file where present (see Model constructor). */
export const DEFAULTS = Object.freeze({ T_free: 60, T_trace: 300, beta: 0.05, T_nudg: 1500, lr_h: 0.025, mom: 0.9, m: 6, lam: 1e-2, a: 0.5 });
/** selftest pass thresholds on maxRel (see selftest). residualAtol: absolute floor for the relative-residual series
 *  (late residuals sit at the float32 noise floor, ~1e-7), i.e. pass iff |d| <= max(iter * |ref|, residualAtol). */
export const TOL = Object.freeze({ map: 1e-4, iter: 1e-3, ep: 1e-3, anderson: 1e-2, energyFloor: 1e-4, residualAtol: 1e-6 });
const BOX_DEFAULT = { 4: [2, 2], 6: [2, 3], 9: [3, 3] };        // Sudoku box shape (box_h, box_w) per board size N

// =====================================================================================================================
// base64 / array decoding  (model.json: float32 arrays = base64 of little-endian C-order bytes)
// =====================================================================================================================
const B64_LUT = (() => {
  const t = new Uint8Array(256).fill(255);
  const s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  for (let i = 0; i < 64; i++) t[s.charCodeAt(i)] = i;
  t[45] = 62; t[95] = 63;                                         // URL-safe '-' and '_'
  return t;
})();
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** base64 (standard or URL-safe alphabet; whitespace ignored; stops at '=') -> Uint8Array. */
export function base64ToBytes(str) {
  const n = str.length;
  const out = new Uint8Array(Math.floor((n * 3) / 4) + 3);
  let o = 0, acc = 0, bits = 0;
  for (let i = 0; i < n; i++) {
    const code = str.charCodeAt(i);
    if (code === 61) break;                                       // '=' padding
    const v = code < 256 ? B64_LUT[code] : 255;
    if (v === 255) continue;                                      // whitespace / newlines
    acc = ((acc << 6) | v) & 0xffff;
    bits += 6;
    if (bits >= 8) { bits -= 8; out[o++] = (acc >> bits) & 0xff; }
  }
  return out.subarray(0, o);
}

/** Little-endian float32 bytes -> Float32Array with its own compact buffer. */
export function bytesToFloat32(bytes) {
  if (bytes.length % 4 !== 0) throw new Error(`bytesToFloat32: ${bytes.length} bytes is not a multiple of 4`);
  const n = bytes.length >> 2;
  if (LITTLE_ENDIAN) return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(4 * i, true);
  return out;
}

/**
 * Decode one model.json array: {shape, b64} | {shape, data} | (nested) number array | typed array | number.
 * Returns a Float32Array; a declared shape is attached as the non-enumerable property `.shape`.
 */
export function decodeArray(obj, what = 'array') {
  if (obj == null) return null;
  let data, shape = null;
  if (obj instanceof Float32Array) data = obj;
  else if (typeof obj === 'number') data = Float32Array.of(obj);
  else if (Array.isArray(obj)) data = Float32Array.from(obj.flat(Infinity));
  else if (ArrayBuffer.isView(obj)) data = Float32Array.from(obj);
  else if (typeof obj === 'object') {
    if (obj.dtype != null && !/^(<?f4|float32)$/.test(String(obj.dtype))) {
      throw new Error(`decodeArray(${what}): unsupported dtype ${obj.dtype}`);
    }
    if (typeof obj.b64 === 'string') data = bytesToFloat32(base64ToBytes(obj.b64));
    else if (obj.data != null) data = decodeArray(obj.data, what);
    else throw new Error(`decodeArray(${what}): expected {shape, b64}`);
    shape = Array.isArray(obj.shape) ? obj.shape.map(Number) : null;
  } else throw new Error(`decodeArray(${what}): cannot decode a ${typeof obj}`);
  if (shape) {
    const n = shape.reduce((a, b) => a * b, 1);
    if (n !== data.length) throw new Error(`decodeArray(${what}): shape [${shape}] needs ${n} values, got ${data.length}`);
    Object.defineProperty(data, 'shape', { value: shape, enumerable: false, configurable: true, writable: true });
  }
  return data;
}

/** Int vector of exactly n entries (accepts nested arrays / typed arrays). */
function toIntVector(arr, n, what) {
  if (arr == null) throw new Error(`${what}: missing`);
  const flat = Array.isArray(arr) ? arr.flat(Infinity) : Array.from(arr);
  if (flat.length !== n) throw new Error(`${what}: expected ${n} entries, got ${flat.length}`);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const v = Number(flat[i]);
    if (!Number.isInteger(v)) throw new Error(`${what}: entry ${i} = ${flat[i]} is not an integer`);
    out[i] = v;
  }
  return out;
}

/** {data: Float32Array, shape} from a Float32Array or a {data, shape} leaf. */
function normLeaf(v, name) {
  if (v instanceof Float32Array) return { data: v, shape: v.shape ? v.shape.slice() : [v.length] };
  if (v && v.data instanceof Float32Array) return { data: v.data, shape: (v.shape || v.data.shape || [v.data.length]).slice() };
  throw new Error(`Model: ${name} missing (expected a Float32Array or {data, shape})`);
}

const withShape = (arr, shape) => {
  Object.defineProperty(arr, 'shape', { value: shape.slice(), enumerable: false, configurable: true, writable: true });
  return arr;
};

// =====================================================================================================================
// activations (generic; the shipped model uses "tanh")
// =====================================================================================================================
const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const softplus = (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));     // jax.nn.softplus = logaddexp(x, 0)
const ACT_BASE = {
  tanh: [Math.tanh, (x) => { const y = Math.tanh(x); return 1 - y * y; }],    // jax tanh JVP: g * (1 - ans^2)
  relu: [(x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0)],
  silu: [(x) => x * sigmoid(x), (x) => { const s = sigmoid(x); return s * (1 + x * (1 - s)); }],
  gelu: [                                                                           // jax.nn.gelu(approximate=True)
    (x) => 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x))),
    (x) => {
      const th = Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x));
      return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x);
    },
  ],
  softplus: [softplus, sigmoid],
};

/** Activation by name: "base" -> base(x); "base@T" -> base(T x) / T. Returns {f, df, name}. */
export function makeActivation(name) {
  const str = String(name);
  let base = str, T = null;
  if (str.includes('@')) {
    const parts = str.split('@');
    base = parts[0];
    if (base !== 'relu') T = parseFloat(parts[1]);
    if (T !== null && !Number.isFinite(T)) throw new Error(`activation "${name}": bad temperature`);
  }
  const pair = ACT_BASE[base];
  if (!pair) throw new Error(`unknown activation "${name}" (tanh|relu|silu|gelu|softplus, optionally @T)`);
  if (T === null) return { f: pair[0], df: pair[1], name: str };
  const f = pair[0], df = pair[1];
  return { f: (x) => f(T * x) / T, df: (x) => df(T * x), name: str };
}

// =====================================================================================================================
// small numerics
// =====================================================================================================================
/** Solve A x = b (n x n, row-major Float64Array) by Gaussian elimination with partial pivoting, in place.
 *  Returns b (now x), or null if singular / non-finite (stands in for jnp.linalg.solve). */
function solveDense(A, b, n) {
  for (let col = 0; col < n; col++) {
    let piv = col, best = Math.abs(A[col * n + col]);
    for (let r = col + 1; r < n; r++) { const v = Math.abs(A[r * n + col]); if (v > best) { best = v; piv = r; } }
    if (!(best > 0) || !Number.isFinite(best)) return null;
    if (piv !== col) {
      for (let c = 0; c < n; c++) { const t = A[col * n + c]; A[col * n + c] = A[piv * n + c]; A[piv * n + c] = t; }
      const t = b[col]; b[col] = b[piv]; b[piv] = t;
    }
    const d = A[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[r * n + c] -= f * A[col * n + c];
      b[r] -= f * b[col];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= A[r * n + c] * b[c];
    b[r] = s / A[r * n + r];
    if (!Number.isFinite(b[r])) return null;
  }
  return b;
}

/** |a - b| / (|b| + 1e-20) over whole vectors (relative residual). */
function relDiff(a, b) {
  let num = 0, den = 0;
  for (let i = 0; i < b.length; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return Math.sqrt(num) / (Math.sqrt(den) + 1e-20);
}

/** Seeded PRNG (mulberry32) -> uniform [0, 1). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fill `out` with N(0, 1) samples (Box-Muller on mulberry32(seed)). */
function randnFill(out, seed) {
  const rnd = mulberry32(seed);
  for (let i = 0; i < out.length; i += 2) {
    const u1 = 1 - rnd(), u2 = rnd(), r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < out.length) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

const nowMs = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());

// =====================================================================================================================
// Model
// =====================================================================================================================
export class Model {
  /**
   * Normally built by decodeModel(json). theta = {Win, bin, W1, b1, W2, b2}, psi = {V, c}: each a Float32Array or a
   * {data: Float32Array, shape} leaf; exposed as model.theta.<name> / model.psi.<name> = {data, shape}.
   */
  constructor({ meta = {}, theta, psi, puzzles = [], json = null } = {}) {
    if (!theta || !psi) throw new Error('Model: theta and psi are required');
    meta = meta || {};
    const T = {}, Q = {};
    for (const k of THETA_NAMES) T[k] = normLeaf(theta[k], `theta.${k}`);
    for (const k of PSI_NAMES) Q[k] = normLeaf(psi[k], `psi.${k}`);
    const H = T.b1.data.length, NOUT = Q.c.data.length, din = T.Win.data.length / H;
    const P = din - NOUT, N = NOUT / P;
    if (!(P > 0) || !Number.isInteger(din) || !Number.isInteger(N) || din !== P * (N + 1)) {
      throw new Error(`Model: inconsistent sizes (Win ${T.Win.data.length} values, H ${H}, readout ${NOUT})`);
    }
    for (const [k, n] of [['W1', H * H], ['W2', H * H], ['bin', H], ['b2', H]]) {
      if (T[k].data.length !== n) throw new Error(`Model: theta.${k} has ${T[k].data.length} values, expected ${n}`);
    }
    if (Q.V.data.length !== H * NOUT) throw new Error(`Model: psi.V has ${Q.V.data.length} values, expected ${H * NOUT}`);
    for (const [k, v] of [['N', N], ['P', P], ['din', din], ['H', H]]) {
      if (meta[k] != null && Number(meta[k]) !== v) throw new Error(`Model: meta.${k} = ${meta[k]} but the weights imply ${v}`);
    }
    if (!Array.isArray(meta.vodes)) meta.vodes = VODES.slice();
    this.meta = meta;
    this.dims = Object.freeze({ N, P, din, H });
    this.N = N; this.P = P; this.din = din; this.H = H; this.NOUT = NOUT;
    const act = makeActivation(meta.act ?? 'tanh');
    this.actName = act.name; this.act = act.f; this.dact = act.df;
    this.theta = T; this.psi = Q;
    this.Win = T.Win.data; this.bin = T.bin.data; this.W1 = T.W1.data; this.b1 = T.b1.data;
    this.W2 = T.W2.data; this.b2 = T.b2.data; this.V = Q.V.data; this.c = Q.c.data;
    this.box = Array.isArray(meta.box) ? meta.box.map(Number) : (BOX_DEFAULT[N] || [1, N]);
    this.puzzles = Array.isArray(puzzles) ? puzzles : [];
    Object.defineProperty(this, 'json', { value: json, enumerable: false, writable: true });
    // defaults: meta.ep (if present) > reference[0].nudge > DEFAULTS
    const ep = meta.ep || meta.ep_cfg || {};
    const ref0 = json && Array.isArray(json.reference) && json.reference.length ? json.reference[0] : null;
    const nd = (ref0 && ref0.nudge) || {};
    const tm = meta.test_metrics || {};
    this.defaults = Object.freeze({
      T_free: Number(tm.sweeps ?? DEFAULTS.T_free),
      T_trace: Number(ep.T_free ?? DEFAULTS.T_trace),
      beta: Number(ep.beta ?? nd.beta ?? DEFAULTS.beta),
      T_nudg: Number(ep.T_nudg ?? nd.steps ?? DEFAULTS.T_nudg),
      lr_h: Number(ep.lr_h ?? nd.lr_h ?? DEFAULTS.lr_h),
      mom: Number(ep.mom ?? nd.mom ?? DEFAULTS.mom),
      m: DEFAULTS.m, lam: DEFAULTS.lam, a: DEFAULTS.a,
    });
    this.vodes = VODES;
    this.formulas = Object.freeze({
      u: `hat u = ${this.actName}(s W1 + b1)`, s: `hat s = ${this.actName}(u W2 + b2 + x_inj)`, x_inj: 'x_inj = x Win + bin',
    });
    this.sizes = Object.freeze({ u: H, s: H });
    this.offsets = Object.freeze({ u: 0, s: H });
    this.D = 2 * H;
    this.shapes = Object.freeze({ u: [H], s: [H] });
    this.thetaShapes = Object.freeze({ Win: [din, H], bin: [H], W1: [H, H], b1: [H], W2: [H, H], b2: [H] });
    const F64 = (n) => new Float64Array(n);
    this._preU = F64(H); this._hatU = F64(H); this._preS = F64(H); this._hatS = F64(H);
    this._eu = F64(H); this._es = F64(H); this._du = F64(H); this._ds = F64(H);
    this._v1 = F64(H); this._v2 = F64(H); this._v3 = F64(H);
    this._logits = F64(NOUT); this._probs = F64(NOUT); this._lse = F64(P); this._gl = F64(NOUT);
    this._predTmp = new Int32Array(P);
    this._viewCache = new WeakMap();
    this._sweepTmp = this.zeroState();
  }

  // ------------------------------------------------------------------ state helpers
  /** Wrap a flat Float32Array(D) as {u, s} subarray views (u first, then s); `.flat` is non-enumerable. */
  _wrap(flat) {
    const H = this.H;
    const z = { u: flat.subarray(0, H), s: flat.subarray(H, 2 * H) };
    Object.defineProperty(z, 'flat', { value: flat, enumerable: false });
    return z;
  }

  /** Cached views of an internal flat buffer (no allocation after the first call). */
  _views(flat) {
    let v = this._viewCache.get(flat);
    if (!v) { v = this._wrap(flat); this._viewCache.set(flat, v); }
    return v;
  }

  /** Accept a state {u, s}, a [u, s] pair, or a flat Float32Array(D) (views, no copy). */
  asState(z) {
    if (z == null) throw new Error('state is null');
    if (z instanceof Float32Array) {
      if (z.length !== this.D) throw new Error(`flat state must have ${this.D} values, got ${z.length}`);
      return this._views(z);
    }
    const s = Array.isArray(z) ? { u: z[0], s: z[1] } : z;
    for (const v of VODES) if (!s[v] || s[v].length !== this.sizes[v]) throw new Error(`state.${v} must have ${this.sizes[v]} values`);
    return s;
  }

  /** z = 0 (initial state): {u: F32(H), s: F32(H)}. */
  zeroState() { return this._wrap(new Float32Array(this.D)); }

  /** Deep copy of a state (new contiguous buffer). */
  cloneState(z) { return this._wrap(this.flatten(z)); }

  /** Flat Float32Array(D) copy of z = concat(u, s). */
  flatten(z, out = new Float32Array(this.D)) {
    const zs = this.asState(z);
    out.set(zs.u, 0); out.set(zs.s, this.H);
    return out;
  }

  /** Views onto a flat Float32Array(D) (no copy). */
  unflatten(flat) { return this.asState(flat); }

  /** The contiguous flat buffer behind z when z is a view state, else a flat copy (into `out`). */
  flatOf(z, out = null) {
    if (z instanceof Float32Array) { this.asState(z); return z; }
    const f = z && z.flat;
    if (f instanceof Float32Array && f.length === this.D && z.u instanceof Float32Array && z.s instanceof Float32Array
      && z.u.buffer === f.buffer && z.s.buffer === f.buffer && z.u.byteOffset === f.byteOffset
      && z.s.byteOffset === f.byteOffset + 4 * this.H && z.u.length === this.H && z.s.length === this.H) return f;
    return this.flatten(z, out || new Float32Array(this.D));
  }

  /** y <- y + a x in place on every vode; returns y. */
  axpy(a, x, y) {
    const xs = this.asState(x), ys = this.asState(y);
    for (const v of VODES) { const X = xs[v], Y = ys[v]; for (let i = 0; i < Y.length; i++) Y[i] += a * X[i]; }
    return y;
  }

  /** out = a x + b y (new state unless `out` is given). */
  lincomb(a, x, b, y, out = this.zeroState()) {
    const xs = this.asState(x), ys = this.asState(y), os = this.asState(out);
    for (const v of VODES) { const X = xs[v], Y = ys[v], O = os[v]; for (let i = 0; i < O.length; i++) O[i] = a * X[i] + b * Y[i]; }
    return out;
  }

  /** out = x - y. */
  sub(x, y, out = this.zeroState()) { return this.lincomb(1, x, -1, y, out); }

  /** <x, y> over the concatenated (u, s). */
  dot(x, y) {
    const xs = this.asState(x), ys = this.asState(y);
    let s = 0;
    for (const v of VODES) { const X = xs[v], Y = ys[v]; for (let i = 0; i < X.length; i++) s += X[i] * Y[i]; }
    return s;
  }

  /** ||z|| over the concatenated (u, s). */
  norm(z) { return Math.sqrt(this.dot(z, z)); }

  /** The s vector of a state, or s itself (length H). */
  _sVec(s, what = 's') {
    if (s && s.length === this.H && (ArrayBuffer.isView(s) || Array.isArray(s))) return s;
    if (s instanceof Float32Array && s.length === this.D) return s.subarray(this.H, this.D);
    if (s && s.s && s.s.length === this.H) return s.s;
    throw new Error(`${what}: expected s (${this.H} values) or a state {u, s}`);
  }

  // ------------------------------------------------------------------ input / stem
  _puzzle(puzzle) {
    const puz = toIntVector(puzzle, this.P, 'puzzle');
    for (let p = 0; p < this.P; p++) if (puz[p] < 0 || puz[p] > this.N) throw new Error(`puzzle[${p}] = ${puz[p]} not in 0..${this.N}`);
    return puz;
  }

  /** x = one-hot(puzzle cell, N+1) flattened row-major -> F32(din). */
  encode(puzzle) {
    const puz = this._puzzle(puzzle), K = this.N + 1, x = new Float32Array(this.din);
    for (let p = 0; p < this.P; p++) x[p * K + puz[p]] = 1;
    return x;
  }

  /** Targets y = solution - 1 -> Int32Array(P) in 0..N-1. */
  label(solution) {
    const sol = toIntVector(solution, this.P, 'solution');
    for (let p = 0; p < this.P; p++) sol[p] -= 1;
    return this._label(sol);
  }

  /**
   * x_inj = x Win + bin, x = encode(puzzle): x_inj[h] = sum_p Win[p(N+1) + puzzle[p], h]
   * + bin[h]. puzzle: P ints, 0 = blank, 1..N = clue. Returns F32(H) with the puzzle attached as the
   * non-enumerable `.puzzle` (thetaGrad recomputes the stem from it).
   */
  embed(puzzle) {
    const puz = this._puzzle(puzzle), H = this.H, K = this.N + 1, Win = this.Win, bin = this.bin, acc = this._v1;
    acc.fill(0);
    for (let p = 0; p < this.P; p++) {
      const w = (p * K + puz[p]) * H;
      for (let h = 0; h < H; h++) acc[h] += Win[w + h];
    }
    const out = new Float32Array(H);
    for (let h = 0; h < H; h++) out[h] = acc[h] + bin[h];
    Object.defineProperty(out, 'puzzle', { value: puz, enumerable: false });
    return out;
  }

  // ------------------------------------------------------------------ prediction kernels
  /** hat u(s)[h] = act(sum_i s[i] W1[i,h] + b1[h]); pre-activation left in _preU. */
  _kU(s, out) {
    const H = this.H, W = this.W1, b = this.b1, pre = this._preU, act = this.act;
    pre.fill(0);
    for (let i = 0; i < H; i++) {
      const a = s[i];
      if (a === 0) continue;
      const w = i * H;
      for (let h = 0; h < H; h++) pre[h] += a * W[w + h];
    }
    for (let h = 0; h < H; h++) { const v = pre[h] + b[h]; pre[h] = v; out[h] = act(v); }
  }

  /** hat s(u)[h] = act(sum_i u[i] W2[i,h] + b2[h] + x_inj[h]); pre-activation in _preS. */
  _kS(u, x, out) {
    const H = this.H, W = this.W2, b = this.b2, pre = this._preS, act = this.act;
    pre.fill(0);
    for (let i = 0; i < H; i++) {
      const a = u[i];
      if (a === 0) continue;
      const w = i * H;
      for (let h = 0; h < H; h++) pre[h] += a * W[w + h];
    }
    for (let h = 0; h < H; h++) { const v = pre[h] + b[h] + x[h]; pre[h] = v; out[h] = act(v); }
  }

  _checkX(x) {
    if (!x || x.length !== this.H) throw new Error(`x_inj must be a Float32Array(${this.H}) from model.embed(puzzle)`);
  }

  /** Jacobi predictions from the CURRENT state into scratch: _hatU = hat u(z.s), _hatS = hat s(z.u). */
  _jacobi(z, x) {
    this._checkX(x);
    this._kU(z.s, this._hatU);
    this._kS(z.u, x, this._hatS);
  }

  /** Jacobi predictions {u: hat u(z.s), s: hat s(z.u)} (new state). */
  predict(z, xinj) {
    const zs = this.asState(z), out = this.zeroState();
    this._jacobi(zs, xinj);
    out.u.set(this._hatU); out.s.set(this._hatS);
    return out;
  }

  /** Prediction errors {u: u - hat u(s), s: s - hat s(u)}. */
  errors(z, xinj) {
    const zs = this.asState(z), out = this.zeroState();
    this._jacobi(zs, xinj);
    for (let h = 0; h < this.H; h++) { out.u[h] = zs.u[h] - this._hatU[h]; out.s[h] = zs.s[h] - this._hatS[h]; }
    return out;
  }

  /** E(z) = 1/2 (|u - hat u(s)|^2 + |s - hat s(u)|^2) -> {E, parts: {u, s}}. */
  energy(z, xinj) {
    const zs = this.asState(z);
    this._jacobi(zs, xinj);
    let pu = 0, ps = 0;
    for (let h = 0; h < this.H; h++) {
      const a = zs.u[h] - this._hatU[h], b = zs.s[h] - this._hatS[h];
      pu += a * a; ps += b * b;
    }
    return { E: 0.5 * (pu + ps), parts: { u: 0.5 * pu, s: 0.5 * ps } };
  }

  _energyScalar(zs, x) {
    this._jacobi(zs, x);
    let e = 0;
    for (let h = 0; h < this.H; h++) {
      const a = zs.u[h] - this._hatU[h], b = zs.s[h] - this._hatS[h];
      e += a * a + b * b;
    }
    return 0.5 * e;
  }

  /** One Gauss-Seidel sweep into `out` (may alias z): u' = hat u(s); s' = hat s(u'). */
  sweepInto(z, xinj, out) {
    const zs = this.asState(z), os = this.asState(out);
    this._checkX(xinj);
    this._kU(zs.s, os.u);
    this._kS(os.u, xinj, os.s);
    return out;
  }

  /** One Gauss-Seidel sweep; returns a NEW state. */
  sweep(z, xinj) { return this.sweepInto(z, xinj, this.zeroState()); }

  /** Free relaxation by sweeps: T sweeps from z0 (default 0; T default 60). */
  relaxFree(xinj, T = null, z0 = null) {
    const n = T ?? this.defaults.T_free;
    const z = z0 ? this.cloneState(z0) : this.zeroState();
    for (let k = 0; k < n; k++) this.sweepInto(z, xinj, z);
    return z;
  }

  /** The composed lap map s -> hat s(hat u(s)) -> F32(H). */
  lap(s, xinj) {
    const S = this._sVec(s, 'lap');
    this._checkX(xinj);
    this._kU(S, this._v1);
    const out = new Float32Array(this.H);
    this._kS(this._v1, xinj, out);
    return out;
  }

  /**
   * rho(J) of the lap's Jacobian at s* = zStar.s: power iteration with
   * the hand JVP  J v = act'(pre_s) * (((act'(pre_u) * (v W1)) W2),  pre_u = s* W1 + b1, pre_s = hat u(s*) W2 + b2 + x_inj;
   * v_0 = normalised N(0, 1) (seeded JS PRNG, not JAX's); returns |J v_last| with |v_last| = 1 after `iters` steps.
   */
  spectralRadius(zStar, xinj, iters = 60, seed = 1) {
    const S = this._sVec(zStar, 'spectralRadius'), H = this.H, dact = this.dact, W1 = this.W1, W2 = this.W2;
    this._checkX(xinj);
    this._kU(S, this._hatU);
    this._kS(this._hatU, xinj, this._hatS);
    const gU = this._du, gS = this._ds, v = this._v1, a = this._v2, b = this._v3;
    for (let h = 0; h < H; h++) { gU[h] = dact(this._preU[h]); gS[h] = dact(this._preS[h]); }
    randnFill(v, seed);
    let nv = 0;
    for (let h = 0; h < H; h++) nv += v[h] * v[h];
    nv = Math.sqrt(nv) + 1e-20;
    for (let h = 0; h < H; h++) v[h] /= nv;
    let lam = 0;
    for (let it = 0; it < iters; it++) {
      a.fill(0);
      for (let i = 0; i < H; i++) { const vi = v[i]; if (vi === 0) continue; const w = i * H; for (let h = 0; h < H; h++) a[h] += vi * W1[w + h]; }
      for (let h = 0; h < H; h++) a[h] *= gU[h];
      b.fill(0);
      for (let i = 0; i < H; i++) { const ai = a[i]; if (ai === 0) continue; const w = i * H; for (let h = 0; h < H; h++) b[h] += ai * W2[w + h]; }
      let nb = 0;
      for (let h = 0; h < H; h++) { b[h] *= gS[h]; nb += b[h] * b[h]; }
      lam = Math.sqrt(nb);
      for (let h = 0; h < H; h++) v[h] = b[h] / (lam + 1e-20);
    }
    return lam;
  }

  /**
   * dE/dz into `g` (may alias z), hand-written VJPs of the energy E:
   *   e_u = u - hat u(s), e_s = s - hat s(u), delta_u = e_u * act'(pre_u), delta_s = e_s * act'(pre_s)  (tanh: 1 - hat^2)
   *   dE/du[i] = e_u[i] - sum_h delta_s[h] W2[i,h]      (= e_u - (e_s (1 - hat s^2)) W2^T)
   *   dE/ds[i] = e_s[i] - sum_h delta_u[h] W1[i,h]      (= e_s - (e_u (1 - hat u^2)) W1^T)
   */
  gradEInto(z, xinj, g) {
    const zs = this.asState(z), gs = this.asState(g), H = this.H, dact = this.dact;
    this._jacobi(zs, xinj);
    const eu = this._eu, es = this._es, du = this._du, ds = this._ds;
    for (let h = 0; h < H; h++) {
      eu[h] = zs.u[h] - this._hatU[h]; es[h] = zs.s[h] - this._hatS[h];
      du[h] = eu[h] * dact(this._preU[h]); ds[h] = es[h] * dact(this._preS[h]);
    }
    const W1 = this.W1, W2 = this.W2, gu = gs.u, gss = gs.s;
    for (let i = 0; i < H; i++) {
      const w = i * H;
      let a2 = 0, a1 = 0;
      for (let h = 0; h < H; h++) { a2 += ds[h] * W2[w + h]; a1 += du[h] * W1[w + h]; }
      gu[i] = eu[i] - a2;
      gss[i] = es[i] - a1;
    }
    return g;
  }

  /** dE/dz = {u, s} (NEW state); see gradEInto. */
  gradE(z, xinj) { return this.gradEInto(z, xinj, this.zeroState()); }

  // ------------------------------------------------------------------ readout / cost
  /** Readout into scratch: logits[o] = sum_i s[i] V[i,o] + c[o], o = p N + k;
   *  probs = per-cell softmax over k; lse[p] = logsumexp_k logits[p N + k]. */
  _head(S) {
    const H = this.H, P = this.P, N = this.N, NOUT = this.NOUT, V = this.V, cb = this.c;
    const L = this._logits, Pr = this._probs, lse = this._lse;
    L.fill(0);
    for (let i = 0; i < H; i++) {
      const a = S[i];
      if (a === 0) continue;
      const w = i * NOUT;
      for (let o = 0; o < NOUT; o++) L[o] += a * V[w + o];
    }
    for (let p = 0; p < P; p++) {
      const lo = p * N;
      let mx = -Infinity;
      for (let k = 0; k < N; k++) { const v = L[lo + k] + cb[lo + k]; L[lo + k] = v; if (v > mx) mx = v; }
      let se = 0;
      for (let k = 0; k < N; k++) { const e = Math.exp(L[lo + k] - mx); Pr[lo + k] = e; se += e; }
      for (let k = 0; k < N; k++) Pr[lo + k] /= se;
      lse[p] = mx + Math.log(se);
    }
  }

  _label(label) {
    const y = toIntVector(label, this.P, 'label');
    for (let p = 0; p < this.P; p++) if (y[p] < 0 || y[p] >= this.N) throw new Error(`label[${p}] = ${y[p]} not in 0..${this.N - 1}`);
    return y;
  }

  /** Per-cell argmax of the readout into pred (first max wins, like jnp.argmax). */
  _argmaxInto(S, pred) {
    this._head(S);
    const N = this.N, L = this._logits;
    for (let p = 0; p < this.P; p++) {
      let best = 0;
      for (let k = 1; k < N; k++) if (L[p * N + k] > L[p * N + best]) best = k;
      pred[p] = best;
    }
    return pred;
  }

  /** Readout of s (F32(H) or a state): {logits F32(P N), probs F32(P N) (cell-major P x N), pred Int32Array(P) in
   *  0..N-1 (digit = pred + 1), conf F32(P) = max prob}. */
  readout(s) {
    const S = this._sVec(s, 'readout'), P = this.P, N = this.N;
    this._head(S);
    const logits = Float32Array.from(this._logits), probs = Float32Array.from(this._probs);
    const pred = new Int32Array(P), conf = new Float32Array(P);
    for (let p = 0; p < P; p++) {
      let best = 0;
      for (let k = 1; k < N; k++) if (this._logits[p * N + k] > this._logits[p * N + best]) best = k;
      pred[p] = best; conf[p] = this._probs[p * N + best];
    }
    return { logits, probs, pred, conf };
  }

  /** C(s, y) = mean_p [lse_p - logits[p, y_p]] (per-cell cross-entropy, mean over cells; batch size 1). */
  cost(s, label) { return this._cost(this._sVec(s, 'cost'), this._label(label)); }

  _cost(S, y) {
    this._head(S);
    let tot = 0;
    for (let p = 0; p < this.P; p++) tot += this._lse[p] - this._logits[p * this.N + y[p]];
    return tot / this.P;
  }

  /** dC/ds into out, times `scale`, accumulated if `acc`:  dlogits[p,k] = (probs[p,k] - [k == y_p]) / P;
   *  dC/ds[i] = sum_o dlogits[o] V[i,o]. */
  _gradCInto(S, y, out, scale = 1, acc = false) {
    const H = this.H, P = this.P, N = this.N, NOUT = this.NOUT, V = this.V, gl = this._gl;
    this._head(S);
    const Pr = this._probs, invP = 1 / P;
    for (let p = 0; p < P; p++) for (let k = 0; k < N; k++) gl[p * N + k] = (Pr[p * N + k] - (k === y[p] ? 1 : 0)) * invP;
    for (let i = 0; i < H; i++) {
      let a = 0;
      const w = i * NOUT;
      for (let o = 0; o < NOUT; o++) a += gl[o] * V[w + o];
      out[i] = acc ? out[i] + scale * a : scale * a;
    }
    return out;
  }

  /** dC/ds (Float32Array(H)); the cost only reads the readout vode s. */
  gradC(s, label) { return this._gradCInto(this._sVec(s, 'gradC'), this._label(label), new Float32Array(this.H)); }

  /** Per-cell / board accuracy of the readout vs label 0..N-1. */
  accuracy(s, label) {
    const y = this._label(label), pred = this._argmaxInto(this._sVec(s, 'accuracy'), this._predTmp);
    const correct = new Uint8Array(this.P);
    let n = 0;
    for (let p = 0; p < this.P; p++) { correct[p] = pred[p] === y[p] ? 1 : 0; n += correct[p]; }
    return { cell: n / this.P, board: n === this.P, correct };
  }

  /** Cells whose (0-based) digit repeats in their row, column or box -> Uint8Array(P) of 0/1. */
  conflicts(pred) {
    const N = this.N, P = this.P, bh = this.box[0], bw = this.box[1], out = new Uint8Array(P);
    const boxOf = (r, c) => Math.floor(r / bh) * Math.floor(N / bw) + Math.floor(c / bw);
    for (let i = 0; i < P; i++) {
      for (let j = i + 1; j < P; j++) {
        if (pred[i] !== pred[j]) continue;
        const ri = Math.floor(i / N), ci = i % N, rj = Math.floor(j / N), cj = j % N;
        if (ri === rj || ci === cj || boxOf(ri, ci) === boxOf(rj, cj)) { out[i] = 1; out[j] = 1; }
      }
    }
    return out;
  }

  /** True iff the (0-based) predicted board is a valid complete Sudoku. */
  validBoard(pred) { return this.conflicts(pred).every((v) => v === 0); }

  /** ||sweep(z) - z|| / (||z|| + 1e-20) over the concatenated (u, s). */
  residual(z, xinj) {
    const zs = this.asState(z), t = this._sweepTmp;
    this.sweepInto(zs, xinj, t);
    let num = 0, den = 0;
    for (const v of VODES) {
      const a = t[v], b = zs[v];
      for (let i = 0; i < b.length; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
    }
    return Math.sqrt(num) / (Math.sqrt(den) + 1e-20);
  }

  // ------------------------------------------------------------------ flat helpers (Solver / energySlice)
  _sweepFlat(inF, x, outF) { this.sweepInto(this._views(inF), x, this._views(outF)); }
  _gradEFlat(inF, x, outF) { this.gradEInto(this._views(inF), x, this._views(outF)); }
  _energyFlat(inF, x) { return this._energyScalar(this._views(inF), x); }
  _costFlat(inF, y) { return this._cost(this._views(inF).s, y); }
  _addGradC(inF, y, beta, gF) { this._gradCInto(this._views(inF).s, y, this._views(gF).s, beta, true); }

  // ------------------------------------------------------------------ theta gradient
  /**
   * g(z) = dE/dtheta at FIXED z with x_inj = x Win + bin recomputed inside, i.e. local
   * outer products (postsynaptic error x presynaptic activity), delta_u = e_u * act'(pre_u), delta_s = e_s * act'(pre_s):
   *   W1[i,h] = -s[i] delta_u[h]    b1 = -delta_u      W2[i,h] = -u[i] delta_s[h]    b2 = -delta_s
   *   Win[f,h] = -x[f] delta_s[h]   bin = -delta_s     (x_inj enters hat s, so the stem gradient is the s error's)
   * Returns {Win, bin, W1, b1, W2, b2}: Float32Arrays with a non-enumerable `.shape` (also in model.thetaShapes).
   */
  thetaGrad(z, xinj, puzzle = xinj && xinj.puzzle) {
    if (!puzzle) throw new Error('thetaGrad: x_inj carries no .puzzle (build it with model.embed) -- pass the puzzle');
    const puz = this._puzzle(puzzle), zs = this.asState(z), H = this.H, K = this.N + 1, dact = this.dact;
    this._jacobi(zs, xinj);
    const du = this._du, ds = this._ds;
    for (let h = 0; h < H; h++) {
      du[h] = (zs.u[h] - this._hatU[h]) * dact(this._preU[h]);
      ds[h] = (zs.s[h] - this._hatS[h]) * dact(this._preS[h]);
    }
    const gW1 = new Float32Array(H * H), gW2 = new Float32Array(H * H);
    for (let i = 0; i < H; i++) {
      const si = zs.s[i], ui = zs.u[i], w = i * H;
      for (let h = 0; h < H; h++) { gW1[w + h] = -si * du[h]; gW2[w + h] = -ui * ds[h]; }
    }
    const gb1 = new Float32Array(H), gb2 = new Float32Array(H);
    for (let h = 0; h < H; h++) { gb1[h] = -du[h]; gb2[h] = -ds[h]; }
    const gWin = new Float32Array(this.din * H);
    for (let p = 0; p < this.P; p++) {
      const w = (p * K + puz[p]) * H;
      for (let h = 0; h < H; h++) gWin[w + h] -= ds[h];
    }
    const gbin = Float32Array.from(gb2);
    const sh = this.thetaShapes;
    return {
      Win: withShape(gWin, sh.Win), bin: withShape(gbin, sh.bin), W1: withShape(gW1, sh.W1),
      b1: withShape(gb1, sh.b1), W2: withShape(gW2, sh.W2), b2: withShape(gb2, sh.b2),
    };
  }

  /**
   * Exact readout gradient dC/dpsi = {V, c}: the cost only reads s, so this is a
   * plain backprop through the linear + softmax readout, not an EP estimate.
   *   dlogits[p,k] = (softmax(logits[p])[k] - [k == y_p]) / P        dC/dV = outer(s, dlogits_flat)   dC/dc = dlogits_flat
   */
  gradPsi(s, label) {
    const S = this._sVec(s, 'gradPsi'), y = this._label(label);
    const H = this.H, P = this.P, N = this.N, NOUT = this.NOUT;
    this._head(S);
    const gl = this._gl, Pr = this._probs, invP = 1 / P;
    for (let p = 0; p < P; p++) for (let k = 0; k < N; k++) gl[p * N + k] = (Pr[p * N + k] - (k === y[p] ? 1 : 0)) * invP;
    const gV = new Float32Array(H * NOUT);
    for (let i = 0; i < H; i++) {
      const si = S[i];
      if (si === 0) continue;
      const w = i * NOUT;
      for (let o = 0; o < NOUT; o++) gV[w + o] = si * gl[o];
    }
    const gc = Float32Array.from(gl);
    return { V: withShape(gV, [H, NOUT]), c: withShape(gc, [NOUT]) };
  }
}

// =====================================================================================================================
// EP-GD in-app learning: free phase (plain or heavy-ball GD on E, early-stop
// at E < tol), +-beta nudge (GD on F = E + beta C), one-sided or centred EP weight update, exact readout update,
// optional spectral re-cap. Pure functions: `theta`/`psi` are mutated on the model IN PLACE (so "Reset weights" in
// the app is just re-decoding the checkpoint's own json), nothing else on the model changes.
// =====================================================================================================================

/** Free-phase loop via Solver "gd", with the exact early-stop rule (E < tol). cfg: {init: 'ff'|'zero',
 *  eta, mom, T, tol}. Returns {z (state), iters, converged, E, traj: [E(z_t), t = 0..iters] (only if opts.traj)}. */
export function runFreeGD(model, xinj, cfg, opts = {}) {
  const z0 = cfg.init === 'zero' ? model.zeroState() : model.sweepInto(model.zeroState(), xinj, model.zeroState());
  const S = new Solver(model, xinj, { kind: 'gd', z0, lr: cfg.eta, mom: cfg.mom || 0, track: true });
  const T = cfg.T, tol = cfg.tol ?? 1e-6;
  let last = { z: S.z, E: model.energy(S.z, xinj).E, residual: NaN, iter: 0 };
  const traj = opts.traj ? [last.E] : null;
  let iters = 0, converged = last.E < tol;
  while (iters < T && !converged) {
    last = S.step();
    iters++;
    if (traj) traj.push(last.E);
    converged = last.E < tol;
  }
  return { z: model.cloneState(last.z), iters, converged, E: last.E, traj };
}

/** Nudge loop via Solver "nudge" (no early stop). cfg: {beta, eta, mom, T}, signed beta (+ or -). */
export function runNudgeGD(model, xinj, zFree, label, cfg) {
  const S = new Solver(model, xinj, { kind: 'nudge', z0: zFree, beta: cfg.beta, lr: cfg.eta, mom: cfg.mom || 0, label, track: false });
  S.run(cfg.T);
  return model.cloneState(S.z);
}

/** One-sided ((g(z+) - g(z0)) / beta) or centred ((g(z+) - g(z-)) / (2 beta)) EP estimate of dE/dtheta. */
export function epThetaGrad(model, xinj, zPlus, zFree, beta, zMinus = null) {
  const gp = model.thetaGrad(zPlus, xinj);
  const gRef = zMinus ? model.thetaGrad(zMinus, xinj) : model.thetaGrad(zFree, xinj);
  const inv = zMinus ? 1 / (2 * beta) : 1 / beta;
  const out = {};
  for (const k of THETA_NAMES) {
    const a = gp[k], b = gRef[k], o = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) o[i] = (a[i] - b[i]) * inv;
    out[k] = withShape(o, model.thetaShapes[k]);
  }
  return out;
}

/** Top singular value of an H x H matrix W (row-vector convention) by power iteration on W^T W; returns sigma and
 *  (if sigma > cap) rescales W in place so ||W||_2 == cap. Deterministic seeded start vector. */
export function capSpectral(W, H, cap, iters = 20, seed = 1) {
  let s = seed >>> 0;
  const rand = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  const v = new Float64Array(H);
  for (let i = 0; i < H; i++) v[i] = rand() - 0.5;
  const norm = (a) => { let t = 0; for (let i = 0; i < a.length; i++) t += a[i] * a[i]; return Math.sqrt(t); };
  let nv = norm(v) || 1;
  for (let i = 0; i < H; i++) v[i] /= nv;
  const Wv = new Float64Array(H), WtWv = new Float64Array(H);
  const applyW = (vec, out) => { out.fill(0); for (let i = 0; i < H; i++) { const vi = vec[i]; if (vi === 0) continue; const w = i * H; for (let h = 0; h < H; h++) out[h] += vi * W[w + h]; } };
  const applyWt = (vec, out) => { for (let i = 0; i < H; i++) { const w = i * H; let acc = 0; for (let h = 0; h < H; h++) acc += vec[h] * W[w + h]; out[i] = acc; } };
  for (let it = 0; it < iters; it++) {
    applyW(v, Wv);
    applyWt(Wv, WtWv);
    const n = norm(WtWv);
    if (!(n > 0)) break;
    for (let i = 0; i < H; i++) v[i] = WtWv[i] / n;
  }
  applyW(v, Wv);
  const sigma = norm(Wv);
  if (cap != null && sigma > cap && sigma > 0) {
    const scale = cap / sigma;
    for (let i = 0; i < W.length; i++) W[i] *= scale;
  }
  return sigma;
}

/**
 * One full "Learn step": free phase -> +beta nudge (and -beta too if centred) ->
 * EP estimate of dE/dtheta -> exact dC/dpsi at z_free -> theta -= lr*g_theta, psi -= lrReadout*g_psi -> optional
 * spectral re-cap -> free phase re-run. Mutates model.theta/psi in place. cfg:
 *   {label, free: {init, eta, mom, T, tol}, nudge: {beta, eta, mom, T}, centred: bool, lr, lrReadout, cap}
 * Returns {zFree, zPlus, zMinus (centred only), gTheta, gPsi, sigma (if capped), after (re-run free result)}.
 */
export function learnStep(model, xinj, cfg) {
  const label = cfg.label;
  const free = runFreeGD(model, xinj, cfg.free);
  const zPlus = runNudgeGD(model, xinj, free.z, label, { ...cfg.nudge, beta: Math.abs(cfg.nudge.beta) });
  let zMinus = null, gTheta;
  if (cfg.centred) {
    zMinus = runNudgeGD(model, xinj, free.z, label, { ...cfg.nudge, beta: -Math.abs(cfg.nudge.beta) });
    gTheta = epThetaGrad(model, xinj, zPlus, free.z, Math.abs(cfg.nudge.beta), zMinus);
  } else {
    gTheta = epThetaGrad(model, xinj, zPlus, free.z, Math.abs(cfg.nudge.beta));
  }
  const gPsi = model.gradPsi(free.z.s, label);
  const lr = cfg.lr, lrReadout = cfg.lrReadout || 0;
  for (const k of THETA_NAMES) {
    const dst = model.theta[k].data, src = gTheta[k];
    for (let i = 0; i < dst.length; i++) dst[i] -= lr * src[i];
  }
  let sigma = null;
  if (lrReadout > 0) {
    const V = model.psi.V.data, c = model.psi.c.data;
    for (let i = 0; i < V.length; i++) V[i] -= lrReadout * gPsi.V[i];
    for (let i = 0; i < c.length; i++) c[i] -= lrReadout * gPsi.c[i];
  }
  if (cfg.cap != null) {
    sigma = { W1: capSpectral(model.W1, model.H, cfg.cap), W2: capSpectral(model.W2, model.H, cfg.cap) };
  }
  const after = runFreeGD(model, xinj, cfg.free);
  return { zFree: free, zPlus, zMinus, gTheta, gPsi, sigma, after };
}

// =====================================================================================================================
// Solver
// =====================================================================================================================
/**
 * Stateful iterator for animation; one step() = one iteration:
 *   "sweep"    z <- sweep(z)                                        (free relaxation by sweeps)
 *   "damped"   z <- (1 - a) z + a sweep(z)                          (whole state)
 *   "gd"       v <- mom v - lr dE/dz;  z <- z + v                   (= the nudged relaxation with beta = 0)
 *   "anderson" Anderson(m, lam) on the flattened sweep map: steps 1..m are the plain
 *              seeding iterations, after T >= m + 1 steps z == anderson(sweep, z0, m, max_iter = T, lam, beta = aaBeta)
 *   "nudge"    v <- mom v - lr (dE/dz + beta dC/dz);  z <- z + v    (nudged relaxation; label + z0 = free state)
 * opts: {kind, z0, a, lr, mom, m, lam, beta, label, aaBeta = 1 (Anderson mixing), track = true}; defaults from
 * model.defaults: a 0.5, lr 0.025, mom 0.9, m 6, lam 1e-2, beta 0.05; "nudge" without z0
 * starts from the free state after T_trace = 300 sweeps.
 * step() -> {z (fresh copy), E, residual = |sweep(z) - z| / (|z| + 1e-20), iter} (+ C, F = E + beta C for "nudge");
 * track = false skips E / residual (NaN). run(n) = n steps, returns the last step's result. reset(z0) restarts.
 */
export class Solver {
  constructor(model, xinj, opts = {}) {
    model._checkX(xinj);
    const d = model.defaults;
    const kind = opts.kind ?? 'sweep';
    if (!SOLVER_KINDS.includes(kind)) throw new Error(`Solver: unknown kind "${kind}" (${SOLVER_KINDS.join(' | ')})`);
    this.model = model; this.xinj = xinj; this.kind = kind;
    this.a = Number(opts.a ?? d.a);
    this.lr = Number(opts.lr ?? d.lr_h);
    this.mom = Number(opts.mom ?? d.mom);
    this.m = Math.max(1, Math.round(Number(opts.m ?? d.m)));
    this.lam = Number(opts.lam ?? d.lam);
    this.aaBeta = Number(opts.aaBeta ?? 1.0);
    this.beta = Number(opts.beta ?? d.beta);
    this.label = opts.label != null ? model._label(opts.label) : null;
    this.track = opts.track ?? true;
    if (kind === 'nudge' && !this.label) throw new Error('Solver "nudge" needs opts.label (P ints 0..N-1 = solution - 1)');
    const D = model.D;
    this._z = new Float32Array(D);
    this._gz = new Float32Array(D);
    this._g = new Float32Array(D);
    this._v = kind === 'gd' || kind === 'nudge' ? new Float32Array(D) : null;
    this._gram = null;
    if (kind === 'anderson') {
      const m = this.m;
      this._X = []; this._F = []; this._G = [];
      for (let i = 0; i < m; i++) {
        this._X.push(new Float32Array(D)); this._F.push(new Float32Array(D)); this._G.push(new Float64Array(D));
      }
      this._gram = new Float64Array(m * m);
      this._Hm = new Float64Array((m + 1) * (m + 1));
      this._rhs = new Float64Array(m + 1);
      this._alpha = new Float64Array(m);
      this._acc = new Float64Array(D);
    }
    this.reset(opts.z0 ?? null);
  }

  /** Restart from z0 (default 0; for "nudge" the 300-sweep free state): zero momentum / Anderson history, iter = 0. */
  reset(z0 = null) {
    const md = this.model;
    this._z.fill(0);
    if (z0) this._z.set(md.flatOf(z0));
    else if (this.kind === 'nudge') {
      const zv = md._views(this._z);
      for (let k = 0; k < md.defaults.T_trace; k++) md.sweepInto(zv, this.xinj, zv);
    }
    if (this._v) this._v.fill(0);
    if (this._gram) { this._gram.fill(0); this.aaFallbacks = 0; }
    this._gzValid = false; this._zPub = null; this.iter = 0; this.last = null;
    return this;
  }

  /** Current state (a copy that later steps never mutate). Assigning z = state restarts from that state. */
  get z() {
    if (!this._zPub) this._zPub = this.model._wrap(new Float32Array(this._z));
    return this._zPub;
  }

  set z(v) { this.reset(v); }

  /** The math of one iteration (no diagnostics). */
  _advance() {
    const md = this.model, x = this.xinj, z = this._z;
    switch (this.kind) {
      case 'sweep':
        if (!this._gzValid) md._sweepFlat(z, x, this._gz);
        z.set(this._gz);
        break;
      case 'damped': {
        if (!this._gzValid) md._sweepFlat(z, x, this._gz);
        const a = this.a, b = 1 - a, gz = this._gz;
        for (let i = 0; i < z.length; i++) z[i] = b * z[i] + a * gz[i];
        break;
      }
      case 'gd':
      case 'nudge': {
        const g = this._g, v = this._v, mom = this.mom, lr = this.lr;
        md._gradEFlat(z, x, g);
        if (this.kind === 'nudge') md._addGradC(z, this.label, this.beta, g);
        for (let i = 0; i < z.length; i++) { v[i] = mom * v[i] - lr * g[i]; z[i] += v[i]; }
        break;
      }
      default:
        this._stepAnderson();
    }
    this._gzValid = false;
    this._zPub = null;
    this.iter++;
  }

  /** One iteration + diagnostics -> {z, E, residual, iter} (+ C, F for "nudge"). */
  step() {
    this._advance();
    const md = this.model, x = this.xinj, z = this._z;
    const out = { z: this.z, E: NaN, residual: NaN, iter: this.iter };
    if (this.track) {
      out.E = md._energyFlat(z, x);
      md._sweepFlat(z, x, this._gz);
      this._gzValid = true;                                       // reused by the next sweep / damped / seeding step
      out.residual = relDiff(this._gz, z);
      if (this.kind === 'nudge') { out.C = md._costFlat(z, this.label); out.F = out.E + this.beta * out.C; }
    }
    this.last = out;
    return out;
  }

  /** n iterations; diagnostics only for the last one. */
  run(n) {
    if (!(n > 0)) return this.last;
    for (let k = 1; k < n; k++) this._advance();
    return this.step();
  }

  /**
   * One Anderson iteration, k = number of map evaluations so far:
   *   k < m:  X_k = z (k = 0: z0; else the previous plain iterate), F_k = g(X_k)            (seeding)
   *   k >= m: G_i = F_i - X_i; gram = G G^T; scale = mean_i gram_ii;
   *           [[0, 1^T], [1, gram + lam (scale + 1e-12) I]] [mu; alpha] = [1; 0]            (mixing weights)
   *           x_k = aaBeta sum_i alpha_i F_i + (1 - aaBeta) sum_i alpha_i X_i; slot = k mod m; X_slot = x_k;
   *           F_slot = g(x_k)
   *   the displayed iterate is F_slot (the value Anderson acceleration returns at the end).
   * A singular / non-finite LS falls back to the plain iterate (counted in aaFallbacks; JAX would produce NaN).
   */
  _stepAnderson() {
    const md = this.model, x = this.xinj, D = md.D, m = this.m, k = this.iter, z = this._z, X = this._X, F = this._F;
    let slot;
    if (k < m) {
      slot = k;
      X[slot].set(z);
      if (this._gzValid) F[slot].set(this._gz);
      else md._sweepFlat(X[slot], x, F[slot]);
    } else {
      slot = k % m;
      const alpha = this._aaAlpha();
      if (alpha) {
        const acc = this._acc, bF = this.aaBeta, bX = 1 - bF;
        acc.fill(0);
        for (let i = 0; i < m; i++) {
          const Fi = F[i], Xi = X[i], cF = bF * alpha[i], cX = bX * alpha[i];
          if (cF !== 0) for (let j = 0; j < D; j++) acc[j] += cF * Fi[j];
          if (cX !== 0) for (let j = 0; j < D; j++) acc[j] += cX * Xi[j];
        }
        X[slot].set(acc);
      } else {
        X[slot].set(F[(k - 1) % m]);
        this.aaFallbacks++;
      }
      md._sweepFlat(X[slot], x, F[slot]);
    }
    this._gramUpdate(slot, Math.min(k + 1, m));
    z.set(F[slot]);
  }

  /** Refresh G_slot = F_slot - X_slot and row / column `slot` of the gram over the `filled` slots. */
  _gramUpdate(slot, filled) {
    const D = this.model.D, m = this.m, G = this._G, Gs = G[slot], Xs = this._X[slot], Fs = this._F[slot], gram = this._gram;
    for (let j = 0; j < D; j++) Gs[j] = Fs[j] - Xs[j];
    for (let i = 0; i < filled; i++) {
      const Gi = G[i];
      let s = 0;
      for (let j = 0; j < D; j++) s += Gs[j] * Gi[j];
      gram[slot * m + i] = s;
      gram[i * m + slot] = s;
    }
  }

  /** Bordered, relatively ridge-regularised LS for the mixing weights. */
  _aaAlpha() {
    const m = this.m, n = m + 1, Hm = this._Hm, b = this._rhs, gram = this._gram, alpha = this._alpha;
    let scale = 0;
    for (let i = 0; i < m; i++) scale += gram[i * m + i];
    scale /= m;
    const ridge = this.lam * (scale + 1e-12);
    Hm.fill(0);
    for (let i = 0; i < m; i++) {
      Hm[1 + i] = 1;
      Hm[(1 + i) * n] = 1;
      for (let j = 0; j < m; j++) Hm[(1 + i) * n + 1 + j] = gram[i * m + j] + (i === j ? ridge : 0);
    }
    b.fill(0);
    b[0] = 1;
    if (!solveDense(Hm, b, n)) return null;
    for (let i = 0; i < m; i++) { alpha[i] = b[1 + i]; if (!Number.isFinite(alpha[i])) return null; }
    return alpha;
  }
}

// =====================================================================================================================
// EP gradient, energy slice
// =====================================================================================================================
/**
 * Centred EP estimate (batch size 1):
 *   (thetaGrad(z+) - thetaGrad(z-)) / (2 beta)  per leaf of {Win, bin, W1, b1, W2, b2}.
 * For W1 / W2 this is a difference of two outer products, e.g. W2: (-u+ delta_s+^T + u- delta_s-^T) / (2 beta).
 */
export function epGradient(model, xinj, zPlus, zMinus, beta) {
  if (!(beta !== 0) || !Number.isFinite(beta)) throw new Error('epGradient: beta must be a non-zero number');
  const gp = model.thetaGrad(zPlus, xinj), gm = model.thetaGrad(zMinus, xinj), inv = 1 / (2 * beta), out = {};
  for (const k of THETA_NAMES) {
    const a = gp[k], b = gm[k], o = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) o[i] = (a[i] - b[i]) * inv;
    out[k] = withShape(o, model.thetaShapes[k]);
  }
  return out;
}

/**
 * Generator form of energySlice (yields the fraction done after every grid row, returns the result) so a page can
 * spread the res^2 energy evaluations over frames.
 */
export function* energySliceIter(model, xinj, zStar, z0, { res = 41, span = 1.3, seed = 7 } = {}) {
  const D = model.D, zs = Float32Array.from(model.flatOf(zStar)), z0f = model.flatOf(z0);
  const u = new Float64Array(D), w = new Float64Array(D);
  let uu = 0;
  for (let i = 0; i < D; i++) { u[i] = z0f[i] - zs[i]; uu += u[i] * u[i]; }
  let nu = Math.sqrt(uu);
  if (!(nu > 1e-12)) {                                           // z0 == z*: random u of norm 1e-2 max(|z*|, 1)
    randnFill(u, seed + 1);
    let n2 = 0, zn = 0;
    for (let i = 0; i < D; i++) { n2 += u[i] * u[i]; zn += zs[i] * zs[i]; }
    nu = 1e-2 * Math.max(Math.sqrt(zn), 1);
    const f = nu / Math.sqrt(n2);
    for (let i = 0; i < D; i++) u[i] *= f;
    uu = nu * nu;
  }
  randnFill(w, seed);
  for (let pass = 0; pass < 2; pass++) {                         // Gram-Schmidt w against u (twice)
    let wu = 0;
    for (let i = 0; i < D; i++) wu += w[i] * u[i];
    const f = wu / uu;
    for (let i = 0; i < D; i++) w[i] -= f * u[i];
  }
  let ww = 0;
  for (let i = 0; i < D; i++) ww += w[i] * w[i];
  const fw = nu / Math.sqrt(ww);
  for (let i = 0; i < D; i++) w[i] *= fw;
  ww = uu;
  const a = [], b = [];
  for (let i = 0; i < res; i++) { const t = res > 1 ? -span + (2 * span * i) / (res - 1) : 0; a.push(t); b.push(t); }
  const zt = new Float32Array(D);
  const evalAt = (aa, bb) => {
    for (let i = 0; i < D; i++) zt[i] = zs[i] + aa * u[i] + bb * w[i];
    return model._energyFlat(zt, xinj);
  };
  const grid = new Float32Array(res * res);
  let Emin = Infinity, Emax = -Infinity;
  for (let ib = 0; ib < res; ib++) {
    for (let ia = 0; ia < res; ia++) {
      const E = evalAt(a[ia], b[ib]);
      grid[ib * res + ia] = E;
      if (E < Emin) Emin = E;
      if (E > Emax) Emax = E;
    }
    yield (ib + 1) / res;
  }
  const coords = (z) => {
    const f = model.flatOf(z);
    let pa = 0, pb = 0;
    for (let i = 0; i < D; i++) { const d = f[i] - zs[i]; pa += d * u[i]; pb += d * w[i]; }
    return [pa / uu, pb / ww, f];
  };
  const project = (z) => { const c = coords(z); return [c[0], c[1]]; };
  const offPlane = (z) => {                                      // |z - z* - a u - b w| / |u|
    const [pa, pb, f] = coords(z);
    let s = 0;
    for (let i = 0; i < D; i++) { const r = f[i] - zs[i] - pa * u[i] - pb * w[i]; s += r * r; }
    return Math.sqrt(s) / nu;
  };
  return {
    grid, a, b, res, span, Emin, Emax, Estar: evalAt(0, 0), E0: evalAt(1, 0),
    u: Float32Array.from(u), w: Float32Array.from(w), project, offPlane, evalAt,
  };
}

/**
 * Real 2-D slice of the energy: grid[ib * res + ia] = E(z* + a[ia] u + b[ib] w), a = b = linspace(-span, span, res),
 * u = z0 - z*, w = seeded N(0, 1) direction Gram-Schmidt-orthogonalised to u and scaled to |u| (so z* is at (0, 0) and
 * z0 at (1, 0)). project(z) -> [a, b] = [<z - z*, u> / |u|^2, <z - z*, w> / |w|^2]; offPlane(z) = distance of z from
 * the plane in units of |u|; evalAt(a, b) -> E. If z0 == z*, u is a random direction of norm 1e-2 max(|z*|, 1).
 */
export function energySlice(model, xinj, zStar, z0, opts = {}) {
  const it = energySliceIter(model, xinj, zStar, z0, opts);
  let r = it.next();
  while (!r.done) r = it.next();
  return r.value;
}

// =====================================================================================================================
// selftest against the JAX references of model.json
// =====================================================================================================================
const fmtE = (x) => (Number.isFinite(x) ? x.toExponential(2) : String(x));

function mkRow(name, maxAbs, maxRel, tol, extra) {
  return Object.assign({ name, maxAbs, maxRel, tol, ok: Number.isFinite(maxRel) && maxRel <= tol }, extra || {});
}

function vecStats(got, ref) {
  if (!got || !ref || got.length !== ref.length) return { maxAbs: NaN, scale: 0, finite: false, mismatch: true };
  let maxAbs = 0, scale = 0, finite = true;
  for (let i = 0; i < ref.length; i++) {
    const r = ref[i], g = got[i];
    if (!Number.isFinite(g) || !Number.isFinite(r)) { finite = false; continue; }
    const d = Math.abs(g - r);
    if (d > maxAbs) maxAbs = d;
    const s = Math.abs(r);
    if (s > scale) scale = s;
  }
  return { maxAbs, scale, finite, mismatch: false };
}

const relOf = (e) => (e.mismatch || !e.finite ? Infinity : e.maxAbs / Math.max(e.scale, 1e-30));

/** maxRel = max|got - ref| / max|ref|. */
function vecRow(name, got, ref, tol, extra) {
  const e = vecStats(got, ref);
  const note = e.mismatch ? `length ${got ? got.length : '-'} vs ${ref ? ref.length : '-'}` : undefined;
  return mkRow(name, e.maxAbs, relOf(e), tol, Object.assign(note ? { note } : {}, extra));
}

/** Per vode max|got - ref| / max|ref|; the row reports the worst vode. */
function stateRow(name, got, ref, tol, extra) {
  let maxAbs = 0, maxRel = 0;
  const parts = [];
  for (const v of VODES) {
    const e = vecStats(got[v], ref[v]), rel = relOf(e);
    maxAbs = Number.isFinite(e.maxAbs) ? Math.max(maxAbs, e.maxAbs) : NaN;
    maxRel = Math.max(maxRel, rel);
    parts.push(`${v} ${fmtE(rel)}`);
  }
  return mkRow(name, maxAbs, maxRel, tol, Object.assign({ note: parts.join(', ') }, extra));
}

/** Energies: maxRel = |dE| / max(|E_ref|, energyFloor * 1/2 |z_ref|^2) (near an equilibrium E is a difference of
 *  nearly equal float32 numbers, so the floor is set by the state's own energy scale). */
function energyRow(model, name, got, ref, zRef, tol) {
  const floor = TOL.energyFloor * 0.5 * model.dot(zRef, zRef), d = Math.abs(got - ref);
  return mkRow(name, d, d / Math.max(Math.abs(ref), floor, 1e-30), tol, { note: `js ${fmtE(got)} jax ${fmtE(ref)}` });
}

/** Per theta leaf max|got - ref| / max|ref|; the row reports the worst leaf. */
function leavesRow(name, got, refObj, tol, extra) {
  let maxAbs = 0, maxRel = 0, n = 0;
  const parts = [];
  for (const k of THETA_NAMES) {
    if (refObj == null || refObj[k] == null) continue;
    const e = vecStats(got[k], decodeArray(refObj[k], k)), rel = relOf(e);
    maxAbs = Number.isFinite(e.maxAbs) ? Math.max(maxAbs, e.maxAbs) : NaN;
    maxRel = Math.max(maxRel, rel);
    n++;
    parts.push(`${k} ${fmtE(rel)}`);
  }
  if (!n) return mkRow(name, NaN, NaN, tol, Object.assign({ note: 'no theta leaves in the reference' }, extra, { ok: false }));
  return mkRow(name, maxAbs, maxRel, tol, Object.assign({ note: parts.join(', ') }, extra));
}

/** Decode a reference state: {u, s} of arrays | [u, s] | one flat array of D values. */
function decodeState(model, obj) {
  if (obj == null) return null;
  const z = model.zeroState();
  const flatLike = (Array.isArray(obj) && obj.length !== 2) || (!Array.isArray(obj) && (obj.b64 != null || obj.data != null));
  if (flatLike) {
    const f = decodeArray(obj, 'state');
    if (f.length !== model.D) throw new Error(`state: ${f.length} values, expected ${model.D}`);
    z.flat.set(f);
    return z;
  }
  const parts = Array.isArray(obj) ? { u: obj[0], s: obj[1] } : obj;
  for (const v of VODES) {
    const a = decodeArray(parts[v], `state.${v}`);
    if (!a || a.length !== model.sizes[v]) throw new Error(`state.${v}: ${a ? a.length : 0} values, expected ${model.sizes[v]}`);
    z[v].set(a);
  }
  return z;
}

/** Best alignment of a reference series ref[i] with got[i + off] (off 1: ref[0] is after the first sweep; off 0). */
function seriesFit(got, ref, floorAt) {
  let best = null;
  for (const off of [1, 0]) {
    if (ref.length + off > got.length) continue;
    let maxAbs = 0, maxRel = 0;
    for (let i = 0; i < ref.length; i++) {
      const g = got[i + off], r = Number(ref[i]), d = Math.abs(g - r);
      const rel = Number.isFinite(d) ? d / Math.max(Math.abs(r), floorAt(i + off), 1e-30) : Infinity;
      if (!(d <= maxAbs)) maxAbs = d;
      if (!(rel <= maxRel)) maxRel = rel;
    }
    if (!best || maxRel < best.maxRel) best = { off, maxAbs, maxRel };
  }
  return best || { off: 0, maxAbs: NaN, maxRel: Infinity };
}

const numArr = (a) => (a == null ? null : Array.isArray(a) ? a.flat(Infinity).map(Number) : Array.from(decodeArray(a)));

function* refChecks(model, r, tag, rows, opts, idx) {
  const guard = (name, fn) => {
    try { const row = fn(); if (row) rows.push(row); } catch (err) { rows.push(mkRow(name, NaN, Infinity, 0, { note: `error: ${err.message}` })); }
  };
  let x;
  try { x = model.embed(r.puzzle); } catch (err) { rows.push(mkRow(`${tag}.puzzle`, NaN, Infinity, 0, { note: `error: ${err.message}` })); return; }
  guard(`${tag}.x_inj`, () => (r.x_inj != null ? vecRow(`${tag}.x_inj`, x, decodeArray(r.x_inj), TOL.map) : null));
  // own free trajectory from z = 0 (K = the largest z_sweep<K> key in the reference, typically 60)
  let K = 1;
  for (const key of Object.keys(r)) { const mt = /^z_sweep(\d+)$/.exec(key); if (mt) K = Math.max(K, Number(mt[1])); }
  const z = model.zeroState();
  let z1 = null;
  for (let k = 1; k <= K; k++) { model.sweepInto(z, x, z); if (k === 1) z1 = model.cloneState(z); }
  guard(`${tag}.z_sweep1`, () => (r.z_sweep1 != null ? stateRow(`${tag}.z_sweep1 (1 sweep from 0)`, z1, decodeState(model, r.z_sweep1), TOL.map) : null));
  guard(`${tag}.E_sweep1`, () => {
    if (r.E_sweep1 == null || r.z_sweep1 == null) return null;
    const zr = decodeState(model, r.z_sweep1);
    return energyRow(model, `${tag}.E_sweep1 (E at ref state)`, model.energy(zr, x).E, Number(r.E_sweep1), zr, TOL.map);
  });
  if (K > 1) {
    const zKey = `z_sweep${K}`, eKey = `E_sweep${K}`, lKey = `logits_sweep${K}`;
    guard(`${tag}.${zKey}`, () => stateRow(`${tag}.${zKey} (${K} sweeps from 0)`, z, decodeState(model, r[zKey]), TOL.iter));
    guard(`${tag}.${eKey}`, () => {
      if (r[eKey] == null) return null;
      const zr = decodeState(model, r[zKey]);
      return energyRow(model, `${tag}.${eKey} (E at ref state)`, model.energy(zr, x).E, Number(r[eKey]), zr, TOL.map);
    });
    guard(`${tag}.${lKey}`, () => {
      if (r[lKey] == null) return null;
      return vecRow(`${tag}.${lKey} (readout of ref s)`, model.readout(decodeState(model, r[zKey]).s).logits, decodeArray(r[lKey]), TOL.map);
    });
    guard(`${tag}.${lKey} own`, () => (r[lKey] != null
      ? vecRow(`${tag}.${lKey} (JS trajectory)`, model.readout(z.s).logits, decodeArray(r[lKey]), TOL.iter) : null));
  }
  yield;
  if (r.z_pert != null) {
    guard(`${tag}.E_pert`, () => {
      if (r.E_pert == null) return null;
      const zp = decodeState(model, r.z_pert);
      return energyRow(model, `${tag}.E_pert`, model.energy(zp, x).E, Number(r.E_pert), zp, TOL.map);
    });
    guard(`${tag}.gradE_pert`, () => (r.gradE_pert != null
      ? stateRow(`${tag}.gradE_pert`, model.gradE(decodeState(model, r.z_pert), x), decodeState(model, r.gradE_pert), TOL.map) : null));
    const tg = r.thetaGrad_pert ?? r.theta_grad_pert;
    if (tg) guard(`${tag}.thetaGrad_pert`, () => leavesRow(`${tag}.thetaGrad_pert`, model.thetaGrad(decodeState(model, r.z_pert), x), tg, TOL.map));
    yield;
  }
  const nd = r.nudge || null;
  const own = {};
  const mode = opts.nudge ?? 'none';                            // light by default (page load); 'first' | 'all' replay
  if (nd && (mode === 'all' || mode === true || (mode === 'first' && idx === 0))) {
    const d = model.defaults;
    const beta = Number(nd.beta ?? d.beta), steps = Number(nd.steps ?? nd.T ?? d.T_nudg);
    const lr = Number(nd.lr_h ?? nd.lr ?? d.lr_h), mom = Number(nd.mom ?? d.mom), Tf = Number(nd.T_free ?? d.T_trace);
    let zf = null;
    guard(`${tag}.nudge.z_free`, () => {
      if (nd.z_free == null) return null;
      zf = decodeState(model, nd.z_free);
      return stateRow(`${tag}.nudge.z_free (${Tf} sweeps from 0)`, model.relaxFree(x, Tf), zf, TOL.iter);
    });
    yield;
    if (!zf) zf = model.relaxFree(x, Tf);
    for (const [sgn, key] of [[1, 'z_plus'], [-1, 'z_minus']]) {
      if (nd[key] == null) continue;
      guard(`${tag}.nudge.${key}`, () => {
        const S = new Solver(model, x, { kind: 'nudge', z0: zf, beta: sgn * beta, lr, mom, label: nd.label, track: false });
        S.run(steps);
        own[key] = S.z;
        return stateRow(`${tag}.nudge ${sgn > 0 ? '+' : '-'}beta (${steps} heavy-ball steps from ref z_free)`, S.z, decodeState(model, nd[key]), TOL.iter);
      });
      yield;
    }
  }
  const epRef = r.ep_grad ?? r.ep_grad_ref ?? (nd ? (nd.ep_grad ?? nd.ep_grad_ref) : null);
  if (epRef && nd && nd.z_plus != null && nd.z_minus != null) {
    const beta = Number(epRef.beta ?? nd.beta ?? model.defaults.beta);
    const leaves = epRef.grad ?? epRef.theta ?? epRef;
    guard(`${tag}.ep_grad`, () => leavesRow(`${tag}.ep_grad (centred, at ref z+ / z-)`,
      epGradient(model, x, decodeState(model, nd.z_plus), decodeState(model, nd.z_minus), beta), leaves, TOL.ep));
    if (own.z_plus && own.z_minus) {
      guard(`${tag}.ep_grad own`, () => leavesRow(`${tag}.ep_grad (from the JS nudged states)`,
        epGradient(model, x, own.z_plus, own.z_minus, beta), leaves, TOL.iter, { info: true }));
    }
    yield;
  }
  if (r.anderson) { guard(`${tag}.anderson`, () => andersonRow(model, x, r.anderson, `${tag}.anderson`)); yield; }
}

/** Optional Anderson reference {m, T | max_iter, lam?, aa_beta?, z0?, z | z_out | z_aa}. */
function andersonRow(model, x, a, tag) {
  const d = model.defaults;
  const zr = decodeState(model, a.z ?? a.z_out ?? a.z_aa ?? a.z_star ?? null);
  if (!zr) return mkRow(tag, NaN, NaN, TOL.anderson, { info: true, ok: false, note: 'unrecognised format (expected {m, T, z})' });
  const m = Number(a.m ?? d.m), T = Number(a.T ?? a.max_iter ?? a.steps ?? a.iters ?? d.T_free);
  const S = new Solver(model, x, {
    kind: 'anderson', m, lam: Number(a.lam ?? d.lam), aaBeta: Number(a.aa_beta ?? 1),
    z0: a.z0 != null ? decodeState(model, a.z0) : null, track: false,
  });
  S.run(T);
  return stateRow(`${tag} (m = ${m}, ${T} iterations)`, S.z, zr, TOL.anderson);
}

/** traj = {E: [...], residual: [...]} after each sweep of one puzzle (reference[0] or puzzles[0]: the best match is
 *  reported); residual definition |sweep(z_k) - z_k| / |z_k| or |z_k - z_(k-1)| / |z_k|, also the best match. */
function* trajChecks(model, json, rows) {
  const tr = json.traj;
  const Es = numArr(tr.E ?? tr.energy), Rs = numArr(tr.residual ?? tr.res);
  if (!Es && !Rs) { rows.push(mkRow('traj', NaN, NaN, 0, { info: true, ok: false, note: 'unrecognised format' })); return; }
  const cands = [];
  if (tr.puzzle) cands.push(['traj.puzzle', tr.puzzle]);
  else {
    if (Array.isArray(json.reference) && json.reference[0] && json.reference[0].puzzle) cands.push(['reference[0]', json.reference[0].puzzle]);
    if (Array.isArray(json.puzzles) && json.puzzles[0] && json.puzzles[0].puzzle) cands.push(['puzzles[0]', json.puzzles[0].puzzle]);
  }
  const K = Math.max(Es ? Es.length : 0, Rs ? Rs.length : 0);
  let bestE = null, bestR = null;
  for (const [src, puz] of cands) {
    let x;
    try { x = model.embed(puz); } catch (err) { continue; }
    const zs = [model.zeroState()];
    for (let k = 1; k <= K + 2; k++) zs.push(model.sweep(zs[k - 1], x));
    const E = zs.map((zz) => model.energy(zz, x).E), nrm = zs.map((zz) => model.norm(zz));
    const dn = (p, q) => { let s = 0; for (let i = 0; i < model.D; i++) { const dd = p.flat[i] - q.flat[i]; s += dd * dd; } return Math.sqrt(s); };
    const resA = zs.map((zz, k) => (k + 1 < zs.length ? dn(zs[k + 1], zz) / (nrm[k] + 1e-20) : NaN));
    const resB = zs.map((zz, k) => (k > 0 ? dn(zz, zs[k - 1]) / (nrm[k] + 1e-20) : NaN));
    if (Es) {
      const f = seriesFit(E, Es, (k) => TOL.energyFloor * 0.5 * nrm[k] * nrm[k]);
      if (!bestE || f.maxRel < bestE.maxRel) bestE = Object.assign(f, { src });
    }
    if (Rs) {
      for (const [def, arr] of [['|sweep(z_k) - z_k| / |z_k|', resA], ['|z_k - z_(k-1)| / |z_k|', resB]]) {
        const f = seriesFit(arr, Rs, () => TOL.residualAtol / TOL.iter);   // maxRel <= iter  <=>  |d| <= max(iter |r|, atol)
        if (!bestR || f.maxRel < bestR.maxRel) bestR = Object.assign(f, { src, def });
      }
    }
    yield;
  }
  if (Es) {
    rows.push(bestE ? mkRow(`traj.E (${Es.length} sweeps)`, bestE.maxAbs, bestE.maxRel, TOL.iter, { note: `${bestE.src}; ref[i] = E(z_(i+${bestE.off}))` })
      : mkRow('traj.E', NaN, Infinity, TOL.iter, { note: 'no puzzle to replay' }));
  }
  if (Rs) {
    rows.push(bestR ? mkRow(`traj.residual (${Rs.length} sweeps)`, bestR.maxAbs, bestR.maxRel, TOL.iter, { note: `${bestR.src}; ${bestR.def} at k = i+${bestR.off}` })
      : mkRow('traj.residual', NaN, Infinity, TOL.iter, { note: 'no puzzle to replay' }));
  }
}

/** Replays the bundled puzzles for T_free sweeps: agreement of model_solves / sweeps_to_solve (informational). */
function* puzzleChecks(model, json, rows, nMax) {
  const list = Array.isArray(json.puzzles) ? json.puzzles : [];
  const n = Math.min(nMax, list.length);
  if (!(n > 0)) return;
  const T = model.defaults.T_free, pred = new Int32Array(model.P), z = model.zeroState();
  let agree = 0, solvedJs = 0, solvedRef = 0, stepsAgree = 0, stepsN = 0, bad = 0;
  for (let i = 0; i < n; i++) {
    const pz = list[i];
    let x, sol;
    try { x = model.embed(pz.puzzle); sol = toIntVector(pz.solution, model.P, 'solution'); } catch (err) { bad++; continue; }
    z.flat.fill(0);
    let first = -1, lastBad = 0, solvedT = false;
    for (let k = 1; k <= T; k++) {
      model.sweepInto(z, x, z);
      model._argmaxInto(z.s, pred);
      let ok = true;
      for (let p = 0; p < model.P; p++) if (pred[p] + 1 !== sol[p]) { ok = false; break; }
      if (ok && first < 0) first = k;
      if (!ok) lastBad = k;
      if (k === T) solvedT = ok;
    }
    if (solvedT) solvedJs++;
    if (pz.model_solves) solvedRef++;
    if (Boolean(pz.model_solves) === solvedT) agree++;
    if (pz.sweeps_to_solve != null) {
      const s = Number(pz.sweeps_to_solve);
      stepsN++;
      if (s === first || (solvedT && s === lastBad + 1) || (s < 0 && first < 0)) stepsAgree++;
    }
    if (i % 8 === 7) yield;
  }
  const m = n - bad;
  rows.push(mkRow(`puzzles: model_solves agree (${m} puzzles, ${T} sweeps)`, m - agree, m ? (m - agree) / m : NaN, 0,
    { info: true, note: `JS solves ${solvedJs}/${m}, JAX ${solvedRef}/${m}${bad ? `, ${bad} malformed` : ''}` }));
  if (stepsN) {
    rows.push(mkRow('puzzles: sweeps_to_solve agree', stepsN - stepsAgree, (stepsN - stepsAgree) / stepsN, 0,
      { info: true, note: `${stepsAgree}/${stepsN} (first solved sweep, or first sweep of the final solved run)` }));
  }
}

function* selftestGen(model, json, opts, rows) {
  if (!json) { rows.push(mkRow('model.json', NaN, Infinity, 0, { note: 'no json given' })); return; }
  const refs = Array.isArray(json.reference) ? json.reference : [];
  if (!refs.length) rows.push(mkRow('json.reference', NaN, Infinity, 0, { note: 'no reference vectors in model.json' }));
  for (let i = 0; i < refs.length; i++) yield* refChecks(model, refs[i], `ref${i}`, rows, opts, i);
  if (json.traj) yield* trajChecks(model, json, rows);
  if (json.anderson && refs[0]) {
    const x = model.embed(json.anderson.puzzle ?? refs[0].puzzle);
    try { rows.push(andersonRow(model, x, json.anderson, 'anderson')); } catch (err) { rows.push(mkRow('anderson', NaN, Infinity, 0, { note: `error: ${err.message}` })); }
    yield;
  }
  yield* puzzleChecks(model, json, rows, Number(opts.puzzles ?? 0));
}

function finish(rows, t0) {
  const counted = rows.filter((r) => !r.info);
  const passed = counted.filter((r) => r.ok).length;
  return { ok: counted.length > 0 && passed === counted.length, rows, passed, total: counted.length, ms: nowMs() - t0 };
}

/**
 * Compare the engine with json.reference (+ json.traj, optional anderson / ep_grad) -> {ok, rows: [{name, maxAbs,
 * maxRel, tol, ok, note?, info?}], passed, total, ms}. maxRel = max|js - jax| / max|jax| per vode / leaf (energies:
 * relative to max(|E|, 1e-4 * 1/2 |z|^2); traj.residual: relative to max(|r|, residualAtol / iter), i.e. an absolute
 * floor of 1e-6). tol: TOL.map = 1e-4 for single maps / energies / gradients at the reference states; TOL.iter = 1e-3
 * for iterated trajectories (60 sweeps, 300 sweeps, 1500 nudge steps, Anderson) and the traj series; TOL.ep = 1e-3 for
 * the centred EP gradient (a difference quotient); TOL.anderson = 1e-2. Rows with info = true are reported but do not
 * affect ok.
 * opts: {nudge: 'none' (DEFAULT, light: ~0.1-0.2 s, safe on page load) | 'first' | 'all' (replay the 300-sweep free
 * state and the two 1500-step heavy-ball nudges of the first / every reference, ~1-2 s each), puzzles: 0 (default) | n
 * | Infinity (replay the bundled puzzles; informational)}. The EP gradient at the reference z+ / z- is always checked.
 */
export function selftest(model, json = model.json, opts = {}) {
  const rows = [], t0 = nowMs(), it = selftestGen(model, json, opts || {}, rows);
  for (let r = it.next(); !r.done; r = it.next()) { /* run to completion */ }
  return finish(rows, t0);
}

/** selftest that yields to the event loop every opts.sliceMs (default 12 ms); opts.onProgress(nRows) is called. */
export async function selftestAsync(model, json = model.json, opts = {}) {
  opts = opts || {};
  const rows = [], t0 = nowMs(), it = selftestGen(model, json, opts, rows), slice = Number(opts.sliceMs ?? 12);
  let last = nowMs();
  for (let r = it.next(); !r.done; r = it.next()) {
    if (opts.onProgress) opts.onProgress(rows.length);
    if (nowMs() - last > slice) { await new Promise((res) => setTimeout(res, 0)); last = nowMs(); }
  }
  return finish(rows, t0);
}

// =====================================================================================================================
// loading
// =====================================================================================================================
/** Parsed model.json (or its text) -> Model: base64 float32 leaves -> {data: Float32Array, shape}. */
export function decodeModel(json) {
  if (typeof json === 'string') json = JSON.parse(json);
  if (!json || typeof json !== 'object') throw new Error('decodeModel: expected the parsed model.json object');
  const meta = Object.assign({}, json.meta || {});
  const theta = {}, psi = {};
  for (const [dst, src, names, what] of [[theta, json.theta, THETA_NAMES, 'theta'], [psi, json.psi, PSI_NAMES, 'psi']]) {
    for (const k of names) {
      const a = decodeArray(src ? src[k] : null, `${what}.${k}`);
      if (!a) throw new Error(`decodeModel: ${what}.${k} missing`);
      dst[k] = { data: a, shape: a.shape ? a.shape.slice() : [a.length] };
    }
  }
  return new Model({ meta, theta, psi, puzzles: json.puzzles || [], json });
}

/** fetch(url) + decodeModel. */
export async function loadModel(url = './model.json') {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`loadModel: HTTP ${res.status} for ${url}`);
  return decodeModel(await res.json());
}
