// sim.js — thin adapter around the inference engine (./model.js) + the simulation loop.
// Every call into the engine and every engine-specific name (vode keys, weight names, shapes) lives in
// this file, so an API mismatch is a one-file fix. Nothing here touches the DOM or three.js.
//
// The model: a 2-layer fully connected tanh network folded into a loop.
//   x (16 cells x 5 one-hot = 80) -> x_inj = x Win + bin   (256)
//   û(s) = tanh(s W1 + b1)        ŝ(u) = tanh(u W2 + b2 + x_inj)        (W1, W2: 256 x 256, x @ W convention)
//   E(z) = ½‖u − û(s)‖² + ½‖s − ŝ(u)‖²        readout: logits = s V + c -> (16 cells, 4 digits)

export let VODES = ['u', 's'];                        // live binding, replaced by model.meta.vodes on load
export const KMAX = 4;                                // unrolled view: s_{k0} -> u -> s -> ... (KMAX sweeps = 2*KMAX+1 layers)

export const VODE_INFO = {
  s: {
    key: 's', title: 's', name: 'main layer (input + output)', shapeTxt: '256 tanh units',
    formula: 'ŝ = tanh(u·W2 + b2 + x_inj)',
    role: 'The clue embedding x_inj enters here on every sweep, and the readout V reads the board from here: the loop closes at s.',
  },
  u: {
    key: 'u', title: 'u', name: 'hidden layer', shapeTxt: '256 tanh units',
    formula: 'û = tanh(s·W1 + b1)',
    role: 'The other half of the loop: a fully connected layer that re-describes s, then feeds it back through W2.',
  },
};
export function vodeInfo(v) {
  return VODE_INFO[v] || { key: v, title: v, name: 'layer', shapeTxt: '', formula: '', role: '' };
}
export const ACT_TXT = 'tanh units · no normalisation layers · ‖W1‖₂ = ‖W2‖₂ = 3';
export const ARCS = [                                 // forward arcs: from -> to, the matrix that carries the signal
  { from: 's', to: 'u', W: 'W1' },
  { from: 'u', to: 's', W: 'W2' },
];

// Free-phase inference is gradient descent on E (predictive-coding inference): u <- u - eta dE/du, s <- s - eta
// dE/ds, either plain or with heavy-ball momentum (a HUD toggle, see Sim#setMode). Every number (init, eta, mom,
// T_free, tol) is read live from the selected model's meta.inference -- nothing is hardcoded here (the app only
// overrides the initial state, see APP_INIT). Models are listed in models/index.json; puzzles.json is shared.
export const GD_DESC_PLAIN = 'Gradient descent on E (predictive-coding inference), plain: z ← z − η·∇E(z). Each layer feels its own error and the error of the layer it predicts, so errors flow backward (rose). Nodes start at zero, then descend E from there.';
export const GD_DESC_MOM = 'Gradient descent on E with heavy-ball momentum: v ← μv − η·∇E(z), z ← z + v. Same dynamics, faster.';
// Playback speeds, 0.25 to 60 iterations per second on a stepped log scale. DEFAULT_SPEED (3) must be a literal
// member of this list, otherwise the slider label snaps to the nearest preset while the engine steps at the real
// value underneath. The nudge phase plays at the same speed as the free phase, so nothing above 60 is needed.
export const SPEEDS = [0.25, 0.5, 1, 2, 3, 4, 6, 10, 15, 20, 30, 45, 60];   // iterations per second
// ONE default speed for every board (fast/medium/slow/custom alike): the visible speed difference between
// showcase categories must come from the dynamics (iterations to an all-green readout), never from playback
// tricks. 3 it/s puts the slowest showcase board (~29 iterations to green, best.json HB mode) at ~10 s, while a
// fast board (2 iterations to green) is on screen for under a second.
export const DEFAULT_SPEED = 3;

// The free phase starts from z = 0 in the app, in both inference modes, for every board (showcase, custom editor
// boards, and the Learn step's free/re-run phases), so the convergence is visible live instead of jumping in one
// sweep to a state that is already mostly right. This overrides meta.inference.init / meta.inference_alt.init
// ("ff" = one feedforward sweep from 0, the initialisation used in training and for the reported test accuracy).
// Zero-init reaches the identical final board as ff-init on all 240 test puzzles, in both modes, always converges
// inside the T_free budget, and takes only a few more E < tol iterations (+6 median), so this is a display choice,
// not a change in what the model computes or how well it scores.
// The nudged phase is unaffected: it still starts from the free-phase equilibrium.
export const APP_INIT = 'zero';

// ------------------------------------------------------------------ small helpers
const num = (v, d) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function cloneZ(z) {
  const o = {};
  for (const v of VODES) o[v] = Float32Array.from(z[v]);
  return o;
}
function zerosLike(z) {
  const o = {};
  for (const v of VODES) o[v] = new Float32Array(z[v].length);
  return o;
}
function copyZ(dst, src) { for (const v of VODES) dst[v].set(src[v]); }
function rms(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s / Math.max(1, a.length));
}
function zNorm(z) {
  let s = 0;
  for (const v of VODES) { const a = z[v]; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; }
  return Math.sqrt(s);
}
function zDiffNorm(x, y) {
  let s = 0;
  for (const v of VODES) { const a = x[v], b = y[v]; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } }
  return Math.sqrt(s);
}
function lerpArr(out, a, b, t) {
  const u = 1 - t;
  for (let i = 0; i < out.length; i++) out[i] = a[i] * u + b[i] * t;
}
function linspace(a, b, n) {
  const o = new Float32Array(n);
  for (let i = 0; i < n; i++) o[i] = n === 1 ? a : a + (b - a) * (i / (n - 1));
  return o;
}

// ------------------------------------------------------------------ 4x4 Sudoku helpers
export function allGrids() {                       // all 288 valid 4x4 grids, digits 1..4
  const out = [];
  const g = new Int8Array(16);
  const ok = (p, d) => {
    const r = p >> 2, c = p & 3;
    for (let k = 0; k < 4; k++) if (g[r * 4 + k] === d || g[k * 4 + c] === d) return false;
    const br = r & 2, bc = c & 2;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) if (g[(br + i) * 4 + bc + j] === d) return false;
    return true;
  };
  const rec = (p) => {
    if (p === 16) { out.push(Int8Array.from(g)); return; }
    for (let d = 1; d <= 4; d++) if (ok(p, d)) { g[p] = d; rec(p + 1); g[p] = 0; }
  };
  rec(0);
  return out;
}
export function isValidBoard(d) {                  // d: 16 digits 1..4
  for (let i = 0; i < 4; i++) {
    let row = 0, col = 0, box = 0;
    for (let j = 0; j < 4; j++) {
      row |= 1 << d[i * 4 + j];
      col |= 1 << d[j * 4 + i];
      const r = ((i >> 1) << 1) + (j >> 1), c = ((i & 1) << 1) + (j & 1);
      box |= 1 << d[r * 4 + c];
    }
    if (row !== 30 || col !== 30 || box !== 30) return false;   // bits 1..4 set
  }
  return true;
}
export function solutionsFor(puzzle, grids) {
  return grids.filter((g) => {
    for (let p = 0; p < 16; p++) if (puzzle[p] && g[p] !== puzzle[p]) return false;
    return true;
  });
}
/** digits0: argmax 0..3 per cell; puzzle 0..4; solution 1..4 or null */
export function boardStatus(digits0, puzzle, solution) {
  const d = new Int8Array(16);
  for (let p = 0; p < 16; p++) d[p] = digits0[p] + 1;
  let clueOk = true;
  for (let p = 0; p < 16; p++) if (puzzle[p] && puzzle[p] !== d[p]) { clueOk = false; break; }
  const valid = clueOk && isValidBoard(d);
  let correct = -1;
  if (solution) { correct = 0; for (let p = 0; p < 16; p++) if (solution[p] === d[p]) correct++; }
  return { valid, clueOk, correct };
}
function countGivens(p) { let k = 0; for (let i = 0; i < p.length; i++) if (p[i]) k++; return k; }

// status: 0 = stalled (E >= tol after T_free), 1 = converged but wrong, 2 = solved
function normPuzzle(p, i) {
  const puzzle = Int8Array.from(Array.isArray(p.puzzle) ? p.puzzle : []);
  const solution = Array.isArray(p.solution) && p.solution.length === 16 ? Int8Array.from(p.solution) : null;
  const status = num(p.status, null);
  return {
    id: p.id ?? i, index: i, puzzle, solution,
    givens: num(p.givens, countGivens(puzzle)),
    source: typeof p.source === 'string' ? p.source : null,
    status, iters: num(p.iters, null), cells: num(p.cells, null), Efree: num(p.E, null),
    custom: false,
  };
}
function fallbackPuzzle(grids) {
  const sol = grids[137 % grids.length];
  const puzzle = new Int8Array(16);
  [0, 3, 5, 10, 12, 15].forEach((p) => { puzzle[p] = sol[p]; });
  return { id: 'builtin', index: 0, puzzle, solution: null, givens: 6, status: null, custom: false };
}
/** which puzzle_status/iters/cells/E arrays to read for `mode` ('on' = momentum, 'off' = plain): these are
 *  TOP-LEVEL on the model json, not under meta. meta.inference_alt + json.puzzle_status_alt (+ _iters/_cells/_E)
 *  hold the results for the other inference mode and are optional -- without them both modes read the single
 *  top-level set. */
function statusSourceFor(json, meta, mode) {
  const haveAlt = meta.inference_alt && json.puzzle_status_alt;
  const base = { puzzle_status: json.puzzle_status, puzzle_iters: json.puzzle_iters, puzzle_cells: json.puzzle_cells, puzzle_E: json.puzzle_E };
  if (!haveAlt) return base;
  const alt = { puzzle_status: json.puzzle_status_alt, puzzle_iters: json.puzzle_iters_alt, puzzle_cells: json.puzzle_cells_alt, puzzle_E: json.puzzle_E_alt };
  const baseIsOn = num((meta.inference || {}).mom, 0) > 0;
  return (mode === 'on') === baseIsOn ? base : alt;
}
/** merge puzzles.json's shared puzzle list with one model's per-puzzle result arrays for the given mode. */
export function mergePuzzles(puzzlesJson, json, meta, mode) {
  const src = statusSourceFor(json || {}, meta || {}, mode);
  const list = (puzzlesJson && Array.isArray(puzzlesJson.puzzles)) ? puzzlesJson.puzzles : [];
  return list.map((p, i) => ({
    ...p,
    status: src.puzzle_status ? src.puzzle_status[i] : null,
    iters: src.puzzle_iters ? src.puzzle_iters[i] : null,
    cells: src.puzzle_cells ? src.puzzle_cells[i] : null,
    E: src.puzzle_E ? src.puzzle_E[i] : null,
  }));
}
/** free-phase settings for `mode` ('on' = momentum, 'off' = plain GD), from meta.inference / meta.inference_alt.
 *  Fallback (no meta.inference_alt): on = eta 0.3, mom 0.9; off = plain GD at meta's own eta. */
export function inferenceFor(meta, mode) {
  const inf = (meta && meta.inference) || {};
  const alt = meta && meta.inference_alt;
  // init is always APP_INIT ('zero'), overriding whatever the checkpoint's own meta.inference(.alt)
  // .init says -- see the APP_INIT comment above. eta/mom/T/tol are untouched, still read live from the model.
  if (alt) {
    const infIsOn = num(inf.mom, 0) > 0;
    const chosen = (mode === 'on') === infIsOn ? inf : alt;
    return { init: APP_INIT, eta: num(chosen.eta, 0.1), mom: num(chosen.mom, 0), T: Math.round(num(chosen.T_free, 150)), tol: num(chosen.tol, 1e-6) };
  }
  if (mode === 'on') return { init: APP_INIT, eta: 0.3, mom: 0.9, T: Math.round(num(inf.T_free, 150)), tol: num(inf.tol, 1e-6) };
  return { init: APP_INIT, eta: num(inf.eta, 0.1), mom: 0, T: Math.round(num(inf.T_free, 150)), tol: num(inf.tol, 1e-6) };
}
function readDims(model, meta) {
  const d = (model && model.dims) || {};
  const N = num(d.N, num(meta.N, 4));
  const P = num(d.P, num(meta.P, N * N));
  const din = num(d.din, num(meta.din, P * (N + 1)));
  const H = num(d.H, num(meta.H, 256));
  return { N, P, din, H };
}
/** {data: Float32Array, shape: [r, c]} from whatever the engine stores */
function asMat(x, fallbackShape) {
  if (!x) return null;
  if (x.data && x.shape) return { data: x.data instanceof Float32Array ? x.data : Float32Array.from(x.data), shape: x.shape.slice() };
  if (ArrayBuffer.isView(x) || Array.isArray(x)) return { data: Float32Array.from(x), shape: fallbackShape ? fallbackShape.slice() : [1, x.length] };
  return null;
}

// ------------------------------------------------------------------ loading (models/index.json lists the models,
// puzzles.json is shared by all of them)
//
// ENABLED_MODELS: the only model ids the app will show in the picker or fetch a weights file for. The app filters
// models/index.json down to this list before anything (including fetch()) touches the other entries; to show
// more models, add their entries to models/index.json and their ids to this array.
export const ENABLED_MODELS = ['best'];

export async function createSim(opts = {}) {
  const say = opts.onProgress || (() => {});
  say('Loading the inference engine…');
  let engine;
  try {
    engine = await import('./model.js');
  } catch (e) {
    throw new Error('Could not load model.js: ' + ((e && e.message) || e));
  }
  if (typeof engine.Solver !== 'function' || typeof engine.decodeModel !== 'function') {
    throw new Error('model.js does not export Solver and decodeModel().');
  }
  say('Loading the model list…');
  const rawModelsIndex = opts.modelsIndex || await (await fetch('./models/index.json')).json();
  // filter to ENABLED_MODELS before anything else reads modelsIndex.models, so no code path downstream (picker,
  // fetch of a weights file, About tab, ...) can see or load a model that is not enabled.
  const modelsIndex = { ...rawModelsIndex, models: (rawModelsIndex.models || []).filter((m) => ENABLED_MODELS.includes(m.id)) };
  say('Loading the puzzle set…');
  const puzzlesJson = opts.puzzlesJson || await (await fetch('./puzzles.json')).json();
  say('Loading the showcase…');
  let showcaseJson = opts.showcaseJson;
  if (!showcaseJson) { try { const r = await fetch('./models/showcase.json'); showcaseJson = r.ok ? await r.json() : null; } catch (e) { showcaseJson = null; } }
  const models = Array.isArray(modelsIndex.models) ? modelsIndex.models : [];
  const pick = (id) => models.find((m) => m.id === id);
  const startId = opts.modelId || (pick('best') || pick('mid') || models[0] || {}).id;
  const sim = new Sim(engine, modelsIndex, puzzlesJson, opts);
  sim.showcaseRaw = (showcaseJson && Array.isArray(showcaseJson.showcase)) ? showcaseJson.showcase : [];
  await sim.loadModelById(startId, say);
  return sim;
}

// ------------------------------------------------------------------ the simulation
export class Sim {
  constructor(engine, modelsIndex, puzzlesJson, opts = {}) {
    this.engine = engine;
    this.modelsIndex = modelsIndex || { models: [] };
    this.puzzlesJson = puzzlesJson || { puzzles: [] };
    this.mobile = !!opts.mobile;
    this.grids = allGrids();
    this.mode = 'off';         // 'off' = plain GD, 'on' = + momentum (toggle, see inferenceFor)
    // Standard nudging only: one-sided +beta, no centred toggle in the app (the model.js engine still supports
    // centred EP, used by the self-test against the JAX reference; the app never enables it).
    this.learnHistory = [];    // [{n, E, cells, solved}]
    this.learnUpdates = 0;
    this.showcaseRaw = [];     // models/showcase.json's 10 curated entries (set by createSim before first load)
    this.showcase = [];        // showcaseRaw + the live puzzle object + this model/mode's status badge
    this.listeners = new Map();
  }

  /** the 10-puzzle showcase (models/showcase.json), tagged with this model+mode's live status; falls back to
   *  a synthesized 10 (sorted by iters, 3 fast/2 medium/3 slow among solved + 2 converged-wrong) if the file is
   *  missing, so the picker always has something to show. */
  _buildShowcase() {
    const byIndex = new Map(this.puzzles.map((p) => [p.index, p]));
    const wrap = (e) => { const p = byIndex.get(e.puzzle_index); return p ? { ...e, puzzle: p, status: p.status } : null; };
    if (this.showcaseRaw.length) { this.showcase = this.showcaseRaw.map(wrap).filter(Boolean); return; }
    const solved = this.puzzles.filter((p) => p.status === 2 && p.iters != null).sort((a, b) => a.iters - b.iters);
    const wrong = this.puzzles.filter((p) => p.status === 1);
    const n = solved.length;
    const pick = (list, category) => list.map((p) => ({ id: p.id, puzzle_index: p.index, category, label: p.id, iters_hb: p.iters, iters_plain: null, cells_correct: 16, presses_to_fix: null, puzzle: p, status: p.status }));
    this.showcase = [
      ...pick(solved.slice(0, 3), 'fast'),
      ...pick([solved[Math.floor(n * 0.45)], solved[Math.floor(n * 0.55)]].filter(Boolean), 'medium'),
      ...pick(solved.slice(-3).reverse(), 'slow'),
      ...pick(wrong.slice(0, 2), 'unsolved'),
    ];
  }

  /** (Re)load a model by models/index.json id: fetch + decode, rebuild the puzzle pool for the current mode,
   *  reset params from its meta.inference (mode) / meta.app_learning, keep the current puzzle id if it exists in
   *  the new pool (else fall back to meta.default_puzzle / the usual heuristic). */
  async loadModelById(id, say = () => {}) {
    const entry = (this.modelsIndex.models || []).find((m) => m.id === id) || (this.modelsIndex.models || [])[0];
    if (!entry) throw new Error('models/index.json has no models');
    say(`Loading "${entry.label || id}"…`);
    const r = await fetch(entry.file.startsWith('.') || entry.file.startsWith('/') ? entry.file : `./${entry.file}`);
    if (!r.ok) throw new Error(`${entry.file}: HTTP ${r.status}`);
    const json = await r.json();
    say('Decoding the weights…');
    const model = this.engine.decodeModel(json);
    const keepId = this.current && !this.current.custom ? this.current.id : null;

    this.modelId = entry.id; this.modelEntry = entry;
    this.model = model;
    this.json = json || {};
    this.meta = model.meta || {};
    if (Array.isArray(this.meta.vodes) && this.meta.vodes.length) VODES = this.meta.vodes.slice();
    this.vodes = VODES;
    this.dims = readDims(model, this.meta);
    this._rebuildPuzzles();
    this._buildShowcase();

    // The nudge phase has no eta/momentum of its own: it uses the same values as the free phase
    // (this.params.lr/mom, set below by _applyMode()/inferenceFor() from meta.inference(.alt)), toggled by the
    // same momentum switch. tNudge (from meta.app_learning.T_nudge) is a cap, not a fixed run length: the nudge
    // phase stops early once it settles (see the tick loop), like the free phase's own E < tol stop.
    const nd = this.meta.app_learning || this.meta.nudge || {};
    this.params = {
      beta: num(nd.beta, 0.05),
      tNudge: Math.round(num(nd.T_nudge, 300)), tFree: 300,   // tFree: only the hidden rho/zStar diagnostic below
      learnLr: num((this.meta.app_learning || {}).lr, 0.05), lrReadout: num((this.meta.app_learning || {}).lr_readout, 0),
      cap: (this.meta.app_learning || {}).cap ?? (this.meta.train || {}).cap ?? null,
    };
    // default inference mode from the checkpoint's own meta (per model)
    this.mode = num((this.meta.inference || {}).mom, 0) > 0 ? 'on' : 'off';
    this._applyMode();

    this.speed = 3;
    this.playing = false;
    this.autoPause = true;
    this.maxStepsPerFrame = 12;
    this.learnHistory = [];
    this.learnUpdates = 0;
    this.phase = 'free';
    this.iter = 0;
    this.phaseIter = 0;
    this.acc = 0;
    this.blend = 1;
    this.blendDur = 0.3;
    this.version = 0;
    this.layerVersion = 0;
    this.sliceToken = 0;
    this.slice = null;
    this.traj = [];
    this._pendingZ = [];
    this.cycle = null;
    this.rho = num(this.meta.rho_J, null);
    this.stats = { valid: false, clueOk: false, correct: -1 };
    this._weights = null;

    let next = keepId != null ? this.puzzles.find((p) => p.id === keepId) : null;
    if (!next) next = this.pickInitialPuzzle();
    this.loadPuzzle(next);
    this.emit('model', entry);
  }

  /** rebuild this.puzzles for the current model + mode (json.puzzle_status, merged with puzzles.json). */
  _rebuildPuzzles() {
    const merged = mergePuzzles(this.puzzlesJson, this.json, this.meta, this.mode);
    this.puzzles = merged.map(normPuzzle).filter((p) => p.puzzle.length === 16);
    if (!this.puzzles.length) this.puzzles = [fallbackPuzzle(this.grids)];
  }
  /** free-phase settings for the current mode; mutates this.params.{lr,mom,init,T,tol} and maxFreeIters. */
  _applyMode() {
    const inf = inferenceFor(this.meta, this.mode);
    this.inference = inf;
    this.params = this.params || {};
    Object.assign(this.params, { lr: inf.eta, mom: inf.mom });
    this.maxFreeIters = Math.max(inf.T, 1);
  }
  /** 'off' = plain gradient descent on E; 'on' = + momentum. Re-tags the puzzle pool and resets the run. */
  setMode(mode) {
    if (mode !== 'on' && mode !== 'off') return;
    this.mode = mode;
    this._applyMode();
    this._rebuildPuzzles();
    this._buildShowcase();
    const keepId = this.current && !this.current.custom ? this.current.id : null;
    const next = (keepId != null && this.puzzles.find((p) => p.id === keepId)) || this.pickInitialPuzzle();
    this.loadPuzzle(next);
    this.emit('params', this.params);
  }

  // ---------------------------------------------------------------- events
  on(ev, fn) {
    if (!this.listeners.has(ev)) this.listeners.set(ev, new Set());
    this.listeners.get(ev).add(fn);
    return () => this.listeners.get(ev).delete(fn);
  }
  emit(ev, data) {
    const set = this.listeners.get(ev);
    if (!set) return;
    for (const fn of set) {
      try { fn(data); } catch (e) { console.error(`[sim] listener for "${ev}" failed`, e); }
    }
  }

  // ---------------------------------------------------------------- engine adapter (one-file fix zone)
  // NB keep the engine's array as is: it carries a non-enumerable .puzzle that thetaGrad / epGradient need
  _engineEmbed(puz) { return this.model.embed(Array.from(puz)); }
  _engineZero() { return cloneZ(this.model.zeroState()); }
  _enginePredict(z) { return cloneZ(this.model.predict(z, this.xinj)); }
  _engineReadout(s) {
    const r = this.model.readout(s) || {};
    const P = this.dims.P;
    const probs = r.probs ? Float32Array.from(r.probs) : softmaxRows(r.logits, P);
    return { probs };
  }
  _engineResidual(z) {
    try {
      const r = this.model.residual(z, this.xinj);
      return Number.isFinite(r) ? r : NaN;
    } catch (e) { return NaN; }
  }
  _engineSolver(kind, o) {
    const opts = { kind, ...o };
    if (opts.z0) opts.z0 = cloneZ(opts.z0);              // never let the engine alias our buffers
    return new this.engine.Solver(this.model, this.xinj, opts);
  }
  _engineStep(s) {
    const r = s.step() || {};
    return { z: r.z || s.z, E: r.E, residual: r.residual, iter: r.iter };
  }
  _engineEpGradient(zp, zm, beta) { return this.engine.epGradient(this.model, this.xinj, zp, zm, beta); }
  _engineSlice(zStar, z0, o) { return this.engine.energySlice(this.model, this.xinj, zStar, z0, o); }
  _engineRho(zStar) {
    if (typeof this.model.spectralRadius !== 'function') return null;
    try { const r = this.model.spectralRadius(zStar, this.xinj); return Number.isFinite(r) ? r : (r && Number.isFinite(r.rho) ? r.rho : null); }
    catch (e) { return null; }
  }
  /** weights as {data, shape} (row-major, x @ W): W1[i][j] connects s_i -> u_j, W2[i][j] connects u_i -> s_j */
  weights() {
    if (this._weights) return this._weights;
    const th = this.model.theta || {}, ps = this.model.psi || {};
    const { H, P, din } = this.dims;
    this._weights = {
      W1: asMat(th.W1, [H, H]), W2: asMat(th.W2, [H, H]), Win: asMat(th.Win, [din, H]),
      b1: asMat(th.b1, [1, H]), b2: asMat(th.b2, [1, H]), bin: asMat(th.bin, [1, H]),
      V: asMat(ps.V, [H, P * 4]), c: asMat(ps.c, [1, P * 4]),
    };
    return this._weights;
  }

  // ---------------------------------------------------------------- derived quantities of one state
  _snap(z, residual) {
    const pred = this._enginePredict(z);
    const err = {}, parts = {}, rmsV = {}, rmsP = {}, rmsE = {};
    let E = 0;
    for (const v of VODES) {
      const a = z[v], p = pred[v], e = new Float32Array(a.length);
      let se = 0, sv = 0, sp = 0;
      for (let i = 0; i < a.length; i++) {
        const d = a[i] - p[i];
        e[i] = d; se += d * d; sv += a[i] * a[i]; sp += p[i] * p[i];
      }
      err[v] = e;
      parts[v] = 0.5 * se;
      E += 0.5 * se;
      rmsE[v] = Math.sqrt(se / a.length);
      rmsV[v] = Math.sqrt(sv / a.length);
      rmsP[v] = Math.sqrt(sp / a.length);
    }
    const { probs } = this._engineReadout(z.s);
    return { z, pred, err, E, parts, rmsV, rmsP, rmsE, probs, residual };
  }
  _energy(z) {
    const pred = this._enginePredict(z);
    let E = 0;
    for (const v of VODES) { const a = z[v], p = pred[v]; for (let i = 0; i < a.length; i++) { const d = a[i] - p[i]; E += 0.5 * d * d; } }
    return E;
  }

  // ---------------------------------------------------------------- view (smoothly interpolated live state for the visuals)
  _allocView(zt) {
    const P = this.dims.P;
    const mk = () => zerosLike(zt);
    this.view = {
      z: mk(), pred: mk(), err: mk(), probs: new Float32Array(P * 4),
      digits: new Int32Array(P), conf: new Float32Array(P),
      rmsV: {}, rmsP: {}, rmsE: {}, parts: {}, E: 0,
    };
    this.from = { z: mk(), pred: mk(), err: mk(), probs: new Float32Array(P * 4) };
  }
  _setImmediate(snap) {
    this.to = snap;
    copyZ(this.from.z, snap.z); copyZ(this.from.pred, snap.pred); copyZ(this.from.err, snap.err);
    this.from.probs.set(snap.probs);
    this.blend = 1;
    this._applyView(1);
  }
  _beginBlend(snap) {
    const V = this.view;
    copyZ(this.from.z, V.z); copyZ(this.from.pred, V.pred); copyZ(this.from.err, V.err);
    this.from.probs.set(V.probs);
    this.to = snap;
    this.blend = 0;
    this.blendDur = this.playing ? clamp(0.9 / this.speed, 0.04, 0.4) : 0.32;
    this._applyView(0);
  }
  _applyView(t) {
    const V = this.view, F = this.from, T = this.to;
    const k = ease(clamp(t, 0, 1));
    for (const v of VODES) {
      lerpArr(V.z[v], F.z[v], T.z[v], k);
      lerpArr(V.pred[v], F.pred[v], T.pred[v], k);
      lerpArr(V.err[v], F.err[v], T.err[v], k);
    }
    lerpArr(V.probs, F.probs, T.probs, k);
    const P = this.dims.P;
    for (let p = 0; p < P; p++) {
      let bi = 0, bv = -1;
      for (let d = 0; d < 4; d++) { const q = V.probs[p * 4 + d]; if (q > bv) { bv = q; bi = d; } }
      V.digits[p] = bi; V.conf[p] = bv;
    }
    V.rmsV = T.rmsV; V.rmsP = T.rmsP; V.rmsE = T.rmsE; V.parts = T.parts; V.E = T.E;
    this.version++;
  }

  // ---------------------------------------------------------------- unrolled-view history: the last KMAX+1 states
  /** layer list for the unrolled network: [{k, u, s}] oldest first, at most KMAX+1 entries, last = current */
  _pushLayer(z) {
    this.layers.push({ k: this.iter, u: Float32Array.from(z.u || []), s: Float32Array.from(z.s || []) });
    if (this.layers.length > KMAX + 1) this.layers.shift();
    this.layerVersion++;
  }

  // ---------------------------------------------------------------- puzzles
  labelFor() {                                     // 0..3 per cell, or null
    if (this.solution) return Array.from(this.solution, (d) => d - 1);
    if (this.solutions && this.solutions.length > 1) {
      let best = this.solutions[0], bestScore = -1;
      for (const g of this.solutions) {
        let sc = 0;
        for (let p = 0; p < 16; p++) if (g[p] === this.view.digits[p] + 1) sc++;
        if (sc > bestScore) { bestScore = sc; best = g; }
      }
      return Array.from(best, (d) => d - 1);
    }
    return null;
  }
  /** true iff the current clue set has EXACTLY one valid completion (a trivial 4x4 brute-force solve, solutionsFor
   *  / allGrids): the only case where Learn step has an unambiguous target on a hand-edited board. */
  get hasUniqueSolution() { return !!(this.solutions && this.solutions.length === 1); }
  /** default load target: json.default_puzzle(_id) if it is itself in the showcase, else a "medium" showcase
   *  board; else any showcase entry; else a solved puzzle with a visible progression; else just the first puzzle
   *  (e.g. for an untrained model, whose free phase may never converge -- the app must still show *something* and
   *  handle the stall gracefully). */
  pickInitialPuzzle() {
    const id = this.json.default_puzzle_id;
    const idx = num(this.json.default_puzzle, null);
    const inShowcase = (pred) => this.showcase.find(pred);
    if (id != null && inShowcase((e) => e.id === id)) return this.puzzles.find((p) => p.id === id) || this.puzzles[0];
    if (idx != null && inShowcase((e) => e.puzzle_index === idx)) { const p = this.puzzles.find((p) => p.index === idx); if (p) return p; }
    const medium = inShowcase((e) => e.category === 'medium');
    if (medium) return medium.puzzle;
    if (this.showcase.length) return this.showcase[0].puzzle;
    if (id != null) { const p = this.puzzles.find((p) => p.id === id); if (p) return p; }
    if (idx != null) { const p = this.puzzles.find((p) => p.index === idx) || this.puzzles[idx]; if (p) return p; }
    const good = this.puzzles.filter((p) => p.status === 2 && p.iters != null && p.iters >= 8 && p.iters <= 60 && p.givens === 6);
    if (good.length) return good[0];
    const anySolved = this.puzzles.find((p) => p.status === 2);
    return anySolved || this.puzzles[0];
  }
  /** next puzzle (from the current one) whose status === 2 (solved) / !== 2 (wrong or stalled), wrapping around. */
  nextByStatus(wantSolved) {
    const list = this.puzzles;
    if (!list.length) return;
    const curIdx = this.basePuzzle ? list.findIndex((p) => p.id === this.basePuzzle.id) : -1;
    for (let k = 1; k <= list.length; k++) {
      const p = list[(curIdx + k + list.length) % list.length];
      const ok = wantSolved ? p.status === 2 : p.status !== 2;
      if (ok) { this.loadPuzzle(p); this.play(); return p; }
    }
    this.emit('notice', wantSolved ? 'No other solved puzzle in this pool.' : 'No failing puzzle in this pool.');
    return null;
  }
  loadPuzzle(p) {
    this.current = p;
    this.puzzle = Int8Array.from(p.puzzle);
    this.solutions = solutionsFor(this.puzzle, this.grids);
    let sol = p.solution && p.solution.length === 16 ? Int8Array.from(p.solution) : null;
    if (!sol && this.solutions.length === 1) sol = this.solutions[0];
    this.solution = sol;
    if (!p.custom) this.basePuzzle = p;
    this.xinj = this._engineEmbed(this.puzzle);
    let mx = 1e-9;
    for (let i = 0; i < this.xinj.length; i++) mx = Math.max(mx, Math.abs(this.xinj[i]));
    this.injNorm = this.xinj.map((x) => Math.abs(x) / mx);           // per-s-unit injection strength, 0..1

    // a reference equilibrium (hidden relaxation from z = 0)
    this.z0 = this._engineZero();
    // the free phase's actual starting point: this.inference.init is always APP_INIT ('zero', see the top of this
    // file) regardless of what the checkpoint's own meta.inference(.alt).init says -- "ff" (one forward sweep from
    // z = 0) remains supported here, just not reachable from the live app.
    this.zInit = (this.inference && this.inference.init) === 'zero' ? cloneZ(this.z0) : this._relaxHidden('sweep', { z0: this.z0 }, 1);
    this.zStar = this._relaxHidden('sweep', { z0: this.z0 }, this.params.tFree, 1e-7);
    this.zStarRes = this._engineResidual(this.zStar);
    this.refRms = {};
    for (const v of VODES) this.refRms[v] = Math.max(1e-3, rms(this.zStar[v]));
    if (!this.view) this._allocView(this.zStar);
    this.slice = null;
    this.sliceToken++;
    this.rho = null;
    // one fixed default speed for every board (the visible fast/slow difference must come from the dynamics
    // themselves, not from playback tricks); the speed slider still applies on top of this.
    this.speed = DEFAULT_SPEED;
    this.resetRun();
    this.emit('puzzle', p);
  }
  /** load a showcase.json entry by its showcase index (not the puzzle index) -- the HUD's chip grid uses this. */
  loadShowcaseEntry(i) {
    const e = this.showcase[i];
    if (!e) return;
    this.loadPuzzle(e.puzzle);
    this.play();
  }
  setClue(p, v) {
    const puz = Int8Array.from(this.puzzle);
    puz[p] = clamp(v | 0, 0, 4);
    this.loadPuzzle({ id: 'custom', index: -1, puzzle: puz, solution: null, givens: countGivens(puz), model_solves: null, sweeps_to_solve: null, custom: true });
  }
  cycleClue(p) { this.setClue(p, (this.puzzle[p] + 1) % 5); }
  clearClues() { this.loadPuzzle({ id: 'custom', index: -1, puzzle: new Int8Array(16), solution: null, givens: 0, model_solves: null, sweeps_to_solve: null, custom: true }); }
  restorePuzzle() { if (this.basePuzzle) this.loadPuzzle(this.basePuzzle); }

  // ---------------------------------------------------------------- runs and phases
  _relaxHidden(kind, opts, steps, tol = 0) {
    if (!tol) {                                     // no stopping test: skip the per-step diagnostics
      const s = this._engineSolver(kind, { ...opts, track: false });
      if (typeof s.run === 'function') { s.run(steps); return cloneZ(s.z); }
      for (let i = 0; i < steps; i++) s.step();
      return cloneZ(s.z);
    }
    const s = this._engineSolver(kind, opts);
    let z = opts.z0;
    for (let i = 0; i < steps; i++) {
      const r = this._engineStep(s);
      z = r.z;
      if (tol && Number.isFinite(r.residual) && r.residual < tol) break;
    }
    return cloneZ(z);
  }
  _makeFreeSolver(z0) {
    const P = this.params;
    return this._engineSolver('gd', { z0, lr: P.lr, mom: P.mom });
  }
  get gradientDriven() { return true; }   // the only free-phase solver is gradient descent on E; nudging is too
  get nudgeBeta() { return this.phase === 'nudge+' ? this.params.beta : 0; }

  resetRun() {
    this.phase = 'free';
    this.z = cloneZ(this.zInit);         // per meta.inference.init: "ff" (one sweep) or "zero"
    this.solver = this._makeFreeSolver(this.z);
    this.iter = 0; this.phaseIter = 0; this.acc = 0; this.stallCount = 0;
    this.converged = false; this.convCount = 0; this.stopReason = '';
    this.zFree = null; this.runs = { plus: null }; this.ep = null;
    this.greenAtIter = null;
    this._greenCandidate = null;   // start of the current unbroken all-correct streak (see _updateStats)
    const snap = this._snap(this.z, this._engineResidual(this.z));
    this.hist = { iter: [0], E: [snap.E], res: [snap.residual], markers: [{ iter: 0, label: 'free', kind: 'free' }] };
    this.layers = [];
    this._pushLayer(this.z);
    this.traj = [];
    this._pendingZ = [];
    this._pushTraj(this.z, snap.E);
    this._setImmediate(snap);
    this._updateStats(snap);
    this.emit('reset');
  }
  setParams(patch) {
    Object.assign(this.params, patch);
    this.emit('params', this.params);
  }
  _ensureFree() {
    if (this.phase === 'free') this.zFree = cloneZ(this.z);
    if (!this.zFree) this.zFree = cloneZ(this.zStar);
    return this.zFree;
  }
  setPhase(ph) {
    if (ph === 'free') {
      this.cycle = null;
      this.resetRun();
      this.emit('phase', ph);
      return true;
    }
    if (ph === 'nudge+') {
      const label = this.labelFor();
      if (!label) { this.emit('notice', 'This clue set has no solution, so there is no label to nudge toward.'); return false; }
      const zf = this._ensureFree();
      const beta = this.params.beta;
      this.solver = this._engineSolver('nudge', { z0: zf, beta, lr: this.params.lr, mom: this.params.mom, label });
      this.phase = ph; this.phaseIter = 0; this.converged = false; this.convCount = 0; this.stopReason = '';
      this.z = cloneZ(zf);
      this.hist.markers.push({ iter: this.iter, label: '+β', kind: ph });
      const snap = this._snap(this.z, this._engineResidual(this.z));
      this._beginBlend(snap);
      this._updateStats(snap);
      this.emit('phase', ph);
      return true;
    }
    if (ph === 'ep') return this.computeEP();
    return false;
  }
  /** dE/dtheta at z, via the engine's thetaGrad (each leaf a Float32Array with a non-enumerable .shape). */
  _engineThetaGrad(z) { return this.model.thetaGrad(z, this.xinj); }
  /** EP estimate of dE/dW, one-sided (standard nudging) only:
   *   g = dC/dtheta estimate = [dE/dW(z^beta) - dE/dW(z^0)] / beta     (z^0 = the free equilibrium; matches
   *   model.js epThetaGrad() exactly -- theta -= lr*g is the update, see applyEPUpdate()).
   *   Always computed from converged equilibria, so Learn is correct however early it is clicked: regardless of
   *   how far the animated free/nudge phases have gotten on screen, z^0 is a synchronous, hidden runFreeGD() to
   *   its own convergence (same init/eta/mom/tol the animation uses -- deterministic, so this is the exact state
   *   the animation would eventually reach, computed in a few ms instead of watched), and z^beta is always a
   *   fresh T_nudge-step relaxation from that z^0. Clicking Learn / EP signal before the on-screen animation
   *   settles still gives the fully converged EP update.
   */
  computeEP() {
    if (typeof this.model.thetaGrad !== 'function') { this.emit('notice', 'thetaGrad() is not available in model.js.'); return false; }
    const label = this.labelFor();
    if (!label) { this.emit('notice', 'This clue set has no solution, so there is no EP signal.'); return false; }
    const beta = this.params.beta, T = this.params.tNudge;
    const wasSettled = this.phase === 'free' && this.converged;
    const full = this.engine.runFreeGD(this.model, this.xinj, { init: this.inference.init, eta: this.inference.eta, mom: this.inference.mom, T: this.inference.T, tol: this.inference.tol });
    const zf = cloneZ(full.z);
    const zp = this._relaxHidden('nudge', { z0: zf, beta, lr: this.params.lr, mom: this.params.mom, label }, T);
    let g;
    try {
      const gp = this._engineThetaGrad(zp);
      const gRef = this._engineThetaGrad(zf);
      const inv = 1 / beta;
      g = {};
      // this must match model.js's own epThetaGrad() sign convention exactly: (gp - gRef) * inv, not negated.
      for (const k of Object.keys(gp)) {
        const a = gp[k], b = gRef[k], o = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) o[i] = (a[i] - b[i]) * inv;
        g[k] = o;
      }
    } catch (e) {
      this.emit('notice', 'EP gradient failed: ' + ((e && e.message) || e));
      return false;
    }
    const { H } = this.dims;
    const mats = {};
    for (const k of ['W1', 'W2']) {
      // the heatmap displays the actual update direction DeltaW = -g (matching the Learn tab's own formula label,
      // "DeltaW prop -[...]"), not the raw gradient g used internally for theta -= lr*g. Negate here, at display
      // time only, so the colours agree with the label while g keeps the sign applyEPUpdate() needs.
      const neg = g && g[k] ? Float32Array.from(g[k], (v) => -v) : null;
      const m = asMat(neg, [H, H]);
      if (!m) continue;
      let s = 0, mx = 0;
      for (let i = 0; i < m.data.length; i++) { s += m.data[i] * m.data[i]; mx = Math.max(mx, Math.abs(m.data[i])); }
      mats[k] = { ...m, fro: Math.sqrt(s), max: mx };
    }
    this.ep = { mats, g, beta, T, zPlus: zp, zFree: zf, label, forPuzzleId: this.current && this.current.id, fastForwarded: !wasSettled };
    this.cycle = null;
    this.phase = 'ep';
    this.pause();
    this.hist.markers.push({ iter: this.iter, label: 'EP', kind: 'ep' });
    this.emit('ep', this.ep);
    this.emit('phase', 'ep');
    if (!wasSettled) this.emit('notice', 'Fast-forwarded to equilibrium to compute the EP signal.');
    return true;
  }

  // ---------------------------------------------------------------- Learn step
  /** Shared bookkeeping after any weight update (learnStep() or applyEPUpdate()): push a row onto learnHistory,
   *  re-derive this puzzle's status against the just-updated weights, and re-animate the free phase from zero. */
  _afterWeightUpdate(label, E, converged) {
    this.learnUpdates++;
    const cells = this.model.accuracy(this._lastAfterS, label).cell * this.dims.P;
    this.learnHistory.push({ n: this.learnUpdates, E, cells, solved: converged && cells === this.dims.P });
    if (this.learnHistory.length > 50) this.learnHistory.shift();
    this._weights = null;                 // weights() cache is stale now
    if (this.current) {
      this.current.status = converged ? (cells === this.dims.P ? 2 : 1) : 0;
      this.current.cells = cells; this.current.Efree = E;
    }
    this.cycle = null;
    this.ep = null;                       // the EP signal was computed against the OLD weights; stale now
    this.loadPuzzle(this.current);        // recompute zInit/zStar with the new weights, animate the free phase
    this.play();
    this.emit('learn', this.learnHistory[this.learnHistory.length - 1]);
  }
  /** One full Learn step on the current puzzle: free -> +beta nudge -> EP update (one-sided only) ->
   *  theta/psi -= lr*grad -> optional spectral re-cap -> free phase re-run. Mutates the live model's weights in
   *  place (model.js learnStep()). This is the "x5 (fast-forward)" bulk path: it reruns free+nudge from scratch
   *  every call, so it does not depend on (or consume) any EP signal already shown in the Learn tab. */
  learnStep() {
    if (!this.hasUniqueSolution) { this.emit('notice', this.solutions.length === 0 ? 'These clues contradict each other: no solution to learn from.' : 'This clue set has more than one valid solution: no unambiguous target to learn from.'); return false; }
    const label = this.labelFor();
    if (!label) { this.emit('notice', 'This clue set has no solution, so there is no label to learn from.'); return false; }
    const cfg = {
      label,
      free: { init: this.inference.init, eta: this.inference.eta, mom: this.inference.mom, T: this.inference.T, tol: this.inference.tol },
      nudge: { beta: this.params.beta, eta: this.params.lr, mom: this.params.mom, T: this.params.tNudge },
      centred: false, lr: this.params.learnLr, lrReadout: this.params.lrReadout, cap: this.params.cap,
    };
    let r;
    try { r = this.engine.learnStep(this.model, this.xinj, cfg); } catch (e) {
      this.emit('notice', 'Learn failed: ' + ((e && e.message) || e));
      return false;
    }
    this._lastAfterS = r.after.z.s;
    if (this.current) this.current.iters = r.after.iters;
    this._afterWeightUpdate(label, r.after.E, r.after.converged);
    return true;
  }
  /** Phase-row "Learn" button: apply theta -= lr * (the CURRENT EP signal), computing that signal first
   *  (free -> nudge +beta -> dE/dW contrast) if it is missing or stale (wrong puzzle, or the weights changed since
   *  it was computed). Does NOT re-run free/nudge if a fresh signal is already sitting in the Learn tab -- it
   *  reuses exactly the zPlus/zFree states shown there. Then restarts the free phase from zero so the effect on
   *  THIS board is visible immediately, same as learnStep(). */
  applyEPUpdate() {
    if (!this.hasUniqueSolution) { this.emit('notice', this.solutions.length === 0 ? 'These clues contradict each other: no solution to learn from.' : 'This clue set has more than one valid solution: no unambiguous target to learn from.'); return false; }
    const label = this.labelFor();
    if (!label) { this.emit('notice', 'This clue set has no solution, so there is no label to learn from.'); return false; }
    if (!this.ep || !this.ep.g || this.ep.forPuzzleId !== (this.current && this.current.id)) {
      if (!this.computeEP()) return false; // computeEP() already emits a notice on failure
    }
    const { g } = this.ep;
    const lr = this.params.learnLr, lrReadout = this.params.lrReadout;
    for (const k of Object.keys(g)) {
      const dst = this.model.theta[k] && this.model.theta[k].data;
      if (!dst) continue;
      const src = g[k];
      for (let i = 0; i < dst.length; i++) dst[i] -= lr * src[i];
    }
    if (lrReadout > 0 && typeof this.model.gradPsi === 'function') {
      const gPsi = this.model.gradPsi(this.ep.zFree.s, label);
      const V = this.model.psi.V.data, c = this.model.psi.c.data;
      for (let i = 0; i < V.length; i++) V[i] -= lrReadout * gPsi.V[i];
      for (let i = 0; i < c.length; i++) c[i] -= lrReadout * gPsi.c[i];
    }
    if (this.params.cap != null && typeof this.engine.capSpectral === 'function') {
      this.engine.capSpectral(this.model.W1, this.model.H, this.params.cap);
      this.engine.capSpectral(this.model.W2, this.model.H, this.params.cap);
    }
    const after = this.engine.runFreeGD(this.model, this.xinj, { init: this.inference.init, eta: this.inference.eta, mom: this.inference.mom, T: this.inference.T, tol: this.inference.tol });
    this._lastAfterS = after.z.s;
    if (this.current) this.current.iters = after.iters;
    this._afterWeightUpdate(label, after.E, after.converged);
    return true;
  }
  /** Reload the selected checkpoint's own weights from its json (undoes every Learn step on this model). */
  resetWeights() {
    if (!this.modelId) return;
    const keep = this.current;
    this.loadModelById(this.modelId).then(() => {
      if (keep && !keep.custom) { const p = this.puzzles.find((p) => p.id === keep.id); if (p) this.loadPuzzle(p); }
      this.emit('notice', 'Weights reset to the checkpoint on disk.');
    });
  }
  /** "Run a full EP cycle": free phase -> nudge +beta -> EP signal -> weight update (lr) -> free phase again,
   *  animated at the current playback speed throughout (no phase changes the speed). */
  runEpCycle() {
    if (!this.labelFor()) { this.emit('notice', 'This clue set has no solution, so there is no label for the EP cycle.'); return; }
    this.setPhase('free');
    this.cycle = { stage: 'free' };
    this.play();
    this.emit('params', this.params);
  }
  _advanceCycle(reason) {
    if (!this.cycle) return;
    if (this.cycle.stage === 'free' && (reason === 'converged' || reason === 'budget')) {
      this.cycle.stage = 'plus'; this.setPhase('nudge+'); this.play();
    } else if (this.cycle.stage === 'plus' && (reason === 'converged' || reason === 'budget')) {
      // nudge settled (early, on its own convergence check -- see the tick loop) or hit its step cap either way:
      // compute the EP signal from exactly this state, apply the weight update, and restart the free phase.
      this.cycle.stage = 'ep';
      this.applyEPUpdate();
    }
  }

  // ---------------------------------------------------------------- transport
  play() {
    if (this.phase === 'ep') { this.emit('notice', 'The EP signal is computed. Pick a phase to keep iterating.'); return; }
    this.playing = true;
    this.emit('play', true);
  }
  pause() { this.playing = false; this.acc = 0; this.emit('play', false); }
  toggle() { if (this.playing) this.pause(); else this.play(); }
  setSpeed(v) { this.speed = clamp(v, 0.1, 1000); this.emit('params', this.params); }
  stepManual(n = 1) {
    if (this.phase === 'ep') return;
    if (this.playing) this.pause();
    for (let i = 0; i < n; i++) this._stepOnce(true);
  }
  _stepOnce(manual = false) {
    if (!this.solver) return;
    const prev = this.z;
    let r;
    try { r = this._engineStep(this.solver); } catch (e) {
      this.pause();
      this.emit('notice', 'Solver step failed: ' + ((e && e.message) || e));
      return;
    }
    const z = cloneZ(r.z);
    this.z = z;
    this.iter++; this.phaseIter++;
    // the solver reports |sweep(z) − z| / |z| (same definition as model.residual); recompute only if missing
    let res = num(r.residual, NaN);
    if (!Number.isFinite(res)) res = this._engineResidual(z);
    const snap = this._snap(z, res);
    if (!Number.isFinite(snap.E)) {
      this.pause();
      this.emit('notice', 'The state diverged (E is not finite). Reset, or use a smaller step size.');
      return;
    }
    const nz = zNorm(z);
    const delta = nz > 0 ? zDiffNorm(z, prev) / nz : 0;
    this.hist.iter.push(this.iter); this.hist.E.push(snap.E); this.hist.res.push(res);
    if (this.hist.iter.length > 8000) for (const k of ['iter', 'E', 'res']) this.hist[k].splice(0, 2000);
    if (this.phase === 'nudge+') this.runs.plus = { z, iters: this.phaseIter };
    if (this.phase === 'free') this._pushLayer(z);
    this._pushTraj(z, snap.E);
    if (manual) this.blendDur = 0.32;
    this._beginBlend(snap);
    this._updateStats(snap);
    this.emit('step', snap);

    // convergence / budget -- E < tol is the model's own stopping rule (meta.inference.tol)
    if (this.phase === 'free') {
      const tol = (this.inference && this.inference.tol) || 1e-6;
      if (!this.converged && snap.E < tol) {
        this.converged = true; this.stopReason = 'converged';
        if (this.cycle) this._advanceCycle('converged');
        else { if (this.autoPause && this.playing) this.pause(); this.emit('converged', this.iter); }
      } else if (!this.converged && this._gdStalled(delta, res)) {
        // gradient descent at rest (∇E ≈ 0) while E > 0: a stationary point of E that is not a fixed point of the loop
        this.stopReason = 'stalled';
        if (this.cycle) this._advanceCycle('budget');
        else { if (this.playing) this.pause(); this.emit('budget', this.iter); }
      } else if (!this.converged && this.phaseIter >= this.maxFreeIters) {
        this.stopReason = 'budget';
        if (this.cycle) this._advanceCycle('budget');
        else { if (this.playing) this.pause(); this.emit('budget', this.iter); }
      } else if (this.cycle && this.phaseIter >= this.params.tFree) {
        this._advanceCycle('budget');
      }
    } else if (this.phase === 'nudge+') {
      // the nudge phase stops on its own convergence (5 consecutive near-stationary steps), whether or not it is
      // part of an EP cycle; tNudge (from meta.app_learning.T_nudge) is a hard cap so a puzzle that never settles
      // still stops.
      if (delta < 1e-7) this.convCount++; else this.convCount = 0;
      if (this.phaseIter === this.params.tNudge || this.convCount === 5) {
        const reason = this.phaseIter === this.params.tNudge ? 'budget' : 'converged';
        this.stopReason = reason;
        if (this.cycle) this._advanceCycle(reason);
        else { if (this.playing && this.autoPause) this.pause(); this.emit(reason === 'budget' ? 'budget' : 'converged', this.iter); }
      }
    }
  }
  /** energy plateau (< 1% change over the last 150 iterations) while still far from a fixed point */
  _gdStalled(delta, res) {
    if (this.stopReason === 'stalled' || this.phaseIter < 200) return false;
    const H = this.hist.E, n = H.length;
    if (n < 152) return false;
    const now = H[n - 1], old = H[n - 151];
    return Number.isFinite(res) && res > 1e-3 && now > 1e-4 && Math.abs(old - now) < 0.01 * now;
  }
  _updateStats(snap) {
    const P = this.dims.P;
    const digits = new Int32Array(P);
    for (let p = 0; p < P; p++) {
      let bi = 0, bv = -1;
      for (let d = 0; d < 4; d++) { const q = snap.probs[p * 4 + d]; if (q > bv) { bv = q; bi = d; } }
      digits[p] = bi;
    }
    this.stats = boardStatus(digits, this.puzzle, this.solution);
    this.stats.digits = digits;
    // "all green at iteration k": the first iteration of the CURRENT unbroken all-correct streak (matches the
    // offline time-to-green methodology -- "first iteration after which the readout is correct and stays correct
    // until convergence" -- not just the first iteration it happens to look right, which can flicker true then
    // false again before the real, stable crossing on some puzzles). Any wrong reading clears the streak.
    const k = this.phase === 'free' ? this.iter : this.phaseIter;
    if (this.stats.valid) { if (this._greenCandidate == null) this._greenCandidate = k; this.greenAtIter = this._greenCandidate; }
    else { this._greenCandidate = null; this.greenAtIter = null; }
  }

  /** advance the simulation clock and the visual interpolation; call once per animation frame */
  frame(dt) {
    dt = clamp(dt, 0, 0.1);
    if (this.playing && this.solver && this.phase !== 'ep') {
      this.acc += dt * this.speed;
      let n = 0;
      while (this.acc >= 1 && n < this.maxStepsPerFrame && this.playing) { this.acc -= 1; this._stepOnce(); n++; }
      if (this.acc > 2) this.acc = 1;
    }
    if (this.blend < 1) {
      this.blend = Math.min(1, this.blend + dt / Math.max(0.02, this.blendDur));
      this._applyView(this.blend);
    }
  }

  // ---------------------------------------------------------------- spectral radius at z* (deferred, optional)
  computeRho() {
    const r = this._engineRho(this.zStar);
    this.rho = r;
    this.emit('rho', r);
    return r;
  }

  // ---------------------------------------------------------------- energy-well slice
  _pushTraj(z, E) {
    if (this.slice && this.slice.project) {
      const ab = this._project(z);
      if (ab) this.traj.push({ a: ab[0], b: ab[1], E });
    } else if (!this.slice) {
      if (this._pendingZ.length < 800) this._pendingZ.push({ z, E });
    }
    if (this.traj.length > 6000) this.traj.splice(0, 2000);
  }
  _project(z) {
    try {
      const ab = this.slice.project(z);
      if (!ab || !Number.isFinite(ab[0]) || !Number.isFinite(ab[1])) return null;
      return [ab[0], ab[1]];
    } catch (e) { return null; }
  }
  async computeSlice() {
    const token = ++this.sliceToken;
    if (typeof this.engine.energySlice !== 'function') {
      this.slice = { error: 'energySlice() is not available in model.js' };
      this.emit('slice', this.slice);
      return;
    }
    const res = this.mobile ? 33 : 49, span = 1.35;
    let raw;
    try {
      if (typeof this.engine.energySliceIter === 'function') {
        // generator form: spread the res^2 energy evaluations over several frames
        const it = this.engine.energySliceIter(this.model, this.xinj, this.zStar, this.z0, { res, span });
        let t0 = performance.now();
        let r = it.next();
        while (!r.done) {
          if (performance.now() - t0 > 8) {
            await new Promise((ok) => setTimeout(ok, 0));
            if (token !== this.sliceToken) return;
            t0 = performance.now();
          }
          r = it.next();
        }
        raw = r.value;
      } else {
        raw = await this._engineSlice(this.zStar, this.z0, { res, span });
      }
    } catch (e) {
      if (token !== this.sliceToken) return;
      this.slice = { error: String((e && e.message) || e) };
      this.emit('slice', this.slice);
      return;
    }
    if (token !== this.sliceToken || !raw || !raw.grid) return;
    const grid = Float32Array.from(raw.grid);
    const n = Math.round(Math.sqrt(grid.length));
    const axis = (v) => {
      if (v && v.length === n) return Float32Array.from(v);
      if (v && typeof v === 'object' && Number.isFinite(v.min) && Number.isFinite(v.max)) return linspace(v.min, v.max, n);
      if (typeof v === 'number' && v > 0) return linspace(-v, v, n);
      return linspace(-span, span, n);
    };
    const sl = { n, grid, a: axis(raw.a), b: axis(raw.b), project: typeof raw.project === 'function' ? raw.project : null, transposed: false };
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < grid.length; i++) {
      const L = Math.log10(Math.max(grid[i], 1e-12));
      if (Number.isFinite(L)) { lo = Math.min(lo, L); hi = Math.max(hi, L); }
    }
    sl.logMin = lo; sl.logMax = Math.max(hi, lo + 1e-3);
    // index-order calibration: which grid layout reproduces E(z0) at project(z0)?
    if (sl.project) {
      try {
        const ab = sl.project(this.z0);
        const Et = Math.log10(Math.max(this._energy(this.z0), 1e-12));
        const eA = Math.abs(Math.log10(Math.max(sampleSlice(sl, ab[0], ab[1], false), 1e-12)) - Et);
        const eB = Math.abs(Math.log10(Math.max(sampleSlice(sl, ab[0], ab[1], true), 1e-12)) - Et);
        sl.transposed = eB + 1e-6 < eA;
        sl.z0ab = [ab[0], ab[1]];
        const abs = sl.project(this.zStar);
        sl.zsab = [abs[0], abs[1]];
      } catch (e) { /* keep defaults */ }
    }
    if (!sl.z0ab) sl.z0ab = [1, 0];
    if (!sl.zsab) sl.zsab = [0, 0];
    this.slice = sl;
    if (sl.project) {
      this.traj = [];
      for (const it of this._pendingZ) { const ab = this._project(it.z); if (ab) this.traj.push({ a: ab[0], b: ab[1], E: it.E }); }
      if (!this._pendingZ.length) { const ab = this._project(this.z); if (ab) this.traj.push({ a: ab[0], b: ab[1], E: this.to ? this.to.E : 0 }); }
    }
    this._pendingZ = [];
    this.emit('slice', sl);
  }

  // ---------------------------------------------------------------- numerics self-test
  async runSelftest() {
    const fn = typeof this.engine.selftestAsync === 'function' ? this.engine.selftestAsync
      : typeof this.engine.selftest === 'function' ? this.engine.selftest : null;
    if (!fn) return { ok: null, rows: [], error: 'selftest() is not exported by model.js' };
    try {
      const r = await fn(this.model, this.json);
      return r || { ok: null, rows: [] };
    } catch (e) {
      return { ok: false, rows: [], error: String((e && e.message) || e) };
    }
  }
}

/** bilinear sample of the slice grid at (a, b) */
export function sampleSlice(sl, a, b, transposed = sl.transposed) {
  const n = sl.n;
  const pos = (ax, v) => {
    const lo = ax[0], hi = ax[n - 1];
    const f = hi === lo ? 0 : ((v - lo) / (hi - lo)) * (n - 1);
    return clamp(f, 0, n - 1);
  };
  const fa = pos(sl.a, a), fb = pos(sl.b, b);
  const ia = Math.min(n - 2, Math.floor(fa)), ib = Math.min(n - 2, Math.floor(fb));
  const ta = fa - ia, tb = fb - ib;
  const get = (i, j) => (transposed ? sl.grid[i * n + j] : sl.grid[j * n + i]);   // i: a-index, j: b-index
  const v00 = get(ia, ib), v10 = get(ia + 1, ib), v01 = get(ia, ib + 1), v11 = get(ia + 1, ib + 1);
  return (v00 * (1 - ta) + v10 * ta) * (1 - tb) + (v01 * (1 - ta) + v11 * ta) * tb;
}

function softmaxRows(logits, P) {
  const out = new Float32Array(P * 4);
  if (!logits) return out.fill(0.25);
  for (let p = 0; p < P; p++) {
    let m = -Infinity;
    for (let d = 0; d < 4; d++) m = Math.max(m, logits[p * 4 + d]);
    let s = 0;
    for (let d = 0; d < 4; d++) { const e = Math.exp(logits[p * 4 + d] - m); out[p * 4 + d] = e; s += e; }
    for (let d = 0; d < 4; d++) out[p * 4 + d] /= s;
  }
  return out;
}
