// hud.js — the DOM overlay: transport, readouts, fold slider, phases, charts, puzzle picker + clue editor, solver and
// EP parameters, heatmaps, view toggles, About (metrics + self-test badge), captions, tooltips, keyboard shortcuts.
import { SPEEDS, VODES, vodeInfo } from './sim.js';
import { COLORS } from './world.js';

const $ = (id) => document.getElementById(id);
const fmt = (x) => {
  if (x == null || !Number.isFinite(x)) return '–';
  const a = Math.abs(x);
  if (a === 0) return '0';
  if (a < 1e-3 || a >= 1e4) return x.toExponential(1).replace('e', 'e');
  return x.toPrecision(3);
};
const log10 = (x) => Math.log10(Math.max(x, 1e-30));
const isTyping = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.isContentEditable);

export function createHud(ctx) {
  const { sim, world, stations, flows } = ctx;
  let tour = null;
  const mqPhone = window.matchMedia('(max-width: 760px)');
  const reduced = world.state.reduced;
  // per-frame dirty flags (declared first: handlers below set them)
  let chartDirty = true, chartT = 0, lastHistLen = -1;
  let captionDirty = true, lastCaption = '';
  let roT = 0;

  // ---------------------------------------------------------------- toast
  let toastTimer = 0;
  function toast(msg, ms = 3800) {
    const t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, ms);
  }
  sim.on('notice', (m) => toast(m, 5000));

  // ---------------------------------------------------------------- tabs
  const tabIds = ['run', 'puzzle', 'learn', 'view', 'about'];
  function selectTab(name, focus = false) {
    for (const t of tabIds) {
      const b = $(`tabbtn-${t}`), p = $(`tab-${t}`);
      const on = t === name;
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
      p.hidden = !on;
      if (on && focus) b.focus();
    }
    if (name === 'run') chartDirty = true;
    if (mqPhone.matches && $('panel').classList.contains('collapsed')) setSheet(true);
  }
  for (const t of tabIds) $(`tabbtn-${t}`).addEventListener('click', () => selectTab(t));
  $('tabs').addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const cur = tabIds.findIndex((t) => $(`tabbtn-${t}`).getAttribute('aria-selected') === 'true');
    const nxt = (cur + (e.key === 'ArrowRight' ? 1 : -1) + tabIds.length) % tabIds.length;
    selectTab(tabIds[nxt], true);
    e.preventDefault();
  });
  selectTab('run');

  // ---------------------------------------------------------------- bottom sheet + safe area for the 3-D view
  function setSheet(open) {
    $('panel').classList.toggle('collapsed', !open);
    $('sheet-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
    updateInsets();
  }
  $('sheet-toggle').addEventListener('click', () => setSheet($('panel').classList.contains('collapsed')));
  function updateInsets() {
    const panel = $('panel');
    // the 3-D view is framed inside the screen area the HUD leaves free (fixed bands, so captions can change freely)
    if (mqPhone.matches) world.setInsets({ left: 0, right: 0, top: 236, bottom: panel.offsetHeight + 14 });
    else world.setInsets({ left: 0, right: panel.offsetWidth + 32, top: 24, bottom: 118 });
  }
  if ('ResizeObserver' in window) new ResizeObserver(updateInsets).observe($('panel'));
  mqPhone.addEventListener ? mqPhone.addEventListener('change', () => { setSheet(!mqPhone.matches); }) : null;
  if (mqPhone.matches) setSheet(false); else updateInsets();

  // ---------------------------------------------------------------- transport
  function syncPlay() {
    const b = $('btn-play');
    b.dataset.state = sim.playing ? 'playing' : 'paused';
    b.setAttribute('aria-label', sim.playing ? 'Pause' : 'Play');
  }
  sim.on('play', syncPlay);
  $('btn-play').addEventListener('click', () => { sim.toggle(); syncPlay(); });
  $('btn-step').addEventListener('click', () => sim.stepManual(1));
  $('btn-reset').addEventListener('click', () => { sim.setPhase('free'); sim.play(); });
  const speedIdx = (v) => { let best = 0; SPEEDS.forEach((s, i) => { if (Math.abs(s - v) < Math.abs(SPEEDS[best] - v)) best = i; }); return best; };
  function syncSpeed() {
    const i = speedIdx(sim.speed);
    $('speed').value = String(i);
    $('speed-label').textContent = `${SPEEDS[i]} it/s`;
  }
  $('speed').addEventListener('input', (e) => { sim.setSpeed(SPEEDS[+e.target.value]); syncSpeed(); });
  sim.on('params', syncSpeed);

  // ---------------------------------------------------------------- fold
  function syncFold(f) {
    $('fold').value = String(f);
    const unrolled = world.state.foldTarget > 0.5;
    $('btn-unfold').setAttribute('aria-pressed', unrolled ? 'true' : 'false');
    $('btn-unfold').textContent = unrolled ? 'Fold' : 'Unfold';
  }
  world.onFold(syncFold);
  $('fold').addEventListener('input', (e) => world.setFoldValue(+e.target.value));
  function toggleUnfold() {
    const to = world.state.foldTarget > 0.5 ? 0 : 1;
    world.animateFoldTo(to);
    syncFold(world.getFold());
    world.preset('overview');
  }
  $('btn-unfold').addEventListener('click', toggleUnfold);

  // ---------------------------------------------------------------- phases
  function syncPhase() {
    const map = { free: 'phase-free', 'nudge+': 'phase-plus', ep: 'phase-ep' };
    for (const [ph, id] of Object.entries(map)) $(id).setAttribute('aria-checked', sim.phase === ph ? 'true' : 'false');
    syncPlay();
    captionDirty = true;
  }
  sim.on('phase', syncPhase);
  sim.on('reset', syncPhase);
  function startNudge() {
    // the nudge phase plays at exactly the same it/s as the free phase, and stops on its own convergence rather
    // than always running the full step cap (see sim.js's tick loop).
    if (!sim.setPhase('nudge+')) return;
    sim.play();
  }
  $('phase-free').addEventListener('click', () => { sim.setPhase('free'); sim.play(); });
  $('phase-plus').addEventListener('click', () => startNudge());
  $('phase-ep').addEventListener('click', () => {
    toast('Computing the EP signal (the missing nudged relaxations run in the background)…', 2500);
    setTimeout(() => { if (sim.setPhase('ep')) selectTab('learn'); }, 30);
  });

  // ---------------------------------------------------------------- readouts + energy parts
  function updateReadouts() {
    const T = sim.to || {};
    const nudging = sim.phase === 'nudge+';
    $('ro-iter').textContent = nudging ? `${sim.phaseIter}/${sim.params.tNudge}` : String(sim.iter);
    $('ro-iter').previousElementSibling.textContent = nudging ? 'nudge step' : 'iteration';
    $('ro-energy').textContent = fmt(T.E);
    $('ro-res').textContent = fmt(T.residual);
    const st = sim.stats;
    const acc = $('ro-acc');
    if (st.correct >= 0) acc.textContent = `${st.correct}/16 ${st.valid ? '✓' : ''}`;
    else acc.textContent = st.valid ? 'valid ✓' : 'not valid';
    acc.style.color = st.valid ? COLORS.green : '';
    const pu = (T.parts && T.parts.u) || 0, ps = (T.parts && T.parts.s) || 0;
    const mx = Math.max(pu, ps, 1e-30);
    $('part-u').style.width = `${(100 * pu) / mx}%`;
    $('part-s').style.width = `${(100 * ps) / mx}%`;
    $('part-u-val').textContent = fmt(pu);
    $('part-s-val').textContent = fmt(ps);
  }
  // correct/wrong vs. the no-unique-solution fallback consistent/conflict (see board.js)
  let legendKnown = null;
  function syncLegend() {
    const known = !!sim.solution;
    if (known === legendKnown) return;
    legendKnown = known;
    $('legend-correct').textContent = known ? 'correct' : 'consistent';
    $('legend-wrong').textContent = known ? 'wrong' : 'conflict';
  }
  sim.on('puzzle', syncLegend);

  // ---------------------------------------------------------------- chart (log10 E and log10 residual vs iteration)
  const chart = $('chart');
  function drawChart() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = chart.clientWidth, h = chart.clientHeight;
    if (!w || !h) return;
    if (chart.width !== Math.round(w * dpr) || chart.height !== Math.round(h * dpr)) { chart.width = Math.round(w * dpr); chart.height = Math.round(h * dpr); }
    const g = chart.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const Hh = sim.hist;
    const n = Hh.iter.length;
    if (!n) return;
    const x0 = Hh.iter[0], x1 = Math.max(Hh.iter[n - 1], x0 + 20);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const a = log10(Hh.E[i]), b = log10(Hh.res[i]);
      if (Number.isFinite(a)) { lo = Math.min(lo, a); hi = Math.max(hi, a); }
      if (Number.isFinite(b) && Hh.res[i] > 0) { lo = Math.min(lo, b); hi = Math.max(hi, b); }
    }
    if (!Number.isFinite(lo)) { lo = -6; hi = 2; }
    lo = Math.floor(lo - 0.2); hi = Math.ceil(hi + 0.2);
    if (hi - lo < 3) hi = lo + 3;
    const L = 34, R = 8, Tp = 8, B = 20;
    const X = (it) => L + ((it - x0) / (x1 - x0)) * (w - L - R);
    const Y = (v) => Tp + (1 - (v - lo) / (hi - lo)) * (h - Tp - B);
    g.font = '10px "JetBrains Mono", ui-monospace, monospace';
    g.fillStyle = 'rgba(139,152,165,0.85)';
    g.strokeStyle = 'rgba(139,152,165,0.14)';
    g.lineWidth = 1;
    const step = Math.max(1, Math.ceil((hi - lo) / 6));
    for (let v = Math.ceil(lo); v <= hi; v += step) {
      const y = Y(v);
      g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle';
      g.fillText(`1e${v}`, L - 4, y);
    }
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    g.fillText('0', L, h - 6);
    g.textAlign = 'right';
    g.fillText(`iteration ${Hh.iter[n - 1]}`, w - R, h - 6);
    // phase markers
    g.setLineDash([3, 3]);
    g.strokeStyle = 'rgba(167,139,250,0.65)';
    g.fillStyle = 'rgba(167,139,250,0.9)';
    g.textAlign = 'left';
    for (const m of Hh.markers) {
      if (m.iter < x0) continue;
      const x = X(m.iter);
      g.beginPath(); g.moveTo(x, Tp); g.lineTo(x, h - B); g.stroke();
      g.fillText(m.label, x + 3, Tp + 10);
    }
    g.setLineDash([]);
    const series = (arr, color, width) => {
      g.strokeStyle = color; g.lineWidth = width; g.beginPath();
      let started = false;
      const stride = Math.max(1, Math.floor(n / ((w - L - R) * 1.5)));
      for (let i = 0; i < n; i += stride) {
        const v = arr[i];
        if (!(v > 0) || !Number.isFinite(v)) { started = false; continue; }
        const x = X(Hh.iter[i]), y = Y(log10(v));
        if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
      }
      g.stroke();
    };
    series(Hh.res, 'rgba(139,152,165,0.95)', 1.25);
    series(Hh.E, COLORS.rose, 1.8);
  }

  // ---------------------------------------------------------------- puzzles: the 10-board showcase
  const STATUS_MARK = { 2: '✓', 1: '✗', 0: '…' };
  const STATUS_COLOR = { 2: 'var(--green)', 1: 'var(--rose)', 0: 'var(--muted)' };
  const GROUPS = [['fast', 'Fast'], ['medium', 'Medium'], ['slow', 'Slow'], ['unsolved', 'Unsolved · teach it']];
  let fittedIds = new Set();   // showcase ids fitted by Learn step this session; cleared on reset/model switch
  function fillFilter() {
    // test-only: sim.puzzles is the full bundled pool (240 source:"test" + 60 source:"generated"); counting the
    // generated split here would inflate both N and "solved" while the label says "test puzzles".
    const testPool = sim.puzzles.filter((p) => p.source === 'test');
    const n = testPool.length;
    const solved = testPool.filter((p) => p.status === 2).length;
    $('puzzle-pool-note').textContent = `${solved} of ${n} test puzzles are solved by this model's free phase (η = ${fmt(sim.params.lr)}${sim.params.mom ? `, μ = ${fmt(sim.params.mom)}` : ''}). The 10 below are curated: 3 fast, 2 medium, 3 slow (solved), 2 unsolved (teach it with Learn).`;
  }
  function renderShowcase() {
    const root = $('showcase');
    root.innerHTML = '';
    for (const [cat, label] of GROUPS) {
      const entries = sim.showcase.map((e, i) => [e, i]).filter(([e]) => e.category === cat);
      if (!entries.length) continue;
      const group = document.createElement('div'); group.className = 'sc-group';
      const h = document.createElement('div'); h.className = 'sc-label'; h.textContent = label; group.appendChild(h);
      const row = document.createElement('div'); row.className = 'sc-row'; group.appendChild(row);
      for (const [e, i] of entries) {
        const b = document.createElement('button'); b.type = 'button'; b.className = 'sc-chip';
        b.dataset.index = String(i);
        const fitted = fittedIds.has(e.id);
        const dot = document.createElement('span'); dot.className = 'dot';
        dot.style.background = fitted ? 'var(--amber)' : (STATUS_COLOR[e.status] || 'var(--muted)');
        const txt = document.createElement('span');
        txt.textContent = fitted ? `✓ ${e.id} (fitted)` : `${STATUS_MARK[e.status] || '?'} ${e.id}`;
        b.title = e.category === 'unsolved'
          ? `${e.cells_correct}/16 cells, ~${e.presses_to_fix ?? '?'} Learn presses to fix`
          : `${e.time_to_green_hb ?? '?'} it to all-green (η/μ mode), ${e.time_to_green_plain ?? '?'} it (plain) — E < tol at ${e.conv_iters_hb ?? '?'} / ${e.conv_iters_plain ?? '?'} it`;
        b.append(dot, txt);
        b.addEventListener('click', () => { sim.loadShowcaseEntry(i); syncShowcaseSelection(); syncLearn(); });
        row.appendChild(b);
      }
      root.appendChild(group);
    }
    syncShowcaseSelection();
  }
  function syncShowcaseSelection() {
    const cur = sim.current;
    const idx = cur && !cur.custom ? sim.showcase.findIndex((e) => e.id === cur.id) : -1;
    for (const b of $('showcase').querySelectorAll('.sc-chip')) b.classList.toggle('on', Number(b.dataset.index) === idx);
    // highlight Learn step when an "unsolved" showcase board is selected (the teach-it playground)
    const isUnsolvedPick = idx >= 0 && sim.showcase[idx].category === 'unsolved';
    $('phase-learn').classList.toggle('highlight', isUnsolvedPick);
  }
  $('puzzle-clear').addEventListener('click', () => { sim.clearClues(); sim.play(); });
  $('puzzle-restore').addEventListener('click', () => { sim.restorePuzzle(); sim.play(); });

  const clueGrid = $('clue-grid');
  const clueBtns = [];
  for (let p = 0; p < 16; p++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.id = `clue-${p}`;
    b.addEventListener('click', () => { sim.cycleClue(p); sim.play(); });
    b.addEventListener('keydown', (e) => {
      if (/^[0-4]$/.test(e.key)) { sim.setClue(p, +e.key); sim.play(); e.preventDefault(); }
      else if (e.key === 'Backspace' || e.key === 'Delete') { sim.setClue(p, 0); sim.play(); e.preventDefault(); }
    });
    b.addEventListener('focus', () => ctx.onHoverCell && ctx.onHoverCell(p));
    b.addEventListener('blur', () => ctx.onHoverCell && ctx.onHoverCell(-1));
    clueGrid.appendChild(b);
    clueBtns.push(b);
  }
  function syncPuzzle() {
    const puz = sim.puzzle;
    for (let p = 0; p < 16; p++) {
      const v = puz[p];
      clueBtns[p].textContent = v ? String(v) : '·';
      clueBtns[p].classList.toggle('clue', !!v);
      clueBtns[p].setAttribute('aria-label', `row ${(p >> 2) + 1}, column ${(p & 3) + 1}: ${v ? 'clue ' + v : 'blank'}`);
    }
    const n = sim.solutions.length;
    const cur = sim.current || {};
    let s = cur.custom ? 'Edited clue set. ' : `${cur.source === 'generated' ? 'Generated' : 'Held-out test'} puzzle ${cur.id}. `;
    if (n === 0) s += 'These clues contradict each other: no valid board exists.';
    else if (n === 1) s += 'Unique solution (accuracy is measured against it).';
    else s += `${n} valid boards fit these clues; the nudge pulls toward the one closest to the current readout.`;
    if (cur.custom) s += ' This edited board has not been checked against the solver.';
    else if (cur.status === 2) s += ` Gradient descent on E solves it (converged by iteration ${cur.iters ?? '?'}).`;
    else if (cur.status === 1) s += ` Gradient descent on E converges but to a wrong board (${cur.cells ?? '?'}/16 cells, iteration ${cur.iters ?? '?'}).`;
    else if (cur.status === 0) s += ' Gradient descent on E stalls on this puzzle (a local minimum of E, E ≥ tol after the step budget) — a good Learn-step target.';
    $('puzzle-status').textContent = s;
    syncShowcaseSelection();
    syncLearnGate();
  }

  // ---------------------------------------------------------------- nudging & learning (beta, lr)
  // the nudge phase has no eta/steps sliders of its own (it always matches the free phase's eta + momentum
  // toggle; T_nudge is a fixed cap read from the model's own meta, not adjustable), and there is no
  // centred (±β) toggle (one-sided nudging only -- see index.html's Learn-tab note on training vs. the app).
  function syncParams() {
    const P = sim.params;
    $('param-beta').value = P.beta; $('param-beta-val').textContent = P.beta.toFixed(3);
    $('param-lr').value = P.learnLr; $('param-lr-val').textContent = fmt(P.learnLr);
    $('tog-mom').checked = sim.mode === 'on';
    captionDirty = true;
  }
  const bindParam = (id, key, parse = parseFloat) => {
    const el = $(id), out = $(`${id}-val`);
    el.addEventListener('input', () => { const v = parse(el.value); out.textContent = fmt(v); });
    el.addEventListener('change', () => { sim.setParams({ [key]: parse(el.value) }); });
  };
  bindParam('param-beta', 'beta');
  bindParam('param-lr', 'learnLr');
  sim.on('params', syncParams);
  $('tog-mom').addEventListener('change', (e) => sim.setMode(e.target.checked ? 'on' : 'off'));
  $('btn-ep-cycle').addEventListener('click', () => {
    sim.runEpCycle();
    toast('EP cycle: free phase → nudge +β → EP signal → weight update (lr) → free phase again.');
  });

  // ---------------------------------------------------------------- model picker
  // The app ships with only the trained model (sim.js ENABLED_MODELS = ['best']), so sim.modelsIndex.models
  // always has length 1 here: hide the dropdown + blurb row entirely rather than show a single-option picker.
  // To show more models, list them in models/index.json and add their ids to ENABLED_MODELS in sim.js; this row
  // will reappear on its own.
  function fillModelSelect() {
    const sel = $('model-select');
    const row = sel.closest('.row') || sel.parentElement;
    const models = sim.modelsIndex.models || [];
    if (models.length <= 1) {
      if (row) row.style.display = 'none';
      const blurbEl = $('model-blurb');
      if (blurbEl) blurbEl.style.display = 'none';
    } else if (row) { row.style.display = ''; }
    sel.innerHTML = '';
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.id;
      const acc = m.test_board != null ? ` · ${(100 * m.test_board).toFixed(0)}% of test boards` : '';
      o.textContent = `${m.label || m.id}${acc}`;
      sel.appendChild(o);
    }
    sel.value = sim.modelId;
    $('model-blurb').textContent = (sim.modelEntry && sim.modelEntry.blurb) || '';
  }
  function refreshForNewModel() {
    fittedIds = new Set();
    fillModelSelect();
    fillFilter();
    renderShowcase();
    syncParams();
    fillAbout();
    syncLearn();
  }
  $('model-select').addEventListener('change', async (e) => {
    toast('Loading weights…', 2000);
    await sim.loadModelById(e.target.value);
    refreshForNewModel();
  });
  sim.on('model', refreshForNewModel);

  // ---------------------------------------------------------------- Learn
  // The single-step action is the phase row's "Learn" button, after "EP signal" (see index.html); the Learn tab
  // has the animated "Run a full EP cycle" (free -> nudge -> EP -> weight update -> free again) and the
  // "x5 (fast-forward)" bulk path, which reruns free+nudge from scratch every call rather than reusing a shown
  // EP signal.
  function syncLearnGate() {
    const ok = sim.hasUniqueSolution;
    $('phase-learn').disabled = !ok;
    $('btn-learn-x5').disabled = !ok;
    if (!ok) $('learn-status').textContent = sim.solutions.length === 0 ? 'No Learn: these clues contradict each other (no solution).' : 'No Learn: this clue set has more than one valid solution (no unambiguous target).';
  }
  function syncLearn() {
    $('learn-updates').textContent = `updates applied: ${sim.learnUpdates}`;
    const tbl = $('learn-history');
    tbl.innerHTML = '';
    const head = document.createElement('tr');
    for (const h of ['step', 'E', 'cells', 'solved']) { const th = document.createElement('th'); th.textContent = h; th.style.textAlign = 'left'; th.style.color = 'var(--muted)'; head.appendChild(th); }
    tbl.appendChild(head);
    for (const row of sim.learnHistory.slice(-8)) {
      const tr = document.createElement('tr');
      for (const v of [row.n, fmt(row.E), `${row.cells}/16`, row.solved ? '✓' : '']) { const td = document.createElement('td'); td.textContent = v; tbl.appendChild(tr); tr.appendChild(td); }
    }
    syncLearnGate();
    syncShowcaseSelection();
  }
  function markFittedIfSolved(last) {
    if (!last || !last.solved || !sim.current || sim.current.custom) return;
    if (sim.showcase.some((e) => e.id === sim.current.id)) { fittedIds.add(sim.current.id); renderShowcase(); }
  }
  function doPhaseLearn() {
    if (!sim.hasUniqueSolution) { syncLearnGate(); return; }
    $('learn-status').textContent = 'Applying the EP signal (computing it first if missing) → weight update → re-run…';
    const ok = sim.applyEPUpdate();
    syncLearn();
    const last = sim.learnHistory[sim.learnHistory.length - 1];
    markFittedIfSolved(last);
    if (ok && last && last.solved) $('learn-status').textContent = `fitted in ${sim.learnUpdates} update${sim.learnUpdates === 1 ? '' : 's'} ✓ — all green. These updates fit this board and may hurt others.`;
    else $('learn-status').textContent = ok ? `Learn update ${sim.learnUpdates} applied. These updates fit this board and may hurt others.` : 'Learn could not run (see the toast).';
  }
  $('phase-learn').addEventListener('click', doPhaseLearn);
  $('btn-learn-x5').addEventListener('click', () => {
    if (!sim.hasUniqueSolution) { syncLearnGate(); return; }
    let last = null;
    for (let i = 0; i < 5; i++) { sim.learnStep(); last = sim.learnHistory[sim.learnHistory.length - 1]; if (last && last.solved) break; }
    syncLearn();
    markFittedIfSolved(last);
    $('learn-status').textContent = last && last.solved ? `fitted in ${sim.learnUpdates} update${sim.learnUpdates === 1 ? '' : 's'} ✓ — all green.` : `${sim.learnUpdates} updates applied.`;
  });
  $('btn-learn-reset').addEventListener('click', () => {
    sim.resetWeights();
    fittedIds = new Set();
    $('learn-status').textContent = 'Weights reset.';
    renderShowcase();
    syncLearn();
  });
  sim.on('learn', syncLearn);

  // ---------------------------------------------------------------- EP heatmaps
  function drawHeatmap(canvas, M) {
    const [r, c] = M.shape;
    canvas.width = c; canvas.height = r;
    const g = canvas.getContext('2d');
    const img = g.createImageData(c, r);
    const abs = Float32Array.from(M.data, Math.abs).sort();
    const vmax = Math.max(abs[Math.floor(abs.length * 0.995)] || 0, 1e-30);
    const amber = [242, 180, 65], steel = [111, 143, 207], bg = [10, 14, 21];
    for (let i = 0; i < r * c; i++) {
      const v = M.data[i] / vmax;
      const m = Math.min(1, Math.abs(v));
      const col = v >= 0 ? amber : steel;
      const k = Math.pow(m, 0.6);
      img.data[i * 4] = bg[0] + (col[0] - bg[0]) * k;
      img.data[i * 4 + 1] = bg[1] + (col[1] - bg[1]) * k;
      img.data[i * 4 + 2] = bg[2] + (col[2] - bg[2]) * k;
      img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return vmax;
  }
  sim.on('ep', (ep) => {
    for (const k of ['W1', 'W2']) {
      const M = ep.mats[k];
      if (!M) continue;
      drawHeatmap($(`hm-${k}`), M);
      const arrow = k === 'W1' ? 's → u' : 'u → s';
      $(`hm-${k}-cap`).textContent = `Δ${k} (${arrow}) ${M.shape[0]}×${M.shape[1]} · ‖·‖F ${fmt(M.fro)} · max ${fmt(M.max)}`;
    }
    const momTxt = sim.mode === 'on' ? 'heavy-ball' : 'plain-GD';
    $('ep-status').textContent = `One-sided EP estimate of ∂C/∂W at β = ${ep.beta}, from the nudged equilibrium after up to ${ep.T} ${momTxt} steps (same η/momentum as the free phase). Rows = presynaptic unit, columns = postsynaptic unit; amber +, steel −. In the 3-D view the strongest synapses glow amber in proportion to |ΔW|.`;
  });

  // ---------------------------------------------------------------- view toggles
  const toggles = {
    'tog-pred': (on) => stations.setShowPredictions(on),
    'tog-err': (on) => stations.setShowErrors(on),
    'tog-synapses': (on) => flows.setOptions({ synapses: on }),
    'tog-particles': (on) => flows.setOptions({ particles: on }),
    'tog-beam': (on) => flows.setOptions({ beam: on }),
    'tog-labels': (on) => world.setLabels(on),
    'tog-bloom': (on) => world.setBloom(on, true),
    'tog-rotate': (on) => world.setAutoRotate(on),
  };
  for (const [id, fn] of Object.entries(toggles)) $(id).addEventListener('change', (e) => fn(e.target.checked));
  if (reduced) { $('tog-rotate').checked = false; world.setAutoRotate(false); }
  world.onQuality((q) => {
    $('tog-bloom').checked = q.bloom;
    $('quality-status').textContent = q.note ? `Adaptive quality: ${q.note}.` : '';
  });

  // ---------------------------------------------------------------- camera buttons
  $('cam-overview').addEventListener('click', () => world.preset('overview'));
  $('cam-board').addEventListener('click', () => world.preset('board'));
  $('cam-ride').addEventListener('click', () => (world.isRiding() ? world.stopRide() : world.preset('ride')));
  $('btn-tour').addEventListener('click', () => tour && (tour.active ? tour.stop() : tour.start()));

  // ---------------------------------------------------------------- About
  function fillAbout() {
    const m = sim.meta || {};
    const mt = m.metrics || {};
    const tm = mt.test || {};
    const tr = m.train || {};
    const rows = [];
    rows.push(['model', `${m.label || m.id || sim.modelId} — ${m.blurb || ''}`]);
    // one-line summary, e.g. "EP centred with GD+momentum on E, test boards 87.7%" -- values read live from this
    // model's own meta, never hardcoded (this app ships only the trained checkpoint: see ENABLED_MODELS in sim.js).
    if ((tr.method || m.inference) && tm.board != null) {
      rows.push(['summary', `${tr.method || 'EP-trained'} with ${m.inference && m.inference.mom ? 'GD+momentum' : 'GD'} on E, test boards ${(100 * tm.board).toFixed(1)}%`]);
    }
    if (tm.board != null) rows.push(['board accuracy (test)', `${(+tm.board).toFixed(4)} on ${tm.n || sim.puzzles.length} puzzles`]);
    if (tm.cell != null) rows.push(['cell accuracy (test)', (+tm.cell).toFixed(4)]);
    if (tm.solved != null) rows.push(['solved (test)', (+tm.solved).toFixed(4)]);
    if (tm.conv_frac != null) rows.push(['converged (test)', (+tm.conv_frac).toFixed(4)]);
    if (mt.rho_J && mt.rho_J.median != null) rows.push(['ρ(J) at z_free (median)', (+mt.rho_J.median).toFixed(3)]);
    if (tr.method) rows.push(['training rule', String(tr.method)]);
    if (tr.epoch != null) rows.push(['checkpoint epoch', String(tr.epoch)]);
    if (tr.cap != null) rows.push(['‖W1‖₂ = ‖W2‖₂ cap', String(tr.cap)]);
    if (m.inference) rows.push(['free-phase inference (training)', `${m.inference.init} init, η=${m.inference.eta}, μ=${m.inference.mom ?? 0}, T≤${m.inference.T_free}, tol=${m.inference.tol}`]);
    // training and the reported accuracy above used a feedforward ("ff") start, but this app always shows the
    // free phase starting from zero (sim.js APP_INIT) so the convergence is visible live; the two reach the
    // identical final board on all 240 test puzzles in both inference modes (zero-init vs ff-init agreement 100%).
    if (m.inference && m.inference.init && m.inference.init !== 'zero') {
      rows.push(['free-phase init (this app)', `zero (live, so you can watch it converge) — reaches the identical equilibrium as the ${m.inference.init} start used in training, on all 240 test boards (measured)`]);
    }
    rows.push(['layers', `${VODES.join(', ')} · ${sim.dims.H} units each · input ${sim.dims.din}`]);
    rows.push(['bundled puzzles', String(sim.puzzles.length)]);
    const dl = $('metrics');
    dl.innerHTML = '';
    for (const [k, v] of rows) {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      dl.append(dt, dd);
    }
  }
  function setSelftest(res) {
    const b = $('selftest-badge');
    const tbl = $('selftest-rows');
    tbl.innerHTML = '';
    if (!res) { b.className = 'badge wait'; b.textContent = 'self-test pending…'; return; }
    const count = res.total != null ? ` (${res.passed}/${res.total} checks${res.ms != null ? `, ${Math.round(res.ms)} ms` : ''})` : '';
    if (res.ok === true) { b.className = 'badge ok'; b.textContent = `✓ numerics verified vs JAX${count}`; }
    else if (res.ok === false) { b.className = 'badge bad'; b.textContent = `✗ numerics mismatch vs JAX${count}`; }
    else { b.className = 'badge wait'; b.textContent = 'self-test unavailable'; }
    const rows = Array.isArray(res.rows) ? res.rows : [];
    for (const r of rows.slice(0, 60)) {
      const tr = document.createElement('tr');
      let cells;
      if (typeof r === 'string') cells = [r];
      else if (Array.isArray(r)) cells = r.map(String);
      else {
        const rel = r.maxRel != null ? `rel ${fmt(r.maxRel)}${r.tol != null ? ` ≤ ${fmt(r.tol)}` : ''}` : r.maxAbs != null ? `max|Δ| ${fmt(r.maxAbs)}` : '';
        cells = [r.name ?? r.check ?? '', rel, r.info ? 'info' : r.ok === true ? '✓' : r.ok === false ? '✗' : ''];
        if (r.note) tr.title = r.note;
      }
      for (const c of cells) { const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); }
      tbl.appendChild(tr);
    }
    if (res.error) { const tr = document.createElement('tr'); const td = document.createElement('td'); td.textContent = res.error; tr.appendChild(td); tbl.appendChild(tr); }
  }

  // ---------------------------------------------------------------- captions
  function captionFor() {
    const ph = sim.phase;
    const unrolled = world.state.foldTarget > 0.5;
    const solvedTxt = sim.stats.valid ? (sim.greenAtIter != null ? ` All green at iteration ${sim.greenAtIter}.` : ' The board is complete and valid.') : '';
    let tag = 'Free phase', cls = 'tag', text = '';
    if (ph === 'free') {
      if (sim.converged) {
        tag = 'Equilibrium';
        const wrong = sim.stats.valid ? '' : sim.stats.correct >= 0
          ? ` But the readout is not the solution (${sim.stats.correct}/16 cells): the loop settled on a wrong fixed point, which is how the model fails on this puzzle.`
          : ' But the readout is not a valid board: the loop settled on a wrong fixed point.';
        text = `E ≈ 0 after ${sim.iter} iterations: every unit equals its prediction. This fixed point is where the unrolled feedforward network converges.${solvedTxt}${wrong}`;
      } else if (sim.stopReason === 'stalled') {
        cls = 'tag gd';
        tag = 'Stalled';
        text = `Gradient descent has come to rest at E = ${fmt(sim.to && sim.to.E)} > 0: a local minimum of E, ∇E ≈ 0 but E > 0 (residual ${fmt(sim.to && sim.to.residual)}), so the board is not the network's answer. Not every puzzle in the pool is solved by this model (see the Puzzle tab); try "Learn" (phase row) to push past it.`;
      } else if (sim.stopReason === 'budget') {
        tag = 'Not converged';
        text = `No fixed point within ${sim.maxFreeIters} iterations (E = ${fmt(sim.to && sim.to.E)}, this model's own step budget). This puzzle is in the "stalled" group for this model/mode; try "Learn" (phase row) or switch puzzle.${solvedTxt}`;
      } else {
        cls = 'tag gd';
        const momTxt = sim.mode === 'on' ? ' with heavy-ball momentum' : ', plain';
        const startTxt = sim.inference && sim.inference.init === 'zero' ? 'start at zero' : `start from ${sim.inference ? sim.inference.init : 'a'} init`;
        text = `The clues enter through x_inj; nodes ${startTxt}, then gradient descent on E${momTxt} fills the board in, moving each unit toward lowering its own error and the error of the unit it predicts, so errors (rose) flow backward as well as predictions forward: u ← u − η·∂E/∂u, s ← s − η·∂E/∂s.${solvedTxt}`;
      }
      if (unrolled) text += ' Unrolled: every step moves every copy a little, and the readout taps every s copy (dim teal, read-only) just as x_inj injects into every s copy.';
    } else if (ph === 'nudge+') {
      cls = 'tag nudge';
      tag = 'Nudge +β';
      text = `A weak pull β·C on the readout toward the label (violet), β = ${sim.params.beta}. Gradient descent on F = E + β·C at the same η/momentum as the free phase, started from the free equilibrium, stopping once it settles (cap ${sim.params.tNudge} steps); errors (rose) flow backward around the ring and the equilibrium shifts.${unrolled ? ' Unrolled: the nudge force −β·∂C/∂s pushes into every s copy, violet, just like x_inj.' : ''}`;
    } else if (ph === 'ep') {
      cls = 'tag ep';
      tag = 'EP signal';
      text = 'ΔW ∝ −[∂E/∂W(z^β) − ∂E/∂W(z⁰)] / β, the one-sided EP estimate (z⁰ = the free equilibrium, z^β = the +β nudged equilibrium; this app shows standard one-sided nudging only). For each synapse this is a local product of postsynaptic error and presynaptic activity. The strongest synapses glow amber; full heatmaps are in Learn.';
    }
    return { tag, cls, text };
  }
  function updateCaption() {
    const c = captionFor();
    const key = c.tag + '|' + c.text;
    if (key === lastCaption) return;
    lastCaption = key;
    $('caption-phase').textContent = c.tag;
    $('caption-phase').className = c.cls;
    $('caption-text').textContent = c.text;
  }
  sim.on('converged', () => { captionDirty = true; });
  sim.on('budget', () => { captionDirty = true; });

  // ---------------------------------------------------------------- tooltip + station card
  function showTooltip(html, x, y) {
    const t = $('tooltip');
    t.innerHTML = html;
    t.hidden = false;
    const r = t.getBoundingClientRect();
    const W = window.innerWidth, Hh = window.innerHeight;
    let left = x + 14, top = y + 14;
    if (left + r.width > W - 8) left = x - r.width - 14;
    if (top + r.height > Hh - 8) top = y - r.height - 14;
    t.style.left = `${Math.max(8, left)}px`;
    t.style.top = `${Math.max(8, top)}px`;
  }
  function hideTooltip() { $('tooltip').hidden = true; }
  function showCard(title, html) {
    $('station-card-title').textContent = title;
    $('station-card-body').innerHTML = html;
    $('station-card').hidden = false;
  }
  function hideCard() { $('station-card').hidden = true; }
  $('station-card-close').addEventListener('click', hideCard);

  // ---------------------------------------------------------------- keyboard
  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isTyping(e.target)) return;
    const onButton = e.target && e.target.tagName === 'BUTTON';
    switch (e.key) {
      case ' ':
        if (onButton) return;
        sim.toggle(); syncPlay(); e.preventDefault(); break;
      case 'ArrowRight':
        if (e.target && e.target.getAttribute && e.target.getAttribute('role') === 'tab') return;
        sim.stepManual(1); e.preventDefault(); break;
      case 'r': sim.setPhase('free'); sim.play(); break;
      case 'n': randomPuzzle(); break;
      case 'f': toggleUnfold(); break;
      case 't': if (tour) (tour.active ? tour.stop() : tour.start()); break;
      case 'o': world.preset('overview'); break;
      case 'b': world.preset('board'); break;
      case 'Escape':
        if (tour && tour.active) tour.stop();
        hideCard(); hideTooltip(); world.stopRide();
        break;
      default: break;
    }
  });

  // ---------------------------------------------------------------- per-frame
  function update(dt) {
    roT -= dt;
    if (roT <= 0) { roT = 0.1; updateReadouts(); captionDirty = true; }
    if (captionDirty) { captionDirty = false; updateCaption(); }
    chartT -= dt;
    const histLen = sim.hist.iter.length + sim.hist.markers.length * 100000;
    if (histLen !== lastHistLen) { lastHistLen = histLen; chartDirty = true; }
    if (chartDirty && chartT <= 0 && !$('tab-run').hidden) { chartDirty = false; chartT = 0.06; drawChart(); }
  }

  sim.on('puzzle', () => { syncPuzzle(); chartDirty = true; captionDirty = true; });
  sim.on('reset', () => { chartDirty = true; });
  window.addEventListener('resize', () => { chartDirty = true; });

  fillModelSelect();
  fillFilter();
  renderShowcase();
  syncPuzzle();
  syncParams();
  syncSpeed();
  syncPhase();
  syncFold(world.getFold());
  fillAbout();
  setSelftest(null);
  updateReadouts();
  syncLegend();
  syncLearn();

  return {
    update, toast, selectTab, setSheet, setSelftest, showTooltip, hideTooltip, showCard, hideCard,
    setTour(t) { tour = t; },
    refreshAbout: fillAbout,
    syncAll() { syncPuzzle(); syncParams(); syncSpeed(); syncPhase(); syncPlay(); },
    setRho(r) { $('rho-line').textContent = r != null && Number.isFinite(r) ? `Spectral radius of the loop's Jacobian at this equilibrium: ρ(J) = ${r.toFixed(3)} (each lap shrinks deviations by about this factor).` : ''; },
    info: (v) => vodeInfo(v),
  };
}
