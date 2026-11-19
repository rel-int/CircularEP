// app.js — entry point: boot the engine, build the world, wire events, run the frame loop.
import { createSim, vodeInfo } from './sim.js';
import { createWorld, redrawAllLabels, COLORS } from './world.js';
import { createStations } from './stations.js';
import { createFlows } from './flows.js';
import { createBoard } from './board.js';
import { createHud } from './hud.js';
import { createTour } from './tour.js';

window.__shidokuStarted = true;
const $ = (id) => document.getElementById(id);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobile = window.matchMedia('(max-width: 760px), (pointer: coarse)').matches;
const fmt = (x) => (x == null || !Number.isFinite(x) ? '–' : Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4 ? x.toExponential(1) : x.toPrecision(3));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function setLoading(msg) { const m = $('loading-msg'); if (m) m.textContent = msg; }
function fail(e) {
  console.error(e);
  const msg = e && e.stack ? `${e.message}\n\n${e.stack}` : String((e && e.message) || e);
  if (window.__shidokuFail) window.__shidokuFail(msg);
}
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

async function boot() {
  const sim = await createSim({ mobile, onProgress: setLoading });
  setLoading('Building the ring…');
  await nextFrame();
  const world = createWorld($('stage'), { reduced, mobile });
  const board = createBoard(world, sim);
  const stations = createStations(world, sim);
  world.scene.add(stations.group);
  const flows = createFlows(world, sim, stations, board);
  const ctx = { sim, world, stations, flows, board };
  ctx.onHoverCell = (p) => { board.setHover(p); stations.setHighlightBlock(p >= 0 ? p : -1); };
  const hud = createHud(ctx);
  const tour = createTour({ sim, world, hud, stations });
  hud.setTour(tour);
  world.snap('overview');                 // re-frame now that the HUD has reported the screen area it covers

  // ---------------------------------------------------------------- deferred heavy work: rho, self-test
  let rhoTimer = 0;
  function scheduleRho(delay) {
    clearTimeout(rhoTimer);
    hud.setRho(null);
    rhoTimer = setTimeout(() => { hud.setRho(sim.computeRho()); }, delay);
  }
  sim.on('puzzle', (p) => {
    flows.setPuzzle();
    flows.setPhase();
    scheduleRho(900);
  });
  sim.on('phase', () => flows.setPhase());
  sim.on('reset', () => flows.setPhase());
  sim.on('ep', (ep) => flows.setEP(ep));
  sim.on('converged', () => {
    if (tour.active || world.isRiding()) return;
    hud.toast(sim.stats.valid ? 'Equilibrium reached: E ≈ 0, every unit equals its prediction.'
      : 'Equilibrium reached (E ≈ 0), but on a wrong board: a wrong fixed point.', 3200);
  });

  // ---------------------------------------------------------------- picking: tooltips, hover highlight, clicks
  const canvas = world.renderer.domElement;
  let pointer = null, down = null, pickPending = false;
  function pickables() {
    return [...board.cells, ...stations.hitMeshes(), ...flows.pickables];
  }
  function describe(hit) {
    const o = hit.object, ud = o.userData || {};
    if (ud.kind === 'cell') {
      const p = ud.cell, v = sim.view;
      const clue = sim.puzzle[p];
      const d = v.digits[p] + 1, conf = v.conf[p];
      const probs = [0, 1, 2, 3].map((k) => v.probs[p * 4 + k].toFixed(2)).join(' · ');
      return { kind: 'cell', id: p, html: `<b>Cell ${(p >> 2) + 1},${(p & 3) + 1}</b> ${clue ? `<span class="violet">clue ${clue}</span>` : '<span class="muted">blank</span>'}<span class="f">readout ${d} (p = ${conf.toFixed(2)})</span><span class="f" style="color: var(--muted)">p(1…4) = ${probs}</span><span class="muted">click to cycle the clue</span>` };
    }
    if (ud.kind === 'layer') {
      const L = stations.layerInfo(ud.layer);
      const vi = vodeInfo(L.vode);
      const part = sim.to && sim.to.parts ? sim.to.parts[L.vode] : null;
      if (L.live || world.getFold() < 0.5) {
        return { kind: 'layer', id: ud.layer, html: `<b>${esc(vi.title)}</b> · ${esc(vi.name)}<span class="f">${esc(vi.formula)}</span><span class="muted">${esc(vi.shapeTxt)} · ½‖${vi.title} − ${vi.title}̂‖² = ${fmt(part)}</span>` };
      }
      return { kind: 'layer', id: ud.layer, html: L.computed
        ? `<b>${esc(vi.title)}<sub>${L.k}</sub></b> · ${esc(vi.title)} after sweep ${L.k}<span class="f">${L.vode === 'u' ? `u_${L.k} = û(s_${L.k - 1})` : L.k === 0 ? 's_0 = 0 (start)' : `s_${L.k} = ŝ(u_${L.k})`}</span>`
        : `<b>${esc(vi.title)}</b> · not computed yet<span class="muted">the next sweeps will fill this layer</span>` };
    }
    if (ud.kind === 'pedestal') {
      return { kind: 'pedestal', id: 0, html: '<b>Input x</b> · 16 cells × 5 one-hot = 80 numbers<span class="f">x_inj = x·Win + bin → injected into ŝ</span><span class="muted">click to look along the beam</span>' };
    }
    return null;
  }
  function doPick() {
    pickPending = false;
    if (!pointer || world.isRiding()) return;
    const hits = world.pick(pointer.x, pointer.y, pickables());
    const d = hits.length ? describe(hits[0]) : null;
    if (d) {
      hud.showTooltip(d.html, pointer.x, pointer.y);
      canvas.classList.add('pointing');
      ctx.onHoverCell(d.kind === 'cell' ? d.id : -1);
    } else {
      hud.hideTooltip();
      canvas.classList.remove('pointing');
      ctx.onHoverCell(-1);
    }
    return d;
  }
  canvas.addEventListener('pointermove', (e) => {
    pointer = { x: e.clientX, y: e.clientY };
    if (e.pointerType === 'touch') return;
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) { hud.hideTooltip(); return; }
    pickPending = true;
  });
  canvas.addEventListener('pointerleave', () => { pointer = null; hud.hideTooltip(); ctx.onHoverCell(-1); canvas.classList.remove('pointing'); });
  canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  canvas.addEventListener('pointerup', (e) => {
    const d0 = down;
    down = null;
    if (!d0 || Math.hypot(e.clientX - d0.x, e.clientY - d0.y) > 6 || performance.now() - d0.t > 450) return;
    const hits = world.pick(e.clientX, e.clientY, pickables());
    if (!hits.length) return;
    const ud = hits[0].object.userData || {};
    if (ud.kind === 'cell') {
      sim.cycleClue(ud.cell);
      sim.play();
      const v = sim.puzzle[ud.cell];
      hud.toast(v ? `Cell ${(ud.cell >> 2) + 1},${(ud.cell & 3) + 1} is now a clue: ${v}. The ring re-settles from z = 0.` : `Cell ${(ud.cell >> 2) + 1},${(ud.cell & 3) + 1} is now blank.`, 2600);
      if (e.pointerType !== 'touch') { pointer = { x: e.clientX, y: e.clientY }; pickPending = true; }
    } else if (ud.kind === 'layer') {
      const L = stations.layerInfo(ud.layer);
      const vi = vodeInfo(L.vode);
      world.flyTo(world.layerPose(ud.layer), 1.4);
      const part = sim.to && sim.to.parts ? sim.to.parts[L.vode] : null;
      const sub = (n) => String(n).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c] || c).join('');
      const title = world.getFold() > 0.5 && L.computed ? `${vi.title}${sub(L.k)} · ${vi.name}` : `${vi.title} · ${vi.name}`;
      hud.showCard(title, `<p class="f">${esc(vi.formula)}</p><p class="note">${esc(vi.shapeTxt)}. ${esc(vi.role)}</p>
        <p class="note">Energy term now: ½‖${vi.title} − ${vi.title}̂‖² = <span class="rose mono">${fmt(part)}</span></p>
        <p class="note">Layout: units are grouped into a 4 × 4 arrangement of blocks${L.vode === 's' ? ' by the board cell their readout weights V favour most' : ' by the s block they feed most strongly through W2'}. Hover a board cell to light its block.</p>`);
      hud.hideTooltip();
    } else if (ud.kind === 'pedestal') {
      world.preset('injection');
      hud.showCard('Input x and the beam x_inj', '<p class="f">x_inj = x·Win + bin</p><p class="note">The puzzle as 16 cells × 5 one-hot slots (blank, 1, 2, 3, 4) = 80 inputs. Win maps it to 256 numbers that are added inside the prediction of s on every sweep: ŝ = tanh(u·W2 + b2 + x_inj). The beam is densest toward the s units with the largest |x_inj|.</p>');
    }
  });

  // ---------------------------------------------------------------- fonts: redraw canvas text once the web fonts are in
  if (document.fonts && document.fonts.load) {
    Promise.all([
      document.fonts.load('600 64px "Fraunces"'),
      document.fonts.load('500 24px "Instrument Sans"'),
      document.fonts.load('500 24px "JetBrains Mono"'),
    ]).then(() => { redrawAllLabels(); board.redraw(); flows.redraw(); }).catch(() => {});
  }

  // ---------------------------------------------------------------- start: the free phase relaxing, the board sharpening
  // sim.speed is already DEFAULT_SPEED here (set by createSim() -> loadPuzzle()), so the speed is not overridden.
  hud.syncAll();
  sim.play();
  hud.syncAll();
  scheduleRho(1200);
  // the self-test replays the JAX references (light mode, ~0.1 s): run it shortly after start
  setTimeout(async () => { hud.setSelftest(await sim.runSelftest()); }, 2500);

  let last = performance.now(), errors = 0;
  function frame(now) {
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    try {
      sim.frame(dt);
      world.update(dt);
      stations.update(dt);
      flows.update(dt);
      board.update(dt, now / 1000);
      if (pickPending) doPick();
      hud.update(dt);
      world.render();
    } catch (e) {
      if (errors++ < 5) console.error('[frame]', e);
      if (errors === 5) hud.toast('Rendering error: see the browser console.', 6000);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  window.__shidokuBooted = true;
  const ld = $('loading');
  ld.classList.add('done');
  setTimeout(() => { ld.hidden = true; }, 800);

  let toured = false;
  try { toured = localStorage.getItem('shidoku-ring-toured') === '1'; } catch (e) { toured = false; }
  if (!toured) setTimeout(() => hud.toast('New here? Press Tour for a guided walk through the ring.', 5200), 2200);

  // expose for debugging from the console (no globals are required by the app)
  window.shidoku = { sim, world, stations, flows, board, hud, tour, COLORS };
}

boot().catch(fail);
