// tour.js — a guided tour in 8 steps: short captions, camera moves, and real actions on the live model.
const $ = (id) => document.getElementById(id);

export function createTour({ sim, world, hud, stations }) {
  const restartFree = (speed) => { sim.setPhase('free'); if (speed) sim.setSpeed(speed); sim.play(); };
  const fold = (to) => { if (Math.abs(world.state.foldTarget - to) > 0.01) world.animateFoldTo(to); };

  const steps = [
    {
      title: 'A network folded into a loop',
      text: 'This is a trained network for 4×4 Sudoku. Two fully connected layers of 256 tanh units, s at the front and u at the back, are joined into a ring. Predictions travel around it (teal). The board floating above is read from s.',
      enter() { fold(0); world.preset('overview'); if (sim.phase !== 'free') restartFree(4); },
    },
    {
      title: 'Two layers, their predictions and errors',
      text: 'Each sphere is a unit: brightness is its value in [−1, 1], and it stands out of its panel by that value. The teal ring is the value its prediction asks for, û = tanh(s·W1 + b1) or ŝ = tanh(u·W2 + b2 + x_inj). The rose segment between them is the prediction error. Watch the gaps close as the free phase runs.',
      enter() { fold(0); restartFree(2); world.flyTo(world.layerPose(stations.liveLayers().s), 1.6); },
    },
    {
      title: 'The clues enter as a beam',
      text: 'The input x is the board as 16 cells × 5 one-hot slots = 80 numbers. Win turns it into x_inj, the violet beam injected into the prediction of s at every step. It is the only place the puzzle enters the loop.',
      enter() { fold(0); world.preset('injection'); },
    },
    {
      title: 'Relaxation is energy descent',
      text: 'E = ½‖u − û‖² + ½‖s − ŝ‖², the sum of squared prediction errors. Inference lowers E to zero at a fixed point. Watch it on the Energy tab: the log E / residual chart falls, and the two per-layer bars (½‖u − û‖², ½‖s − ŝ‖²) shrink toward zero as the gap between every unit and its own prediction closes.',
      enter() { fold(0); restartFree(3); world.preset('overview'); hud.selectTab('run'); },
    },
    {
      title: 'Unfold: a deep feedforward network',
      text: 'Unrolled, the ring is a deep feedforward network s₀ = 0 → u₁ → s₁ → u₂ → … with W1 and W2 repeated. Gradient descent moves every copy a little every step. Input and output both attach to s, so when unrolled they both reach every s copy: x_inj injects into every copy (violet), and the readout now taps every copy too (dim teal, read-only in the free phase — it watches but sends nothing back, since the readout is not part of E). Fold it back and you get the loop: the same computation in constant memory.',
      enter() { fold(1); restartFree(2); world.preset('overview'); },
    },
    {
      title: 'The solved board',
      text: 'The readout is logits = s·V + c with a softmax per cell. Opacity is confidence, the four bars are p(1)…p(4), and clues are violet. Every predicted digit is rose while it differs from the solution (or conflicts with another cell, if no unique solution is known) and flips to green the instant it matches — watch cells turn green as the free phase converges. When all 16 digits form a valid board it pulses green. Click a cell to change a clue and watch the ring re-settle.',
      enter() { fold(0); if (!sim.converged) sim.play(); world.preset('board'); },
    },
    {
      title: 'Nudging and learning',
      text: 'Equilibrium propagation: nudge the readout toward the label with +β (violet pull, rose errors flowing backward around the ring — and unrolled, into every s copy, exactly like x_inj), and contrast that equilibrium with the free one. For every synapse the update is a local product of error and activity (amber). These weights were trained this way — by equilibrium propagation, not backprop (the training recipe used a centred ±β rule; this app shows standard one-sided nudging) — and you can watch it happen on this exact board next (this applies one weight update; Reset weights in the Learn tab undoes it).',
      enter() { fold(0); world.preset('overview'); hud.selectTab('learn'); sim.runEpCycle(); },
    },
    {
      title: 'Teach it this board',
      text: 'Pick one of the two "teach it" puzzles this trained model still gets wrong (Puzzle tab, bottom group). Press "Learn" (next to "EP signal" in the phase row) a few times: each press reads off the equilibrium-propagation update and applies it to the weights, then re-solves. Watch the board turn from red to green over a handful of presses — the model is learning, live, in your browser.',
      enter() {
        fold(0); world.preset('board'); hud.selectTab('learn');
        // sim.nextByStatus() walks the full 300-puzzle pool, not the showcase -- it would pick an arbitrary
        // non-showcase "wrong" puzzle, not one of the two curated "teach it" boards this step's text promises.
        // Pick directly from the showcase's 'unsolved' entries instead.
        if (!sim.current || sim.current.status === 2 || !sim.showcase.some((e) => e.category === 'unsolved' && e.id === sim.current.id)) {
          const entry = sim.showcase.find((e) => e.category === 'unsolved');
          const idx = entry ? sim.showcase.indexOf(entry) : -1;
          if (idx >= 0) sim.loadShowcaseEntry(idx);
        }
      },
    },
  ];

  const state = { active: false, i: 0 };
  function render() {
    const s = steps[state.i];
    $('tour-step').textContent = `Tour · ${state.i + 1} / ${steps.length}`;
    $('tour-title').textContent = s.title;
    $('tour-text').textContent = s.text;
    $('tour-prev').disabled = state.i === 0;
    $('tour-next').textContent = state.i === steps.length - 1 ? 'finish' : 'next';
  }
  function go(i) {
    state.i = Math.max(0, Math.min(steps.length - 1, i));
    render();
    try { steps[state.i].enter(); } catch (e) { console.error('[tour] step failed', e); }
  }
  function start() {
    state.active = true;
    $('tour-card').hidden = false;
    $('caption').hidden = true;
    hud.hideCard();
    world.stopRide();
    go(0);
    $('tour-next').focus({ preventScroll: true });
    api.active = true;
  }
  function stop() {
    state.active = false;
    api.active = false;
    $('tour-card').hidden = true;
    $('caption').hidden = false;
    try { localStorage.setItem('shidoku-ring-toured', '1'); } catch (e) { /* storage is optional */ }
  }
  $('tour-next').addEventListener('click', () => { if (state.i >= steps.length - 1) stop(); else go(state.i + 1); });
  $('tour-prev').addEventListener('click', () => go(state.i - 1));
  $('tour-exit').addEventListener('click', stop);
  document.addEventListener('keydown', (e) => {
    if (!state.active || e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); e.stopImmediatePropagation(); if (state.i < steps.length - 1) go(state.i + 1); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); e.stopImmediatePropagation(); go(state.i - 1); }
  }, true);

  const api = { start, stop, next: () => go(state.i + 1), prev: () => go(state.i - 1), active: false, steps };
  return api;
}
