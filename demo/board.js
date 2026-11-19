// board.js — the large readable readout board floating above the ring: logits = s·V + c -> (16 cells x 4 digits).
// Each cell: argmax digit (opacity = confidence), 4 tiny probability bars, clue cells VIOLET, GREEN pulse when the
// board is complete and valid. Click a cell to cycle its clue (handled by app.js through pickCell()).
import * as THREE from 'three';
import { COLORS, FONTS, BOARD_POS, BOARD_POS_UNF, BOARD_SCALE, anchors, makeLabel } from './world.js';
import { chainUniforms } from './chain.js';
import { boardStatus } from './sim.js';

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

const CELL = 1.62;
const GAP = 0.12;
const BOXGAP = 0.16;

/** true if the predicted digit at p (1..4, from `full`) duplicates another cell in its row, column or 2x2 box,
 *  or contradicts a clue: the no-known-solution fallback for "wrong" ("conflict") vs "right" ("consistent"). */
function hasConflict(full, puzzle, p) {
  const r = p >> 2, c = p & 3, d = full[p];
  if (puzzle[p] && puzzle[p] !== d) return true;
  for (let k = 0; k < 4; k++) {
    if (k !== c && full[r * 4 + k] === d) return true;
    if (k !== r && full[k * 4 + c] === d) return true;
  }
  const br = r & 2, bc = c & 2;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    const q = (br + i) * 4 + (bc + j);
    if (q !== p && full[q] === d) return true;
  }
  return false;
}

function cellCentre(p) {
  const r = p >> 2, c = p & 3;
  const x = (c - 1.5) * (CELL + GAP) + (c >= 2 ? 0.5 : -0.5) * BOXGAP;
  const y = (1.5 - r) * (CELL + GAP) + (r >= 2 ? -0.5 : 0.5) * BOXGAP;
  return new THREE.Vector2(x, y);
}

function makeAtlas() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024; canvas.height = 256;
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.anisotropy = 4;
  const draw = () => {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 1024, 256);
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `600 214px ${FONTS.display}`;
    for (let d = 0; d < 4; d++) ctx.fillText(String(d + 1), d * 256 + 128, 140);
    tex.needsUpdate = true;
  };
  draw();
  return { tex, draw };
}

// canvas height is taller than its "visual" (W * 1240/1024) ratio by BOTTOM_PAD_PX: that extra band is empty space
// below the grid (board.js positions the mesh so the top edge is unchanged and the bottom edge moves down by it),
// giving the caption line clear air above the cells instead of grazing the last row.
const BACKDROP_W = 1024, BACKDROP_H_BASE = 1240, BACKDROP_BOTTOM_PAD_PX = 150;
const BACKDROP_H = BACKDROP_H_BASE + BACKDROP_BOTTOM_PAD_PX;

function makeBackdrop() {
  const W = BACKDROP_W, Hh = BACKDROP_H;
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = Hh;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const draw = () => {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, Hh);
    const r = 46;
    const g = ctx.createLinearGradient(0, 0, 0, Hh);
    g.addColorStop(0, 'rgba(16,22,33,0.92)');
    g.addColorStop(1, 'rgba(9,13,21,0.92)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(r, 0); ctx.arcTo(W, 0, W, Hh, r); ctx.arcTo(W, Hh, 0, Hh, r); ctx.arcTo(0, Hh, 0, 0, r); ctx.arcTo(0, 0, W, 0, r);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = 'rgba(139,152,165,0.35)';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = COLORS.ink;
    ctx.font = `600 58px ${FONTS.display}`;
    ctx.fillText('Readout', 56, 96);
    ctx.fillStyle = COLORS.muted;
    ctx.font = `500 30px ${FONTS.mono}`;
    ctx.textAlign = 'right';
    ctx.fillText('softmax(s·V + c), per cell', W - 56, 92);
    ctx.textAlign = 'center';
    ctx.font = `500 26px ${FONTS.mono}`;
    ctx.fillStyle = 'rgba(139,152,165,0.9)';
    // sits well inside the bottom padding band, clear of the last row of cells (see BACKDROP_BOTTOM_PAD_PX above)
    ctx.fillText('opacity = confidence · bars = p(1) … p(4)', W / 2, BACKDROP_H_BASE - 42 + BACKDROP_BOTTOM_PAD_PX * 0.72);
    tex.needsUpdate = true;
  };
  draw();
  return { tex, draw };
}

export function createBoard(world, sim) {
  const group = new THREE.Group();
  group.name = 'board';
  group.position.copy(BOARD_POS);
  group.scale.setScalar(BOARD_SCALE);
  world.scene.add(group);

  const atlas = makeAtlas();
  const backdrop = makeBackdrop();
  const span = 4 * CELL + 3 * GAP + BOXGAP;
  const bgW = span + 1.0;
  const bgHBase = bgW * (BACKDROP_H_BASE / BACKDROP_W);          // height the backdrop used to be (sets the top edge)
  const bottomPad = bgW * (BACKDROP_BOTTOM_PAD_PX / BACKDROP_W); // extra clearance below the grid, in world units
  const bgH = bgHBase + bottomPad;
  const bg = new THREE.Mesh(
    new THREE.PlaneGeometry(bgW, bgH),
    new THREE.MeshBasicMaterial({ map: backdrop.tex, transparent: true, depthWrite: false, fog: false, side: THREE.DoubleSide }),
  );
  // top edge unchanged (bgHBase-based), all of the extra height goes to the bottom, below the last row of cells
  bg.position.set(0, (bgHBase - bgW) * 0.5 - 0.02 - bottomPad / 2, -0.04);
  bg.renderOrder = 6;
  group.add(bg);

  // green glow behind the board when solved
  const glowTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(128, 128, 30, 128, 128, 128);
    g.addColorStop(0, 'rgba(255,255,255,0.9)'); g.addColorStop(0.55, 'rgba(255,255,255,0.25)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c); return t;
  })();
  const glow = new THREE.Mesh(
    new THREE.PlaneGeometry(bgW * 1.9, bgH * 1.7),
    new THREE.MeshBasicMaterial({ map: glowTex, color: new THREE.Color(COLORS.green), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }),
  );
  glow.position.set(0, bg.position.y, -0.2);
  glow.renderOrder = 5;
  group.add(glow);

  const colors = {
    uInk: { value: new THREE.Color(COLORS.ink).multiplyScalar(0.92) },
    uViolet: { value: new THREE.Color(COLORS.violet) },
    uGreen: { value: new THREE.Color(COLORS.green) },
    uRose: { value: new THREE.Color(COLORS.rose) },
    uMuted: { value: new THREE.Color(COLORS.muted) },
    uBg: { value: new THREE.Color('#0F1622') },
  };
  const cellGeo = new THREE.PlaneGeometry(CELL, CELL);
  const cells = [];
  for (let p = 0; p < 16; p++) {
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      uniforms: {
        ...colors,
        uAtlas: { value: atlas.tex },
        uDigit: { value: 0 }, uConf: { value: 0.25 }, uProbs: { value: new THREE.Vector4(0.25, 0.25, 0.25, 0.25) },
        uClue: { value: 0 }, uSolved: { value: 0 }, uPulse: { value: 1 }, uHover: { value: 0 }, uCorrect: { value: 0 },
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D uAtlas;
        uniform float uDigit, uConf, uClue, uSolved, uPulse, uHover, uCorrect;
        uniform vec4 uProbs;
        uniform vec3 uInk, uViolet, uGreen, uRose, uMuted, uBg;
        varying vec2 vUv;
        float rbox(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
        void main() {
          vec2 p = vUv - 0.5;
          float d = rbox(p, vec2(0.5), 0.09);
          float fill = 1.0 - smoothstep(-0.004, 0.004, d);
          if (fill < 0.01) discard;
          float border = smoothstep(-0.035, -0.014, d) * fill;
          // predicted (non-clue) digits: rose while wrong / conflicting, green the instant they are right / consistent
          vec3 predCol = mix(uRose, uGreen, uCorrect);
          vec3 edge = mix(predCol * 0.6, uViolet * 0.95, uClue);
          edge = mix(edge, uGreen * 1.25, uSolved * uPulse);
          vec3 col = uBg * (1.0 + 0.5 * uHover) + edge * border * (0.75 + 0.5 * uHover);
          vec2 g = vec2((vUv.x - 0.5) / 0.66 + 0.5, (vUv.y - 0.25) / 0.72);
          float glyph = 0.0;
          if (g.x > 0.0 && g.x < 1.0 && g.y > 0.0 && g.y < 1.0) glyph = texture2D(uAtlas, vec2((uDigit + g.x) * 0.25, g.y)).a;
          vec3 gc = mix(predCol, uViolet * 1.05, uClue);
          gc = mix(gc, uGreen * 1.1, uSolved * 0.5);
          col = mix(col, gc, glyph * (0.08 + 0.92 * uConf));
          float bx = (vUv.x - 0.17) / 0.66;
          if (bx > 0.0 && bx < 1.0 && vUv.y > 0.075 && vUv.y < 0.215) {
            float slot = floor(bx * 4.0);
            float fx = fract(bx * 4.0);
            float pr = slot < 0.5 ? uProbs.x : (slot < 1.5 ? uProbs.y : (slot < 2.5 ? uProbs.z : uProbs.w));
            float h = (vUv.y - 0.075) / 0.14;
            float inBar = step(0.16, fx) * step(fx, 0.84);
            float bar = inBar * step(h, max(pr, 0.0));
            float base = inBar * step(h, 0.07);
            vec3 bc = abs(slot - uDigit) < 0.5 ? gc : uMuted * 0.85;
            col = mix(col, bc, max(bar * 0.95, base * 0.45));
          }
          gl_FragColor = vec4(col, fill * 0.96);
          #include <colorspace_fragment>
        }`,
    });
    const mesh = new THREE.Mesh(cellGeo, m);
    const cc = cellCentre(p);
    mesh.position.set(cc.x, cc.y, 0);
    mesh.renderOrder = 7;
    mesh.userData = { kind: 'cell', cell: p };
    group.add(mesh);
    cells.push(mesh);
  }

  // the correct/wrong/clue legend lives in the HUD (next to the BOARD readout, always crisp at any zoom and at
  // phone width); this in-world sprite stays a short instruction, same as before.
  const caption = makeLabel([{ text: 'click a cell to cycle its clue', font: 'body', size: 15, color: COLORS.muted }], { worldH: 0.34 });
  caption.sprite.position.set(0, -span / 2 - 0.95, 0.05);
  caption.sprite.material.opacity = 0.75;
  group.add(caption.sprite);

  // ---------------------------------------------------------------- state
  let solvedT = 0, hover = -1, yaw = 0;
  const reduced = world.state.reduced;
  const target = new THREE.Vector3();
  const full = new Int8Array(16);
  const TWEEN_RATE = 1 / 0.2;   // ~200 ms colour tween, within the 150-250 ms readability window

  function update(dt, time) {
    // folded: up and to the right of the ring; unfolded: above the middle of the chain
    group.position.lerpVectors(BOARD_POS, BOARD_POS_UNF, smooth(0.15, 0.85, chainUniforms.uFold.value));
    anchors.board.copy(group.position);
    // face the camera (yaw only) so the board stays readable from any orbit angle
    const cam = world.camera.position;
    target.copy(cam).sub(group.position);
    const want = Math.atan2(target.x, target.z);
    let d = want - yaw;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    yaw += d * (1 - Math.exp(-dt * (reduced ? 12 : 5)));
    group.rotation.y = yaw;

    const V = sim.view;
    if (!V) return;
    const st = boardStatus(V.digits, sim.puzzle, sim.solution);
    const solved = st.valid ? 1 : 0;
    solvedT += (solved - solvedT) * Math.min(1, dt * 2.5);
    const pulse = reduced ? 0.85 : 0.6 + 0.4 * Math.sin(time * 3.2);

    // correctness of every predicted (non-clue) digit label: green the instant it matches, rose otherwise.
    // Ground truth: the known unique solution if there is one; else the row/column/box consistency fallback.
    const known = !!sim.solution;
    for (let p = 0; p < 16; p++) full[p] = V.digits[p] + 1;
    const tweenK = Math.min(1, dt * TWEEN_RATE);
    for (let p = 0; p < 16; p++) {
      const u = cells[p].material.uniforms;
      u.uDigit.value = V.digits[p];
      u.uConf.value = V.conf[p];
      u.uProbs.value.set(V.probs[p * 4], V.probs[p * 4 + 1], V.probs[p * 4 + 2], V.probs[p * 4 + 3]);
      const isClue = sim.puzzle && sim.puzzle[p] ? 1 : 0;
      u.uClue.value = isClue;
      u.uSolved.value = solvedT;
      u.uPulse.value = pulse;
      u.uHover.value = p === hover ? 1 : 0;
      if (!isClue) {
        const target = known ? (full[p] === sim.solution[p] ? 1 : 0) : (hasConflict(full, sim.puzzle, p) ? 0 : 1);
        u.uCorrect.value += (target - u.uCorrect.value) * tweenK;
      }
    }
    glow.material.opacity = solvedT * (reduced ? 0.22 : 0.16 + 0.12 * pulse);
  }

  return {
    group, cells,
    update,
    redraw() { atlas.draw(); backdrop.draw(); },
    pickCell(hits) {
      for (const h of hits) if (h.object.userData && h.object.userData.kind === 'cell') return h.object.userData.cell;
      return -1;
    },
    setHover(p) { hover = p; },
    /** board-local position of digit bar d in cell p (where the V synapses land) */
    barLocal(p, d) {
      const cc = cellCentre(p);
      const x = cc.x + (-0.5 + 0.17 + ((d + 0.5) / 4) * 0.66) * CELL;
      const y = cc.y + (-0.5 + 0.075) * CELL;
      return new THREE.Vector3(x, y, 0.01);
    },
    worldOfCell(p) { const cc = cellCentre(p); return group.localToWorld(new THREE.Vector3(cc.x, cc.y, 0)); },
  };
}
