# Circular Predictive Coding — 4×4 Sudoku demo

An interactive 3D, in-browser view of a trained circular predictive-coding network solving 4×4 Sudoku. The network relaxes by gradient descent on its energy, live, and you can apply equilibrium-propagation nudging and weight updates to the board on screen.

## Running it

The demo is a static page: plain HTML and JavaScript ES modules, with no build step. Serve this folder with any static file server, for example

```
cd demo
python -m http.server 8000   # then open http://localhost:8000
```

or publish the folder with GitHub Pages. Opening `index.html` directly from disk does not work, because browsers block ES modules and `fetch()` on `file://`.

Everything is bundled except the web fonts, which `index.html` imports from Google Fonts. Offline, the page still works and falls back to system fonts.

## The model

- A dense two-layer circular network (2 × 256 tanh units, `u` and `s`) on 4×4 boards with 6 givens. The clues enter `s` through an input embedding, and a linear readout of `s` gives the 16 × 4 digit logits.
- Trained with centred (±β) equilibrium propagation on heavy-ball gradient-descent free phases. It solves 87.7% of the 1000 test boards.
- The script that trained this checkpoint is not included in this repository. `model.js` is a JavaScript port of the JAX model in this repository.

## Files

- `models/best.json`: the trained weights and their metadata. `models/index.json` is the model list (the app shows only entries whose ids are in `ENABLED_MODELS` in `sim.js`).
- `puzzles.json`: the bundled boards (240 test puzzles and 60 generated ones). `models/showcase.json`: the 10 curated boards in the Puzzle tab.
- `model.js` and `sim.js`: inference, nudging and EP updates. The other modules handle rendering and UI.

## Third-party code

three.js r169 is vendored in `vendor/three/` under the MIT license (see `vendor/three/LICENSE`).
