# Live drawing demo

[Open the public demo](https://neural-network-from-scratch-one.vercel.app/).

Draw one digit (0–9) using a mouse, touch, or pen. Predictions update during the stroke. The original C++ model runs in a Web Worker through WebAssembly, keeping the drawing interface responsive. No drawing or prediction request leaves the browser.

## Run locally

```sh
python3 demo/server.py
```

Open http://127.0.0.1:8765. Python only serves static files; no compiler or Python packages are needed to use the checked-in demo. Stop with Ctrl+C; use `--port 8766` if necessary.

## Original network, unchanged

`wasm.cpp` includes `../main.cpp` unchanged, renames its training entry point at compile time, initializes the original 784 → 10 → 10 architecture, and loads the original `saved_network.txt` with `load_network`. Each prediction calls the original `predict` and `forward_prop`; probabilities come directly from its output neurons. There is no JavaScript reimplementation, retraining, weight conversion, or change to activation/inference math.

`compat/bits/stdc++.h` supplies standard C++ headers because Emscripten uses libc++ instead of GCC's convenience header. It does not change network operations. The linker removes unused training functions/data. The original checkpoint is embedded verbatim into the compiled module's virtual filesystem.

The model uses double precision. Cross-platform math libraries may differ by tiny rounding amounts, so parity tests require identical predicted digits and absolute probability differences below 1e-10.

## Canvas → input contract

- Drawing surface: 336 × 336 backing pixels, square at every display size. Pointer coordinates account for display scaling and borders.
- Fit the ink's bounding box proportionally inside 20 × 20 pixels, preserving aspect ratio, with black padding in a 28 × 28 image.
- Area-average grayscale and center the ink's center of mass without clipping. This is drawing-input preparation, separate from the unchanged network.
- Pass exactly 784 values in row-major order (`y * 28 + x`), black 0 through white 255. No division by 255, inversion, or transpose: training loads raw MNIST values.
- The pixelated preview displays the exact input array.
- Predictions are throttled while drawing; responses for older strokes/cleared canvases are discarded.

## Build the browser model

The generated `static/network.mjs` and `static/network.wasm` are checked in so Vercel does not need a C++ compiler. Rebuild only when the model or adapter changes:

```sh
# Install/activate Emscripten 4.0.15 using https://emscripten.org/docs/getting_started/downloads.html
# Then source your emsdk_env.sh, so em++ is on PATH.
python3 demo/build_wasm.py
python3 demo/test_parity.py
```

Alternatively set `EMXX` to the absolute path of `em++`. The build records source and output SHA-256 hashes in `wasm-manifest.json`. `python3 demo/check_build.py` prevents deploying stale generated assets.

Parity testing requires Node.js, Python, a GCC C++17 compiler, and the repository's existing `digit_recognizer/train.csv`. It compares the native adapter (`inference.cpp`) with the actual generated module over all 8,400 validation examples plus blank, full-intensity, and seeded random images. It checks predicted digits, every probability, finite outputs, and softmax sums.

## Deploy to Vercel

From the repository root:

```sh
vercel link
vercel --prod
```

`vercel.json` verifies generated model hashes and publishes only `demo/static`. The training dataset, native binaries, Python tools, and C++ sources are not served. Inference requires no server, credentials, or paid model API. Model weights are included in the publicly downloadable WebAssembly module.

## Saved-model selection

Compared the two distinct checkpoints found on the repository's current remote branches (master, maybe-less-broken, fix-fading-gradients), using unchanged native inference on the last 8,400 rows of `digit_recognizer/train.csv` (the intended 80/20 split):

| Checkpoint | Correct | Accuracy |
| --- | ---: | ---: |
| Root `saved_network.txt` (also shared by branch copies) | 7,785 / 8,400 | 92.68% |
| `fix-fading-gradients` experimental `saved_network_x.txt` | 1,073 / 8,400 | 12.77% |

The demo uses the winning root checkpoint. SHA-256: `7a68f774699fc949c74d85ceea82c3c5b093b961038cc5397cb88038302787b1`.

This is an existing validation split, not a new independent test set. Freehand accuracy may differ. Evaluation streams CSV rows to the adapter instead of invoking the original training CSV loader; no original neural-network source is edited.

## Verification results

Native/WebAssembly comparison passed on 8,424 inputs: all predicted digits matched, with a maximum absolute probability difference of 2.33e-15. WASM validation accuracy remained 7,785/8,400 (92.68%).

The production URL was verified without authentication. Its model assets matched the tested local files byte-for-byte. Browser drawing at desktop and mobile widths produced live predictions, the preview matched all 784 submitted pixels, and no browser errors were reported.
