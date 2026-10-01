#!/usr/bin/env python3
"""Prevent deploying a generated model that no longer matches its sources."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / 'demo/wasm-manifest.json').read_text())
for section in ['sources', 'outputs']:
    for name, expected in manifest[section].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'{name} changed. Run python3 demo/build_wasm.py with Emscripten, then re-run parity tests.')
print('Verified: WebAssembly build matches original network, weights, and adapter.')
