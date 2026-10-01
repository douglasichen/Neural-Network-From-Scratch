#!/usr/bin/env python3
"""Compare native C++ and deployed WASM using all 8,400 validation images."""
import csv
import json
import math
from pathlib import Path
import random
import subprocess
from server import build

ROOT = Path(__file__).resolve().parent.parent
subprocess.run(['python3', str(ROOT / 'demo/check_build.py')], check=True)
with (ROOT / 'digit_recognizer/train.csv').open() as f:
    rows = list(csv.reader(f))[33601:]
pixels = [r[1:] for r in rows]
rng = random.Random(42)
pixels += [[str(v)] * 784 for v in [0, 1, 127, 255]]
pixels += [[str(rng.randrange(256)) for _ in range(784)] for _ in range(20)]
payload = '\n'.join(' '.join(p) for p in pixels) + '\n'

def run(command):
    result = subprocess.run(command, input=payload, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return [json.loads(line) for line in result.stdout.splitlines()]

native = run([str(build()), str(ROOT / 'saved_network.txt')])
wasm = run(['node', str(ROOT / 'demo/wasm_predict.mjs')])
assert len(native) == len(wasm) == len(pixels)
max_difference = 0
for a, b in zip(native, wasm):
    assert a['prediction'] == b['prediction'], (a, b)
    assert len(b['probabilities']) == 10
    assert abs(sum(b['probabilities']) - 1) < 1e-10
    for x, y in zip(a['probabilities'], b['probabilities']):
        assert math.isfinite(y)
        max_difference = max(max_difference, abs(x-y))
        assert abs(x-y) < 1e-10, (x, y)
correct = sum(b['prediction'] == int(r[0]) for b, r in zip(wasm, rows))
print(f'PASS: {len(pixels)} native/WASM predictions match; max probability difference {max_difference:.3g}.')
print(f'Validation accuracy: {correct}/{len(rows)} = {correct/len(rows):.4%}.')
