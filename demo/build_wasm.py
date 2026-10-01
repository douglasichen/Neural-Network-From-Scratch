#!/usr/bin/env python3
"""Build with Emscripten 4.0.15; keep original C++ and saved weights unchanged."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = ['main.cpp', 'saved_network.txt', 'demo/wasm.cpp', 'demo/compat/bits/stdc++.h',
           'demo/build_wasm.py', 'csv2/reader.hpp', 'csv2/mio.hpp', 'csv2/parameters.hpp']
OUTPUTS = ['demo/static/network.mjs', 'demo/static/network.wasm']

def hashes(paths):
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}

if __name__ == '__main__':
    compiler = os.environ.get('EMXX', 'em++')
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    subprocess.run([
        compiler, str(HERE / 'wasm.cpp'), '-I' + str(HERE / 'compat'), '-std=c++17',
        '-O2', '-flto', '-Wno-return-type', '-sMODULARIZE=1', '-sEXPORT_ES6=1',
        '-sEXPORT_NAME=createDigitNetwork', '-sENVIRONMENT=web,worker,node',
        '-sEXPORTED_RUNTIME_METHODS=["HEAPU8"]', '-sALLOW_MEMORY_GROWTH=1', '-sINITIAL_MEMORY=16777216',
        '-sEXPORTED_FUNCTIONS=["_initialize_model","_input_buffer","_output_buffer","_predict_digit"]',
        '--embed-file', str(ROOT / 'saved_network.txt') + '@/saved_network.txt',
        '-o', str(HERE / 'static' / 'network.mjs')
    ], check=True)
    (HERE / 'wasm-manifest.json').write_text(json.dumps({
        'compiler': version, 'sources': hashes(SOURCES), 'outputs': hashes(OUTPUTS)
    }, indent=2) + '\n')
