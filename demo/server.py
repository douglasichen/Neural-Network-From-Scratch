#!/usr/bin/env python3
"""Serve the static browser demo locally; no inference backend is required."""
import argparse
from pathlib import Path
import shutil
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build():
    binary = HERE / 'build' / 'inference'
    sources = [HERE / 'inference.cpp', ROOT / 'main.cpp', * (ROOT / 'csv2').glob('*.hpp')]
    if not binary.exists() or any(p.stat().st_mtime > binary.stat().st_mtime for p in sources):
        compiler = next((shutil.which(c) for c in ['g++-15', 'g++-14', 'g++-13', 'g++'] if shutil.which(c)), None)
        if not compiler:
            raise SystemExit('A GCC C++17 compiler is required.')
        binary.parent.mkdir(exist_ok=True)
        subprocess.run([compiler, '-std=c++17', '-O2', str(HERE / 'inference.cpp'), '-o', str(binary)], check=True)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    # Only static files are served. Inference runs inside each visitor's browser.
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE / 'static'), **kw)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Draw a digit at http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
