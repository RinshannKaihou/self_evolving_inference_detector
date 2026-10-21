#!/usr/bin/env python3
"""Write src/v35_detector/MANIFEST.json: sha256 of the shipped config and tables, and where they were copied from.
tests/test_manifest.py checks the files against it, so an accidental edit of a table or of m4.json fails the tests.
  python3 tools/make_manifest.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35
"""
import argparse
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'src', 'v35_detector')
SHIPPED = ['configs/m4.json', 'tables/qwen3.npz', 'tables/glm.npz', 'tables/dsv4.npz']
LOGIC = ['core.py', 'offline.py', 'trees.py']                 # byte copies of the research repo's detector/
EDITED = {'stream.py': 'module docstring; RepDetector.__init__ accepts a .npz table path (think_end must then be given)',
          'vocab.py': 'tokenizer paths from the command line instead of the research repo; load() accepts a .npz path'}


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--research', required=True, help='research repo (detector_v35) the files were copied from')
    a = ap.parse_args()
    src = os.path.join(a.research, 'detector')
    commit = subprocess.run(['git', '-C', a.research, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    files = {}
    for rel in SHIPPED + LOGIC:
        mine, theirs = sha(os.path.join(PKG, rel)), sha(os.path.join(src, rel))
        if mine != theirs:
            raise SystemExit(f'{rel} differs from the research copy')
        files[rel] = mine
    out = {'source': {'repo': 'detector_v35', 'commit': commit, 'dir': 'detector/'},
           'sha256': files, 'edited': EDITED}
    with open(os.path.join(PKG, 'MANIFEST.json'), 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
