#!/usr/bin/env python3
"""Write src/v35_detector/MANIFEST.json: sha256 of the shipped configs, tables and byte-copied logic, and where they were copied
from. tests/test_manifest.py checks the files against it, so an accidental edit of a table or of a config fails the tests.
rc3 (rare_chars) was not committed in the research repo when it was copied: its files are the ones frozen by sha256 for its
one-shot holdout / external run (research runs/holdout_ledger.jsonl, 2026-10-22).
  python3 tools/make_manifest.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35
"""
import argparse
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'src', 'v35_detector')
SHIPPED = ['configs/m4.json', 'tables/qwen3.npz', 'tables/glm.npz', 'tables/dsv4.npz',
           'configs/rc3.json', 'tables/qwen3_rare.npz', 'tables/glm_rare.npz', 'tables/dsv4_rare.npz', 'tables/qwen3_rcref.npz',
           'tables/glm_rcref.npz', 'tables/rc3_gb.npz']
LOGIC = ['core.py', 'offline.py', 'trees.py', 'rare_gb.py']   # byte copies of the research repo's detector/
EDITED = {'stream.py': 'module docstring; RepDetector.__init__ accepts a .npz table path (think_end must then be given)',
          'vocab.py': 'tokenizer paths from the command line instead of the research repo; load() accepts a .npz path',
          'rare.py': "module docstring example; the config's model path ('detector/tables/<file>') resolves to the package's tables/; "
                     "RareDetector.__init__ accepts a .npz table path (think_end must then be given)",
          'rare_vocab.py': 'relative imports; tokenizer path from the command line; load() accepts a .npz path',
          'rare_ref.py': 'relative imports; the table builder (needs labelled research data) removed, loading unchanged'}


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
    note = ('rare_chars rc3 (rare*.py, configs/rc3.json, tables/*_rare.npz, *_rcref.npz, rc3_gb.npz) was copied from the research '
            "working tree before it was committed; these are the files frozen by sha256 for rc3's one-shot holdout / external run "
            '(research runs/holdout_ledger.jsonl, 2026-10-22)')
    out = {'source': {'repo': 'detector_v35', 'commit': commit, 'dir': 'detector/', 'note': note},
           'sha256': files, 'edited': EDITED}
    with open(os.path.join(PKG, 'MANIFEST.json'), 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
