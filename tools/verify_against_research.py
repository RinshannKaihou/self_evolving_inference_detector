#!/usr/bin/env python3
"""The shipped detector reproduces the evaluated m4 alerts record by record: runs this package's RepDetector (64-token
chunks, as the research repo's tools/run_detector.py) over the research repo's dev records and compares (tokens, alert
boundary, component) with the research repo's saved runs/m4_dev_fires.jsonl. Needs the research repo and its corpora;
dev only (holdout / external are guarded there). Exit status 1 on any difference.
  python3 tools/verify_against_research.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
from v35_detector import CONFIG_M4, RepDetector  # noqa: E402

_D = {}


def _run(a):
    key, corpus, tok = a
    if corpus not in _D:
        _D[corpus] = RepDetector(corpus, CONFIG_M4)
    det = _D[corpus]
    st = det.start()
    res = None
    for i in range(0, len(tok), 64):
        res = det.feed(st, tok[i:i + 64])
        if res is not None:
            break
    res = res or det.finish(st)
    return key, len(tok), res.fired_at, res.reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--research', required=True)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--out', default=os.path.join(REPO, 'runs', 'verify_research_dev.json'))
    a = ap.parse_args()
    sys.path.insert(0, a.research)
    from dataset.load import iter_records              # the research repo's loader (dev is unguarded)
    want = {}
    for line in open(os.path.join(a.research, 'runs', 'm4_dev_fires.jsonl')):
        r = json.loads(line)
        want[tuple(r['key'])] = (r['n'], r['fired_at'], r['reason'])
    t0 = time.time()
    items = ((r.key, r.corpus, np.asarray(r.token_ids)) for r in iter_records('dev'))
    got, diff = {}, []
    with Pool(a.procs) as p:
        for key, n, at, reason in p.imap_unordered(_run, items, chunksize=16):
            got[key] = (n, at, reason)
            if want.get(key) != (n, at, reason):
                diff.append({'key': key, 'research': want.get(key), 'shipped': (n, at, reason)})
    missing = sorted(set(want) - set(got))
    out = {'records': len(got), 'research_records': len(want), 'alerts': sum(v[1] >= 0 for v in got.values()),
           'differences': len(diff), 'missing': len(missing), 'first_differences': diff[:20], 'seconds': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=list)
    print(json.dumps({k: v for k, v in out.items() if k != 'first_differences'}))
    sys.exit(1 if diff or missing else 0)


if __name__ == '__main__':
    main()
