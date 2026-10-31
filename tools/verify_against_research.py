#!/usr/bin/env python3
"""The shipped detectors reproduce the evaluated alerts record by record: runs this package's detector (64-token chunks, as the
research repo's tools/run_detector.py) over the research repo's dev records and compares (tokens, alert position, rule) with
the research repo's saved runs. Needs the research repo and its corpora; dev only (holdout / external are guarded there).
Exit status 1 on any difference.
  m4  (repetition):  runs/m4_dev_fires.jsonl
  rc3 / rc4 / rc1t (rare_chars): runs/<rc>_dev_fires.jsonl and runs/<rc>_monitor_dev_fires.jsonl. rc3 / rc4 use the same
                     logprob inputs as run_detector.py (the generated token's logprob; where absent, the 20th); rc1t uses token ids only
  python3 tools/verify_against_research.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35 [--detector rc3|rc4|rc1t]
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
from v35_detector import CONFIG_M4, CONFIG_RC1T, CONFIG_RC3, CONFIG_RC4, RareDetector, RepDetector  # noqa: E402

RARE = {'rc3': CONFIG_RC3, 'rc4': CONFIG_RC4, 'rc1t': CONFIG_RC1T}

_D = {}


def _run_m4(a):
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
    return key, {'strict': (len(tok), res.fired_at, res.reason)}


def _run_rare(a):
    name, key, corpus, tok, lp1, lpc = a
    if (name, corpus) not in _D:
        _D[(name, corpus)] = RareDetector(corpus, RARE[name])
    det = _D[(name, corpus)]
    st = det.start()
    res = None
    for i in range(0, len(tok), 64):
        ch = tok[i:i + 64]
        res = det.feed(st, ch) if det.token_only else det.feed(st, ch, lp1[i:i + 64], lpc[i:i + 64])
        if res is not None:
            break
    res = res or det.finish(st)
    mon = det.monitor(st)
    return key, {'strict': (len(tok), res.fired_at, res.reason), 'monitor': (len(tok), mon.fired_at, mon.reason)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--research', required=True)
    ap.add_argument('--detector', default='m4', choices=('m4', 'rc3', 'rc4', 'rc1t'))
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    name = 'verify_research_dev.json' if a.detector == 'm4' else f'verify_research_{a.detector}_dev.json'
    out_path = a.out or os.path.join(REPO, 'runs', name)
    sys.path.insert(0, a.research)
    from dataset.load import iter_records              # the research repo's loader (dev is unguarded)
    stems = {'strict': 'm4_dev_fires.jsonl'} if a.detector == 'm4' else \
        {'strict': f'{a.detector}_dev_fires.jsonl', 'monitor': f'{a.detector}_monitor_dev_fires.jsonl'}
    want = {t: {} for t in stems}
    for t, stem in stems.items():
        for line in open(os.path.join(a.research, 'runs', stem)):
            r = json.loads(line)
            want[t][tuple(r['key'])] = (r['n'], r['fired_at'], r['reason'])
    t0 = time.time()
    if a.detector == 'm4':
        fn, items = _run_m4, ((r.key, r.corpus, np.asarray(r.token_ids)) for r in iter_records('dev'))
    else:
        fn = _run_rare
        needs_lp = a.detector in ('rc3', 'rc4')
        items = ((a.detector, r.key, r.corpus, np.asarray(r.token_ids), r.top_lps[:, 0] if needs_lp else None,
                  np.where(np.isnan(r.chosen_lp), r.top_lps[:, -1], r.chosen_lp) if needs_lp else None)
                 for r in iter_records('dev'))
    got, diff = {t: {} for t in stems}, []
    with Pool(a.procs) as p:
        for key, res in p.imap_unordered(fn, items, chunksize=16):
            for t, v in res.items():
                got[t][key] = v
                if want[t].get(key) != v:
                    diff.append({'tier': t, 'key': key, 'research': want[t].get(key), 'shipped': v})
    missing = sorted({(t, k) for t in stems for k in set(want[t]) - set(got[t])})
    out = {'detector': a.detector, 'records': len(got['strict']), 'research_records': len(want['strict']),
           'alerts': {t: sum(v[1] >= 0 for v in got[t].values()) for t in stems},
           'differences': len(diff), 'missing': len(missing), 'first_differences': diff[:20], 'seconds': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, 'w'), indent=1, default=list)
    print(json.dumps({k: v for k, v in out.items() if k != 'first_differences'}))
    sys.exit(1 if diff or missing else 0)


if __name__ == '__main__':
    main()
