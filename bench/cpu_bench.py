#!/usr/bin/env python3
"""CPU cost per token (one thread) and per-request state of the shipped detector, on the deployment machine's CPU, next to
vLLM's repetition check (upstream transcription, called after every appended token as its scheduler does).

Inputs are the synthetic streams of tests/synth.py (no real trajectories in this repo): a mix of random text, verbatim and
near-verbatim loops, character runs and circling sentences. The detector is fed 64-token chunks and stops at its first
alert, as in production. Best of --rounds rounds. The research repo measured the same detector on its real dev records
(3.43 us/token on an Apple M3 Pro); running this script on both machines translates that figure to the server.

  python bench/cpu_bench.py [--table qwen3] [--rounds 3] [--vllm] [--out runs/cpu_bench.json]
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))
from synth import KINDS, make  # noqa: E402
from test_state import state_bytes  # noqa: E402
from test_vllm_rule import upstream  # noqa: E402
from v35_detector import CONFIG_M4, RepDetector  # noqa: E402


def run_detector(det, tok):
    st = det.start()
    res, fed = None, 0
    for i in range(0, len(tok), 64):
        res = det.feed(st, tok[i:i + 64])
        fed = min(i + 64, len(tok))
        if res is not None:
            break
    if res is None:
        det.finish(st)
    return fed, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='qwen3')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--vllm', action='store_true', help="also time vLLM's check ([50,500]x3 and [32,2000]x3); slow")
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'runs', 'cpu_bench.json'))
    a = ap.parse_args()
    streams = [(k, make(a.table, k, s)) for k in KINDS for s in range(1, a.seeds + 1)]
    det = RepDetector(a.table, CONFIG_M4)
    best, fed_total, states = None, 0, []
    for _ in range(a.rounds):
        t0, fed_total, states = time.perf_counter(), 0, []
        for _, tok in streams:
            fed, st = run_detector(det, tok)
            fed_total += fed
            states.append(st)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    out = dict(machine=platform.platform(), processor=platform.processor(), python=sys.version.split()[0],
               numpy=np.__version__, table=a.table, streams=len(streams), tokens_fed=fed_total,
               us_per_token=1e6 * best / fed_total,
               state_bytes_max=max(state_bytes(st) for st in states),
               load_avg_1m=os.getloadavg()[0] if hasattr(os, 'getloadavg') else None)
    if a.vllm:
        try:                                             # the installed vLLM's own function, as its scheduler calls it
            from vllm.sampling_params import RepetitionDetectionParams
            from vllm.v1.core.sched.utils import check_sequence_repetition

            def check(ids, mn, mx, c):
                return check_sequence_repetition(ids, RepetitionDetectionParams(max_pattern_size=mx, min_pattern_size=mn,
                                                                                min_count=c))
            out['vllm_check_source'] = 'installed vllm'
        except ImportError:
            check = upstream
            out['vllm_check_source'] = 'transcription in tests/test_vllm_rule.py (vllm not installed): indicative only'
        for mn, mx, c in ((50, 500, 3), (32, 2000, 3)):
            n, t0 = 0, time.perf_counter()
            for _, tok in streams[::4]:                  # a subset: the check is ~100x slower
                ids = []
                for x in tok.tolist():
                    ids.append(x)
                    n += 1
                    if check(ids, mn, mx, c):
                        break
            out[f'vllm_[{mn},{mx}]x{c}_us_per_token'] = 1e6 * (time.perf_counter() - t0) / n
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
