#!/usr/bin/env python3
"""CPU cost per token (one thread) and per-request state of the shipped detectors, on the deployment machine's CPU, next to
vLLM's repetition check (upstream transcription, called after every appended token as its scheduler does).

Inputs are the synthetic streams of tests/synth.py (no real trajectories in this repo): a mix of random text, verbatim and
near-verbatim loops, character runs and circling sentences. The detector is fed 64-token chunks and stops at its first
alert, as in production. Best of --rounds rounds. The research repo measured the same detector on its real dev records
(3.43 us/token on an Apple M3 Pro); running this script on both machines translates that figure to the server.

rc3 and rc4 (garbled-output monitor) get the same token streams with synthetic logprobs (no real logprobs in this repo). Random
word sequences put far more tokens through the learned score's gate than real outputs do (rare_gb_gate_share here, about 1%
on real outputs), so its per-token figure overstates the real cost (research repo, real dev records: 0.76 us/token on an
Apple M3 Pro). 'hooks' times the whole sync path of serving/hooks.py per token (buffering, 64-token blocks, events) for m4,
rc3 / rc4 and rc1t, driven by 1-token engine steps. rc3 / rc4 carry EngineCoreOutput.v35_rare ([top1, chosen]); rc1t carries no
values and disables sampler collection. The streams are loop-free with greedy values so detectors run to the end of every request
(hooks_alerts counts exceptions). rc3_collect is V35_RARE=collect for logprob configs: the hook only receives the values.

  python bench/cpu_bench.py [--table qwen3] [--rounds 3] [--vllm] [--out runs/cpu_bench.json]
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))
from synth import KINDS, make  # noqa: E402
from test_state import state_bytes  # noqa: E402
from test_rare import synth as rare_synth  # noqa: E402
from test_vllm_rule import upstream  # noqa: E402
from v35_detector import CONFIG_M4, CONFIG_RC1T, CONFIG_RC3, CONFIG_RC4, RareDetector, RepDetector  # noqa: E402

RARE_CFG = {'rc3': CONFIG_RC3, 'rc4': CONFIG_RC4, 'rc1t': CONFIG_RC1T}
from v35_detector import rare as R  # noqa: E402
from v35_detector.serving import hooks  # noqa: E402

LN10 = np.log(10.0)
NOTE_RC3 = ('rc3 inputs are synthetic: random word sequences with synthetic logprobs put far more tokens through the learned '
            "score's gate (rare_gb_gate_share) than real outputs (about 1%), so rare_us_per_token overstates the real cost: "
            'a conservative upper figure (research repo, real dev records: 0.76 us/token on an Apple M3 Pro)')


def synthetic_logprobs(n, rng):
    """natural-log top-1 and generated-token logprobs: the generated token is the top-1 at 85% of positions (sampling)"""
    lp1 = -rng.exponential(0.3, n) * LN10
    return lp1, np.where(rng.random(n) < 0.85, lp1, lp1 - rng.exponential(1.0, n) * LN10)


def run_rare(det, tok, lp1, lpc):
    st = det.start()
    fed = 0
    for i in range(0, len(tok), 64):
        ch = tok[i:i + 64]
        res = det.feed(st, ch) if det.token_only else det.feed(st, ch, lp1[i:i + 64], lpc[i:i + 64])
        fed = min(i + 64, len(tok))
        if res is not None:
            break
    det.finish(st)
    return fed, st


def rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


STATIC = '''
import json, os, sys, tracemalloc
sys.path.insert(0, sys.argv[1])
import numpy as np
from v35_detector import CONFIG_RC1T, CONFIG_RC3, CONFIG_RC4, RareDetector
try:
    import psutil
    m0 = psutil.Process().memory_info().rss
except ImportError:
    psutil = None
tracemalloc.start()
det = RareDetector(sys.argv[2], {'rc3': CONFIG_RC3, 'rc4': CONFIG_RC4, 'rc1t': CONFIG_RC1T}[sys.argv[3]])
st, n = det.start(), 2000
tok = np.random.default_rng(0).integers(0, 150000, n)
if det.token_only:
    det.feed(st, tok)
else:
    det.feed(st, tok, np.full(n, -0.1), np.full(n, -0.5))
det.finish(st)
kept, peak = tracemalloc.get_traced_memory()
print(json.dumps([kept, peak, None if psutil is None else psutil.Process().memory_info().rss - m0]))
'''


def static_memory(table, name):
    """tables of one configuration in a fresh process (rc3 and rc4 share caches): retained and peak (tracemalloc), RSS growth"""
    import subprocess
    out = subprocess.run([sys.executable, '-c', STATIC, os.path.join(HERE, '..', 'src'), table, name], capture_output=True,
                         text=True, check=True).stdout
    kept, peak, grown = json.loads(out.strip().splitlines()[-1])
    return kept / 2 ** 20, peak / 2 ** 20, None if grown is None else grown / 2 ** 20


def bench_rare(table, streams, rounds, name='rc3'):
    """per-token cost and static memory of one rare_chars configuration; keys rare_* for rc3, other names as prefixes"""
    rng = np.random.default_rng(0)
    data = [(tok, None, None) if name == 'rc1t' else (tok, *synthetic_logprobs(len(tok), rng)) for _, tok in streams]
    kept, peak, grown = static_memory(table, name)
    det = RareDetector(table, RARE_CFG[name])
    run_rare(det, *data[0])                              # first use builds the per-token input table
    gate = [0, 0]
    mask = R.gb_gate_mask

    def counted(F):
        g = mask(F)
        gate[0] += int(g.sum())
        gate[1] += len(g)
        return g
    R.gb_gate_mask = counted                             # one untimed pass: how many tokens reach the learned score
    states = [run_rare(det, *d)[1] for d in data]
    R.gb_gate_mask = mask
    best, fed_total = None, 0
    for _ in range(rounds):
        t0, fed_total = time.perf_counter(), 0
        for d in data:
            fed_total += run_rare(det, *d)[0]
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    k = 'rare' if name == 'rc3' else name
    out = {f'{k}_us_per_token': 1e6 * best / fed_total, f'{k}_tokens_fed': fed_total,
           f'{k}_gb_gate_share': gate[0] / max(gate[1], 1), f'{k}_state_bytes_max': max(state_bytes(st) for st in states),
           f'{k}_static_mb': kept, f'{k}_load_peak_mb': peak, f'{k}_load_rss_mb': grown}
    if name == 'rc3':
        out['rare_note'] = NOTE_RC3
    else:
        out[f'{k}_note'] = (f'{name} stops at its first strict alert, which comes early on these mixed-script synthetic streams '
                            f'({k}_tokens_fed): compare rc3 and {name} on hooks_us_per_token (English-like streams, no alert) and '
                            'on the online run instead')
    return out


def bench_hooks(table, rounds, seeds=4):
    """us per token of the whole sync hook path (register, 1-token feeds with rc3's values, finish) for m4, rc3, both; on
    streams without loops, so that the detectors run to the end of every request (alerts are counted in the output)"""
    data = []
    for s in range(1, seeds + 1):                        # English-like: Latin words, punctuation, digits, a think-end
        tok, lp1, _ = rare_synth(table, 6000, s, han_share=0.0)
        lp1 = (lp1 / 6).astype(np.float32).tolist()      # greedy and confident (the research Qwen3 data is greedy): rc3 still
        rows = [[a, a] for a in lp1]                     # scores every gated token, it just rarely fires
        data.append((tok.tolist(), rows))
    n = sum(len(t) for t, _ in data)
    out, alerts = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, m4, rare, with_lp in (('m4', 'shadow', 'off', False), ('m4_values_ignored', 'shadow', 'off', True),
                                        ('rc3_collect', 'off', 'collect', True),
                                        ('rc3', 'off', 'shadow', True), ('rc3_buffering_only', 'off', 'shadow', True),
                                        ('m4_rc3', 'shadow', 'shadow', True),
                                        ('m4_rc3_no_values', 'shadow', 'shadow', False),
                                        ('rc4', 'off', 'shadow', True), ('m4_rc4', 'shadow', 'shadow', True),
                                        ('rc1t', 'off', 'shadow', False), ('m4_rc1t', 'shadow', 'shadow', False)):
            svc = hooks.Service(hooks.settings({'V35_DETECTOR': m4, 'V35_RARE': rare, 'V35_DETECTOR_TABLE': table,
                                                'V35_RARE_CONFIG': 'rc1t' if 'rc1t' in name else 'rc4' if 'rc4' in name else 'rc3',
                                                'V35_DETECTOR_EVENTS': os.path.join(tmp, name + '.jsonl')}))
            if name == 'rc3_buffering_only':             # the hook's own work: the detector is replaced by a no-op
                svc.register('-')
                svc.abort('-')
                svc.runner.rare_start = lambda rid: None
                svc.runner.rare_feed = lambda rid, ids, top1, chosen: ((), False)
                svc.runner.rare_finish = lambda rid, ids, top1, chosen: ()
            best = None
            for r in range(rounds):
                t0 = time.perf_counter()
                for j, (tok, rows) in enumerate(data):
                    rid = f'{r}-{j}'
                    svc.register(rid)
                    for k, t in enumerate(tok):
                        svc.feed(rid, [t], rows[k] if with_lp else None)
                    svc.finish(rid, 'length')
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            svc.close()
            out[name] = 1e6 * best / n
            alerts[name] = {k: v for k, v in svc.n.items() if k in ('alerts', 'rare_strict', 'rare_monitor')}
    return dict(hooks_us_per_token=out, hooks_tokens=n, hooks_requests=len(data) * rounds, hooks_alerts=alerts)


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
    out.update(bench_rare(a.table, streams, a.rounds))
    out.update(bench_rare(a.table, streams, a.rounds, 'rc4'))
    out.update(bench_rare(a.table, streams, a.rounds, 'rc1t'))
    out.update(bench_hooks(a.table, a.rounds))
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
