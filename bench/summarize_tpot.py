#!/usr/bin/env python3
"""Plain-text summary of online_tpot.py results, short enough to paste into a message.

Per output length and concurrency: server TPOT of the baseline, the session drift (baseline_end - baseline), and every
other profile's TPOT change against the baseline (us/token); throughput change; server CPU; validity flags (preemptions,
early stops, v35 coverage). r3 (collect_only ran): the collection cost (collect_only - baseline: sampler kernels and
transport), the rc3 increment of each rc3 profile (profile - collect_only), and vLLM's own logprobs path, r2's way of
getting the same inputs (baseline_logprobs - baseline), next to the drift, for server / client TPOT, throughput and CPU.
r2 results (rc3_shadow_* profiles) get r2's tables. Then each detector's own time per token and the memory of the server
processes.

  python bench/summarize_tpot.py runs/tpot_qwen3_1k/results.json [more results.json ...]
"""
import json
import sys

ORDER = ['baseline', 'collect_only', 'rc3_sync', 'rc3_async', 'baseline_logprobs', 'rc3_shadow_sync', 'rc3_shadow_async',
         'm4_rc3_sync', 'm4_rc3_async', 'vllm_default', 'vllm_tuned', 'v35_sync', 'v35_async', 'baseline_end']
RC3 = ['rc3_shadow_sync', 'rc3_shadow_async', 'm4_rc3_sync', 'm4_rc3_async']      # r2
RC3_R3 = ['rc3_sync', 'rc3_async', 'm4_rc3_sync', 'm4_rc3_async']
W = 19


def mb(x):
    return f'{x / 2 ** 20:.0f}' if x is not None else '-'


def proc_mem(snapshot):
    """API server (the root process), the v35 detector process (named v35-detector by setproctitle; without it, the
    multiprocessing spawn child: EngineCore and vLLM's workers name themselves VLLM::...), and the whole tree; PSS where
    available, else RSS"""
    if not snapshot:
        return None
    rows = list(snapshot.values())
    val = [r['pss'] if r.get('pss') is not None else r['rss'] for r in rows]
    root = val[0]                                        # memory() lists the root process first
    det = sum(v for r, v in zip(rows, val) if any(k in (r.get('cmdline') or '') + r.get('name', '')
                                                  for k in ('v35-detector', 'multiprocessing.spawn import spawn_main')))
    return root, det, sum(val), rows[0].get('pss') is not None


def table(title, rows, profiles, cell):
    lines = [title, 'len    C  ' + ''.join(f'{p:>{W}s}' for p in profiles)]
    for (length, conc), by_prof in sorted(rows.items()):
        lines.append(f'{length:<5d}{conc:>4d}  ' + ''.join(f'{cell(by_prof.get(p), by_prof):>{W}s}' for p in profiles))
    return lines


def detector_lines(prof, s):
    if not s:
        return [f'{prof}: no v35 summary (server not stopped cleanly?)']
    c, w, out = s['counters'], s.get('worker'), []
    if s.get('mode', 'shadow') != 'off':
        if w:
            out.append(f'{prof}: m4 {1e6 * w["detector_seconds"] / max(w["tokens"], 1):.2f} us/token in the detector process '
                       f'({w["tokens"]} tokens)')
        else:
            out.append(f'{prof}: m4 {1e6 * s["detector_seconds_sync"] / max(c.get("tokens", 0), 1):.2f} us/token in the '
                       f'output loop ({c.get("tokens", 0)} tokens)')
    if s.get('rare', 'off') == 'collect':
        out.append(f'{prof}: rc3 collect only: sampler values for {c.get("rare_values", 0)} tokens, '
                   f'{c.get("rare_requests", 0)} requests')
    elif s.get('rare', 'off') != 'off':
        if w:
            out.append(f'{prof}: rc3 {1e6 * w.get("rare_seconds", 0) / max(w.get("rare_tokens", 0), 1):.2f} us/token in the '
                       f'detector process ({w.get("rare_tokens", 0)} tokens)')
        else:
            out.append(f'{prof}: rc3 {1e6 * s.get("rare_seconds_sync", 0) / max(c.get("rare_tokens", 0), 1):.2f} us/token in '
                       f'the output loop ({c.get("rare_tokens", 0)} tokens)')
    out.append(f'{prof}: counters {c}')
    return out


def summarize(path):
    d = json.load(open(path))
    out = [f'== {path}  status={d["status"]}  vllm={d["environment"]["packages"].get("vllm")}  '
           f'v35-detector={d["environment"]["packages"].get("v35-detector")} ==']
    rows = {}
    for r in d['summary']:
        rows.setdefault((r['output_length'], r['concurrency']), {})[r['profile']] = r
    profiles = [p for p in ORDER if any(p in v for v in rows.values())]

    def tpot(r, by):
        if r is None:
            return '-'
        if r['profile'] == 'baseline':
            return f'{r["server_tpot_ms"]:.3f}ms'
        return f'{r.get("server_tpot_delta_us_vs_baseline", float("nan")):+.1f}us'

    def thr(r, by):
        return '-' if r is None else f'{r.get("throughput_loss_pct_vs_baseline", float("nan")):+.2f}%'

    def cpu(r, by):
        if r is None:
            return '-'
        return f'{r["server_cpu_cores"]:.2f}/{r.get("api_server_cpu_cores", float("nan")):.2f}'

    def flags(r, by):
        if r is None:
            return '-'
        bad = []
        if r.get('preemptions'):
            bad.append(f'pre{r["preemptions"]}')
        if r.get('early_stops'):
            bad.append(f'stop{r["early_stops"]}')
        if not r.get('v35_coverage_ok', True):
            bad.append('COV!')
        return ','.join(bad) or 'ok'

    if 'collect_only' in profiles:
        out += r3_tables(rows, [p for p in RC3_R3 if p in profiles])
    elif 'baseline_logprobs' in profiles:
        out += r2_tables(rows, [p for p in RC3 if p in profiles])
    out += table('server TPOT: baseline absolute; others = change vs baseline (baseline_end = session drift)', rows,
                 profiles, tpot)
    out += table('throughput change vs baseline (positive = slower)', rows, profiles, thr)
    out += table('server CPU (cores during the timed waves): whole process tree / API server process', rows, profiles, cpu)
    out += table('validity: preemptions / early stops (vLLM check) / v35 coverage (incl. rc3: every request checked)', rows,
                 profiles, flags)
    for prof, s in (d.get('v35_summaries') or {}).items():
        out += detector_lines(prof, s)
    out.append('memory (MB; PSS where available, else RSS): API server process / v35-detector process / whole tree')
    for prof in profiles:
        snaps = (d.get('memory') or {}).get(prof) or {}
        for key in sorted(snaps):
            m = proc_mem(snaps[key])
            if m:
                out.append(f'  {prof:<19s}{key:<15s}{mb(m[0]):>8s}{mb(m[1]):>8s}{mb(m[2]):>8s}{"" if m[3] else "  (RSS)"}')
    return '\n'.join(out)


def r3_tables(rows, rc3):
    """r3: what collecting rc3's inputs in the sampler costs, what rc3 adds on top, and r2's way (vLLM's logprobs path) for
    comparison, all next to the drift"""
    out = []
    cols = ['drift', 'collect'] + [f'+{p}' for p in rc3] + ['logprobs_path']

    def diff(a, b, key):
        return None if a is None or b is None else a[key] - b[key]

    for title, key, unit, base_fmt, fmt, scale in (
            ('r3: server TPOT; drift = baseline_end - baseline; collect = collect_only - baseline (sampler kernels + '
             'transport); +profile = profile - collect_only (rc3 increment); logprobs_path = baseline_logprobs - baseline '
             "(r2's way)", 'server_tpot_ms', 'us/token', lambda r: f'{r["server_tpot_ms"]:.3f}ms', lambda v: f'{v:+.1f}',
             1e3),
            ('r3: client TPOT, same differences', 'client_tpot_ms', 'us/token', lambda r: f'{r["client_tpot_ms"]:.3f}ms',
             lambda v: f'{v:+.1f}', 1e3),
            ('r3: throughput, same differences', 'tokens_per_second', 'tokens/s', lambda r: f'{r["tokens_per_second"]:.1f}',
             lambda v: f'{v:+.1f}', 1.0),
            ('r3: server CPU (whole tree), same differences', 'server_cpu_cores', 'cores',
             lambda r: f'{r["server_cpu_cores"]:.2f}', lambda v: f'{v:+.3f}', 1.0),
            ('r3: API server process CPU, same differences', 'api_server_cpu_cores', 'cores',
             lambda r: f'{r["api_server_cpu_cores"]:.2f}', lambda v: f'{v:+.3f}', 1.0)):
        out += [title, f'len    C  {"baseline":>{W}s}' + ''.join(f'{c:>{W}s}' for c in cols) + f'   ({unit})']
        for (length, conc), by in sorted(rows.items()):
            b, co = by.get('baseline'), by.get('collect_only')
            vals = ([diff(by.get('baseline_end'), b, key), diff(co, b, key)] + [diff(by.get(p), co, key) for p in rc3]
                    + [diff(by.get('baseline_logprobs'), b, key)])
            out.append(f'{length:<5d}{conc:>4d}  {"-" if b is None else base_fmt(b):>{W}s}' +
                       ''.join(f'{"-" if v is None else fmt(v * scale):>{W}s}' for v in vals))
        if key == 'server_tpot_ms':
            for name, prof in (('collect', 'collect_only'), ('logprobs_path', 'baseline_logprobs')):
                out.append(f'  {name} in % of the baseline TPOT: ' + ', '.join(
                    f'len {length} C {conc}: {100 * (by[prof][key] / by["baseline"][key] - 1):+.2f}%'
                    for (length, conc), by in sorted(rows.items()) if 'baseline' in by and prof in by))
    return out


def r2_tables(rows, rc3):
    """the two questions of r2: what logprobs=1 costs by itself, and what rc3 adds on top of it (both next to the drift)"""
    out = []
    cols = ['drift', 'logprobs_cost'] + [f'+{p}' for p in rc3]

    def head(title, unit):
        return [title, f'len    C  {"baseline":>{W}s}' + ''.join(f'{c:>{W}s}' for c in cols) + f'   ({unit})']

    def line(length, conc, base, vals, fmt):
        return f'{length:<5d}{conc:>4d}  {base:>{W}s}' + ''.join(f'{"-" if v is None else fmt(v):>{W}s}' for v in vals)

    def diff(a, b, key):
        return None if a is None or b is None else a[key] - b[key]

    for title, key, unit, base_fmt, fmt, scale in (
            ('r2: server TPOT; logprobs_cost = baseline_logprobs - baseline; +profile = profile - baseline_logprobs '
             '(rc3 increment); drift = baseline_end - baseline', 'server_tpot_ms', 'us/token',
             lambda r: f'{r["server_tpot_ms"]:.3f}ms', lambda v: f'{v:+.1f}', 1e3),
            ('r2: client TPOT, same differences', 'client_tpot_ms', 'us/token', lambda r: f'{r["client_tpot_ms"]:.3f}ms',
             lambda v: f'{v:+.1f}', 1e3),
            ('r2: throughput, same differences', 'tokens_per_second', 'tokens/s', lambda r: f'{r["tokens_per_second"]:.1f}',
             lambda v: f'{v:+.1f}', 1.0),
            ('r2: server CPU (whole tree), same differences', 'server_cpu_cores', 'cores',
             lambda r: f'{r["server_cpu_cores"]:.2f}', lambda v: f'{v:+.3f}', 1.0),
            ('r2: API server process CPU, same differences', 'api_server_cpu_cores', 'cores',
             lambda r: f'{r["api_server_cpu_cores"]:.2f}', lambda v: f'{v:+.3f}', 1.0)):
        out += head(title, unit)
        for (length, conc), by in sorted(rows.items()):
            b, bl = by.get('baseline'), by.get('baseline_logprobs')
            vals = [diff(by.get('baseline_end'), b, key), diff(bl, b, key)] + [diff(by.get(p), bl, key) for p in rc3]
            vals = [None if v is None else v * scale for v in vals]
            out.append(line(length, conc, '-' if b is None else base_fmt(b), vals, fmt))
        if key == 'server_tpot_ms':
            out.append('  logprobs_cost in % of the baseline TPOT: ' + ', '.join(
                f'len {length} C {conc}: {100 * (by["baseline_logprobs"][key] / by["baseline"][key] - 1):+.2f}%'
                for (length, conc), by in sorted(rows.items()) if 'baseline' in by and 'baseline_logprobs' in by))
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print('\n\n'.join(summarize(p) for p in sys.argv[1:]))
