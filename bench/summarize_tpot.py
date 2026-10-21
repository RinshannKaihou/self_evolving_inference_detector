#!/usr/bin/env python3
"""Plain-text summary of online_tpot.py results, short enough to paste into a message.

Per output length and concurrency: server TPOT of the baseline, the session drift (baseline_end - baseline), and every
other profile's TPOT change against the baseline (us/token); throughput change; server CPU; validity flags (preemptions,
early stops, v35 coverage). Then the detector's own time per token and the memory of the server processes.

  python bench/summarize_tpot.py runs/tpot_qwen3_1k/results.json [more results.json ...]
"""
import json
import sys

ORDER = ['baseline', 'vllm_default', 'vllm_tuned', 'v35_sync', 'v35_async', 'baseline_end']


def mb(x):
    return f'{x / 2 ** 20:.0f}' if x is not None else '-'


def proc_mem(snapshot):
    """API server (the root process), the v35 detector process, and the whole tree; PSS where available, else RSS"""
    if not snapshot:
        return None
    rows = list(snapshot.values())
    val = [r['pss'] if r.get('pss') is not None else r['rss'] for r in rows]
    root = val[0]                                        # memory() lists the root process first
    det = sum(v for r, v in zip(rows, val) if 'v35-detector' in (r.get('cmdline') or '') + r.get('name', ''))
    return root, det, sum(val), rows[0].get('pss') is not None


def table(title, rows, profiles, cell):
    lines = [title, 'len    C  ' + ''.join(f'{p:>14s}' for p in profiles)]
    for (length, conc), by_prof in sorted(rows.items()):
        lines.append(f'{length:<5d}{conc:>4d}  ' + ''.join(f'{cell(by_prof.get(p), by_prof):>14s}' for p in profiles))
    return lines


def summarize(path):
    d = json.load(open(path))
    out = [f'== {path}  status={d["status"]}  vllm={d["environment"]["packages"].get("vllm")} ==']
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
        return '-' if r is None else f'{r["server_cpu_cores"]:.2f}'

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

    out += table('server TPOT: baseline absolute; others = change vs baseline (baseline_end = session drift)', rows,
                 profiles, tpot)
    out += table('throughput change vs baseline (positive = slower)', rows, profiles, thr)
    out += table('server CPU (cores, whole process tree, during the timed waves)', rows, profiles, cpu)
    out += table('validity: preemptions / early stops (vLLM check) / v35 coverage', rows, profiles, flags)
    for prof, s in (d.get('v35_summaries') or {}).items():
        if not s:
            out.append(f'{prof}: no v35 summary (server not stopped cleanly?)')
            continue
        if s.get('worker'):
            w = s['worker']
            out.append(f'{prof}: detector {1e6 * w["detector_seconds"] / max(w["tokens"], 1):.2f} us/token in the '
                       f'detector process ({w["tokens"]} tokens); counters {s["counters"]}')
        else:
            tok = s['counters'].get('tokens', 0)
            out.append(f'{prof}: detector {1e6 * s["detector_seconds_sync"] / max(tok, 1):.2f} us/token in the output loop '
                       f'({tok} tokens); counters {s["counters"]}')
    out.append('memory (MB; PSS where available, else RSS): API server process / v35-detector process / whole tree')
    for prof in profiles:
        snaps = (d.get('memory') or {}).get(prof) or {}
        for key in sorted(snaps):
            m = proc_mem(snaps[key])
            if m:
                out.append(f'  {prof:<13s}{key:<15s}{mb(m[0]):>8s}{mb(m[1]):>8s}{mb(m[2]):>8s}{"" if m[3] else "  (RSS)"}')
    return '\n'.join(out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print('\n\n'.join(summarize(p) for p in sys.argv[1:]))
