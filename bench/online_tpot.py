#!/usr/bin/env python3
"""Online TPOT, CPU and memory of the v35 detectors inside vLLM: m4 (repetition) next to vLLM's own repetition check, and the
garbled-output monitor rc3 next to the cost of asking vLLM for logprobs at all.

Slimmed from the previous online tests (v32_online/online_tpot_run.py): one vLLM server per profile, started and stopped
by this script; every request generates exactly the requested number of tokens (ignore_eos), so all profiles do the same
work; the detectors run in shadow mode (they detect and log, never stop a request).

Profiles (default: the r2 set, marked *; r1 ran baseline, vllm_*, v35_*, baseline_end)
  baseline, baseline_end *  detectors off, no logprobs; run first and last to measure the drift of the session
  baseline_logprobs *       every request asks for logprobs=1, detectors off: the cost of logprobs itself (GPU sampler,
                            engine-to-frontend transfer, the frontend's logprob objects and their JSON in the response)
  rc3_shadow_sync *         logprobs=1, V35_RARE=shadow (m4 off), sync runner (inside the output-processing loop)
  rc3_shadow_async *        logprobs=1, V35_RARE=shadow (m4 off), async runner (a separate detector process)
  m4_rc3_sync *             logprobs=1, V35_DETECTOR=shadow and V35_RARE=shadow, sync runner
  m4_rc3_async *            logprobs=1, V35_DETECTOR=shadow and V35_RARE=shadow, async runner
  vllm_default              vLLM's check per request, repetition_detection [50,500]x3 (vLLM's default sizes)
  vllm_tuned                vLLM's check per request, [32,2000]x3 (tuned on dev under the same 0.1% cap)
  v35_sync                  V35_DETECTOR=shadow, sync runner (no logprobs)
  v35_async                 V35_DETECTOR=shadow, async runner (no logprobs)
vLLM's check stops a request when it finds a loop; such requests are counted (early_stops) and their tokens kept.
The rc3 increment is a logprobs profile against baseline_logprobs; the cost of logprobs is baseline_logprobs against
baseline (bench/summarize_tpot.py prints both).

Per (profile, output length, concurrency), over timed waves after warm-up waves:
  server TPOT (vllm:request_time_per_output_token_seconds) and ITL (vllm:inter_token_latency_seconds), /metrics deltas;
  client TPOT from SSE chunk arrivals; throughput; CPU seconds of every server process during the wave;
  memory (RSS / PSS / USS) of every server process when ready and after each output length;
  v35 coverage from its events file: one 'finish' per request with all its tokens, no 'error' event, and with rc3 on,
  every request checked by rc3 (finish 'rare': 'checked'); logprob profiles: a logprob for every streamed token.

  python bench/online_tpot.py --output runs/tpot_qwen3_h20 --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5
Needs: the v35 patch applied to this environment's vLLM (integrations/vllm_0_27), `pip install -e <repo>`, aiohttp, psutil,
transformers. --server-cmd replaces `vllm serve` (bench/fake_vllm_server.py runs the whole protocol without a GPU).
"""
import argparse
import asyncio
import json
import os
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import aiohttp
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_tpot_metrics import fetch_metrics, metric_total, summarize_metrics_delta  # noqa: E402

REQUEST_TPOT = 'vllm:request_time_per_output_token_seconds'
ITL = 'vllm:inter_token_latency_seconds'
PREEMPTIONS = 'vllm:num_preemptions_total'
SAMPLING = dict(temperature=.6, top_k=20, top_p=.95, min_p=0., repetition_penalty=1., frequency_penalty=0., presence_penalty=0.)
PROFILES = {                                             # name: (server environment, vLLM repetition check, logprobs)
    'baseline': ({'V35_DETECTOR': 'off'}, None, None),
    'baseline_logprobs': ({'V35_DETECTOR': 'off'}, None, 1),
    'rc3_shadow_sync': ({'V35_DETECTOR': 'off', 'V35_RARE': 'shadow', 'V35_DETECTOR_RUNNER': 'sync'}, None, 1),
    'rc3_shadow_async': ({'V35_DETECTOR': 'off', 'V35_RARE': 'shadow', 'V35_DETECTOR_RUNNER': 'async'}, None, 1),
    'm4_rc3_sync': ({'V35_DETECTOR': 'shadow', 'V35_RARE': 'shadow', 'V35_DETECTOR_RUNNER': 'sync'}, None, 1),
    'm4_rc3_async': ({'V35_DETECTOR': 'shadow', 'V35_RARE': 'shadow', 'V35_DETECTOR_RUNNER': 'async'}, None, 1),
    'baseline_end': ({'V35_DETECTOR': 'off'}, None, None),
    'vllm_default': ({'V35_DETECTOR': 'off'}, {'min_pattern_size': 50, 'max_pattern_size': 500, 'min_count': 3}, None),
    'vllm_tuned': ({'V35_DETECTOR': 'off'}, {'min_pattern_size': 32, 'max_pattern_size': 2000, 'min_count': 3}, None),
    'v35_sync': ({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_RUNNER': 'sync'}, None, None),
    'v35_async': ({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_RUNNER': 'async'}, None, None),
}
R2_PROFILES = ['baseline', 'baseline_logprobs', 'rc3_shadow_sync', 'rc3_shadow_async', 'm4_rc3_sync', 'm4_rc3_async',
               'baseline_end']
NOTES = ('A database keeps a write-ahead log so committed changes can be recovered after a crash. Replication improves '
         'availability, but network latency and consistency requirements affect throughput. A cache reduces repeated reads. '
         'Measurements should distinguish warm caches, cold caches, request concurrency, queueing time, and memory usage.\n')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def prompt_tokens(tokenizer, length, index):
    """a fixed-length prompt (token ids), different per request index (same construction as the previous tests)"""
    if tokenizer is None:                                # --synthetic-prompts: ids only, for the fake server
        return [(index * 7919 + 13 * i) % 30000 + 100 for i in range(length)]
    prefix = tokenizer.encode(f'Task {index}: Read the reference notes and explain the engineering tradeoffs.\n',
                              add_special_tokens=False)
    notes = tokenizer.encode(NOTES, add_special_tokens=False)
    suffix = tokenizer.encode('\nUsing these notes, write a detailed technical explanation with examples and a numbered '
                              'checklist.\nAnswer:', add_special_tokens=False)
    count = length - len(prefix) - len(suffix)
    return prefix + (notes * (count // len(notes) + 1))[:count] + suffix


async def request(session, url, prompt, length, index, rep, logprobs=None):
    payload = dict(model='bench', prompt=prompt, max_tokens=length, seed=17 + index, ignore_eos=True, stream=True,
                   stream_options={'include_usage': True}, return_token_ids=True, **SAMPLING)
    if rep is not None:
        payload['repetition_detection'] = rep
    if logprobs is not None:
        payload['logprobs'] = logprobs
    start, arrivals, ids, usage, finish, stop_reason = time.perf_counter(), [], [], None, None, None
    lp_positions = 0
    async with session.post(url + '/v1/completions', json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:2000]}')
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            obj = json.loads(line[6:])
            if 'error' in obj:
                raise RuntimeError(str(obj['error']))
            usage = obj.get('usage') or usage
            for choice in obj.get('choices', []):
                toks = choice.get('token_ids') or []
                if toks:
                    arrivals.append((time.perf_counter() - start, len(toks)))
                    ids.extend(toks)
                lp = choice.get('logprobs')
                if lp:
                    lp_positions += len(lp.get('token_logprobs') or [])
                finish = choice.get('finish_reason') or finish
                stop_reason = choice.get('stop_reason') or stop_reason
    n = len(ids)
    if not usage or usage.get('completion_tokens') != n or usage.get('prompt_tokens') != len(prompt):
        raise RuntimeError(f'token accounting: {n} streamed, usage {usage}')
    if n != length and rep is None:
        raise RuntimeError(f'request {index} produced {n} tokens, expected {length} (ignore_eos)')
    if logprobs is not None and lp_positions != n:
        raise RuntimeError(f'request {index}: logprobs for {lp_positions} of {n} tokens (logprobs={logprobs} requested)')
    first, last = arrivals[0][0], arrivals[-1][0]
    return dict(index=index, output_tokens=n, early_stop=n != length, finish_reason=finish, stop_reason=stop_reason,
                ttft_seconds=first, client_tpot_seconds=(last - first) / (n - 1) if n > 1 else None,
                chunks=len(arrivals), multi_token_chunks=sum(k > 1 for _, k in arrivals), logprob_positions=lp_positions)


async def wave(url, prompts, length, indices, rep, logprobs=None):
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=len(prompts))) as session:
        return await asyncio.gather(*(request(session, url, p, length, i, rep, logprobs) for p, i in zip(prompts, indices)))


def processes(pid):
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def cpu_seconds(pid):
    out = {}
    for p in processes(pid):
        try:
            t = p.cpu_times()
            out[p.pid] = t.user + t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def memory(pid):
    """RSS / PSS / USS per server process (PSS and USS need /proc/<pid>/smaps: Linux; elsewhere RSS only)"""
    out = {}
    for p in processes(pid):
        try:
            try:
                m = p.memory_full_info()
            except psutil.AccessDenied:
                m = p.memory_info()
            out[str(p.pid)] = dict(name=p.name(), cmdline=' '.join(p.cmdline())[:300], rss=m.rss,
                                   pss=getattr(m, 'pss', None), uss=getattr(m, 'uss', None))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def wait_ready(proc, url, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited with {proc.returncode} (see server.log)')
        try:
            with urllib.request.urlopen(url + '/health', timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError('server start-up timed out')


def stop_server(proc):
    children = processes(proc.pid)[1:]
    if proc.poll() is None:
        proc.terminate()                                 # graceful: the detector writes its summary at exit
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    _, alive = psutil.wait_procs(children, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def wait_metric_count(url, expected, timeout=60):
    start = time.monotonic()
    while True:
        text = fetch_metrics(url)
        count = metric_total(text, REQUEST_TPOT + '_count')
        if count == expected:
            return text
        if count is not None and count > expected:
            raise RuntimeError(f'server request count {count}, expected {expected}')
        if time.monotonic() - start > timeout:
            raise RuntimeError(f'metric flush timeout: {count}, expected {expected}')
        time.sleep(.2)


def read_events(path, offset):
    if not path.exists():
        return [], offset
    with open(path, encoding='utf-8') as f:
        f.seek(offset)
        data = f.read()
    lines = data.split('\n')
    complete = lines[:-1]                                # a partially written last line waits for the next read
    return [json.loads(x) for x in complete if x], offset + len(data) - len(lines[-1])


def wait_coverage(path, offset, responses, rare=False, timeout=20):
    """v35 events of one wave: a 'finish' per request with all its tokens, no 'error', and with rc3 on, every request
    checked by rc3"""
    want = sorted(r['output_tokens'] for r in responses)
    events, deadline = [], time.monotonic() + timeout
    while True:
        new, offset = read_events(path, offset)
        events += new
        finished = sorted(e['tokens'] for e in events if e['event'] == 'finish')
        if len(finished) >= len(want) or time.monotonic() > deadline:
            break
        time.sleep(.2)
    errors = [e for e in events if e['event'] == 'error']
    rare_status = Counter(e.get('rare') for e in events if e['event'] == 'finish') if rare else None
    rare_ok = not rare or set(rare_status) == {'checked'}
    return dict(ok=finished == want and not errors and rare_ok, finished_tokens=finished, errors=errors[:5],
                alerts=sum(e['event'] == 'alert' for e in events), rare_status=rare_status,
                rare_alerts=Counter(e['tier'] for e in events if e['event'] == 'rare_alert')), offset


def summarize(report):
    rows = []
    for (profile, length, conc), items in group(report['measurements']).items():
        resp = [r for it in items for r in it['responses']]
        wall = sum(it['wall_seconds'] for it in items)
        row = dict(profile=profile, output_length=length, concurrency=conc, requests=len(resp),
                   server_tpot_ms=1e3 * sum(it['server_tpot_sum'] for it in items) / sum(it['server_tpot_count'] for it in items),
                   server_itl_ms=1e3 * sum(it['server_itl_sum'] for it in items) / sum(it['server_itl_count'] for it in items),
                   client_tpot_ms=1e3 * statistics.mean(r['client_tpot_seconds'] for r in resp),
                   tokens_per_second=sum(r['output_tokens'] for r in resp) / wall,
                   server_cpu_cores=sum(it['cpu_seconds'] for it in items) / wall,
                   early_stops=sum(r['early_stop'] for r in resp),
                   preemptions=sum(it.get('preemptions') or 0 for it in items),
                   v35_coverage_ok=all(it['v35'] is None or it['v35']['ok'] for it in items),
                   logprobs=PROFILES[profile][2], api_server_cpu_cores=sum(it.get('api_server_cpu_seconds') or 0
                                                                          for it in items) / wall)
        rows.append(row)
    for row in rows:
        for base in ('baseline', 'baseline_end', 'baseline_logprobs'):
            b = next((r for r in rows if r['profile'] == base and r['output_length'] == row['output_length']
                      and r['concurrency'] == row['concurrency']), None)
            if b:
                row[f'server_tpot_delta_us_vs_{base}'] = 1e3 * (row['server_tpot_ms'] - b['server_tpot_ms'])
                row[f'throughput_loss_pct_vs_{base}'] = 100 * (1 - row['tokens_per_second'] / b['tokens_per_second'])
    return rows


def group(items):
    out = {}
    for it in items:
        out.setdefault((it['profile'], it['output_length'], it['concurrency']), []).append(it)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--table', default='qwen3', help='V35_DETECTOR_TABLE for the v35 profiles')
    ap.add_argument('--think-end', default='auto', help='V35_DETECTOR_THINK_END')
    ap.add_argument('--profiles', nargs='+', default=R2_PROFILES, choices=list(PROFILES))
    ap.add_argument('--concurrencies', type=int, nargs='+', default=[1, 8, 32, 64])
    ap.add_argument('--output-lengths', type=int, nargs='+', default=[1024])
    ap.add_argument('--input-length', type=int, default=1024)
    ap.add_argument('--waves', type=int, default=2)
    ap.add_argument('--warmup-waves', type=int, default=1)
    ap.add_argument('--max-num-seqs', type=int)
    ap.add_argument('--max-model-len', type=int, default=10240)
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES')
    ap.add_argument('--port', type=int, default=18790)
    ap.add_argument('--serve-args', default='', help='extra arguments for vllm serve (quoted)')
    ap.add_argument('--server-cmd', help='replaces "vllm serve MODEL ..." (e.g. "python bench/fake_vllm_server.py")')
    ap.add_argument('--synthetic-prompts', action='store_true', help='token-id prompts without a tokenizer (fake server)')
    a = ap.parse_args()
    if (a.output / 'results.json').exists():
        ap.error('results.json exists: choose a new --output directory')
    if a.input_length + max(a.output_lengths) > a.max_model_len:
        ap.error('--max-model-len must hold --input-length + the longest --output-lengths')
    max_seqs = a.max_num_seqs or max(a.concurrencies)
    a.output.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if not a.synthetic_prompts:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if a.server_cmd:
        base_cmd = shlex.split(a.server_cmd) + ['--port', str(a.port)]
    else:
        base_cmd = [str(Path(sys.executable).with_name('vllm')), 'serve', a.model, '--served-model-name', 'bench',
                    '--host', '127.0.0.1', '--port', str(a.port), '--tensor-parallel-size', str(a.tensor_parallel_size),
                    '--max-model-len', str(a.max_model_len), '--max-num-seqs', str(max_seqs),
                    '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '.85', '--no-enable-prefix-caching']
    cmd = base_cmd + shlex.split(a.serve_args)
    versions = {}
    for pkg in ('vllm', 'torch', 'numpy', 'v35-detector', 'aiohttp', 'psutil'):
        try:
            import importlib.metadata as md
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = None
    report = dict(status='running', started_at=now(), arguments={k: str(v) for k, v in vars(a).items()}, server_command=cmd,
                  sampling=SAMPLING, environment=dict(python=sys.version, platform=platform.platform(), packages=versions),
                  memory={}, server_pids={}, v35_summaries={}, measurements=[], warmups=[], summary=[])
    target = a.output / 'results.json'

    def save():
        report['summary'] = summarize(report)
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
        tmp.replace(target)

    url = f'http://127.0.0.1:{a.port}'
    save()
    try:
        for profile in a.profiles:
            env_extra, rep, logprobs = PROFILES[profile]
            folder = a.output / profile
            folder.mkdir()
            events = folder / 'v35_events.jsonl'
            env = {k: v for k, v in os.environ.items() if not k.startswith(('V35_', 'VLLM_STABILITY_'))}
            env.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='1', **env_extra)
            rare = env_extra.get('V35_RARE', 'off') != 'off'
            detecting = env_extra['V35_DETECTOR'] != 'off' or rare
            if detecting:
                env.update(V35_DETECTOR_TABLE=a.table, V35_DETECTOR_THINK_END=a.think_end, V35_DETECTOR_EVENTS=str(events.resolve()))
            print(now(), 'starting', profile, flush=True)
            with open(folder / 'server.log', 'w') as log:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                wait_ready(proc, url)
                report['server_pids'][profile] = proc.pid       # the API server; EngineCore and the detector are children
                report['memory'][profile] = {'ready': memory(proc.pid)}
                expected = metric_total(fetch_metrics(url), REQUEST_TPOT + '_count') or 0
                if expected:
                    raise RuntimeError(f'fresh server already counts {expected} requests')
                offset = 0
                for length in a.output_lengths:
                    for conc in a.concurrencies:
                        for wi in range(-a.warmup_waves, a.waves):
                            warm = wi < 0
                            indices = [100000 * length + 1000 * conc + 100 * (wi + a.warmup_waves) + i for i in range(conc)]
                            prompts = [prompt_tokens(tokenizer, a.input_length, i) for i in indices]
                            before = wait_metric_count(url, expected) if expected else fetch_metrics(url)
                            cpu0, t0 = cpu_seconds(proc.pid), time.perf_counter()
                            responses = asyncio.run(wave(url, prompts, length, indices, rep, logprobs))
                            wall = time.perf_counter() - t0
                            cpu1 = cpu_seconds(proc.pid)
                            expected += conc
                            after = wait_metric_count(url, expected)
                            delta = summarize_metrics_delta(before, after)
                            if not warm and not delta.get('valid'):
                                raise RuntimeError(f'invalid metric delta: {delta["errors"][:3]}')
                            d = {m + p: metric_total(after, m + p) - (metric_total(before, m + p) or 0)
                                 for m in (REQUEST_TPOT, ITL) for p in ('_sum', '_count')}
                            # preempted requests (KV cache too small for concurrency x length) distort TPOT
                            preempt = ((metric_total(after, PREEMPTIONS) or 0) - (metric_total(before, PREEMPTIONS) or 0)
                                       if metric_total(after, PREEMPTIONS) is not None else None)
                            cov = None
                            if detecting:
                                cov, offset = wait_coverage(events, offset, responses, rare)
                            item = dict(profile=profile, output_length=length, concurrency=conc, wave=wi, started_at=now(),
                                        wall_seconds=wall, responses=responses,
                                        server_tpot_sum=d[REQUEST_TPOT + '_sum'], server_tpot_count=d[REQUEST_TPOT + '_count'],
                                        server_itl_sum=d[ITL + '_sum'], server_itl_count=d[ITL + '_count'],
                                        cpu_seconds=sum(cpu1.get(p, 0) - cpu0.get(p, 0) for p in cpu1),
                                        cpu_seconds_by_pid={str(p): cpu1[p] - cpu0.get(p, 0) for p in cpu1}, v35=cov,
                                        api_server_cpu_seconds=cpu1.get(proc.pid, 0) - cpu0.get(proc.pid, 0),
                                        preemptions=preempt, logprobs=logprobs)
                            report['warmups' if warm else 'measurements'].append(item)
                            save()
                            tpot = 1e3 * d[REQUEST_TPOT + '_sum'] / max(d[REQUEST_TPOT + '_count'], 1)
                            print(now(), profile, f'len={length} c={conc} wave={wi} server_tpot={tpot:.4f}ms '
                                  f'early_stops={sum(r["early_stop"] for r in responses)} v35={cov and cov["ok"]} '
                                  f'preemptions={preempt}' + (f' rc3={dict(cov["rare_status"])} rc3_alerts='
                                                              f'{dict(cov["rare_alerts"])}' if cov and rare else ''),
                                  flush=True)
                            if cov is not None and not cov['ok'] and not warm:
                                raise RuntimeError(f'v35 coverage failed: {cov}')
                    report['memory'][profile][f'after_len{length}'] = memory(proc.pid)
                    save()
            finally:
                stop_server(proc)
            if events.exists():
                summaries = [e for e in read_events(events, 0)[0] if e['event'] == 'summary']
                report['v35_summaries'][profile] = summaries[-1] if summaries else None
            save()
        report.update(status='complete', finished_at=now())
        save()
        print('COMPLETE', target, flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), finished_at=now())
        save()
        raise


if __name__ == '__main__':
    main()
