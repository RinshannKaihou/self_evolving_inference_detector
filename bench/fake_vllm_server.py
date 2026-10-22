#!/usr/bin/env python3
"""A stand-in for `vllm serve` that speaks the parts of its API that bench/online_tpot.py uses, so the whole benchmark
protocol (profiles, waves, /metrics deltas, coverage of the v35 events) can be exercised without a GPU. Its timings mean
nothing. It drives the v35 hooks exactly as the patched OutputProcessor does (register / feed per step / finish), from
the same environment variables, and applies vLLM's repetition check (upstream transcription) when a request asks for it.
Every token gets a synthetic top-1 logprob and generated-token logprob. As the patched engine does (r3), they reach the
feed hook as EngineCoreOutput.v35_rare ([top1, chosen]) for every request when V35_RARE is collect or shadow (read by
v35_detector.serving.collect, as the patched sampler reads it), whatever the request asks; a request with "logprobs": k
also gets the same values back in the completions format (top_logprobs keyed "token_id:N" with
"return_tokens_as_token_ids"), as vLLM returns its raw logprobs. "logit_bias" with a bias >= 50 makes that token the output
at every step (the rc3 smoke uses it to produce a lone UTF-8 continuation byte).

  python bench/fake_vllm_server.py --port 18790 [--step-ms 2]
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from v35_detector.serving import hooks  # noqa: E402  (reads V35_* at import, as in the patched vLLM)
from v35_detector.serving.collect import SAMPLER  # noqa: E402  (the engine side: collects when V35_RARE is on)

LABELS = '{engine="0",model_name="bench"}'
LN10 = 2.302585092994046
STATS = {k: 0.0 for k in ('tpot_sum', 'tpot_count', 'itl_sum', 'itl_count', 'ttft_sum', 'ttft_count')}


def vllm_check(ids, mn, mx, c):
    """vllm/v1/core/sched/utils.py check_sequence_repetition (v0.18.0), on the output so far"""
    for L in range(max(mn, 1), mx + 1):
        if L * c > len(ids):
            return False
        if all(ids[-(L * m + n)] == ids[-n] for n in range(1, L + 1) for m in range(1, c)):
            return True
    return False


async def health(_):
    return web.Response(text='ok')


async def metrics(_):
    lines = []
    for name, key in (('vllm:request_time_per_output_token_seconds', 'tpot'), ('vllm:inter_token_latency_seconds', 'itl'),
                      ('vllm:time_to_first_token_seconds', 'ttft')):
        lines += [f'# TYPE {name} histogram', f'{name}_sum{LABELS} {STATS[key + "_sum"]}',
                  f'{name}_count{LABELS} {STATS[key + "_count"]}']
    return web.Response(text='\n'.join(lines) + '\n')


async def completions(request):
    body = await request.json()
    n_max, prompt, rep = int(body['max_tokens']), body['prompt'], body.get('repetition_detection')
    k_lp = body.get('logprobs')
    as_ids = bool(body.get('return_tokens_as_token_ids'))
    forced = [int(t) for t, b in (body.get('logit_bias') or {}).items() if b >= 50]
    rng = np.random.default_rng(int(body.get('seed', 0)))
    rid = f'cmpl-{id(request)}-{time.monotonic_ns()}'
    resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
    await resp.prepare(request)
    hooks.register(rid)
    arrived, times, ids, finish, stop_reason = time.perf_counter(), [], [], 'length', None
    cycle = rng.integers(100, 150000, 40)                # --loop-after: the output turns into a 40-token loop
    for k in range(n_max):
        await asyncio.sleep(request.app['step'])        # one engine step
        loop_after = request.app['loop_after']
        tok = int(cycle[k % 40]) if loop_after is not None and k >= loop_after else int(rng.integers(100, 150000))
        if forced:
            tok = forced[0]
        ids.append(tok)
        times.append(time.perf_counter())
        # the raw distribution's top-1 logprob and the generated token's (float32, as the sampler computes them); the
        # generated token is the top-1 85% of the time
        top1 = float(np.float32(-float(rng.exponential(0.3)) * LN10))
        chosen = top1 if rng.random() < 0.85 else float(np.float32(top1 - float(rng.exponential(1.0)) * LN10))
        if forced:                                       # raw logprobs ignore the logit bias: the forced token is unlikely
            chosen = float(np.float32(top1 - 5 * LN10))
        out_lp = None
        if k_lp is not None:                             # what vLLM returns for logprobs=k (raw_logprobs)
            key = (lambda t: f'token_id:{t}') if as_ids else (lambda t: '.' if t == tok else ',')
            top = {key(tok if chosen == top1 else tok + 1): top1}
            if k_lp and chosen != top1:
                top[key(tok)] = chosen
            out_lp = {'tokens': [key(tok)], 'token_logprobs': [chosen], 'top_logprobs': [top] if k_lp else [None],
                      'text_offset': [k]}
        stop = hooks.feed(rid, [tok], [top1, chosen] if SAMPLER is not None else None)
        done = k + 1 == n_max
        if stop is not None:
            finish, stop_reason, done = 'stop', stop, True
        elif rep and vllm_check(ids, rep['min_pattern_size'], rep['max_pattern_size'], rep['min_count']):
            finish, done = 'stop', True
        chunk = {'id': rid, 'choices': [{'index': 0, 'text': '.', 'token_ids': [tok], 'logprobs': out_lp,
                                         'finish_reason': finish if done else None,
                                         'stop_reason': stop_reason if done else None}]}
        await resp.write(f'data: {json.dumps(chunk)}\n\n'.encode())
        if done:
            break
    hooks.finish(rid, finish)
    n = len(ids)
    STATS['ttft_sum'] += times[0] - arrived
    STATS['ttft_count'] += 1
    STATS['itl_sum'] += times[-1] - times[0]
    STATS['itl_count'] += n - 1
    STATS['tpot_sum'] += (times[-1] - times[0]) / (n - 1) if n > 1 else 0.0
    STATS['tpot_count'] += 1
    usage = {'id': rid, 'choices': [], 'usage': {'prompt_tokens': len(prompt), 'completion_tokens': n}}
    await resp.write(f'data: {json.dumps(usage)}\n\ndata: [DONE]\n\n'.encode())
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--step-ms', type=float, default=2.0)
    ap.add_argument('--loop-after', type=int, help='every output turns into a 40-token loop after this many tokens')
    a = ap.parse_args()
    app = web.Application()
    app['step'] = a.step_ms / 1e3
    app['loop_after'] = a.loop_after
    app.add_routes([web.get('/health', health), web.get('/metrics', metrics), web.post('/v1/completions', completions)])
    web.run_app(app, host='127.0.0.1', port=a.port, print=None)


if __name__ == '__main__':
    main()
