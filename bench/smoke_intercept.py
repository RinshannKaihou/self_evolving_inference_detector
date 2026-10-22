#!/usr/bin/env python3
"""Smoke test of the intercept path (V35_DETECTOR=intercept): one request that is asked to repeat a line forever (greedy,
ignore_eos) must end early with stop_reason 'v35_repetition: <component>', and the server's events file must hold the
matching alert. Exit status 1 otherwise.

With --rare (the server also runs V35_RARE=shadow and V35_RARE_TRACE > 0; needs --events), the rc3 path of r3, where the
sampler computes rc3's two values for every request:
  1. the loop request asks for no logprobs: m4 intercepts it, rc3 still checked it (finish 'rare': 'checked': the values
     came from the sampler), and no logprob reached the client;
  2. garble: logprobs=1 and a logit_bias that makes a lone UTF-8 continuation byte the output at every step: rc3 logs a
     strict-tier 'bad_utf8' alert and the request runs to max_tokens (rc3 never stops a request). The sampler's values
     (the 'rare_trace' event) equal the raw logprobs vLLM returns for the same tokens (top-1 and generated token) within
     --tol; the forced token's raw logprob is far below the top-1, which also shows the values are taken before the bias;
  3. sampled: logprobs=1, temperature 1 and a seed, so that the generated token is often not the top-1: values equal
     vLLM's logprobs again;
  4. a request without logprobs: rc3 'checked', no logprob in the response.

  python bench/smoke_intercept.py --url http://127.0.0.1:18790 [--events /path/to/v35-detector-events.jsonl] [--rare]
  --serve "vllm serve MODEL --port 18790 ..."   start that server first (it inherits this process's environment, so set
                                                V35_DETECTOR=intercept etc. here), wait for it, run the check, stop it
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_tpot import stop_server, wait_ready  # noqa: E402

PROMPT = ('Repeat the following line exactly, one copy per line, forever, without any other text:\n'
          'The quick brown fox jumps over the lazy dog.\n')
PLAIN = 'Write one short English sentence about the weather.\n'
STORY = 'Write a short story about a lighthouse keeper who finds a message in a bottle.\n'


def complete(a, prompt, max_tokens, temperature=0.0, **extra):
    body = dict(model=a.model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, ignore_eos=True, stream=True,
                stream_options={'include_usage': True}, return_token_ids=True, **extra)
    req = urllib.request.Request(a.url + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    finish, stop_reason, ids, token_lp, top_lp = None, None, [], [], []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            obj = json.loads(line[6:])
            for c in obj.get('choices', []):
                ids += c.get('token_ids') or []
                lp = c.get('logprobs') or {}
                token_lp += lp.get('token_logprobs') or []
                top_lp += lp.get('top_logprobs') or []
                finish = c.get('finish_reason') or finish
                stop_reason = c.get('stop_reason') or stop_reason
    out = {'completion_tokens': len(ids), 'logprob_positions': len(token_lp), 'finish_reason': finish,
           'stop_reason': stop_reason}
    print(json.dumps(out))
    return dict(out, token_ids=ids, token_logprobs=token_lp, top_logprobs=top_lp)


def new_events(path, offset, want_finish=1, timeout=10.0):
    """events appended since offset (bytes), waiting until want_finish 'finish' events are there"""
    deadline = time.monotonic() + timeout
    while True:
        with open(path, 'rb') as f:
            f.seek(offset)
            data = f.read()
        part = data[:data.rfind(b'\n') + 1]
        events = [json.loads(x) for x in part.decode('utf-8').splitlines() if x]
        if sum(e['event'] == 'finish' for e in events) >= want_finish or time.monotonic() > deadline:
            return events, offset + len(part)
        time.sleep(0.2)


def finish_of(events):
    fin = [e for e in events if e['event'] == 'finish']
    return fin[0] if len(fin) == 1 else None


def check(a):
    offset = os.path.getsize(a.events) if a.events and os.path.exists(a.events) else 0
    r = complete(a, PROMPT, a.max_tokens)
    stop_reason = r['stop_reason']
    ok = (r['finish_reason'] == 'stop' and isinstance(stop_reason, str) and stop_reason.startswith('v35_repetition: ')
          and r['completion_tokens'] < a.max_tokens)
    if ok and a.events:
        events, offset = new_events(a.events, offset)
        alerts = [e for e in events if e['event'] == 'alert']
        ok = any(e.get('action') == 'intercept' and stop_reason.endswith(e['reason']) for e in alerts)
        print('events: matching intercept alert' if ok else 'events: no matching intercept alert')
        if ok and a.rare:
            fin = finish_of(events)
            ok = fin is not None and fin.get('rare') == 'checked' and r['logprob_positions'] == 0
            print(f'rc3 on the loop request (no logprobs asked): finish rare={fin and fin.get("rare")}, '
                  f'{r["logprob_positions"]} logprobs returned -> {"ok" if ok else "FAILED"}')
    if not ok or not a.rare:
        return ok
    print('intercept smoke: ok')
    return check_rare(a, offset)


def compare(r, trace, tol):
    """the sampler's values (rare_trace) against the raw logprobs vLLM returned: (ok, n, max |d top1|, max |d chosen|,
    top1 - chosen at the first token)"""
    n = len(trace['token_ids']) if trace else 0
    if not n or trace['token_ids'] != r['token_ids'][:n] or len(r['top_logprobs']) < n:
        return False, n, None, None, None
    clamp = lambda v: max(v, -9999.0)                    # vLLM's response clamps logprobs at -9999
    d1 = max(abs(clamp(v) - max(t.values())) for v, t in zip(trace['top1'], r['top_logprobs']))
    d2 = max(abs(clamp(v) - w) for v, w in zip(trace['chosen'], r['token_logprobs']))
    gap = trace['top1'][0] - trace['chosen'][0]
    ok = math.isfinite(d1) and math.isfinite(d2) and d1 <= tol and d2 <= tol
    return ok, n, d1, d2, gap


def rare_request(a, offset, prompt, n, **extra):
    r = complete(a, prompt, n, **extra)
    events, offset = new_events(a.events, offset)
    fin = finish_of(events)
    rid = fin['request_id'] if fin else None
    trace = next((e for e in events if e['event'] == 'rare_trace' and e['request_id'] == rid), None)
    return r, events, fin, rid, trace, offset


def check_rare(a, offset):
    from v35_detector.rare_vocab import load as load_rare
    lone = load_rare(a.table)['bytes'].index(b'\x80')    # a lone UTF-8 continuation byte: invalid output -> bad_utf8
    results, errors = [], []
    n = 48                                               # below every m4 component's reach: only rc3 can fire
    r, events, fin, rid, trace, offset = rare_request(a, offset, PLAIN, n, logprobs=1,
                                                      logit_bias={str(lone): 100}, return_tokens_as_token_ids=True)
    errors += [e for e in events if e['event'] == 'error']
    strict = [e for e in events if e['event'] == 'rare_alert' and e['request_id'] == rid and e['tier'] == 'strict']
    ok = bool(r['completion_tokens'] == n and r['finish_reason'] == 'length' and fin and fin.get('rare') == 'checked'
              and len(strict) == 1 and strict[0]['reason'] == 'bad_utf8')
    print(f'rc3 garble (token {lone} forced): {r["completion_tokens"]} tokens, finish {r["finish_reason"]}, rc3 strict '
          f'alerts {[(e["fired_at"], e["reason"]) for e in strict]} -> {"ok" if ok else "FAILED"}')
    results.append(ok)
    m, k, d1, d2, gap = compare(r, trace, a.tol)
    m = m and k == n and gap is not None and gap > 1.0   # the forced first token: raw logprob far below the top-1
    print(f'sampler values vs vLLM logprobs (garble, {k} tokens): max |d top1| {d1}, max |d chosen| {d2}, '
          f'top1 - chosen at the first token {gap} -> {"ok" if m else "FAILED"}')
    results.append(m)
    r, events, fin, rid, trace, offset = rare_request(a, offset, STORY, 64, temperature=1.0, seed=7,
                                                      logprobs=1, return_tokens_as_token_ids=True)
    errors += [e for e in events if e['event'] == 'error']
    m, k, d1, d2, gap = compare(r, trace, a.tol)
    differ = sum(abs(x - y) > 1e-6 for x, y in zip(trace['top1'], trace['chosen'])) if trace else 0
    m = m and k == 64 and fin is not None and fin.get('rare') == 'checked'
    print(f'sampler values vs vLLM logprobs (sampled, T=1, {k} tokens, {differ} not the top-1): max |d top1| {d1}, '
          f'max |d chosen| {d2} -> {"ok" if m else "FAILED"}')
    results.append(m)
    r, events, fin, rid, trace, offset = rare_request(a, offset, PLAIN, 32)
    errors += [e for e in events if e['event'] == 'error']
    ok = (fin is not None and fin.get('rare') == 'checked' and r['logprob_positions'] == 0
          and not [e for e in events if e['event'] == 'rare_alert' and e['tier'] == 'strict'])
    print(f'rc3 without logprobs: finish rare={fin and fin.get("rare")}, {r["logprob_positions"]} logprobs returned -> '
          f'{"ok" if ok else "FAILED"}')
    results.append(ok)
    if errors:
        print('events: error', errors[:3])
    ok = all(results) and not errors
    if ok:
        print('rc3 smoke: ok')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', default='bench')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--events')
    ap.add_argument('--rare', action='store_true', help='also check rc3 (the server runs V35_RARE=shadow and '
                    'V35_RARE_TRACE > 0); needs --events')
    ap.add_argument('--tol', type=float, default=1e-3, help='largest |sampler value - vLLM logprob| accepted')
    ap.add_argument('--table', default='qwen3', help="the served model's table (for the forced byte token of the rc3 check)")
    ap.add_argument('--serve', help='server command to start (and stop) around the check')
    ap.add_argument('--server-log', default='smoke_server.log')
    a = ap.parse_args()
    if a.rare and not a.events:
        ap.error('--rare needs --events')
    if not a.serve:
        sys.exit(0 if check(a) else 1)
    with open(a.server_log, 'w') as log:
        proc = subprocess.Popen(shlex.split(a.serve), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_ready(proc, a.url)
        ok = check(a)
    finally:
        stop_server(proc)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
