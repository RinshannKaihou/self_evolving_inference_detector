#!/usr/bin/env python3
"""Smoke test of the intercept path (V35_DETECTOR=intercept): one request that is asked to repeat a line forever (greedy,
ignore_eos) must end early with stop_reason 'v35_repetition: <component>', and the server's events file must hold the
matching alert. Exit status 1 otherwise.

With --rare (the server also runs V35_RARE=shadow; needs --events), three more checks of the rc3 wiring:
  1. the loop request asks for logprobs=1: m4 still intercepts it, and rc3 checked it (finish event 'rare': 'checked');
  2. garble: logprobs=1 and a logit_bias that makes a lone UTF-8 continuation byte the output at every step: rc3 logs a
     strict-tier 'bad_utf8' alert, and the request still runs to max_tokens (rc3 never stops a request);
  3. a request without logprobs: not checked by rc3 (finish event 'rare': 'no_logprobs'), no rc3 alert.

  python bench/smoke_intercept.py --url http://127.0.0.1:18790 [--events /path/to/v35-detector-events.jsonl] [--rare]
  --serve "vllm serve MODEL --port 18790 ..."   start that server first (it inherits this process's environment, so set
                                                V35_DETECTOR=intercept etc. here), wait for it, run the check, stop it
"""
import argparse
import json
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


def complete(a, prompt, max_tokens, **extra):
    body = dict(model=a.model, prompt=prompt, max_tokens=max_tokens, temperature=0, ignore_eos=True, stream=True,
                stream_options={'include_usage': True}, return_token_ids=True, **extra)
    req = urllib.request.Request(a.url + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    finish, stop_reason, n, lp = None, None, 0, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            obj = json.loads(line[6:])
            for c in obj.get('choices', []):
                n += len(c.get('token_ids') or [])
                lp += len((c.get('logprobs') or {}).get('token_logprobs') or [])
                finish = c.get('finish_reason') or finish
                stop_reason = c.get('stop_reason') or stop_reason
    return {'completion_tokens': n, 'logprob_positions': lp, 'finish_reason': finish, 'stop_reason': stop_reason}


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


def check(a):
    offset = os.path.getsize(a.events) if a.events and os.path.exists(a.events) else 0
    r = complete(a, PROMPT, a.max_tokens, **({'logprobs': 1} if a.rare else {}))
    print(json.dumps(r))
    stop_reason = r['stop_reason']
    ok = (r['finish_reason'] == 'stop' and isinstance(stop_reason, str) and stop_reason.startswith('v35_repetition: ')
          and r['completion_tokens'] < a.max_tokens)
    if ok and a.events:
        events, offset = new_events(a.events, offset)
        alerts = [e for e in events if e['event'] == 'alert']
        ok = any(e.get('action') == 'intercept' and stop_reason.endswith(e['reason']) for e in alerts)
        print('events: matching intercept alert' if ok else 'events: no matching intercept alert')
        if ok and a.rare:
            fin = [e for e in events if e['event'] == 'finish']
            ok = len(fin) == 1 and fin[0].get('rare') == 'checked' and r['logprob_positions'] == r['completion_tokens']
            print(f'rc3 on the loop request: finish rare={fin[0].get("rare") if fin else None}, logprobs for '
                  f'{r["logprob_positions"]} of {r["completion_tokens"]} tokens -> {"ok" if ok else "FAILED"}')
    if not ok or not a.rare:
        return ok
    print('intercept smoke: ok')
    return check_rare(a, offset)


def check_rare(a, offset):
    from v35_detector.rare_vocab import load as load_rare
    lone = load_rare(a.table)['bytes'].index(b'\x80')    # a lone UTF-8 continuation byte: invalid output -> bad_utf8
    n = 48                                               # below every m4 component's reach: only rc3 can fire
    r = complete(a, PLAIN, n, logprobs=1, logit_bias={str(lone): 100})
    print(json.dumps(r))
    events, offset = new_events(a.events, offset)
    fin = [e for e in events if e['event'] == 'finish']
    rid = fin[0]['request_id'] if len(fin) == 1 else None
    strict = [e for e in events if e['event'] == 'rare_alert' and e['request_id'] == rid and e['tier'] == 'strict']
    ok2 = bool(r['completion_tokens'] == n and r['finish_reason'] == 'length' and fin and fin[0].get('rare') == 'checked'
               and len(strict) == 1 and strict[0]['reason'] == 'bad_utf8')
    print(f'rc3 garble (token {lone} forced): {r["completion_tokens"]} tokens, finish {r["finish_reason"]}, rc3 strict '
          f'alerts {[(e["fired_at"], e["reason"]) for e in strict]} -> {"ok" if ok2 else "FAILED"}')
    r = complete(a, PLAIN, 32)
    print(json.dumps(r))
    events2, offset = new_events(a.events, offset)
    fin = [e for e in events2 if e['event'] == 'finish']
    ok3 = (len(fin) == 1 and fin[0].get('rare') == 'no_logprobs' and r['logprob_positions'] == 0
           and not [e for e in events2 if e['event'] == 'rare_alert'])
    print(f'rc3 without logprobs: finish rare={fin[0].get("rare") if fin else None} -> {"ok" if ok3 else "FAILED"}')
    errors = [e for e in events + events2 if e['event'] == 'error']
    if errors:
        print('events: error', errors[:3])
    if ok2 and ok3 and not errors:
        print('rc3 smoke: ok')
    return ok2 and ok3 and not errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', default='bench')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--events')
    ap.add_argument('--rare', action='store_true', help='also check rc3 (the server runs V35_RARE=shadow); needs --events')
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
