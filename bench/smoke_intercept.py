#!/usr/bin/env python3
"""Smoke test of the intercept path (V35_DETECTOR=intercept): one request that is asked to repeat a line forever (greedy,
ignore_eos) must end early with stop_reason 'v35_repetition: <component>', and the server's events file must hold the
matching alert. Exit status 1 otherwise.

  python bench/smoke_intercept.py --url http://127.0.0.1:18790 [--events /path/to/v35-detector-events.jsonl]
  --serve "vllm serve MODEL --port 18790 ..."   start that server first (it inherits this process's environment, so set
                                                V35_DETECTOR=intercept etc. here), wait for it, run the check, stop it
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_tpot import stop_server, wait_ready  # noqa: E402

PROMPT = ('Repeat the following line exactly, one copy per line, forever, without any other text:\n'
          'The quick brown fox jumps over the lazy dog.\n')


def check(a):
    body = dict(model=a.model, prompt=PROMPT, max_tokens=a.max_tokens, temperature=0, ignore_eos=True, stream=True,
                stream_options={'include_usage': True}, return_token_ids=True)
    req = urllib.request.Request(a.url + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    finish, stop_reason, n = None, None, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            obj = json.loads(line[6:])
            for c in obj.get('choices', []):
                n += len(c.get('token_ids') or [])
                finish = c.get('finish_reason') or finish
                stop_reason = c.get('stop_reason') or stop_reason
    print(json.dumps({'completion_tokens': n, 'finish_reason': finish, 'stop_reason': stop_reason}))
    ok = finish == 'stop' and isinstance(stop_reason, str) and stop_reason.startswith('v35_repetition: ') and n < a.max_tokens
    if ok and a.events:
        alerts = [json.loads(x) for x in open(a.events, encoding='utf-8') if '"alert"' in x]
        ok = any(e.get('action') == 'intercept' and stop_reason.endswith(e['reason']) for e in alerts)
        print('events: matching intercept alert' if ok else 'events: no matching intercept alert')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', default='bench')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--events')
    ap.add_argument('--serve', help='server command to start (and stop) around the check')
    ap.add_argument('--server-log', default='smoke_server.log')
    a = ap.parse_args()
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
