"""Serving hooks (src/v35_detector/serving/hooks.py) without vLLM: requests are driven through register / feed / finish
exactly as the patched OutputProcessor calls them. The alert boundary must be the offline one under any step size, in
both runners; shadow never stops; errors fail open."""
import json
import time

import numpy as np
import pytest

from synth import make
from v35_detector import CONFIG_M4
from v35_detector.offline import decide
from v35_detector.serving import hooks
from v35_detector.vocab import THINK_END, load

M4 = json.load(open(CONFIG_M4))
CASES = [('qwen3', 'loop40'), ('qwen3', 'noisy150'), ('glm', 'circling'), ('glm', 'restate'), ('dsv4', 'answer'),
         ('qwen3', 'random')]


def service(tmp_path, corpus, mode='intercept', runner='sync'):
    env = {'V35_DETECTOR': mode, 'V35_DETECTOR_TABLE': corpus, 'V35_DETECTOR_RUNNER': runner,
           'V35_DETECTOR_EVENTS': str(tmp_path / f'events_{corpus}_{mode}_{runner}.jsonl')}
    return hooks.Service(hooks.settings(env))


def events(svc):
    return [json.loads(line) for line in open(svc.s['events'])]


def steps(tok, rng):
    """engine steps: mostly 1 token, sometimes several (speculative decoding)"""
    i = 0
    while i < len(tok):
        k = int(rng.choice([1, 1, 1, 2, 4]))
        yield tok[i:i + k].tolist()
        i += k


@pytest.mark.parametrize('corpus,kind', CASES)
def test_sync_intercept_stops_at_the_offline_alert(tmp_path, corpus, kind):
    tok = make(corpus, kind, 1)
    at, reason = decide(tok, load(corpus), M4, THINK_END[corpus])
    svc = service(tmp_path, corpus)
    svc.register('r1')
    seen, stop = 0, None
    for step in steps(tok, np.random.default_rng(5)):
        seen += len(step)
        stop = svc.feed('r1', step)
        if stop is not None:
            break
    svc.finish('r1', 'stop' if stop else 'length')
    svc.close()
    if at < 0:
        assert stop is None
        assert [e['tokens'] for e in events(svc) if e['event'] == 'finish'] == [len(tok)]
    else:
        assert stop == hooks.STOP_PREFIX + reason
        assert at <= seen < at + 4                       # stopped in the engine step that delivered the alert boundary
        alert = next(e for e in events(svc) if e['event'] == 'alert')
        assert (alert['fired_at'], alert['reason'], alert['action']) == (at, reason, 'intercept')


def test_shadow_logs_but_never_stops(tmp_path):
    tok = make('qwen3', 'noisy150', 1)
    at, reason = decide(tok, load('qwen3'), M4, THINK_END['qwen3'])
    svc = service(tmp_path, 'qwen3', mode='shadow')
    svc.register('r1')
    assert all(svc.feed('r1', step) is None for step in steps(tok, np.random.default_rng(1)))
    svc.finish('r1', 'length')
    svc.close()
    ev = events(svc)
    assert [(e['fired_at'], e['reason'], e['action']) for e in ev if e['event'] == 'alert'] == [(at, reason, 'shadow')]
    assert [e['tokens'] for e in ev if e['event'] == 'finish'] == [len(tok)]
    assert next(e for e in ev if e['event'] == 'summary')['counters'].get('intercepts', 0) == 0


@pytest.mark.parametrize('corpus,kind', [('qwen3', 'loop300'), ('glm', 'restate'), ('qwen3', 'random')])
def test_async_intercept(tmp_path, corpus, kind):
    tok = make(corpus, kind, 2)
    at, reason = decide(tok, load(corpus), M4, THINK_END[corpus])
    svc = service(tmp_path, corpus, runner='async')
    svc.register('r1')
    seen, stop = 0, None
    for step in steps(tok, np.random.default_rng(2)):
        seen += len(step)
        stop = svc.feed('r1', step)
        if stop is not None:
            break
        time.sleep(0.0002)                               # an engine step is milliseconds; the worker keeps up
    svc.finish('r1', 'stop' if stop else 'length')
    svc.close()
    ev = events(svc)
    if at < 0:
        assert stop is None and not [e for e in ev if e['event'] == 'alert']
    else:
        assert stop == hooks.STOP_PREFIX + reason and seen >= at   # async: the stop lands a few steps after the alert
        alert = next(e for e in ev if e['event'] == 'alert')
        assert (alert['fired_at'], alert['reason']) == (at, reason)
    assert next(e for e in ev if e['event'] == 'summary')['worker']['tokens'] > 0


def test_alert_on_the_last_partial_block_is_logged_after_the_end(tmp_path):
    base = make('qwen3', 'loop40', 3)
    T = load('qwen3')
    n = next(n for n in range(700, 5000) if n % 64 and decide(base[:n], T, M4, THINK_END['qwen3'])[0] == n)
    tok = base[:n]
    svc = service(tmp_path, 'qwen3')
    svc.register('r1')
    assert all(svc.feed('r1', [t]) is None for t in tok.tolist())   # the boundary at n is only reached by finish
    svc.finish('r1', 'length')
    svc.close()
    alert = next(e for e in events(svc) if e['event'] == 'alert')
    assert alert['fired_at'] == n and alert['after_end'] is True


def test_detector_error_fails_open(tmp_path):
    svc = service(tmp_path, 'glm')
    svc.register('r1')
    svc.register('r2')

    def boom(rid, block):
        raise RuntimeError('injected')

    svc.runner.feed = boom
    tok = make('glm', 'loop5', 1)
    assert all(svc.feed('r1', step) is None for step in steps(tok, np.random.default_rng(3)))
    svc.finish('r1', 'length')
    svc.abort('r2')
    svc.close()
    ev = events(svc)
    assert [e['where'] for e in ev if e['event'] == 'error'] == ['feed']   # one error, then detection is off for r1
    assert [e['tokens'] for e in ev if e['event'] == 'finish'] == [len(tok)]
    assert next(e for e in ev if e['event'] == 'summary')['counters']['aborted'] == 1


def test_settings_are_checked_at_start_up(tmp_path):
    assert hooks.settings({}) is None
    with pytest.raises(ValueError):
        hooks.settings({'V35_DETECTOR': 'on'})
    with pytest.raises(ValueError):
        hooks.settings({'V35_DETECTOR': 'shadow'})
    with pytest.raises(ValueError):
        hooks.settings({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': 'qwen3', 'V35_DETECTOR_RUNNER': 'thread'})
    table = tmp_path / 'copy.npz'
    np.savez(table, **load('qwen3'))
    with pytest.raises(ValueError):                      # a custom table needs an explicit think_end
        hooks.settings({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': str(table)})
    s = hooks.settings({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': str(table), 'V35_DETECTOR_THINK_END': '151668'})
    assert s['think_end'] == 151668
