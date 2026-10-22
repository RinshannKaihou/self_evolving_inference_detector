"""Serving hooks (src/v35_detector/serving/hooks.py) without vLLM: requests are driven through register / feed / finish
exactly as the patched OutputProcessor calls them. The alert boundary must be the offline one under any step size, in
both runners; shadow never stops; errors fail open."""
import json
import time
import types

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


# ------------------------------------------------------------------ rc3 (V35_RARE=shadow | collect): monitor only
from test_rare import _pool, synth as rare_synth  # noqa: E402
from v35_detector import CONFIG_RC3, CONFIG_RC4  # noqa: E402
from v35_detector import rare as R  # noqa: E402
from v35_detector.rare_vocab import load as load_rare  # noqa: E402

RC3 = json.load(open(CONFIG_RC3))
RC4 = json.load(open(CONFIG_RC4))
RARE_CFG = {'rc3': RC3, 'rc4': RC4}


def values(top1, chosen):
    """one step's EngineCoreOutput.v35_rare (r3): [top1_0, chosen_0, top1_1, chosen_1, ...], Python floats as msgspec
    decodes them"""
    out = np.empty(2 * len(top1))
    out[0::2], out[1::2] = top1, chosen
    return out.tolist()


def rare_stream(corpus, seed, n=1500):
    """an rc3 test stream (float32 values, as the sampler computes them) with planted garble: a lone UTF-8 continuation
    byte, or a Han token glued to a Latin word in the answer"""
    T = load_rare(corpus)
    tok, lp1, lpc = rare_synth(corpus, n, seed, han_share=0.05 * (seed % 3))
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    if seed % 3 == 0:
        tok[700] = next(i for i, b in enumerate(T['bytes']) if b == b'\x80')
    elif seed % 3 == 1:
        tok[900], tok[901] = word, int(_pool(T, 'han', False)[seed])
    return tok, lp1.astype(np.float32).astype(np.float64), lpc.astype(np.float32).astype(np.float64)


def rare_service(tmp_path, corpus='qwen3', mode='off', runner='sync', rare='shadow', trace=0, config='rc3'):
    env = {'V35_DETECTOR': mode, 'V35_RARE': rare, 'V35_DETECTOR_TABLE': corpus, 'V35_DETECTOR_RUNNER': runner,
           'V35_RARE_TRACE': str(trace), 'V35_RARE_CONFIG': config,
           'V35_DETECTOR_EVENTS': str(tmp_path / f'events_{corpus}_{mode}_{rare}_{runner}_{trace}_{config}.jsonl')}
    return hooks.Service(hooks.settings(env))


def drive(svc, rid, tok, lp1, lpc, rng, with_values=True, sleep=0.0):
    """engine steps of 1-4 tokens; returns (tokens fed, the stop reason or None)"""
    svc.register(rid)
    seen, stop, i = 0, None, 0
    while i < len(tok):
        k = int(rng.choice([1, 1, 1, 2, 4]))
        v = values(lp1[i:i + k], lpc[i:i + k]) if with_values else None
        stop = svc.feed(rid, tok[i:i + k].tolist(), v)
        seen = min(i + k, len(tok))
        i += k
        if sleep:
            time.sleep(sleep)
        if stop is not None:
            break
    svc.finish(rid, 'stop' if stop else 'length')
    return seen, stop


def rare_alerts(ev, rid):
    return {e['tier']: (e['fired_at'], e['reason']) for e in ev if e['event'] == 'rare_alert' and e['request_id'] == rid}


@pytest.mark.parametrize('runner,config', [('sync', 'rc3'), ('async', 'rc3'), ('sync', 'rc4'), ('async', 'rc4')])
def test_rare_shadow_equals_offline_and_never_stops(tmp_path, runner, config):
    """V35_RARE_CONFIG picks the configuration; the online alerts equal the offline reference of that configuration"""
    svc = rare_service(tmp_path, runner=runner, config=config)
    want, rng = {}, np.random.default_rng(11)
    for seed in range(6):
        tok, lp1, lpc = rare_stream('qwen3', seed)
        seen, stop = drive(svc, f'r{seed}', tok, lp1, lpc, rng, sleep=0.0001 if runner == 'async' else 0.0)
        assert stop is None and seen == len(tok)         # rc3 never ends a request
        want[f'r{seed}'] = {t: R.decide(tok, lp1, lpc, 'qwen3', RARE_CFG[config], tier=tier)
                            for t, tier in (('strict', 'intercept'), ('monitor', 'monitor'))}
    svc.close()
    ev = events(svc)
    for rid, w in want.items():
        assert rare_alerts(ev, rid) == {t: v for t, v in w.items() if v[0] >= 0}, rid
    assert sum(w['strict'][0] >= 0 for w in want.values()) >= 2        # the comparison covered strict alerts
    assert [e['rare'] for e in ev if e['event'] == 'finish'] == ['checked'] * 6
    assert not [e for e in ev if e['event'] in ('alert', 'error', 'rare_trace')]
    s = [e for e in ev if e['event'] == 'summary'][-1]
    assert s['counters']['rare_requests'] == 6 and s['counters'].get('intercepts', 0) == 0 and s['rare_config'] == config
    if runner == 'sync':
        assert s['rare_seconds_sync'] > 0 and s['detector_seconds_sync'] == 0.0       # m4 off: only rc3 time
    else:
        assert s['worker']['rare_tokens'] > 0 and s['worker']['tokens'] == 0


def test_requests_without_values_skip_rc3_and_m4_is_unchanged(tmp_path):
    """m4 intercept with rc3 on: a request whose engine sends no values (no collection on that path) is not checked by
    rc3, and m4 stops it exactly where it does with rc3 off"""
    tok = make('qwen3', 'loop40', 1)
    at, reason = decide(tok, load('qwen3'), M4, THINK_END['qwen3'])
    runs = {}
    for rare in ('off', 'collect', 'shadow'):
        svc = rare_service(tmp_path, mode='intercept', rare=rare)
        lp = np.full(len(tok), -0.05)
        runs[rare] = drive(svc, 'r1', tok, lp, lp, np.random.default_rng(4), with_values=False)
        svc.close()
        ev = events(svc)
        alert = [(e['fired_at'], e['reason'], e['action']) for e in ev if e['event'] == 'alert']
        assert alert == [(at, reason, 'intercept')]
        fin = next(e for e in ev if e['event'] == 'finish')
        assert fin.get('rare') == (None if rare == 'off' else 'no_values')
        assert not [e for e in ev if e['event'] in ('rare_alert', 'error')]
    assert runs['off'] == runs['collect'] == runs['shadow'] and runs['off'][1] == hooks.STOP_PREFIX + reason


@pytest.mark.parametrize('rare', ['collect', 'shadow'])
def test_m4_is_unchanged_when_values_come(tmp_path, rare):
    """the m4 intercept point and alert are the same with rc3 off, collecting, or detecting"""
    tok = make('qwen3', 'loop40', 5)
    at, reason = decide(tok, load('qwen3'), M4, THINK_END['qwen3'])
    lp = np.full(len(tok), -0.02)
    got = {}
    for r in ('off', rare):
        svc = rare_service(tmp_path, mode='intercept', rare=r)
        got[r] = drive(svc, 'r1', tok, lp, lp, np.random.default_rng(9), with_values=r != 'off')
        svc.close()
        assert [(e['fired_at'], e['reason']) for e in events(svc) if e['event'] == 'alert'] == [(at, reason)]
    assert got['off'] == got[rare]


@pytest.mark.parametrize('runner', ['sync', 'async'])
def test_m4_and_rc3_together(tmp_path, runner):
    """one stream with a loop (m4) and a lone continuation byte (rc3): m4 intercepts at its offline boundary, rc3 logs its
    offline alert and stops nothing"""
    T = load_rare('qwen3')
    tok = make('qwen3', 'loop40', 2)
    tok[150] = next(i for i, b in enumerate(T['bytes']) if b == b'\x80')
    at, reason = decide(tok, load('qwen3'), M4, THINK_END['qwen3'])
    lp1 = np.full(len(tok), -0.01)
    svc = rare_service(tmp_path, mode='intercept', runner=runner)
    seen, stop = drive(svc, 'r1', tok, lp1, lp1, np.random.default_rng(6), sleep=0.0002 if runner == 'async' else 0.0)
    svc.close()
    ev = events(svc)
    assert stop == hooks.STOP_PREFIX + reason and (at <= seen < at + 4 if runner == 'sync' else seen >= at)
    strict = R.decide(tok[:seen], lp1[:seen], lp1[:seen], 'qwen3', RC3)
    assert strict[1] == 'bad_utf8' and rare_alerts(ev, 'r1')['strict'] == strict
    assert next(e for e in ev if e['event'] == 'finish')['rare'] == 'checked'


def test_collect_mode_records_coverage_without_rc3(tmp_path):
    """V35_RARE=collect alone: no table, no detector, no runner; every request with values is 'checked'"""
    s = hooks.settings({'V35_RARE': 'collect', 'V35_DETECTOR_EVENTS': str(tmp_path / 'collect.jsonl')})
    assert s['mode'] == 'off' and s['rare'] == 'collect' and s['table'] is None
    svc = hooks.Service(s)
    tok, lp1, lpc = rare_stream('qwen3', 0, 800)         # holds a lone continuation byte: rc3 would fire on it
    for rid in ('a', 'b'):
        drive(svc, rid, tok, lp1, lpc, np.random.default_rng(1))
    drive(svc, 'c', tok, lp1, lpc, np.random.default_rng(1), with_values=False)
    assert svc.runner is None
    svc.close()
    ev = events(svc)
    assert [e['rare'] for e in ev if e['event'] == 'finish'] == ['checked', 'checked', 'no_values']
    assert not [e for e in ev if e['event'] in ('rare_alert', 'alert', 'error')]
    c = [e for e in ev if e['event'] == 'summary'][-1]['counters']
    assert c['rare_values'] == 1600 and c.get('rare_tokens', 0) == 0 and c['rare_no_values'] == 1


def test_rc3_lost_when_values_stop_or_are_malformed(tmp_path):
    svc = rare_service(tmp_path, mode='shadow')
    tok, lp1, lpc = rare_stream('qwen3', 5, 400)
    svc.register('a')                                    # values for 100 tokens, then none
    for i in range(0, 400, 2):
        svc.feed('a', tok[i:i + 2].tolist(), values(lp1[i:i + 2], lpc[i:i + 2]) if i < 100 else None)
    svc.finish('a', 'length')
    svc.register('b')                                    # one value per token instead of two
    for i in range(0, 400, 2):
        svc.feed('b', tok[i:i + 2].tolist(), lp1[i:i + 2].tolist())
    svc.finish('b', 'length')
    svc.register('c')                                    # r2's LogprobsLists (an output processor patched by r2)
    svc.feed('c', tok[:2].tolist(), types.SimpleNamespace(logprobs=np.zeros((2, 2))))
    svc.finish('c', 'length')
    svc.close()
    ev = events(svc)
    assert [e['rare'] for e in ev if e['event'] == 'finish'] == ['lost', 'lost', 'lost']
    assert not [e for e in ev if e['event'] == 'error']
    assert [e for e in ev if e['event'] == 'summary'][-1]['counters']['rare_lost'] == 3


def test_rc3_error_fails_open_and_m4_goes_on(tmp_path):
    tok = make('qwen3', 'loop40', 3)
    at, reason = decide(tok, load('qwen3'), M4, THINK_END['qwen3'])
    svc = rare_service(tmp_path, mode='intercept')
    svc.register('warm')                                 # builds the runner
    svc.abort('warm')

    def boom(*a):
        raise RuntimeError('injected')

    svc.runner.rare_feed = boom
    lp1 = np.full(len(tok), -0.01)
    seen, stop = drive(svc, 'r1', tok, lp1, lp1, np.random.default_rng(8))
    svc.close()
    ev = events(svc)
    assert [e['where'] for e in ev if e['event'] == 'error'] == ['rare_feed']   # once, then rc3 is off for r1
    assert stop == hooks.STOP_PREFIX + reason and at <= seen < at + 4
    assert next(e for e in ev if e['event'] == 'finish')['rare'] == 'error'


@pytest.mark.parametrize('rare', ['collect', 'shadow'])
def test_trace_logs_the_values_as_received(tmp_path, rare):
    """V35_RARE_TRACE=N (the smoke test compares these with the logprobs vLLM returns): the first N tokens' values of every
    request, also after rc3's strict alert"""
    tok, lp1, lpc = rare_stream('qwen3', 0, 900)         # strict alert at the lone byte (token 700)
    svc = rare_service(tmp_path, rare=rare, trace=800)
    drive(svc, 'a', tok, lp1, lpc, np.random.default_rng(2))
    svc.close()
    ev = events(svc)
    tr = [e for e in ev if e['event'] == 'rare_trace']
    assert len(tr) == 1 and tr[0]['request_id'] == 'a'
    n = len(tr[0]['token_ids'])
    assert 800 <= n < 804 and tr[0]['token_ids'] == tok[:n].tolist()
    assert tr[0]['top1'] == lp1[:n].tolist() and tr[0]['chosen'] == lpc[:n].tolist()
    if rare == 'shadow':
        assert rare_alerts(ev, 'a')['strict'][1] == 'bad_utf8'


def test_rare_settings(tmp_path):
    assert hooks.settings({'V35_RARE': 'off'}) is None
    with pytest.raises(ValueError):                      # monitor only: no intercept mode
        hooks.settings({'V35_RARE': 'intercept', 'V35_DETECTOR_TABLE': 'qwen3'})
    with pytest.raises(ValueError):
        hooks.settings({'V35_RARE': 'shadow'})
    table = tmp_path / 'copy.npz'
    np.savez(table, **load('qwen3'))
    with pytest.raises(ValueError):                      # rc3 needs a shipped table (rare and reference tables)
        hooks.settings({'V35_RARE': 'shadow', 'V35_DETECTOR_TABLE': str(table), 'V35_DETECTOR_THINK_END': '151668'})
    s = hooks.settings({'V35_RARE': 'shadow', 'V35_DETECTOR_TABLE': 'glm'})
    assert s['mode'] == 'off' and s['rare'] == 'shadow' and s['think_end'] == THINK_END['glm'] and s['trace'] == 0
    s = hooks.settings({'V35_DETECTOR': 'shadow', 'V35_RARE': 'shadow', 'V35_DETECTOR_TABLE': 'qwen3',
                        'V35_DETECTOR_THINK_END': 'none', 'V35_DETECTOR_RUNNER': 'async'})
    assert s['think_end'] is None and s['runner'] == 'async'
    s = hooks.settings({'V35_DETECTOR': 'intercept', 'V35_RARE': 'collect', 'V35_DETECTOR_TABLE': 'qwen3',
                        'V35_RARE_TRACE': '64'})
    assert s['rare'] == 'collect' and s['trace'] == 64
    with pytest.raises(ValueError):                      # m4 still needs its table
        hooks.settings({'V35_DETECTOR': 'shadow', 'V35_RARE': 'collect'})
    assert s['rare_name'] == 'rc3' and s['rare_config'] == CONFIG_RC3          # default: rc3
    s = hooks.settings({'V35_RARE': 'shadow', 'V35_RARE_CONFIG': 'rc4', 'V35_DETECTOR_TABLE': 'qwen3'})
    assert s['rare_name'] == 'rc4' and s['rare_config'] == CONFIG_RC4
    with pytest.raises(ValueError):
        hooks.settings({'V35_RARE': 'shadow', 'V35_RARE_CONFIG': 'rc5', 'V35_DETECTOR_TABLE': 'qwen3'})
