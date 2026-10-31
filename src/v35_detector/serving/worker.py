"""Detector process of the async runner (hooks.py, V35_DETECTOR_RUNNER=async): the output-processing loop only queues
64-token blocks; this process scores them and sends back alerts. It runs m4 (config is None when V35_DETECTOR=off) and the
selected rare_chars config (rare_config is None when V35_RARE=off) side by side. rc1t messages carry None for both logprob arrays.

Input messages:  m4   ('start', rid) | ('feed', rid, block) | ('finish', rid, rest) | ('drop', rid)
                 rare ('rstart', rid) | ('rfeed', rid, ids, top1_logprobs, chosen_logprobs) | ('rfinish', rid, ids, top1, chosen)
                      | ('rdrop', rid)
                 ('stop',)
Output messages: ('ready',) | ('alert', rid, fired_at, reason) | ('final', rid, fired_at, reason)  [an alert on the last
                 partial block, after the request ended] | ('error', rid, text)
                 | ('ralert', rid, tier, fired_at, reason, after_end) | ('rerror', rid, text)
                 | ('stats', tokens, seconds, open, rare_tokens, rare_seconds, rare_open) | ('stopped',)
"""
import time

from ..rare import RareDetector
from ..stream import RepDetector

STATS_EVERY_S = 10.0
RARE_OPS = ('rstart', 'rfeed', 'rfinish', 'rdrop')


def rare_step(rare, ent, ids, top1, chosen, last):
    """feed one block of a request to rare_chars (and end its stream if last); returns new alerts as (tier, fired_at, reason),
    tier 'monitor' (the monitoring tier's first alert) or 'strict' (the strict tier's, which ends rare_chars for the request:
    then ent[0].done). ent = [rare state, monitoring-tier alert already reported]; rc1t gets None for both logprob arrays"""
    st = ent[0]
    if len(ids):
        rare.feed(st, ids, top1, chosen)
    if last and not st.done:
        rare.finish(st)
    out = []
    if st.monitor is not None and not ent[1]:
        ent[1] = True
        out.append(('monitor', st.monitor.fired_at, st.monitor.reason))
    if st.result is not None:
        out.append(('strict', st.result.fired_at, st.result.reason))
    return out


def worker_main(inq, outq, table, config, think_end, rare_config=None):
    try:                                                 # shows as 'v35-detector' in ps / psutil (vLLM ships setproctitle)
        import setproctitle
        setproctitle.setproctitle('v35-detector')
    except ImportError:
        pass
    try:
        det = RepDetector(table, config, think_end) if config is not None else None
        rare = RareDetector(table, rare_config, think_end) if rare_config is not None else None
    except Exception as e:                               # reported to the parent, which fails at start-up
        outq.put(('error', None, f'init: {type(e).__name__}: {e}'))
        return
    outq.put(('ready',))
    states, tokens, seconds, last = {}, 0, 0.0, time.monotonic()
    rstates, rtokens, rseconds = {}, 0, 0.0
    while True:
        msg = inq.get()
        op = msg[0]
        if op == 'stop':
            break
        rid = msg[1]
        try:
            if op == 'start':
                states[rid] = det.start()
            elif op == 'drop':
                states.pop(rid, None)
            elif op in ('feed', 'finish'):
                st = states.get(rid) if op == 'feed' else states.pop(rid, None)
                if st is not None and not st.done:
                    block = msg[2]
                    t0 = time.perf_counter()
                    res = det.feed(st, block) if len(block) else None
                    if op == 'finish' and res is None:
                        res = det.finish(st)
                    seconds += time.perf_counter() - t0
                    tokens += len(block)
                    if res is not None and res.is_ill:
                        outq.put(('alert' if op == 'feed' else 'final', rid, res.fired_at, res.reason))
                        states.pop(rid, None)
            elif op == 'rstart':
                rstates[rid] = [rare.start(), False]
            elif op == 'rdrop':
                rstates.pop(rid, None)
            elif op in ('rfeed', 'rfinish'):
                ent = rstates.get(rid) if op == 'rfeed' else rstates.pop(rid, None)
                if ent is not None:                      # None: rc3 ended for the request (strict alert, error) or never started
                    ids = msg[2]
                    t0 = time.perf_counter()
                    hits = rare_step(rare, ent, ids, msg[3], msg[4], op == 'rfinish')
                    rseconds += time.perf_counter() - t0
                    rtokens += len(ids)
                    for tier, at, reason in hits:
                        outq.put(('ralert', rid, tier, at, reason, op == 'rfinish'))
                    if ent[0].done:
                        rstates.pop(rid, None)
        except Exception as e:                           # fail-open: this request loses detection, the process goes on
            if op in RARE_OPS:
                rstates.pop(rid, None)
                outq.put(('rerror', rid, f'{type(e).__name__}: {e}'))
            else:
                states.pop(rid, None)
                outq.put(('error', rid, f'{type(e).__name__}: {e}'))
        if time.monotonic() - last >= STATS_EVERY_S:
            outq.put(('stats', tokens, seconds, len(states), rtokens, rseconds, len(rstates)))
            last = time.monotonic()
    outq.put(('stats', tokens, seconds, len(states), rtokens, rseconds, len(rstates)))
    outq.put(('stopped',))
