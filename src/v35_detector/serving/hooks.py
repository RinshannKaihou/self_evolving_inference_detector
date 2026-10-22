"""Hooks called by the patched vLLM OutputProcessor (integrations/vllm_0_27/patch_output_processor.py); one service per
process that serves requests (each API server process).

    register(request_id)                      a request enters the output processor
    feed(request_id, token_ids, logprobs=None) -> str|None
                                              the new tokens of one engine step and their sample logprobs (vLLM's
                                              EngineCoreOutput.new_logprobs: a LogprobsLists whose .logprobs is [tokens,
                                              1 + k], the generated token in column 0 and the top-k after it; None when the
                                              request did not ask for logprobs); in intercept mode, once m4 fires, returns the
                                              stop_reason the patch finishes the request with (as a stop string:
                                              finish_reason 'stop', and the request is aborted in the engine core)
    finish(request_id, finish_reason)         the request finished for any reason; the last partial block is scored for the log
    abort(request_id)                         the request was cancelled; its state is dropped

Environment, read when this module is imported (a misconfiguration stops the server at start-up):
  V35_DETECTOR            off (default) | shadow (detect and log) | intercept (detect, log and stop the request): m4
  V35_RARE                off (default) | shadow: the garbled-output monitor rc3; it detects and logs, it never stops a
                          request. Only requests with logprobs >= 1 are checked (it reads the top-1 and the generated
                          token's logprob); the others are skipped by rc3, m4 still checks them
  V35_DETECTOR_TABLE      qwen3 | glm | dsv4 | /path/to/table.npz (built by v35_detector.vocab); required unless both are off;
                          rc3 needs a shipped name (its rare and reference tables come with it)
  V35_DETECTOR_THINK_END  auto (default; shipped tables only) | none (chat mode) | the '</think>' token id
  V35_DETECTOR_RUNNER     sync (default: inside the output-processing loop) | async (a separate detector process); both
                          detectors run in the same place
  V35_DETECTOR_CONFIG     m4 config JSON (default: the shipped m4); rc3 always uses the shipped rc3
  V35_DETECTOR_EVENTS     JSONL of alerts (m4: 'alert', rc3: 'rare_alert'), interceptions, errors, finished requests and
                          summaries (default ./v35-detector-events.jsonl)

Tokens (and, for rc3, their two logprobs) are buffered per request and handed over in whole 64-token blocks, so the alerts
equal the evaluated detectors' under any chunking. Fail-open: an error inside a detector is logged and ends that detector
for that request; a dead or saturated async worker ends detection for the affected requests. Serving never fails
because of the detectors. With V35_RARE=off the m4 path is the one tested on the H20 (code tag h20-r1).
"""
import atexit
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter

import numpy as np

from .. import CONFIG_M4, CONFIG_RC3
from ..core import BLOCK
from ..rare import RareDetector
from ..stream import RepDetector
from ..vocab import THINK_END
from .worker import rare_step, worker_main

MODES = ('off', 'shadow', 'intercept')
RARE_MODES = ('off', 'shadow')                           # rc3 monitors only: there is no intercept mode
RUNNERS = ('sync', 'async')
STOP_PREFIX = 'v35_repetition: '
SUMMARY_EVERY_S = 10.0
QUEUE_MAX = 32767                                        # pending messages to the async worker (a block is 64 tokens);
                                                         # the largest semaphore value macOS allows


def settings(env):
    """validated settings from the environment; None when both detectors are off"""
    mode = env.get('V35_DETECTOR', 'off')
    if mode not in MODES:
        raise ValueError(f'V35_DETECTOR={mode!r}: expected one of {MODES}')
    rare = env.get('V35_RARE', 'off')
    if rare not in RARE_MODES:
        raise ValueError(f'V35_RARE={rare!r}: expected one of {RARE_MODES} (rc3 only monitors, it never stops a request)')
    if mode == 'off' and rare == 'off':
        return None
    table = env.get('V35_DETECTOR_TABLE')
    if not table:
        raise ValueError('V35_DETECTOR_TABLE is required (qwen3 | glm | dsv4 | path to a .npz table)')
    te = env.get('V35_DETECTOR_THINK_END', 'auto')
    think_end = te if te == 'auto' else None if te == 'none' else int(te)
    runner = env.get('V35_DETECTOR_RUNNER', 'sync')
    if runner not in RUNNERS:
        raise ValueError(f'V35_DETECTOR_RUNNER={runner!r}: expected one of {RUNNERS}')
    s = dict(mode=mode, rare=rare, table=table, think_end=think_end, runner=runner,
             config=env.get('V35_DETECTOR_CONFIG', CONFIG_M4), rare_config=CONFIG_RC3,
             events=env.get('V35_DETECTOR_EVENTS', './v35-detector-events.jsonl'))
    if mode != 'off':
        det = RepDetector(table, s['config'], think_end)     # fails now on a bad table, think_end or config
        s['think_end'] = det.think_end
    if rare != 'off':
        if table not in THINK_END:
            raise ValueError(f'V35_RARE needs a shipped table in V35_DETECTOR_TABLE ({" | ".join(THINK_END)}), got {table!r}: '
                             'rc3 also needs the model\'s rare and reference tables')
        if s['think_end'] == 'auto':
            s['think_end'] = THINK_END[table]
        if runner == 'sync':                             # loads its tables now; the async worker loads its own copy
            RareDetector(table, CONFIG_RC3, s['think_end'])
    return s


class _SyncRunner:
    def __init__(self, s, on_result):
        self.det = RepDetector(s['table'], s['config'], s['think_end']) if s['mode'] != 'off' else None
        self.rare = RareDetector(s['table'], s['rare_config'], s['think_end']) if s['rare'] != 'off' else None
        self.st = {}
        self.rst = {}                                    # rid -> [rc3 state, monitoring-tier alert reported]
        self.seconds = 0.0
        self.rare_seconds = 0.0

    def start(self, rid):
        self.st[rid] = self.det.start()

    def feed(self, rid, block):
        t0 = time.perf_counter()
        res = self.det.feed(self.st[rid], block)
        self.seconds += time.perf_counter() - t0
        return None if res is None else (res.fired_at, res.reason)

    def finish(self, rid, rest):
        st = self.st.pop(rid)
        t0 = time.perf_counter()
        res = (self.det.feed(st, rest) if len(rest) else None) or self.det.finish(st)
        self.seconds += time.perf_counter() - t0
        return (res.fired_at, res.reason) if res.is_ill else None

    def drop(self, rid):
        self.st.pop(rid, None)

    def rare_start(self, rid):
        self.rst[rid] = [self.rare.start(), False]

    def rare_feed(self, rid, ids, top1, chosen):
        """new alerts, and whether rc3 is done with the request (its strict tier fired)"""
        ent = self.rst[rid]
        t0 = time.perf_counter()
        hits = rare_step(self.rare, ent, ids, top1, chosen, False)
        self.rare_seconds += time.perf_counter() - t0
        if ent[0].done:
            del self.rst[rid]
        return hits, ent[0].done

    def rare_finish(self, rid, ids, top1, chosen):
        ent = self.rst.pop(rid)
        t0 = time.perf_counter()
        hits = rare_step(self.rare, ent, ids, top1, chosen, True)
        self.rare_seconds += time.perf_counter() - t0
        return hits

    def rare_drop(self, rid):
        self.rst.pop(rid, None)

    def alive(self):
        return True

    def close(self):
        pass


class _AsyncRunner:
    def __init__(self, s, on_result):
        ctx = mp.get_context('spawn')
        self.inq, self.outq = ctx.Queue(maxsize=QUEUE_MAX), ctx.Queue()
        self.proc = ctx.Process(target=worker_main, args=(self.inq, self.outq, s['table'],
                                                          s['config'] if s['mode'] != 'off' else None, s['think_end'],
                                                          s['rare_config'] if s['rare'] != 'off' else None),
                                daemon=True, name='v35-detector')
        # spawn re-runs the parent's entry script in the child (for vLLM: its CLI, i.e. torch and all of vLLM); the worker
        # needs only this package, so the entry point is hidden while the child is prepared
        main = sys.modules.get('__main__')
        saved = (getattr(main, '__spec__', None), getattr(main, '__file__', None)) if main is not None else None
        if saved is not None:
            main.__spec__ = None
            if saved[1] is not None:
                del main.__file__
        try:
            self.proc.start()
        finally:
            if saved is not None:
                main.__spec__ = saved[0]
                if saved[1] is not None:
                    main.__file__ = saved[1]
        msg = self.outq.get(timeout=60)
        if msg[0] != 'ready':
            raise RuntimeError(f'v35 detector worker failed to start: {msg}')
        self.on_result = on_result
        self.seconds = 0.0                               # detector time is spent in the worker (its 'stats' messages)
        self.rare_seconds = 0.0
        self.closed = False
        self.thread = threading.Thread(target=self._collect, daemon=True, name='v35-detector-collector')
        self.thread.start()
        # multiprocessing registered its exit handler (it terminates daemon children) just now, after the Service's;
        # atexit runs last-registered first, so this stops the worker cleanly, with its final stats, before that handler
        atexit.register(self.close)

    def _collect(self):
        while True:
            try:
                msg = self.outq.get()
            except (EOFError, OSError):
                return
            if msg[0] == 'stopped':
                return
            self.on_result(msg)

    def start(self, rid):
        self.inq.put_nowait(('start', rid))

    def feed(self, rid, block):
        self.inq.put_nowait(('feed', rid, block))
        return None

    def finish(self, rid, rest):
        self.inq.put_nowait(('finish', rid, rest))
        return None

    def drop(self, rid):
        self.inq.put_nowait(('drop', rid))

    def rare_start(self, rid):
        self.inq.put_nowait(('rstart', rid))

    def rare_feed(self, rid, ids, top1, chosen):         # the block's two logprob arrays travel with its token ids
        self.inq.put_nowait(('rfeed', rid, ids, top1, chosen))
        return (), False                                 # alerts come back through the collector

    def rare_finish(self, rid, ids, top1, chosen):
        self.inq.put_nowait(('rfinish', rid, ids, top1, chosen))
        return ()

    def rare_drop(self, rid):
        self.inq.put_nowait(('rdrop', rid))

    def alive(self):
        return self.proc.is_alive()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.inq.put(('stop',), timeout=1)
            self.thread.join(timeout=5)
        except Exception:
            pass
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()


class Service:
    def __init__(self, s):
        self.s = s
        self.lock = threading.Lock()
        self.events = open(s['events'], 'a', buffering=1, encoding='utf-8')
        self.pending = {}                                # rid -> buffered token ids (< BLOCK), only while detecting
        self.tokens = {}                                 # rid -> tokens seen, until the request ends
        self.alerts = {}                                 # rid -> (fired_at, reason) from the async worker, not yet acted on
        self.n = Counter()
        self.worker_stats = None
        self.runner = None                               # built at the first request: no detector process at import time
        self.broken = False                              # the runner could not be built: detection is off for the process
        self.last_summary = time.monotonic()
        self.m4 = s['mode'] != 'off'
        self.rare = s['rare'] != 'off'
        self.rare_wait = set()                           # rc3: registered, no token yet (whether it has logprobs is unknown)
        self.rare_pending = {}                           # rc3: rid -> (ids, top-1 logprobs, chosen logprobs) (< BLOCK each)
        self.rare_status = {}                            # rc3 coverage for the finish event (see rare_finish)
        atexit.register(self.close)

    # ---------------------------------------------------------------- bookkeeping
    def event(self, kind, **kw):
        line = json.dumps({'ts': round(time.time(), 6), 'pid': os.getpid(), 'event': kind, **kw}, ensure_ascii=False)
        with self.lock:
            self.events.write(line + '\n')

    def summary(self):
        extra = {}
        if self.rare:                                    # rc3's own detector time, apart from m4's
            extra = dict(rare=self.s['rare'], rare_seconds_sync=round(self.runner.rare_seconds, 6) if self.runner else 0.0)
        self.event('summary', mode=self.s['mode'], runner=self.s['runner'], counters=dict(self.n),
                   detector_seconds_sync=round(self.runner.seconds, 6) if self.runner else 0.0, worker=self.worker_stats,
                   open_requests=len(self.tokens), **extra)
        self.last_summary = time.monotonic()

    def fail(self, rid, error, what):
        self.n['errors'] += 1
        self.event('error', request_id=rid, where=what, error=f'{type(error).__name__}: {error}')
        self.pending.pop(rid, None)
        try:
            if self.runner is not None:
                self.runner.drop(rid)
        except Exception:
            pass

    def on_result(self, msg):                            # async worker -> collector thread
        kind = msg[0]
        if kind == 'alert':
            _, rid, at, reason = msg
            with self.lock:
                still_open = rid in self.tokens
                if still_open:
                    self.alerts[rid] = (at, reason)      # acted on at the request's next feed
            if not still_open:                           # the request ended before the alert came back
                self.n['alerts_after_end'] += 1
                self.event('alert', request_id=rid, fired_at=at, reason=reason, after_end=True)
        elif kind == 'final':
            _, rid, at, reason = msg
            self.n['alerts_after_end'] += 1
            self.event('alert', request_id=rid, fired_at=at, reason=reason, after_end=True)
        elif kind == 'error':
            self.n['errors'] += 1
            self.event('error', request_id=msg[1], where='worker', error=msg[2])
        elif kind == 'ralert':
            _, rid, tier, at, reason, after_end = msg
            self.rare_log(rid, tier, at, reason, after_end or rid not in self.tokens)
        elif kind == 'rerror':
            self.n['errors'] += 1
            self.event('error', request_id=msg[1], where='rare_worker', error=msg[2])
            with self.lock:
                if msg[1] in self.rare_status:
                    self.rare_status[msg[1]] = 'error'
        elif kind == 'stats':
            self.worker_stats = {'tokens': msg[1], 'detector_seconds': round(msg[2], 6), 'open_requests': msg[3]}
            if self.rare:
                self.worker_stats.update(rare_tokens=msg[4], rare_seconds=round(msg[5], 6), rare_open_requests=msg[6])

    def act(self, rid, hit):
        at, reason = hit
        self.pending.pop(rid, None)                      # first alert ends detection for the request
        try:
            self.runner.drop(rid)
        except Exception:
            pass
        self.n['alerts'] += 1
        self.event('alert', request_id=rid, fired_at=at, reason=reason, tokens_seen=self.tokens.get(rid, 0),
                   action=self.s['mode'])
        if self.s['mode'] != 'intercept':
            return None
        self.n['intercepts'] += 1
        return STOP_PREFIX + reason

    # ---------------------------------------------------------------- rc3 (monitor only: nothing here ends a request)
    def rare_log(self, rid, tier, at, reason, after_end=False):
        self.n['rare_' + tier] += 1
        if after_end:
            self.event('rare_alert', request_id=rid, tier=tier, fired_at=at, reason=reason, after_end=True)
        else:
            self.event('rare_alert', request_id=rid, tier=tier, fired_at=at, reason=reason,
                       tokens_seen=self.tokens.get(rid, 0), action=self.s['rare'])

    def rare_stop(self, rid, status):
        """rc3 ends for the request (status: lost | error); m4 is not affected"""
        self.rare_pending.pop(rid, None)
        self.rare_wait.discard(rid)
        with self.lock:
            self.rare_status[rid] = status
        try:
            if self.runner is not None:
                self.runner.rare_drop(rid)
        except Exception:
            pass

    def rare_fail(self, rid, error, what):
        self.n['errors'] += 1
        self.event('error', request_id=rid, where=what, error=f'{type(error).__name__}: {error}')
        self.rare_stop(rid, 'error')

    def rare_feed(self, rid, token_ids, logprobs):
        if not len(token_ids):
            return
        buf = self.rare_pending.get(rid)
        if buf is None:
            if rid not in self.rare_wait:                # not checked by rc3 (no logprobs, ended, error, not registered)
                return
            self.rare_wait.discard(rid)                  # the request's first tokens decide whether rc3 can check it
            if logprobs is None:
                self.rare_status[rid] = 'no_logprobs'
                self.n['rare_no_logprobs'] += 1
                return
            try:
                self.runner.rare_start(rid)
            except Exception as e:
                self.rare_fail(rid, e, 'rare_start')
                return
            buf = self.rare_pending[rid] = ([], [], [])
            self.rare_status[rid] = 'checked'
            self.n['rare_requests'] += 1
        try:
            lp = None if logprobs is None else logprobs.logprobs
            if lp is None or lp.ndim != 2 or lp.shape[0] != len(token_ids) or lp.shape[1] < 2:
                self.n['rare_lost'] += 1                 # logprobs stopped coming, or without a top-1 (logprobs=0)
                self.rare_stop(rid, 'lost')
                return
            ids, top1, chosen = buf
            ids.extend(token_ids)
            for row in lp.tolist():                      # vLLM's sampler: column 0 the generated token, then the top-k
                chosen.append(row[0])
                top1.append(row[1])
            if len(ids) >= BLOCK:
                k = len(ids) - len(ids) % BLOCK
                hits, done = self.runner.rare_feed(rid, np.asarray(ids[:k], np.int64), np.asarray(top1[:k]),
                                                   np.asarray(chosen[:k]))
                self.n['rare_tokens'] += k
                del ids[:k], top1[:k], chosen[:k]
                for h in hits:
                    self.rare_log(rid, *h)
                if done:                                 # strict tier fired: rc3 has nothing left to check here
                    self.rare_pending.pop(rid, None)
        except Exception as e:
            self.rare_fail(rid, e, 'rare_feed')

    def rare_finish(self, rid):
        """rc3 coverage of the request: checked (every token, or up to its strict alert) | no_logprobs (the request did not
        ask for logprobs) | lost (logprobs stopped coming, or came without a top-1) | error | none (no token, or detection
        unavailable)"""
        self.rare_wait.discard(rid)
        buf = self.rare_pending.pop(rid, None)
        if buf is not None:
            try:
                ids, top1, chosen = buf
                hits = self.runner.rare_finish(rid, np.asarray(ids, np.int64), np.asarray(top1, np.float64),
                                               np.asarray(chosen, np.float64))
                self.n['rare_tokens'] += len(ids)
                for h in hits:
                    self.rare_log(rid, *h, after_end=True)
            except Exception as e:
                self.rare_fail(rid, e, 'rare_finish')
        with self.lock:
            return self.rare_status.pop(rid, 'none')

    # ---------------------------------------------------------------- hooks
    def register(self, rid):
        self.tokens[rid] = 0
        self.n['requests'] += 1
        if self.broken:
            return
        try:
            if self.runner is None:
                try:
                    self.runner = (_AsyncRunner if self.s['runner'] == 'async' else _SyncRunner)(self.s, self.on_result)
                except Exception:
                    self.broken = True
                    raise
            if not self.runner.alive():
                self.n['worker_down'] += 1
                return
            if self.m4:
                self.runner.start(rid)
                self.pending[rid] = []
            if self.rare:
                self.rare_wait.add(rid)
        except Exception as e:
            self.fail(rid, e, 'register')

    def feed(self, rid, token_ids, logprobs=None):
        if rid in self.tokens:
            self.tokens[rid] += len(token_ids)
        if self.rare:
            self.rare_feed(rid, token_ids, logprobs)
        buf = self.pending.get(rid)
        if buf is None:
            return None
        try:
            buf.extend(token_ids)
            hit = None
            if len(buf) >= BLOCK:
                k = len(buf) - len(buf) % BLOCK
                hit = self.runner.feed(rid, np.asarray(buf[:k], np.int64))
                del buf[:k]
            if hit is None and self.alerts:
                with self.lock:
                    hit = self.alerts.pop(rid, None)
        except Exception as e:
            self.fail(rid, e, 'feed')
            return None
        return None if hit is None else self.act(rid, hit)

    def finish(self, rid, finish_reason):
        buf = self.pending.pop(rid, None)
        if buf is not None:
            try:
                hit = self.runner.finish(rid, np.asarray(buf, np.int64))
                if hit is not None:
                    self.n['alerts_after_end'] += 1
                    self.event('alert', request_id=rid, fired_at=hit[0], reason=hit[1], after_end=True)
            except Exception as e:
                self.fail(rid, e, 'finish')
        extra = {'rare': self.rare_finish(rid)} if self.rare else {}
        with self.lock:
            late = self.alerts.pop(rid, None)            # came back from the worker, but the request ended first
            n = self.tokens.pop(rid, 0)
        if late is not None:
            self.n['alerts_after_end'] += 1
            self.event('alert', request_id=rid, fired_at=late[0], reason=late[1], after_end=True)
        self.n['finished'] += 1
        self.n['tokens'] += n
        self.event('finish', request_id=rid, tokens=n, finish_reason=finish_reason, **extra)
        if time.monotonic() - self.last_summary >= SUMMARY_EVERY_S:
            self.summary()

    def abort(self, rid):
        if self.pending.pop(rid, None) is not None:
            try:
                self.runner.drop(rid)
            except Exception:
                pass
        if self.rare:
            self.rare_wait.discard(rid)
            if self.rare_pending.pop(rid, None) is not None:
                try:
                    self.runner.rare_drop(rid)
                except Exception:
                    pass
            with self.lock:
                self.rare_status.pop(rid, None)
        with self.lock:
            self.alerts.pop(rid, None)
            known = self.tokens.pop(rid, None) is not None
        if known:
            self.n['aborted'] += 1

    def close(self):
        if self.events.closed:                           # explicit close, then atexit
            return
        if self.n['requests']:
            if self.runner is not None:
                self.runner.close()
            self.summary()
        self.events.close()


_SERVICE = None
_s = settings(os.environ)
if _s is not None:
    _SERVICE = Service(_s)


def register(request_id):
    if _SERVICE is not None:
        _SERVICE.register(request_id)


def feed(request_id, token_ids, logprobs=None):
    return _SERVICE.feed(request_id, token_ids, logprobs) if _SERVICE is not None else None


def finish(request_id, finish_reason):
    if _SERVICE is not None:
        _SERVICE.finish(request_id, finish_reason)


def abort(request_id):
    if _SERVICE is not None:
        _SERVICE.abort(request_id)
