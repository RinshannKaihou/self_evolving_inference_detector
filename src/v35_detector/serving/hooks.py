"""Hooks called by the patched vLLM OutputProcessor (integrations/vllm_0_27/patch_output_processor.py); one service per
process that serves requests (each API server process).

    register(request_id)                      a request enters the output processor
    feed(request_id, token_ids) -> str|None   the new tokens of one engine step; in intercept mode, once the detector fires,
                                              returns the stop_reason the patch finishes the request with (as a stop string:
                                              finish_reason 'stop', and the request is aborted in the engine core)
    finish(request_id, finish_reason)         the request finished for any reason; the last partial block is scored for the log
    abort(request_id)                         the request was cancelled; its state is dropped

Environment, read when this module is imported (a misconfiguration stops the server at start-up):
  V35_DETECTOR            off (default) | shadow (detect and log) | intercept (detect, log and stop the request)
  V35_DETECTOR_TABLE      qwen3 | glm | dsv4 | /path/to/table.npz (built by v35_detector.vocab); required unless off
  V35_DETECTOR_THINK_END  auto (default; shipped tables only) | none (chat mode) | the '</think>' token id
  V35_DETECTOR_RUNNER     sync (default: inside the output-processing loop) | async (a separate detector process)
  V35_DETECTOR_CONFIG     detector config JSON (default: the shipped m4)
  V35_DETECTOR_EVENTS     JSONL of alerts, interceptions, errors, finished requests and summaries
                          (default ./v35-detector-events.jsonl)

Tokens are buffered per request and handed over in whole 64-token blocks (the detector's decision points), so the alerts
equal the evaluated detector's under any chunking. Fail-open: an error inside the detector is logged and ends detection
for that request; a dead or saturated async worker ends detection for the affected requests. Serving never fails
because of the detector.
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

from .. import CONFIG_M4
from ..core import BLOCK
from ..stream import RepDetector
from .worker import worker_main

MODES = ('off', 'shadow', 'intercept')
RUNNERS = ('sync', 'async')
STOP_PREFIX = 'v35_repetition: '
SUMMARY_EVERY_S = 10.0
QUEUE_MAX = 32767                                        # pending messages to the async worker (a block is 64 tokens);
                                                         # the largest semaphore value macOS allows


def settings(env):
    """validated settings from the environment; None when off"""
    mode = env.get('V35_DETECTOR', 'off')
    if mode not in MODES:
        raise ValueError(f'V35_DETECTOR={mode!r}: expected one of {MODES}')
    if mode == 'off':
        return None
    table = env.get('V35_DETECTOR_TABLE')
    if not table:
        raise ValueError('V35_DETECTOR_TABLE is required (qwen3 | glm | dsv4 | path to a .npz table)')
    te = env.get('V35_DETECTOR_THINK_END', 'auto')
    think_end = te if te == 'auto' else None if te == 'none' else int(te)
    runner = env.get('V35_DETECTOR_RUNNER', 'sync')
    if runner not in RUNNERS:
        raise ValueError(f'V35_DETECTOR_RUNNER={runner!r}: expected one of {RUNNERS}')
    s = dict(mode=mode, table=table, think_end=think_end, runner=runner,
             config=env.get('V35_DETECTOR_CONFIG', CONFIG_M4),
             events=env.get('V35_DETECTOR_EVENTS', './v35-detector-events.jsonl'))
    det = RepDetector(table, s['config'], think_end)     # fails now on a bad table, think_end or config
    s['think_end'] = det.think_end
    return s


class _SyncRunner:
    def __init__(self, s, on_result):
        self.det = RepDetector(s['table'], s['config'], s['think_end'])
        self.st = {}
        self.seconds = 0.0

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

    def alive(self):
        return True

    def close(self):
        pass


class _AsyncRunner:
    def __init__(self, s, on_result):
        ctx = mp.get_context('spawn')
        self.inq, self.outq = ctx.Queue(maxsize=QUEUE_MAX), ctx.Queue()
        self.proc = ctx.Process(target=worker_main, args=(self.inq, self.outq, s['table'], s['config'], s['think_end']),
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
        atexit.register(self.close)

    # ---------------------------------------------------------------- bookkeeping
    def event(self, kind, **kw):
        line = json.dumps({'ts': round(time.time(), 6), 'pid': os.getpid(), 'event': kind, **kw}, ensure_ascii=False)
        with self.lock:
            self.events.write(line + '\n')

    def summary(self):
        self.event('summary', mode=self.s['mode'], runner=self.s['runner'], counters=dict(self.n),
                   detector_seconds_sync=round(self.runner.seconds, 6) if self.runner else 0.0, worker=self.worker_stats,
                   open_requests=len(self.tokens))
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
        elif kind == 'stats':
            self.worker_stats = {'tokens': msg[1], 'detector_seconds': round(msg[2], 6), 'open_requests': msg[3]}

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
            self.runner.start(rid)
            self.pending[rid] = []
        except Exception as e:
            self.fail(rid, e, 'register')

    def feed(self, rid, token_ids):
        if rid in self.tokens:
            self.tokens[rid] += len(token_ids)
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
        with self.lock:
            late = self.alerts.pop(rid, None)            # came back from the worker, but the request ended first
            n = self.tokens.pop(rid, 0)
        if late is not None:
            self.n['alerts_after_end'] += 1
            self.event('alert', request_id=rid, fired_at=late[0], reason=late[1], after_end=True)
        self.n['finished'] += 1
        self.n['tokens'] += n
        self.event('finish', request_id=rid, tokens=n, finish_reason=finish_reason)
        if time.monotonic() - self.last_summary >= SUMMARY_EVERY_S:
            self.summary()

    def abort(self, rid):
        if self.pending.pop(rid, None) is not None:
            try:
                self.runner.drop(rid)
            except Exception:
                pass
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


def feed(request_id, token_ids):
    return _SERVICE.feed(request_id, token_ids) if _SERVICE is not None else None


def finish(request_id, finish_reason):
    if _SERVICE is not None:
        _SERVICE.finish(request_id, finish_reason)


def abort(request_id):
    if _SERVICE is not None:
        _SERVICE.abort(request_id)
