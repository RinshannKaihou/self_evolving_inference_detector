"""Streaming repetition detector (v3.5 labels; config m4: 14 components in four families). Token ids only.

    det = RepDetector('qwen3', CONFIG_M4)                   # table name or .npz path; config: dict or JSON path; None -> probe mode
    st = det.start()                                         # one state per request
    for chunk in chunks:                                     # any chunk size; processed in BLOCK (64) token blocks
        res = det.feed(st, chunk)                            # Result once an intercept component fires, else None
    res = det.finish(st)                                     # flushes the last partial block; Result() if nothing fired
    mon = det.monitor(st)                                    # first alert of the monitoring tier (components with "tier": "monitor"
                                                             # only record it; scoring continues until an intercept alert)

Component kinds (config "components", evaluated in order at every boundary; a component fires once its condition has held
for S consecutive boundaries):
  M    one feature >= theta, optional separator guard (sep <= guard), novelty cap (nov) and extra bounds (and)
  V    vLLM check_sequence_repetition([mn, mx] x c), exact, reported at the first boundary at or after the prefix
  RUN  longest run of one repeated non-space character >= min_chars
  WS   whitespace share of the last RAW_W raw tokens >= pct with >= min_chars characters
  G    gradient-boosted trees (trees.py) score >= theta
Feature families (core.py): verbatim n-gram copies (A2 / A3 / A5 / P with novelty N), lexical no-progress (wrep, mark),
sentence restatement (sdup).

Bounded per-request state: per n-gram size a sorted key array of the last H normalised positions (+ one block), the last
ng-1 normalised ids, cumulative flag counts of the last max(W) positions; the last RAW_KEEP raw tokens (vLLM-equivalent
rule); the seen-id bitmap and cumulative word-reuse / marker counts; the last SENT_HIST sentence signatures; a few counters.
Features are defined in core.py; offline.py computes the same per-boundary rows for a whole record.
Probe mode (config None) computes every feature and stores each boundary's row in st.rows.
"""
import json
from dataclasses import dataclass, field

import numpy as np

from . import core as C
from .trees import Trees, inputs
from .vocab import SEP, THINK_END, load as load_tables


@dataclass
class Result:
    is_ill: bool = False
    fired_at: int = -1          # raw-token boundary at which the alert is raised
    reason: str = ''


class Ring:
    """the last `keep` values of a 1-d series, addressed by absolute index (compacted once per `slack` appends)"""
    __slots__ = ('a', 'base', 'end', 'keep')

    def __init__(self, keep, dtype, slack=256):
        self.a = np.zeros(keep + slack, dtype)
        self.base = 0
        self.end = 0
        self.keep = keep

    def append(self, x):
        n = len(x)
        if self.end - self.base + n > len(self.a):
            k = min(self.keep, self.end - self.base)
            self.a[:k] = self.a[self.end - self.base - k:self.end - self.base]
            self.base = self.end - k
        s = self.end - self.base
        self.a[s:s + n] = x
        self.end += n

    def at(self, i):
        return self.a[i - self.base]

    def span(self, lo, hi):
        return self.a[lo - self.base:hi - self.base]


class NgramStream:
    """c / d1 / d2 of the normalised positions for one n-gram size; cumulative counts of the needed flags"""

    def __init__(self, ng, flags, wmax):
        self.ng = ng
        self.keys = np.zeros(0, np.uint64)
        self.ctx = np.zeros(0, np.int64)                 # last ng-1 normalised ids
        self.cum = {f: Ring(wmax + 1, np.int32) for f in flags}      # int32: window differences stay exact
        for r in self.cum.values():
            r.append(np.zeros(1, np.int64))              # cum[0] = 0
        self.M = 0

    def push(self, x, answer_at=None):
        ng, m, M0 = self.ng, len(x), self.M
        c, d1, d2 = (np.zeros(m, np.int64) for _ in range(3))
        w = np.concatenate([self.ctx, x])
        j0 = max(M0, ng - 1)                             # first new position holding a full n-gram
        k = M0 + m - j0
        if k > 0:
            j = np.arange(j0, M0 + m, dtype=np.int64)
            hi = C.ngram_hi(w[len(w) - k - ng + 1:], k, ng)
            kn = hi | j.astype(np.uint64)
            order = np.argsort(kn, kind='stable')
            ks = kn[order]
            keys = np.insert(self.keys, np.searchsorted(self.keys, ks), ks)
            h = np.searchsorted(keys, kn)
            floor = np.maximum(j - C.H, 0)
            if answer_at is not None:
                floor = np.where(j >= answer_at, np.maximum(floor, answer_at), floor)
            lo = np.searchsorted(keys, hi | floor.astype(np.uint64))
            cnt = h - lo
            pos = (keys & np.uint64(C.POSMASK)).astype(np.int64)
            a = j - pos[np.maximum(h - 1, 0)]
            b = j - pos[np.maximum(h - 2, 0)]
            s = j0 - M0
            c[s:] = np.minimum(cnt, C.CCAP)
            d1[s:] = np.where(cnt >= 1, a, 0)
            d2[s:] = np.where(cnt >= 2, b, 0)
            cut = M0 + m - C.H                           # later positions only look back to >= cut
            self.keys = keys[pos >= cut] if cut > 0 else keys
        self.ctx = w[max(len(w) - (ng - 1), 0):] if ng > 1 else self.ctx
        self.M = M0 + m
        fl = C.flags(c, d1, d2)
        for f, r in self.cum.items():
            r.append(r.at(r.end - 1) + np.cumsum(fl[f]))

    def share(self, f, w):
        if self.M < w:
            return np.nan
        r = self.cum[f]
        return (r.at(self.M) - r.at(self.M - w)) / w


def exact_block(buf, base, end, n_new, pmin, pmax, c):
    """vLLM check_sequence_repetition([pmin, pmax] x c) holds at some prefix ending in the last n_new tokens.
    buf holds raw tokens of absolute positions [base, end). For lag p, eq(q) = tok[q] == tok[q-p]; the check holds at prefix
    t iff eq on [t-(c-1)p, t-1] (proof in the previous repo's scripts/vllm_native.py). Exact prefilter: with need_p =
    (c-1)p >= n_new, a qualifying run ending in the block covers every position of [end-need_p, B0] (B0 = end - n_new), so
    lag p must match at 6 points spread over that interval; the survivors are verified exactly."""
    B0 = end - n_new
    hi_p = min(pmax, (end - base) // c)                  # a run needs c*p tokens of history
    if n_new <= 0 or hi_p < pmin:
        return False
    ps = np.arange(pmin, hi_p + 1)
    if (c - 1) * pmin >= n_new:
        # point B0 (k = 0) for every lag at once: buf[B0 - p] for p = pmin..hi_p is a reversed contiguous slice
        if B0 - hi_p < base:
            return False if B0 - pmin < base else exact_block(buf, base, end, n_new, pmin, B0 - base, c)
        ps = ps[buf[B0 - hi_p - base:B0 - pmin + 1 - base][::-1] == buf[B0 - base]]
        for k in range(1, 6):                            # further points only on the survivors
            if not ps.size:
                return False
            x = B0 - (((c - 1) * ps - n_new) * k) // 5   # covered interval of lag p: [B0 - ((c-1)p - n_new), B0]
            src = x - ps
            v = src >= base                              # (then x >= base too)
            ps = ps[v & (buf[np.maximum(src, base) - base] == buf[np.maximum(x, base) - base])]
    for p in ps.tolist():
        need = (c - 1) * p
        lo = max(B0 - need + 1, base + p)                # earliest q a run ending in the block can start at
        q = np.arange(lo, end)
        if len(q) < need:
            continue
        eq = buf[q - base] == buf[q - p - base]
        cs = np.concatenate(([0], np.cumsum(eq)))
        e = np.arange(max(B0, lo + need - 1), end) - lo  # run end indices inside the block
        if e.size and (cs[e + 1] - cs[e + 1 - need] == need).any():
            return True
    return False


class State:
    def __init__(self, det):
        self.t = 0                                       # raw tokens consumed into blocks
        self.pending = np.zeros(0, np.int64)             # tokens not yet forming a full block
        self.streams = {ng: NgramStream(ng, det.need[ng], max(C.WINS)) for ng in det.need}
        self.raw = Ring(det.raw_keep, np.int32) if det.raw_keep else None
        self.sepcum = Ring(C.RAW_W + 1, np.int32)
        self.chcum = Ring(C.RAW_W + 1, np.int32)
        self.wscum = Ring(C.RAW_W + 1, np.int32)
        for r in (self.sepcum, self.chcum, self.wscum):
            r.append(np.zeros(1, np.int64))
        self.run_char, self.run_len, self.run_max = -1, 0, 0
        self.answer_at = None                            # normalised position of the first think-end token
        self.nx = 0                                      # normalised positions consumed
        if det.sentences:
            self.cur = []                                            # normalised ids of the open sentence (<= SENT_MAX)
            self.shist = np.zeros((C.SENT_HIST, len(C.SEEDS)), np.uint64)   # signatures of the last SENT_HIST sentences
            self.nh = 0                                              # counted sentences so far
            self.dupcum = Ring(max(C.SDUP_NS) + 1, np.int32)
            self.dupcum.append(np.zeros(1, np.int64))
        if det.lexical:
            self.seen = np.zeros(det.nbits // 8 + 1, np.uint8)       # bitmap of normalised ids seen so far
            self.repcum = Ring(max(C.WREP_WINS) + 1, np.int32)
            self.markcum = Ring(max(C.MARK_WINS) + 1, np.int32)
            for r in (self.repcum, self.markcum):
                r.append(np.zeros(1, np.int64))
        self.streak = [0] * len(det.comps)
        self.result = None
        self.monitor = None                              # first alert of the monitoring tier (any component)
        self.done = False
        self.rows = []                                   # probe mode: (t, features)


class RepDetector:
    def __init__(self, corpus, config=None, think_end='auto'):
        """corpus: shipped table name (qwen3 | glm | dsv4) or a .npz built by vocab.py.
        think_end: token id after which the answer starts ('auto': the shipped table's, vocab.THINK_END; None: chat mode)"""
        self.T = load_tables(corpus)
        if think_end == 'auto' and corpus not in THINK_END:
            raise ValueError(f"think_end='auto' needs a shipped table ({sorted(THINK_END)}); pass the '</think>' id or None")
        self.think_end = THINK_END[corpus] if think_end == 'auto' else think_end
        if isinstance(config, str):
            config = json.load(open(config))
        self.probe = config is None
        self.comps = [] if self.probe else config['components']
        if self.probe:
            self.need = {ng: list(C.FLAGS) for ng in C.NGS}
            self.lexical = self.sentences = True
        else:
            self.need, self.lexical, self.sentences = {}, False, False
            for comp in self.comps:
                if comp['kind'] == 'M':
                    feats = [comp['feat']] + ([comp['nov']['feat']] if comp.get('nov') else []) + [a['feat'] for a in comp.get('and', [])]
                elif comp['kind'] == 'G':
                    feats = comp['feats']
                else:
                    continue
                for feat in feats:
                    if feat.startswith(('wrep_', 'mark_')):
                        self.lexical = True
                        continue
                    if feat.startswith('sdup_'):
                        self.sentences = True
                        continue
                    if feat == 'tpos':
                        continue
                    if feat in ('sep', 'ws_frac', 'chars', 'run'):
                        continue
                    f, ng, _ = feat.split('_')
                    self.need.setdefault(int(ng[1:]), [])
                    if f not in self.need[int(ng[1:])]:
                        self.need[int(ng[1:])].append(f)
        self.trees = {i: Trees(c['trees']) for i, c in enumerate(self.comps) if c['kind'] == 'G'}
        self.nbits = int(self.T['norm'].max()) + 1
        vs = [c for c in self.comps if c['kind'] == 'V']
        self.raw_keep = max((c['c'] * c['mx'] for c in vs), default=0) + C.BLOCK if vs else 0

    def start(self):
        return State(self)

    def feed(self, st, tokens):
        if st.done:
            return st.result
        tok = np.asarray(tokens, np.int64)
        if len(st.pending):
            tok = np.concatenate([st.pending, tok])
        i = 0
        while len(tok) - i >= C.BLOCK and not st.done:
            self._block(st, tok[i:i + C.BLOCK])
            i += C.BLOCK
        st.pending = tok[i:].copy() if not st.done else tok[:0]
        return st.result if st.result is not None and not self.probe else None

    def finish(self, st):
        if not st.done and len(st.pending):
            blk, st.pending = st.pending, st.pending[:0]
            self._block(st, blk)
        st.done = True
        return st.result if st.result is not None else Result()

    @staticmethod
    def monitor(st):
        """first alert of the monitoring tier (components with tier 'monitor' plus every intercept component)"""
        return st.monitor if st.monitor is not None else Result()

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _close_sentence(st):
        if len(st.cur) >= C.SENT_MIN:
            mh = C.minhash(st.cur)
            h = st.shist[:min(st.nh, C.SENT_HIST)]
            dup = bool(((h == mh).sum(1) >= C.SENT_AGREE).any()) if len(h) else False
            st.shist[st.nh % C.SENT_HIST] = mh
            st.nh += 1
            st.dupcum.append(np.array([st.dupcum.at(st.dupcum.end - 1) + dup], np.int64))
        st.cur = []

    def _block(self, st, tok):
        T = self.T
        norm = T['norm'][tok]
        keep = norm != SEP
        x = norm[keep].astype(np.int64)
        if st.answer_at is None and self.think_end is not None:
            hit = np.flatnonzero(tok == self.think_end)
            if len(hit):
                st.answer_at = st.nx + int(np.count_nonzero(keep[:hit[0]]))
        for s in st.streams.values():
            s.push(x, st.answer_at)
        if self.lexical:
            rep = np.ones(len(x), bool)                  # an id is new at its first occurrence in the output
            if len(x):
                u, first = np.unique(x, return_index=True)
                unseen = ((st.seen[u >> 3] >> (u & 7).astype(np.uint8)) & 1) == 0
                rep[first[unseen]] = False
                np.bitwise_or.at(st.seen, u >> 3, (np.uint8(1) << (u & 7).astype(np.uint8)))
            st.repcum.append(st.repcum.at(st.repcum.end - 1) + np.cumsum(rep))
            st.markcum.append(st.markcum.at(st.markcum.end - 1) + np.cumsum(T['marker'][tok]))
        if self.sentences:
            ends = np.flatnonzero(T['send'][tok])
            lo = 0
            for e in ends.tolist() + [None]:                         # segments closed by a sentence-end token
                hi = len(tok) if e is None else e + 1
                seg = norm[lo:hi]
                ids = seg[seg != SEP].tolist()
                while len(st.cur) + len(ids) >= C.SENT_MAX:           # closed by length (the id that reaches SENT_MAX)
                    take = C.SENT_MAX - len(st.cur)
                    st.cur += ids[:take]
                    ids = ids[take:]
                    self._close_sentence(st)
                st.cur += ids
                if e is not None:
                    self._close_sentence(st)
                lo = hi
        st.nx += len(x)
        if st.raw is not None:
            st.raw.append(tok)
        ch = T['chars'][tok].astype(np.int64)
        st.sepcum.append(st.sepcum.at(st.sepcum.end - 1) + np.cumsum(~keep))
        st.chcum.append(st.chcum.at(st.chcum.end - 1) + np.cumsum(ch))
        st.wscum.append(st.wscum.at(st.wscum.end - 1) + np.cumsum(T['ws'][tok].astype(np.int64)))
        rc = T['run_char'][tok]
        for c, n in zip(rc.tolist(), ch.tolist()):        # longest run of one repeated non-space character
            if c < 0:
                st.run_char, st.run_len = -1, 0
            elif c == st.run_char:
                st.run_len += n
            else:
                st.run_char, st.run_len = c, n
            if st.run_len > st.run_max:
                st.run_max = st.run_len
        n_new = len(tok)
        st.t += n_new
        t = st.t
        lo = max(t - C.RAW_W, 0)
        span = max(t - lo, 1)
        raw = {'sep': (st.sepcum.at(t) - st.sepcum.at(lo)) / span}
        chars = st.chcum.at(t) - st.chcum.at(lo)
        raw['ws_frac'] = (st.wscum.at(t) - st.wscum.at(lo)) / max(chars, 1) if chars > 0 else 0.0
        raw['chars'] = float(chars)
        raw['run'] = float(st.run_max)

        memo = {}

        def val(f):
            if f not in memo:
                memo[f] = _val(f)
            return memo[f]

        def _val(f):
            if f in raw:
                return raw[f]
            if f.startswith('wrep_'):
                w = int(f[6:])
                return (st.repcum.at(st.nx) - st.repcum.at(st.nx - w)) / w if st.nx >= w else np.nan
            if f.startswith('mark_'):
                w = int(f[6:])
                return (st.markcum.at(t) - st.markcum.at(t - w)) / w if t >= w else np.nan
            if f.startswith('sdup_N'):
                n = int(f[6:])
                return (st.dupcum.at(st.nh) - st.dupcum.at(st.nh - n)) / n if st.nh >= n else np.nan
            if f == 'tpos':
                return float(t)
            fl, ng, w = f.split('_')
            return st.streams[int(ng[1:])].share(fl, int(w[1:]))

        if self.probe:
            st.rows.append((t, np.array([val(f) for f in C.FEATS], np.float32)))
            return
        for i, comp in enumerate(self.comps):
            k = comp['kind']
            if k == 'M':
                v = val(comp['feat'])
                ok = (not np.isnan(v)) and float(np.float32(v)) >= comp['theta']     # float32: as the calibration cache
                if ok and comp.get('guard') is not None:
                    ok = float(np.float32(raw['sep'])) <= comp['guard']
                if ok and comp.get('nov'):
                    v = val(comp['nov']['feat'])
                    ok = (not np.isnan(v)) and float(np.float32(v)) <= comp['nov']['max']
                for a in comp.get('and', []):
                    if not ok:
                        break
                    v = val(a['feat'])
                    ok = (not np.isnan(v)) and (float(np.float32(v)) >= a['min'] if 'min' in a else float(np.float32(v)) <= a['max'])
            elif k == 'G':
                x = inputs([None if f == 'tpos' else val(f) for f in comp['feats']], t)
                ok = bool(self.trees[i].score(x[None])[0] >= comp['theta'])
            elif k == 'V':
                ok = exact_block(st.raw.a, st.raw.base, st.raw.end, n_new, comp['mn'], comp['mx'], comp['c'])
            elif k == 'RUN':
                ok = raw['run'] >= comp['min_chars']
            elif k == 'WS':
                ok = raw['ws_frac'] >= comp['pct'] and raw['chars'] >= comp['min_chars']
            else:
                raise ValueError(k)
            st.streak[i] = st.streak[i] + 1 if ok else 0
            if st.streak[i] >= comp.get('S', 1):
                if st.monitor is None:                   # the monitoring tier includes every component
                    st.monitor = Result(True, t, describe(comp))
                if comp.get('tier', 'intercept') != 'monitor' and st.result is None:
                    st.result = Result(True, t, describe(comp))
                    st.done = True                       # an intercept alert ends scoring
                    return


def describe(comp):
    k = comp['kind']
    if k == 'M':
        return f"{comp['feat']}>={comp['theta']}" + (f" S{comp['S']}" if comp.get('S', 1) > 1 else '') + \
            (f" sep<={comp['guard']}" if comp.get('guard') is not None else '') + \
            (f" {comp['nov']['feat']}<={comp['nov']['max']}" if comp.get('nov') else '') + \
            ''.join(f" & {a['feat']}{'>=' if 'min' in a else '<='}{a.get('min', a.get('max'))}" for a in comp.get('and', []))
    if k == 'V':
        return f"vllm[{comp['mn']},{comp['mx']}]x{comp['c']}"
    if k == 'G':
        return f"trees{len(comp['trees']['feat'])}>={comp['theta']:.4f} S{comp['S']}"
    if k == 'RUN':
        return f"run>={comp['min_chars']}"
    return f"ws>={comp['pct']}"
