"""Whole-record reference of the R-A per-boundary features (core.py definitions), vectorised over the record.
Used for calibration and as the parity oracle of the streaming detector. Causal: the row of boundary t only reads tokens < t.

  features(token_ids, tables) -> (boundaries int64 [B], F float32 [B, len(FEATS)])
Feature columns (core.FEATS):
  {A2,A3,P}_n{ng}_W{w}  share of flagged positions among the last w normalised positions (NaN while fewer than w exist)
  sep                   share of separator tokens (table norm == SEP) among the last min(t, RAW_W) raw tokens
  ws_frac, chars        whitespace characters / characters, and characters, over the same raw tokens
  run                   longest run so far of one repeated non-space character across consecutive tokens, in characters
"""
import numpy as np

from . import core as C
from .vocab import SEP


def position_stats(x, ng, answer_at=None):
    """c, d1, d2 of every normalised position (core.py); answer_at = normalised position where the answer starts"""
    m = len(x)
    c, d1, d2 = (np.zeros(m, np.int64) for _ in range(3))
    if m < ng:
        return c, d1, d2
    j = np.arange(ng - 1, m, dtype=np.int64)
    keys = C.ngram_hi(x, m - ng + 1, ng) | j.astype(np.uint64)
    s = np.sort(keys)
    hi = np.searchsorted(s, keys)
    floor = np.maximum(j - C.H, 0)
    if answer_at is not None:
        floor = np.where(j >= answer_at, np.maximum(floor, answer_at), floor)
    lo = np.searchsorted(s, (keys & C._HI) | floor.astype(np.uint64))
    cnt = hi - lo
    pos = (s & np.uint64(C.POSMASK)).astype(np.int64)
    a = j - pos[np.maximum(hi - 1, 0)]
    b = j - pos[np.maximum(hi - 2, 0)]
    c[ng - 1:] = np.minimum(cnt, C.CCAP)
    d1[ng - 1:] = np.where(cnt >= 1, a, 0)
    d2[ng - 1:] = np.where(cnt >= 2, b, 0)
    return c, d1, d2


def _win(cs, M, w):
    out = np.full(len(M), np.nan)
    ok = M >= w
    out[ok] = (cs[M[ok]] - cs[M[ok] - w]) / w
    return out


def runs_max(rc, chars):
    """cumulative longest run (chars) of equal run_char >= 0 over consecutive tokens, per token"""
    n = len(rc)
    if n == 0:
        return np.zeros(0, np.int64)
    brk = np.ones(n, bool)                             # a segment starts where run_char changes or is -1
    brk[1:] = (rc[1:] != rc[:-1]) | (rc[1:] < 0)
    seg = np.cumsum(brk) - 1
    val = np.where(rc >= 0, chars, 0).astype(np.int64)
    cs = np.cumsum(val)
    start = np.flatnonzero(brk)
    seg_base = np.r_[0, cs][start]                     # cumulative chars before each segment
    run_here = cs - seg_base[seg]                      # chars of the current run up to and incl. token i
    run_here[rc < 0] = 0
    return np.maximum.accumulate(run_here)


def answer_start(tok, keep, think_end):
    """normalised position where the answer starts (first think_end token), None if none / chat mode"""
    if think_end is None:
        return None
    hit = np.flatnonzero(tok == think_end)
    return int(np.count_nonzero(keep[:hit[0]])) if len(hit) else None


def features(tok, T, think_end=None):
    tok = np.asarray(tok, np.int64)
    n = len(tok)
    bnd = C.boundaries(n)
    norm = T['norm'][tok]
    keep = norm != SEP
    x = norm[keep].astype(np.int64)
    M = np.r_[0, np.cumsum(keep)][bnd]                 # normalised positions before each boundary
    A = answer_start(tok, keep, think_end)
    cols = {}
    for ng in C.NGS:
        fl = C.flags(*position_stats(x, ng, A))
        for f in C.FLAGS:
            cs = np.r_[0, np.cumsum(fl[f])]
            for w in C.WINS:
                cols[f'{f}_n{ng}_W{w}'] = _win(cs, M, w)
    lo = np.maximum(bnd - C.RAW_W, 0)
    span = np.maximum(bnd - lo, 1)
    sep_cs = np.r_[0, np.cumsum(~keep)]
    ch = T['chars'][tok].astype(np.int64)
    ws = T['ws'][tok].astype(np.int64)
    ch_cs, ws_cs = np.r_[0, np.cumsum(ch)], np.r_[0, np.cumsum(ws)]
    cols['sep'] = (sep_cs[bnd] - sep_cs[lo]) / span
    chars = ch_cs[bnd] - ch_cs[lo]
    cols['ws_frac'] = np.where(chars > 0, (ws_cs[bnd] - ws_cs[lo]) / np.maximum(chars, 1), 0.0)
    cols['chars'] = chars.astype(np.float64)
    rm = runs_max(T['run_char'][tok].astype(np.int64), ch)
    cols['run'] = rm[bnd - 1].astype(np.float64) if n else np.zeros(0)
    rep = np.ones(len(x), bool)
    rep[np.unique(x, return_index=True)[1]] = False     # first occurrence of each id is new
    rcs = np.r_[0, np.cumsum(rep)]
    for w in C.WREP_WINS:
        cols[f'wrep_W{w}'] = _win(rcs, M, w)
    mcs = np.r_[0, np.cumsum(T['marker'][tok])]
    for w in C.MARK_WINS:
        cols[f'mark_W{w}'] = _win(mcs, bnd, w)
    sf = C.sentences(norm, T['send'][tok], SEP)
    S = np.searchsorted(np.array([e for e, _ in sf], np.int64), bnd, side='left')   # sentences completed before the boundary
    dcs = np.r_[0, np.cumsum([d for _, d in sf])]
    for nn in C.SDUP_NS:
        cols[f'sdup_N{nn}'] = _win(dcs, S, nn)
    return bnd, np.stack([cols[f] for f in C.FEATS], 1).astype(np.float32) if n else np.zeros((0, len(C.FEATS)), np.float32)


def vllm_first_prefix(tok, mn, mx, c):
    """smallest prefix length t (checking every prefix 1..n) at which vLLM check_sequence_repetition([mn, mx] x c) holds,
    -1 if never. For lag p, eq(q) = tok[q] == tok[q-p]; it holds at t iff eq on [t-(c-1)p, t-1] (q >= p)."""
    tok = np.asarray(tok, np.int64)
    n, best = len(tok), -1
    for p in range(max(mn, 1), mx + 1):
        need = (c - 1) * p
        if c * p > n:
            break
        eq = tok[p:] == tok[:-p]                       # index i <-> q = i + p
        d = np.diff(np.r_[0, eq.view(np.int8), 0])
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        k = (e - s) >= need
        if k.any():
            t = int(s[k][0]) + p + need                # first run long enough: ends at q = s + p + need - 1
            best = t if best < 0 else min(best, t)
    return best


def decide(tok, T, config, think_end=None, tier='intercept'):
    """first alert (boundary t, component description) of a config, replayed on offline features; (-1, '') if none.
    tier 'intercept': components without "tier": "monitor"; tier 'monitor': every component.
    Ties at one boundary go to the earlier component of the config (as in the streaming detector)."""
    from .stream import describe
    bnd, F = features(tok, T, think_end)
    col = {f: i for i, f in enumerate(C.FEATS)}
    best = (-1, '')
    for comp in config['components']:
        if tier == 'intercept' and comp.get('tier', 'intercept') == 'monitor':
            continue
        k = comp['kind']
        if k == 'M':
            cond = np.nan_to_num(F[:, col[comp['feat']]].astype(np.float64), nan=-1) >= comp['theta']
            if comp.get('guard') is not None:
                cond &= F[:, col['sep']].astype(np.float64) <= comp['guard']
            if comp.get('nov'):
                cond &= np.nan_to_num(F[:, col[comp['nov']['feat']]].astype(np.float64), nan=2) <= comp['nov']['max']
            for a in comp.get('and', []):
                v = F[:, col[a['feat']]].astype(np.float64)
                cond &= (np.nan_to_num(v, nan=-1) >= a['min']) if 'min' in a else (np.nan_to_num(v, nan=2) <= a['max'])
        elif k == 'G':
            from .trees import Trees
            X = np.stack([bnd.astype(np.float64) if f == 'tpos' else np.nan_to_num(F[:, col[f]].astype(np.float64), nan=-1.0)
                          for f in comp['feats']], 1)
            cond = Trees(comp['trees']).score(X) >= comp['theta']
        elif k == 'RUN':
            cond = F[:, col['run']] >= comp['min_chars']
        elif k == 'WS':
            cond = (F[:, col['ws_frac']].astype(np.float64) >= comp['pct']) & (F[:, col['chars']] >= comp['min_chars'])
        elif k == 'V':
            t = vllm_first_prefix(tok, comp['mn'], comp['mx'], comp['c'])
            cond = np.zeros(len(bnd), bool)
            if t >= 0:
                cond[np.searchsorted(bnd, t):] = True  # reported at the first boundary >= t
        streak, hit = 0, -1
        for i, ok in enumerate(cond.tolist()):
            streak = streak + 1 if ok else 0
            if streak >= comp.get('S', 1):
                hit = int(bnd[i])
                break
        if hit >= 0 and (best[0] < 0 or hit < best[0]):
            best = (hit, describe(comp))
    return best
