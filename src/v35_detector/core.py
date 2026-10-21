"""Shared definitions of the repetition family R-A (verbatim loops). The streaming detector (stream.py) and the whole-record
reference (offline.py) both use these and must give identical per-boundary features (tests/test_detector.py).

Normalised stream x: the tokens whose table `norm` is not SEP, in order; position j indexes it.
n-gram at j (j >= NG-1): norm[j-NG+1..j]; key_j = (hash >> POSBITS) << POSBITS | j.
  c_j   number of earlier positions p in [lo_j, j-1] with the same n-gram hash (capped at CCAP); 0 for j < NG-1.
        lo_j = max(j-H, 0); for positions of the answer (j >= A, A = normalised position of the first '</think>' token)
        lo_j = max(j-H, A): an answer restating the reasoning is legitimate (v3.5), a loop inside the answer still counts
  d1_j  j - (most recent such p), 0 if c_j < 1;  d2_j  j - (second most recent), 0 if c_j < 2
Position flags:
  A2 = c >= 2, A3 = c >= 3, A5 = c >= 5        (the n-gram is at least the 3rd / 4th / 6th copy within H; drafting rarely
                                               re-writes a passage 6 times, a loop keeps adding copies)
  P  = c >= 2 and |d2 - 2*d1| <= d1 >> 3       (periodic: the two previous copies sit at one and two periods back)
  N  = c == 0                                  (novel: no earlier copy within H; "progress" -- loops have almost none)
Lexical features (no-progress reasoning loops; DESIGN.md §10):
  wrep_W{w}  share of the last w normalised positions whose id already occurred earlier in the output (1 - word novelty;
             streaming: a bitmap over normalised ids)
  mark_W{w}  share of the last w raw tokens that are circling markers (vocab.MARKERS)
Sentence restatement (circling reasoning; DESIGN.md §11):
  a sentence = the normalised ids since the previous sentence end, closed by a token holding a sentence end (vocab 'send',
  the token's own id included) or when it reaches SENT_MAX ids; it counts if it has >= SENT_MIN ids.
  signature = MinHash of the sentence's id set with SEEDS; dup = one of the previous SENT_HIST counted sentences agrees on
  >= SENT_AGREE of the hashes.
  sdup_N{n}  share of the last n counted sentences completed before the boundary that are dup (NaN while fewer than n)
Boundaries: every BLOCK raw tokens (t = 64, 128, ...) and once more at the stream end if it is not on a block edge.
"""
import numpy as np

BLOCK = 64
H = 4096                 # history of the n-gram counts, in normalised positions
POSBITS = 24             # position bits in a key (a request is scored up to 2^24 normalised tokens)
POSMASK = (1 << POSBITS) - 1
CCAP = 255
NGS = (4, 8)             # n-gram sizes computed (the config picks one per component)
FLAGS = ('A2', 'A3', 'A5', 'P', 'N')
WINS = (256, 512, 1024)  # window shares, in normalised positions
RAW_W = 256              # raw-token window of the structure / fill features
_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_PRIME = np.uint64(0x100000001B3)
_HI = np.uint64(~POSMASK & 0xFFFFFFFFFFFFFFFF)


def mix(x):
    x = np.asarray(x, np.int64).view(np.uint64) + _M1
    x = (x ^ (x >> np.uint64(30))) * _M2
    x = (x ^ (x >> np.uint64(27))) * _M3
    return x ^ (x >> np.uint64(31))


def ngram_hi(ids_window, m, ng):
    """high (hash) part of the keys of the m n-grams ending at the last m positions of ids_window (len = m + ng - 1)"""
    x = mix(ids_window)
    h = np.zeros(m, np.uint64)
    with np.errstate(over='ignore'):
        for i in range(ng):
            h = h * _PRIME + x[i:i + m]
    return h & _HI


def flags(c, d1, d2):
    """position flags (dict of bool arrays) from c, d1, d2"""
    c, d1, d2 = (np.asarray(a, np.int64) for a in (c, d1, d2))
    return {'A2': c >= 2, 'A3': c >= 3, 'A5': c >= 5, 'P': (c >= 2) & (np.abs(d2 - 2 * d1) <= (d1 >> 3)), 'N': c == 0}


def boundaries(n):
    """raw-token boundaries of a stream of n tokens"""
    b = list(range(BLOCK, n + 1, BLOCK))
    if not b or b[-1] != n:
        b.append(n)
    return np.array(b, np.int64) if n else np.zeros(0, np.int64)


SENT_MIN, SENT_MAX, SENT_HIST, SENT_AGREE = 4, 256, 512, 3
SEEDS = np.array([0x51ED27, 0x9E3779, 0x7F4A7C, 0x2545F4], np.uint64)
SDUP_NS = (16,)
WREP_WINS = (1024, 2048)
MARK_WINS = (1024,)
FEATS = ([f'{f}_n{ng}_W{w}' for ng in NGS for f in FLAGS for w in WINS] + ['sep', 'ws_frac', 'chars', 'run']
         + [f'wrep_W{w}' for w in WREP_WINS] + [f'mark_W{w}' for w in MARK_WINS] + [f'sdup_N{n}' for n in SDUP_NS])


_SEED_MUL = SEEDS | np.uint64(1)


def minhash(ids):
    """MinHash signature (len(SEEDS) uint64) of the set of normalised ids (repeats do not change a minimum)"""
    x = mix(np.asarray(ids, np.int64))
    with np.errstate(over='ignore'):
        return (x[:, None] * _SEED_MUL + SEEDS).min(0)


def sentences(norm, send, sep):
    """[(raw index of the closing token, dup)] for every counted sentence of a token stream (whole-record reference)"""
    out, cur, hist = [], [], []
    for i, (n, e) in enumerate(zip(norm.tolist(), send.tolist())):
        if n != sep:
            cur.append(n)
        if e or len(cur) >= SENT_MAX:
            if len(cur) >= SENT_MIN:
                mh = minhash(cur)
                H = np.array(hist[-SENT_HIST:], np.uint64).reshape(-1, len(SEEDS))
                out.append((i, bool(((H == mh).sum(1) >= SENT_AGREE).any())))
                hist.append(mh)
            cur = []
    return out
