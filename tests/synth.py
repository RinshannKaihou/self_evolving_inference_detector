"""Synthetic token streams for the tests (no real trajectories in this repo). Token ids are drawn from the shipped tables so
that every feature family sees the token classes it reads: plain words, circling markers, sentence ends, repeated-character
tokens, and the corpus's '</think>'.

  random      random words with sentence ends: no alert expected
  loop<P>     an exact loop of period P after a random prefix (vLLM rule)
  noisy<P>    a loop of period P whose copies each differ in one token (n-gram copy family; the vLLM rule needs exact copies)
  run         one repeated-character token 120 times (RUN)
  circling    the same 12 words reshuffled in every sentence, one marker each (word reuse + markers)
  restate     the same 12 words plus one fresh word per sentence, one marker each (sentence restatement)
  answer      the answer restates the reasoning draft (allowed), then loops inside the answer
"""
import numpy as np

from v35_detector.vocab import SEP, THINK_END, load

KINDS = ('random', 'loop5', 'loop40', 'loop300', 'loop700', 'noisy80', 'noisy150', 'noisy300', 'run', 'circling',
         'restate', 'answer')


def pools(corpus):
    T = load(corpus)
    n, ids = T['norm'], np.arange(len(T['norm']))
    word = ids[(n > 0) & T['letter'] & ~T['marker'] & ~T['send'] & (T['run_char'] < 0) & (T['chars'] >= 3) & (T['chars'] <= 8)]
    marker = ids[T['marker'] & ~T['send']]
    end = ids[T['send'] & (n == SEP) & (T['chars'] <= 2)]
    run = ids[(T['run_char'] >= 0) & (T['chars'] >= 8) & (n == SEP)]
    return word, marker, end, run


def make(corpus, kind, seed):
    rng = np.random.default_rng(seed)
    word, marker, end, run = pools(corpus)

    def text(n):
        out = rng.choice(word, n)
        out[rng.random(n) < 1 / 15] = rng.choice(end)
        return out

    pre = text(int(rng.integers(300, 1200)))
    if kind == 'random':
        return text(6000)
    if kind.startswith('loop'):
        P = int(kind[4:])
        return np.concatenate([pre, np.tile(text(P), 4000 // P + 3)])[:5000]
    if kind.startswith('noisy'):
        P = int(kind[5:])
        seg, copies = text(P), []
        for _ in range(5000 // P + 2):
            c = seg.copy()
            c[rng.integers(0, P)] = rng.choice(word)
            copies.append(c)
        return np.concatenate([pre, *copies])[:6000]
    if kind == 'run':
        return np.concatenate([pre, np.repeat(rng.choice(run), 120), text(500)])
    if kind in ('circling', 'restate'):
        S, mk = rng.choice(word, 12, replace=False), rng.choice(marker, 3)
        fresh = iter(rng.permutation(np.setdiff1d(word, S)))
        body = []
        for _ in range(600):
            s = list(rng.permutation(S if kind == 'circling' else np.r_[S, next(fresh)]))
            s.insert(int(rng.integers(0, len(s))), rng.choice(mk))
            body += s + [end[0]]
        return np.concatenate([pre, np.array(body)])[:7000 if kind == 'circling' else 9000]
    if kind == 'answer':
        draft, seg = text(700), text(60)
        te = THINK_END[corpus] if THINK_END[corpus] is not None else end[0]
        return np.concatenate([pre, draft, draft[:300], [te], draft, np.tile(seg, 40)])
    raise ValueError(kind)


def streams(corpora=('qwen3', 'glm', 'dsv4'), seeds=(1, 2)):
    return [(c, k, s, make(c, k, s)) for c in corpora for k in KINDS for s in seeds]


def chunks(tok, rng):
    """random chunk sizes, as a server delivers them (1 token per step, several with speculative decoding, or batches)"""
    i = 0
    while i < len(tok):
        k = int(rng.choice([1, 3, 7, 64, 100, 333]))
        yield tok[i:i + k]
        i += k
