"""rc3 (rare_chars) on synthetic token streams: streaming equals the whole-record reference under any chunking, both tiers;
planted bad tokens fire the expected rules with the one-token look-ahead; a vocabulary without a reference table never uses the
learned score; the per-request state is bounded. tools/verify_against_research.py checks the evaluated dev alerts instead."""
import json
import sys

import numpy as np
import pytest

from v35_detector import CONFIG_RC1T, CONFIG_RC3, RareDetector
from v35_detector.rare import decide
from v35_detector.rare_vocab import CI, load as load_rare
from v35_detector.vocab import THINK_END

RC3 = json.load(open(CONFIG_RC3))
RC1T = json.load(open(CONFIG_RC1T))
LN10 = np.log(10)


def _pool(T, cls, lead):
    return np.flatnonzero((T['rcls'] == CI[cls]) & (T['lead'] == lead))


def synth(corpus, n, seed, han_share=0.3):
    """a plausible-looking stream: Latin words, Han characters, punctuation and digits, a think-end at 200 (thinking models);
    natural-log logprobs, the generated token at or below the top-1"""
    T = load_rare(corpus)
    rng = np.random.default_rng(seed)
    pools = [_pool(T, 'latin', True)[:3000], _pool(T, 'han', False)[:3000], _pool(T, 'space_punct', False)[:200],
             _pool(T, 'digit', False)[:50]]
    kind = rng.choice(4, n, p=[0.6 - han_share / 2, han_share, 0.3 - han_share / 2, 0.1])
    tok = np.array([rng.choice(pools[k]) for k in kind], np.int64)
    if THINK_END[corpus] is not None:
        tok[200] = THINK_END[corpus]
    lp1 = -rng.exponential(0.3, n) * LN10
    lpc = lp1 - rng.exponential(0.05, n) * LN10
    return tok, lp1, lpc


def stream(det, tok, lp1, lpc, rng):
    st = det.start()
    i = 0
    while i < len(tok):
        k = int(rng.choice([1, 5, 64, 100, 333]))
        ch = tok[i:i + k]
        res = det.feed(st, ch) if det.token_only else det.feed(st, ch, lp1[i:i + k], lpc[i:i + k])
        if res is not None:
            break
        i += k
    res = det.finish(st)
    mon = det.monitor(st)
    return (res.fired_at, res.reason), (mon.fired_at, mon.reason)


@pytest.mark.parametrize('corpus', ['qwen3', 'glm', 'dsv4'])
def test_stream_equals_whole_record(corpus):
    det = RareDetector(corpus, RC3)
    rng = np.random.default_rng(7)
    fired = 0
    T = load_rare(corpus)
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    for seed in range(12):
        tok, lp1, lpc = synth(corpus, 1500, seed, han_share=0.05 * (seed % 4))
        if seed % 4 == 0:                                # an English stream with one Han token glued to a word in the answer
            tok[700], tok[701] = word, int(_pool(T, 'han', False)[seed])
        strict, mon = stream(det, tok, lp1, lpc, rng)
        assert strict == decide(tok, lp1, lpc, corpus, RC3), (corpus, seed)
        assert mon == decide(tok, lp1, lpc, corpus, RC3, tier='monitor'), (corpus, seed)
        fired += mon[0] >= 0
    assert fired > 0                                     # the comparison covered alerts, not only silent streams


def test_bad_utf8_fires_one_token_late():
    """a lone UTF-8 continuation byte between two Latin words: the strict rule bad_utf8, raised when the next token arrives"""
    T = load_rare('qwen3')
    lone = next(i for i, b in enumerate(T['bytes']) if b == b'\x80')
    tok, lp1, lpc = synth('qwen3', 800, 1, han_share=0.0)
    lp1[:], lpc[:] = -0.01, -0.01                        # confident text: nothing else fires
    tok[500] = lone
    assert decide(tok, lp1, lpc, 'qwen3', RC3) == (502, 'bad_utf8')
    assert stream(RareDetector('qwen3', RC3), tok, lp1, lpc, np.random.default_rng(0))[0] == (502, 'bad_utf8')


def test_cjk_glued_to_latin_in_answer():
    """a rare Han token glued to a Latin word in the answer of an English text: the strict rule
    fs<0.01 & caseless_after_latin & answer"""
    T = load_rare('qwen3')
    tok, lp1, lpc = synth('qwen3', 1200, 2, han_share=0.0)
    lp1[:], lpc[:] = -0.01, -0.01
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    han = int(_pool(T, 'han', False)[0])
    tok[900], tok[901] = word, han
    at, why = decide(tok, lp1, lpc, 'qwen3', RC3)
    assert why == 'fs<0.01 & caseless_after_latin & answer' and at == 903, (at, why)


def test_no_reference_table_no_learned_score():
    """DeepSeek has no dev data, hence no reference table: the learned score is never computed and rc3 is its 4 other rules"""
    det = RareDetector('dsv4', RC3)
    assert det.ref is None and det.gbm is None
    tok, lp1, lpc = synth('dsv4', 3000, 3)
    for tier in ('intercept', 'monitor'):
        assert not decide(tok, lp1, lpc, 'dsv4', RC3, tier=tier)[1].startswith('gb')


def test_custom_table_needs_think_end(tmp_path):
    import os
    import shutil
    from v35_detector.vocab import TABLES
    path = str(tmp_path / 'my_model_rare.npz')
    shutil.copy(os.path.join(TABLES, 'qwen3_rare.npz'), path)
    with pytest.raises(ValueError):
        RareDetector(path, RC3)
    det = RareDetector(path, RC3, think_end=None)        # chat mode; no reference table for this vocabulary
    assert det.ref is None and det.gbm is None and det.think_end is None


def _bytes(obj, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(_bytes(k, seen) + _bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(_bytes(v, seen) for v in obj)
    if hasattr(obj, '__dict__'):
        return sys.getsizeof(obj) + sum(_bytes(v, seen) for v in vars(obj).values())
    return sys.getsizeof(obj)


def test_state_is_bounded():
    det = RareDetector('qwen3', RC3)
    T = load_rare('qwen3')
    rng = np.random.default_rng(3)
    st = det.start()
    sizes = []
    for n in (1000, 20000):
        tok = rng.integers(0, len(T['rcls']), n)
        det.feed(st, tok, np.full(n, -0.1), np.full(n, -0.1))
        sizes.append(_bytes(st))
    assert len(st.f['ring']) <= 64 and len(st.hold) == 3
    assert sizes[1] <= sizes[0] + 64 and sizes[1] < 16 * 1024      # does not grow with the stream


# ------------------------------------------------------------------ rc4 (research commit 5118334): "gate": "fs", "ref": "cats"
from v35_detector import CONFIG_RC4  # noqa: E402

RC4 = json.load(open(CONFIG_RC4))


@pytest.mark.parametrize('corpus', ['qwen3', 'glm', 'dsv4'])
def test_rc4_stream_equals_whole_record(corpus):
    """rc4 streams like rc3 (one token behind, any chunking); its learned score needs no reference table, so it also runs on
    DeepSeek"""
    det = RareDetector(corpus, RC4)
    assert det.gbm is not None and det.gbm.gate == 'fs' and 'u' not in det.ref
    rng = np.random.default_rng(9)
    T = load_rare(corpus)
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    fired = 0
    for seed in range(8):
        tok, lp1, lpc = synth(corpus, 1500, seed, han_share=0.05 * (seed % 4))
        if seed % 2 == 0:
            tok[700], tok[701] = word, int(_pool(T, 'han', False)[seed])
        strict, mon = stream(det, tok, lp1, lpc, rng)
        assert strict == decide(tok, lp1, lpc, corpus, RC4), (corpus, seed)
        assert mon == decide(tok, lp1, lpc, corpus, RC4, tier='monitor'), (corpus, seed)
        fired += mon[0] >= 0
    assert fired > 0


def test_rc4_gate_is_minority_letters_and_bad_utf8():
    from v35_detector import rare as R
    det = RareDetector('qwen3', RC4)
    tok, lp1, lpc = synth('qwen3', 2000, 4, han_share=0.1)
    F = R.record_features(det, tok, lp1, lpc)
    assert (R.gb_gate_mask(F) == ((F['fs'] < R.GB_GATE['fs']) | F['bad_utf8'])).all()


def test_rc4_config_keys_are_checked():
    with pytest.raises(ValueError):                      # "ref": "cats" needs "gate": "fs"
        RareDetector('qwen3', dict(RC4, gate='full'))
    with pytest.raises(ValueError):
        RareDetector('qwen3', dict(RC4, gate='other'))
    with pytest.raises(ValueError):                      # no rule on a reference feature without reference tables
        RareDetector('qwen3', dict(RC4, rules=RC4['rules'] + [{'and': [{'f': 'u', 'lt': -6}]}]))
    rc3 = RareDetector('qwen3', RC3)                     # rc3 (no gate / ref keys): full gate, reference tables
    assert rc3.gate == 'full' and 'u' in rc3.ref and rc3.gbm.gate == 'full'


# ------------------------------------------------------------------ rc1t: token ids only
@pytest.mark.parametrize('corpus', ['qwen3', 'glm', 'dsv4'])
def test_rc1t_stream_equals_whole_record(corpus):
    det = RareDetector(corpus, RC1T)
    assert det.token_only and det.gbm is not None and det.gbm.gate == 'fs'
    if corpus == 'dsv4':
        assert not np.any(det.ref['u']) and not np.any(det.ref['cp']) and not np.any(det.ref['cn'])
    rng = np.random.default_rng(31)
    T = load_rare(corpus)
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    fired = 0
    for seed in range(8):
        tok, _, _ = synth(corpus, 1500, seed, han_share=0.05 * (seed % 4))
        if seed % 2 == 0:
            tok[700], tok[701] = word, int(_pool(T, 'han', False)[seed])
        strict, mon = stream(det, tok, None, None, rng)
        assert strict == decide(tok, None, None, corpus, RC1T), (corpus, seed)
        assert mon == decide(tok, None, None, corpus, RC1T, tier='monitor'), (corpus, seed)
        fired += mon[0] >= 0
    assert fired > 0


def test_rc1t_bad_utf8_empty_single_and_last_token():
    T = load_rare('qwen3')
    lone = next(i for i, b in enumerate(T['bytes']) if b == b'\x80')
    space = next(i for i, b in enumerate(T['bytes']) if b == b' ')
    tok, _, _ = synth('qwen3', 800, 1, han_share=0.0)
    tok[500] = lone
    det = RareDetector('qwen3', RC1T)
    assert decide(tok, None, None, 'qwen3', RC1T) == (502, 'bad_utf8')
    assert stream(det, tok, None, None, np.random.default_rng(0))[0] == (502, 'bad_utf8')
    for ids in (np.array([], np.int64), np.array([0]), np.array([lone]), np.array([lone, space])):
        want = decide(ids, None, None, 'qwen3', RC1T)
        assert stream(det, ids, None, None, np.random.default_rng(1))[0] == want


def test_rc1t_rejects_logprobs_and_bad_configs():
    det = RareDetector('qwen3', RC1T)
    with pytest.raises(ValueError):
        det.feed(det.start(), [0], [0.0], [0.0])
    with pytest.raises(ValueError):
        RareDetector('qwen3', dict(RC1T, rules=[{'and': [{'f': 'lp1', 'lt': -1.0}]}]))
    with pytest.raises(ValueError):
        RareDetector('qwen3', dict(RC1T, input='token_ids_logprobs'))
    tok, _, _ = synth('qwen3', 300, 3)
    from v35_detector import rare as R
    F = R.record_features(det, tok)
    assert 'lp1' not in F and 'lpc' not in F and F['gb'].values(np.arange(len(tok))).shape == (len(tok),)


def test_rc1t_state_is_bounded():
    cfg = dict(RC1T, rules=[dict(r, tier='monitor') for r in RC1T['rules']])
    det = RareDetector('qwen3', cfg)
    T = load_rare('qwen3')
    rng = np.random.default_rng(3)
    st, sizes = det.start(), []
    for n in (1000, 20000):
        tok = rng.integers(0, len(T['rcls']), n)
        det.feed(st, tok)
        sizes.append(_bytes(st))
    assert isinstance(st.hold, int) and len(st.f['ring']) <= 64
    assert sizes[1] <= sizes[0] + 64 and sizes[1] < 16 * 1024
