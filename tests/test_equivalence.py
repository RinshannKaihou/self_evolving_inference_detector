"""The streaming detector equals the whole-record reference (offline.py) under any chunking: every feature at every boundary
(probe mode), and the m4 alert (boundary and component). Plus: the synthetic streams exercise every fire path of m4."""
import json

import numpy as np
import pytest

from synth import chunks, streams
from v35_detector import CONFIG_M4, RepDetector
from v35_detector import core as C
from v35_detector.offline import decide, features
from v35_detector.vocab import THINK_END, load

M4 = json.load(open(CONFIG_M4))
STREAMS = streams()


def _run(det, tok, rng):
    st = det.start()
    res = None
    for ch in chunks(tok, rng):
        res = det.feed(st, ch)
        if res is not None:
            break
    return res or det.finish(st), st


@pytest.mark.parametrize('corpus,kind,seed,tok', STREAMS, ids=lambda v: v if isinstance(v, (str, int)) else '')
def test_m4_stream_equals_offline(corpus, kind, seed, tok):
    res, _ = _run(RepDetector(corpus, M4), tok, np.random.default_rng(seed))
    assert (res.fired_at, res.reason if res.is_ill else '') == decide(tok, load(corpus), M4, THINK_END[corpus])


@pytest.mark.parametrize('corpus,kind,seed,tok', [s for s in STREAMS if s[2] == 1], ids=lambda v: v if isinstance(v, (str, int)) else '')
def test_probe_rows_equal_offline(corpus, kind, seed, tok):
    det = RepDetector(corpus, None)
    st = det.start()
    for ch in chunks(tok, np.random.default_rng(seed + 10)):
        det.feed(st, ch)
    det.finish(st)
    bnd, F = features(tok, load(corpus), THINK_END[corpus])
    assert [t for t, _ in st.rows] == bnd.tolist()
    G = np.stack([r for _, r in st.rows]) if st.rows else np.zeros((0, len(C.FEATS)), np.float32)
    assert np.array_equal(np.isnan(G), np.isnan(F)) and np.array_equal(np.nan_to_num(G), np.nan_to_num(F))


def test_every_fire_path_is_exercised():
    """first alerts over the synthetic streams cover the four families: vLLM rule, n-gram copies, word reuse, sentence
    restatement, plus the character run; random text never fires"""
    first = {}
    for corpus, kind, seed, tok in STREAMS:
        det = RepDetector(corpus, M4)
        st = det.start()
        res = det.feed(st, tok) or det.finish(st)
        first.setdefault(kind, set()).add(res.reason.split('[')[0].split('_')[0].split('>')[0] if res.is_ill else '')
    assert first['random'] == {''}
    fired = set().union(*first.values())
    assert {'vllm', 'A5', 'run', 'wrep', 'sdup'} <= fired, fired


def test_chunking_does_not_change_the_alert():
    corpus, _, _, tok = next(s for s in STREAMS if s[1] == 'noisy150')
    want = decide(tok, load(corpus), M4, THINK_END[corpus])
    for size in (1, 2, 63, 64, 65, 4096, len(tok)):
        det = RepDetector(corpus, M4)
        st = det.start()
        res = None
        for i in range(0, len(tok), size):
            res = det.feed(st, tok[i:i + size])
            if res is not None:
                break
        res = res or det.finish(st)
        assert (res.fired_at, res.reason) == want
