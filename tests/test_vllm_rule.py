"""The V components equal vLLM's check_sequence_repetition, called after every appended token as the scheduler does.
`upstream` below is a transcription of vllm/v1/core/sched/utils.py (v0.18.0: _has_repeating_pattern +
check_sequence_repetition); the research repo checked the same fast paths against the upstream source itself."""
import numpy as np
import pytest

from v35_detector import RepDetector
from v35_detector import core as C
from v35_detector.offline import vllm_first_prefix


def upstream(token_ids, mn, mx, c):
    for L in range(max(mn, 1), mx + 1):
        if L * c > len(token_ids):
            return False
        if all(token_ids[-(L * m + n)] == token_ids[-n] for n in range(1, L + 1) for m in range(1, c)):
            return True
    return False


def first_prefix(tok, mn, mx, c):
    """shortest prefix on which the check holds (-1: none)"""
    tok = [int(x) for x in tok]
    for t in range(1, len(tok) + 1):
        if upstream(tok[:t], mn, mx, c):
            return t
    return -1


def streams(mn, mx, c, rng, n_max):
    for trial in range(30):
        n = int(rng.integers(50, n_max))
        tok = rng.integers(0, 3 if trial % 2 else 40, n)
        if trial % 3 == 0:
            P = int(rng.integers(max(mn - 2, 1), mx + 4))
            at = int(rng.integers(0, n))
            tok = np.concatenate([tok[:at], np.tile(rng.integers(0, 40, P), int(rng.integers(1, c + 2))), tok[at:]])
        yield tok


@pytest.mark.parametrize('cfg', [(2, 20, 3), (10, 40, 2), (5, 30, 5), (32, 60, 3)])
def test_fast_replay_and_stream_equal_upstream(cfg):
    mn, mx, c = cfg
    rng = np.random.default_rng(mn * 100 + mx + c)
    for tok in streams(mn, mx, c, rng, 700):
        t = first_prefix(tok, mn, mx, c)
        assert vllm_first_prefix(tok, mn, mx, c) == t
        b = C.boundaries(len(tok))
        want = -1 if t < 0 else int(b[np.searchsorted(b, t)])  # reported at the first boundary at or after the prefix
        det = RepDetector('glm', {'components': [{'kind': 'V', 'mn': mn, 'mx': mx, 'c': c}]})
        st = det.start()
        res = det.feed(st, tok) or det.finish(st)
        assert res.fired_at == want, (cfg, len(tok), res.fired_at, want)


@pytest.mark.parametrize('cfg', [(32, 500, 5), (32, 2000, 3)])
def test_m4_rules_on_long_streams(cfg):
    """the two V components of m4 on longer streams: stream == fast replay (fast replay == upstream is tested above)"""
    mn, mx, c = cfg
    rng = np.random.default_rng(mx + c)
    for tok in streams(mn, mx, c, rng, 7000):
        t = vllm_first_prefix(tok, mn, mx, c)
        b = C.boundaries(len(tok))
        want = -1 if t < 0 else int(b[np.searchsorted(b, t)])
        det = RepDetector('qwen3', {'components': [{'kind': 'V', 'mn': mn, 'mx': mx, 'c': c}]})
        st = det.start()
        res = det.feed(st, tok) or det.finish(st)
        assert res.fired_at == want


@pytest.mark.parametrize('cfg', [(32, 500, 5), (32, 2000, 3), (2, 20, 3)])
def test_loop_from_token_zero(cfg):
    """a loop that starts at the first token (history shorter than c * mx)"""
    mn, mx, c = cfg
    rng = np.random.default_rng(11)
    for P in (mn, mn + 1, 37, 70, 129, 400):
        for reps in (c - 1, c, c + 3):
            tok = np.concatenate([np.tile(rng.integers(0, 9, P), reps), rng.integers(0, 50, int(rng.integers(0, 90)))])
            t = vllm_first_prefix(tok, mn, mx, c)
            b = C.boundaries(len(tok))
            want = -1 if t < 0 else int(b[np.searchsorted(b, t)])
            det = RepDetector('glm', {'components': [{'kind': 'V', 'mn': mn, 'mx': mx, 'c': c}]})
            st = det.start()
            res = det.feed(st, tok) or det.finish(st)
            assert res.fired_at == want, (cfg, P, reps)
