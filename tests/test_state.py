"""Per-request state is bounded: it stops growing once the windows are full, and stays under the 256 KB budget."""
import json
import sys

import numpy as np

from v35_detector import CONFIG_M4, RepDetector
from v35_detector import core as C

M4 = json.load(open(CONFIG_M4))


def state_bytes(obj, seen=None):
    """bytes held by a request state: numpy buffers + Python containers / scalars (recursive)"""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(state_bytes(k, seen) + state_bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(state_bytes(v, seen) for v in obj)
    if hasattr(obj, '__dict__') or hasattr(obj, '__slots__'):
        names = list(getattr(obj, '__dict__', {})) + list(getattr(obj, '__slots__', ()))
        return sys.getsizeof(obj) + sum(state_bytes(getattr(obj, k), seen) for k in names if hasattr(obj, k))
    return sys.getsizeof(obj)


def test_state_is_bounded():
    for corpus in ('qwen3', 'glm'):
        det = RepDetector(corpus, M4)
        st = det.start()
        rng = np.random.default_rng(3)
        sizes = []
        for _ in range(8):                               # 8 x 4,096 random tokens: never fires
            assert det.feed(st, rng.integers(0, 150000, 4096)) is None
            sizes.append(state_bytes(st))
        assert all(len(s.keys) <= C.H + C.BLOCK for s in st.streams.values())
        # once the 4,096-position history and the rings are full the size only moves with ring compaction and the open
        # sentence (about 1.5 KB), it does not grow with the output
        assert max(sizes[3:]) - min(sizes[3:]) <= 0.02 * min(sizes[3:]), sizes
        assert sizes[-1] <= 256 * 1024, sizes[-1]
