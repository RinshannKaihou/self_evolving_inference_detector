"""rc3's values collected in the sampler (v35_detector/serving/collect.py, r3). Without a GPU: the two kernels' arithmetic
(partitions of 4096, max and sum-exp, combined, the sampled token's raw logit) in numpy against log_softmax, on edge cases
(masked logits, a fully masked partition, ids out of range); the torch reference and the Collector's CPU path against
torch.log_softmax; prepare() reads bf16 logits in place and copies float32 ones; failures are fail-open; V35_RARE decides
whether the sampler collects. The Triton kernels themselves are checked on the GPU by bench/collect_gpu_check.py."""
import os
import subprocess
import sys

import numpy as np
import pytest

from v35_detector.serving import collect

try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason='torch is not installed')


def log_softmax64(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def logits(rows, vocab, seed, scale=4.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, vocab)) * scale).astype(np.float32)


@pytest.mark.parametrize('rows,vocab', [(1, 151936), (5, 154880), (3, 4096), (4, 4097), (2, 1000)])
def test_partitioned_arithmetic_equals_log_softmax(rows, vocab):
    x = logits(rows, vocab, vocab + rows)
    rng = np.random.default_rng(1)
    sampled = rng.integers(0, vocab, rows)
    sampled[0] = x[0].argmax()                           # greedy: chosen == top-1
    got = collect.partitioned_np(x, sampled)
    ref = log_softmax64(x)
    assert got.dtype == np.float32
    assert np.allclose(got[:, 0], ref.max(axis=1), atol=2e-5)
    assert np.allclose(got[:, 1], ref[np.arange(rows), sampled], atol=2e-5)
    assert got[0, 0] == got[0, 1]
    assert np.all(got[:, 1] <= got[:, 0])


def test_masked_logits_and_bad_ids():
    """-inf logits (a structured-output bitmask is applied before the sampler, as for vLLM's raw logprobs), a fully masked
    partition, a fully masked row, and an out-of-range sampled id never read out of bounds"""
    vocab = 3 * collect.BLOCK + 17
    x = logits(4, vocab, 7)
    x[0, collect.BLOCK:2 * collect.BLOCK] = -np.inf       # one whole partition masked
    x[1, ::2] = -np.inf                                   # half the vocabulary masked
    x[2, :] = -np.inf                                     # nothing allowed
    sampled = np.array([5, 2, 3, vocab + 9])             # row 1 samples a masked token, row 3 an id out of range
    got = collect.partitioned_np(x, sampled)
    ref = log_softmax64(x[:2])
    assert np.allclose(got[:2, 0], ref.max(axis=1), atol=2e-5)
    assert np.isclose(got[0, 1], ref[0, 5], atol=2e-5)
    assert got[1, 1] == -np.inf                           # a masked token's raw logprob, as log_softmax gives it
    assert np.isnan(got[2]).all()                         # no finite logit: no values (NaN), not +inf
    assert np.isfinite(got[3, 0]) and np.isnan(got[3, 1])


@needs_torch
@pytest.mark.parametrize('dtype', ['bfloat16', 'float16', 'float32'])
def test_reference_and_collector_cpu_path_equal_torch_log_softmax(dtype):
    t = torch.from_numpy(logits(6, 20000, 3)).to(getattr(torch, dtype))
    sampled = torch.tensor([0, 1, 19999, 77, 5, 123], dtype=torch.int32)[:, None]   # [rows, 1] int32, as the sampler has
    ref = t.float().log_softmax(-1)
    want = torch.stack((ref.max(-1).values, ref.gather(-1, sampled.long())[:, 0]), 1)
    col = collect.Collector()
    raw = col.prepare(t, predict_bonus_token=False)
    got = col.finish(raw, sampled.reshape(-1))
    assert torch.allclose(got, want, atol=1e-6) and torch.allclose(collect.reference(t, sampled), want, atol=1e-6)
    np_got = collect.partitioned_np(t.float().numpy(), sampled.numpy())
    assert np.allclose(np_got, want.numpy(), atol=2e-5)  # the kernels' arithmetic on the same (bf16 / fp16) values


@needs_torch
def test_prepare_reads_bf16_in_place_and_copies_float32():
    col = collect.Collector()
    b = torch.zeros(2, 10, dtype=torch.bfloat16)
    assert col.prepare(b) is b                           # the sampler processes a float32 copy: b stays raw
    f = torch.zeros(2, 10)
    kept = col.prepare(f)
    assert kept is not f and kept.data_ptr() != f.data_ptr() and torch.equal(kept, f)
    f.add_(1.0)                                          # the processors' in-place updates do not reach the kept copy
    assert float(kept.sum()) == 0.0
    assert col.prepare(f, predict_bonus_token=True) is None
    assert col.finish(torch.zeros(0, 10), torch.zeros(0, dtype=torch.int32)) is None   # an empty batch


def test_failures_are_fail_open():
    col = collect.Collector()
    assert col.prepare(object()) is None                 # not a tensor
    assert col.finish(object(), None) is None
    assert col.errors == 2


@pytest.mark.parametrize('mode,on', [(None, False), ('off', False), ('collect', True), ('shadow', True), ('bogus', False)])
def test_v35_rare_decides_whether_the_sampler_collects(mode, on):
    env = {} if mode is None else {'V35_RARE': mode}
    code = 'from v35_detector.serving import collect; print(collect.SAMPLER is not None)'
    full = {k: v for k, v in os.environ.items() if k != 'V35_RARE'}
    full.update(env)
    full['PYTHONPATH'] = os.pathsep.join([os.path.join(os.path.dirname(__file__), '..', 'src'), full.get('PYTHONPATH', '')])
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=full)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(on)
