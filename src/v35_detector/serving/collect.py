"""rc3's inputs collected in vLLM's sampler (r3): called by the patched vllm/v1/sample/sampler.py
(integrations/vllm_0_27/patch_vllm.py), in the process that samples (for TP=1 the EngineCore process).

rc3 needs, per generated token, the top-1 logprob and the generated token's logprob of the model's raw distribution: vLLM's
default --logprobs-mode raw_logprobs, i.e. the logits before allowed-token masks, bad words, logit bias, min-tokens,
penalties, temperature and top-k/top-p (rc3 was evaluated on exactly these). Instead of asking vLLM for logprobs (a
full-vocabulary float32 log-softmax, top-k, the sampled token's rank, k + 1 columns sent to the frontend, Logprob objects),
two Triton kernels reduce each row of raw logits once the token is sampled:

  1. grid (rows, partitions of 4096): the partition's max m_p and s_p = sum exp(x - m_p)
  2. grid (rows,): M = max m_p, S = sum s_p exp(m_p - M), lse = M + log S, and the sampled token's raw logit x_c:
         top1   = M - lse         = -log S
         chosen = x_c - lse       = (x_c - M) - log S                      -> float32 [rows, 2]

No full-vocabulary tensor is written and nothing is sorted. The pair travels in its own field (ModelRunnerOutput.v35_rare ->
EngineCoreOutput.v35_rare -> the frontend feed hook), never through the client-visible logprobs.

Raw logits: vLLM's sampler converts bf16/fp16 logits to a float32 copy before any processor runs (Sampler.forward,
`logits = logits.to(torch.float32)`), so the input tensor is still raw after sampling and is read in place: no copy. float32
logits (an fp32 lm_head) would be modified in place by the processors, so prepare() keeps a copy of them (rows x vocab x 4
bytes per step). Structured-output bitmasks are applied before the sampler, as for vLLM's own raw logprobs.

Fail-open: an exception in prepare() / finish() leaves that step without values (logged a few times), sampling goes on. A
fault inside a kernel cannot be caught; every load and store is masked to the tensors' bounds. Speculative decoding is not
collected in r3: the rejection sampler's bonus-token call (predict_bonus_token=True) is skipped, so requests under MTP get
values only for steps without drafts (rc3 reports them as 'lost').

V35_RARE and V35_RARE_CONFIG are read at import: off (default) or rc1t -> SAMPLER is None and the patched sampler does no
collection; collect | shadow with a logprob config (rc3 / rc4) -> SAMPLER is a Collector.
"""
import logging
import os

import numpy as np

try:                                                     # vLLM's process has torch and triton; the detector package does not
    import torch                                         # need them (the Mac tests run without)
except ImportError:
    torch = None
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

BLOCK = 4096                                             # logits per partition (kernel 1)
MAX_LOGGED_ERRORS = 3
log = logging.getLogger('v35_detector.collect')


def reference(logits, sampled):
    """torch reference: [rows, 2] float32 (top-1 logprob, sampled token's logprob) from log_softmax of the raw logits, as
    vLLM's Sampler.compute_logprobs computes them"""
    lp = logits.log_softmax(dim=-1, dtype=torch.float32)
    idx = sampled.reshape(-1).long()
    return torch.stack((lp.max(dim=-1).values, lp.gather(-1, idx[:, None])[:, 0]), dim=1)


def partitioned_np(logits, sampled, block=BLOCK):
    """the two kernels' arithmetic in numpy (float32, same partitions, same treatment of -inf / out-of-range ids): what the
    Mac tests check against log_softmax"""
    x = np.asarray(logits, np.float32)
    rows, n = x.shape
    parts = -(-n // block)
    pad = np.full((rows, parts * block), -np.inf, np.float32)
    pad[:, :n] = x
    pad = pad.reshape(rows, parts, block)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        pm = pad.max(axis=2)
        ps = np.exp(pad - pm[:, :, None]).sum(axis=2, dtype=np.float32)
        ps = np.where(pm == -np.inf, np.float32(0), ps)
        m = pm.max(axis=1)
        s = np.where(pm == -np.inf, np.float32(0), ps * np.exp(pm - m[:, None])).sum(axis=1, dtype=np.float32)
        log_s = np.log(s)
        idx = np.asarray(sampled).reshape(-1).astype(np.int64)
        ok = (idx >= 0) & (idx < n)
        xc = np.where(ok, x[np.arange(rows), np.clip(idx, 0, n - 1)], np.float32(np.nan))
        out = np.stack((-log_s, (xc - m) - log_s), axis=1).astype(np.float32)
    out[np.isinf(m)] = np.nan
    return out


if triton is not None:
    @triton.jit
    def _v35_partials(X, stride, PMax, PSum, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        part = tl.program_id(1)
        offs = part * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row.to(tl.int64) * stride + offs, mask=offs < N, other=-float('inf')).to(tl.float32)
        m = tl.max(x, axis=0)
        s = tl.sum(tl.exp(x - m), axis=0)
        s = tl.where(m == -float('inf'), 0.0, s)         # an all -inf partition adds nothing
        tl.store(PMax + row * PARTS + part, m)
        tl.store(PSum + row * PARTS + part, s)

    @triton.jit
    def _v35_finish(PMax, PSum, X, stride, Sampled, Out, N: tl.constexpr, PARTS: tl.constexpr, BLOCK_P: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_P)
        pm = tl.load(PMax + row * PARTS + offs, mask=offs < PARTS, other=-float('inf'))
        ps = tl.load(PSum + row * PARTS + offs, mask=offs < PARTS, other=0.0)
        m = tl.max(pm, axis=0)
        s = tl.sum(tl.where(pm == -float('inf'), 0.0, ps * tl.exp(pm - m)), axis=0)
        log_s = tl.log(s)
        tok = tl.load(Sampled + row).to(tl.int64)
        ok = (tok >= 0) & (tok < N)
        xc = tl.load(X + row.to(tl.int64) * stride + tok, mask=ok, other=float('nan')).to(tl.float32)
        bad = (m == -float('inf')) | (m == float('inf'))
        tl.store(Out + row * 2, tl.where(bad, float('nan'), -log_s))
        tl.store(Out + row * 2 + 1, tl.where(bad, float('nan'), (xc - m) - log_s))


class Collector:
    """prepare(logits, predict_bonus_token) before the sampler's processors; finish(raw, sampled) after sampling ->
    float32 [rows, 2] on the logits' device, or None (not collected, or a failure: fail-open)"""

    def __init__(self):
        self.workspace = {}                              # device -> (partial max, partial sum) [capacity, partitions]
        self.errors = 0

    def prepare(self, logits, predict_bonus_token=False):
        if predict_bonus_token:                          # speculative decoding: not collected in r3
            return None
        try:
            return logits.clone() if logits.dtype == torch.float32 else logits
        except Exception as e:
            self.fail('prepare', e)
            return None

    def finish(self, raw, sampled):
        try:
            if raw.shape[0] == 0:
                return None
            if not raw.is_cuda:                          # CPU backends and the tests
                return reference(raw, sampled)
            return self.triton(raw, sampled)
        except Exception as e:
            self.fail('finish', e)
            return None

    def triton(self, raw, sampled):
        """the two kernels on a CUDA tensor (raises on failure; finish() catches)"""
        rows, n = raw.shape
        if raw.stride(1) != 1:
            raw = raw.contiguous()
        parts = -(-n // BLOCK)
        ws = self.workspace.get(raw.device)
        if ws is None or ws[0].shape[0] < rows or ws[0].shape[1] != parts:
            cap = -(-rows // 64) * 64
            ws = self.workspace[raw.device] = (torch.empty((cap, parts), dtype=torch.float32, device=raw.device),
                                               torch.empty((cap, parts), dtype=torch.float32, device=raw.device))
        idx = sampled.reshape(-1)
        if idx.stride(0) != 1:
            idx = idx.contiguous()
        # a fresh output each step: the previous step's may still be in its copy to the host (async scheduling); the
        # partial buffers are reused, they are only touched by these two kernels, in stream order
        out = torch.empty((rows, 2), dtype=torch.float32, device=raw.device)
        _v35_partials[(rows, parts)](raw, raw.stride(0), ws[0], ws[1], n, parts, BLOCK, num_warps=4)
        _v35_finish[(rows,)](ws[0], ws[1], raw, raw.stride(0), idx, out, n, parts, max(16, triton.next_power_of_2(parts)),
                             num_warps=1)
        return out

    def fail(self, where, e):
        self.errors += 1
        if self.errors <= MAX_LOGGED_ERRORS:
            log.warning('v35 rc3 collection failed in %s (%d so far), this step has no values: %s: %s', where,
                        self.errors, type(e).__name__, e, exc_info=self.errors == 1)


MODE = os.environ.get('V35_RARE', 'off')
CONFIG = os.environ.get('V35_RARE_CONFIG', 'rc3')
SAMPLER = Collector() if MODE in ('collect', 'shadow') and CONFIG != 'rc1t' else None
