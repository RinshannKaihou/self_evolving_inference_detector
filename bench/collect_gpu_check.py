#!/usr/bin/env python3
"""GPU check of rc3's sampler kernels (v35_detector/serving/collect.py, r3), run on the H20 before the server tests.

Correctness: the two Triton kernels (per-partition max and sum-exp, then combine and read the sampled token's raw logit)
against torch.log_softmax(logits, dtype=float32) - what vLLM's Sampler.compute_logprobs returns as raw logprobs - on random
logits: bf16 / fp16 / fp32, 1 to 257 rows, Qwen3's vocabulary (151,936) and odd sizes, rows with half the vocabulary masked
to -inf (a structured-output bitmask), a row with a single allowed token, sampled ids that are the argmax, random, masked,
or out of range (must give NaN, never a fault). Reports the largest absolute error of each value.

Cost: per call (rows 1 / 8 / 32 / 64, bf16, 151,936): CPU time to launch (no synchronisation in the loop) and GPU time
(CUDA events), next to the ops of vLLM's own logprobs path for logprobs=1 (float32 log_softmax, top-1, gather of the
sampled token, rank) on the same logits.

  python bench/collect_gpu_check.py [--out runs/collect_gpu_check.json] [--tol 1e-3]
Exit status 1 when an error exceeds --tol or the non-finite results disagree with the reference.
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from v35_detector.serving import collect  # noqa: E402

VOCAB = 151936


def case(col, rows, vocab, dtype, seed, masked=False, ids='random'):
    g = torch.Generator(device='cuda').manual_seed(seed)
    logits = (torch.randn(rows, vocab, device='cuda', generator=g) * 4.0).to(dtype)
    if masked:
        logits[::2, ::2] = float('-inf')                 # half the vocabulary of every other row
        logits[-1, :] = float('-inf')
        logits[-1, 3] = 1.5                              # a row with one allowed token
    if ids == 'argmax':
        sampled = logits.float().argmax(-1).to(torch.int32)
    else:
        sampled = torch.randint(0, vocab, (rows,), device='cuda', generator=g, dtype=torch.int32)
        if masked:
            sampled[0] = 0                               # a masked token (-inf logit) in row 0
        if ids == 'bad':
            sampled[-1] = vocab + 5                      # out of range: NaN, no read
    raw = col.prepare(logits)                            # bf16 / fp16: the same tensor; fp32: a copy
    got = col.triton(raw, sampled)
    torch.cuda.synchronize()
    ref = collect.reference(logits, sampled.clamp(0, vocab - 1))
    if ids == 'bad':
        ref[-1, 1] = float('nan')
    same_nonfinite = bool(torch.equal(torch.isnan(got), torch.isnan(ref)) and
                          torch.equal(torch.isinf(got) & ~torch.isnan(got), torch.isinf(ref) & ~torch.isnan(ref)))
    fin = torch.isfinite(got) & torch.isfinite(ref)
    err = (got - ref).abs().where(fin, torch.zeros_like(got)).amax(dim=0).tolist()
    return dict(rows=rows, vocab=vocab, dtype=str(dtype).replace('torch.', ''), masked=masked, ids=ids,
                max_abs_err_top1=err[0], max_abs_err_chosen=err[1], nonfinite_agree=same_nonfinite)


def vllm_logprobs_path(logits, sampled):
    """the ops vLLM runs for logprobs=1 in raw_logprobs mode (compute_logprobs + gather_logprobs), for comparison"""
    lp = logits.log_softmax(dim=-1, dtype=torch.float32)
    top, idx = torch.topk(lp, 1, dim=-1)
    tok = lp.gather(-1, sampled.long()[:, None])
    rank = (lp > tok).sum(-1)
    return top, idx, tok, rank


def timing(col, rows, n=200):
    g = torch.Generator(device='cuda').manual_seed(rows)
    logits = (torch.randn(rows, VOCAB, device='cuda', generator=g) * 4.0).to(torch.bfloat16)
    sampled = torch.randint(0, VOCAB, (rows,), device='cuda', generator=g, dtype=torch.int32)
    out = {'rows': rows}
    for name, fn in (('v35_kernels', lambda: col.finish(col.prepare(logits), sampled)),
                     ('vllm_logprobs1_ops', lambda: vllm_logprobs_path(logits, sampled))):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        cpu = (time.perf_counter() - t0) / n
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n):
            fn()
        end.record()
        end.synchronize()
        out[f'{name}_cpu_us_per_call'] = 1e6 * cpu
        out[f'{name}_gpu_us_per_call'] = 1e3 * start.elapsed_time(end) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs',
                                                  'collect_gpu_check.json'))
    ap.add_argument('--tol', type=float, default=1e-3)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit('no CUDA device: run this on the GPU server')
    import triton
    col = collect.Collector()
    cases = []
    seed = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for rows, vocab in ((1, VOCAB), (8, VOCAB), (64, VOCAB), (257, VOCAB), (3, 4097), (5, 1000)):
            for masked, ids in ((False, 'random'), (False, 'argmax'), (True, 'random'), (True, 'bad')):
                seed += 1
                cases.append(case(col, rows, vocab, dtype, seed, masked, ids))
    e1 = max(c['max_abs_err_top1'] for c in cases)
    e2 = max(c['max_abs_err_chosen'] for c in cases)
    agree = all(c['nonfinite_agree'] for c in cases)
    times = [timing(col, rows) for rows in (1, 8, 32, 64)]
    ok = e1 <= a.tol and e2 <= a.tol and agree and col.errors == 0
    report = dict(ok=ok, tol=a.tol, max_abs_err_top1=e1, max_abs_err_chosen=e2, nonfinite_agree=agree, cases=len(cases),
                  collector_errors=col.errors, timing=times, details=cases, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, triton=triton.__version__, python=platform.python_version())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, 'w'), indent=1)
    print(f'{report["gpu"]}, torch {report["torch"]}, triton {report["triton"]}: {len(cases)} cases')
    print(f'max |error| vs torch.log_softmax: top1 {e1:.3g}, chosen {e2:.3g} (tolerance {a.tol}); non-finite results '
          f'agree: {agree}')
    print('rows   v35 kernels cpu/gpu us per call   vLLM logprobs=1 ops cpu/gpu us per call')
    for t in times:
        print(f'{t["rows"]:>4d}   {t["v35_kernels_cpu_us_per_call"]:8.1f} / {t["v35_kernels_gpu_us_per_call"]:6.1f}'
              f'                  {t["vllm_logprobs1_ops_cpu_us_per_call"]:8.1f} / '
              f'{t["vllm_logprobs1_ops_gpu_us_per_call"]:6.1f}')
    print('collect GPU check: ' + ('ok' if ok else 'FAILED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
