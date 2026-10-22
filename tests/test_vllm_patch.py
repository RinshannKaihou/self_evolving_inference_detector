"""The vLLM patch (integrations/vllm_0_27/patch_vllm.py, r3: five files) on stand-ins that hold the anchors in vLLM 0.27.0's
layout: it applies once, compiles, refuses to apply twice, reverts exactly, replaces the hook lines of the r1 / r2 versions
(output_processor.py only), and the patched stand-ins carry rc3's values from the sampler to the feed hook while the
client-visible tokens and logprobs stay exactly vLLM's. The sampler and model-runner plumbing runs with torch (CPU) when it
is installed. Real vLLM files are checked when these variables point at them (skipped otherwise):
  V35_VLLM_SAMPLER            vllm/v1/sample/sampler.py
  V35_VLLM_GPU_MODEL_RUNNER   vllm/v1/worker/gpu_model_runner.py
  V35_VLLM_SCHEDULER          vllm/v1/core/sched/scheduler.py
  V35_VLLM_ENGINE             vllm/v1/engine/__init__.py
  V35_VLLM_OUTPUT_PROCESSOR   vllm/v1/engine/output_processor.py
"""
import ast
import importlib.util
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from v35_detector.serving import collect

try:
    import torch
except ImportError:
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, '..', 'integrations', 'vllm_0_27', 'patch_vllm.py')
spec = importlib.util.spec_from_file_location('patch_vllm', SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
needs_torch = pytest.mark.skipif(torch is None, reason='torch is not installed')

# ---------------------------------------------------------------- stand-ins (anchors exactly as in vLLM 0.27.0)
SAMPLER = '''import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler


class SamplerOutput:                                   # a dataclass without slots in vLLM: attributes can be added
    def __init__(self, sampled_token_ids, logprobs_tensors):
        self.sampled_token_ids = sampled_token_ids
        self.logprobs_tensors = logprobs_tensors


class Sampler:
    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        # NOTE: the steps of vLLM 0.27.0's Sampler.forward that matter here
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None:
            raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)       # compute_logprobs, raw_logprobs mode
        logits = logits.to(torch.float32)
        if sampling_metadata.logit_bias is not None:                           # processors, in place as vLLM's
            logits.add_(sampling_metadata.logit_bias)
        logits.div_(sampling_metadata.temperature.unsqueeze(dim=1))
        if sampling_metadata.generator is None:
            sampled = logits.argmax(dim=-1).view(-1)
        else:
            sampled = torch.multinomial(logits.softmax(dim=-1), 1, generator=sampling_metadata.generator).view(-1)
        sampled = sampled.long()
        logprobs_tensors = None
        if num_logprobs is not None:                                           # gather_logprobs: sampled, then top-k
            topk_logprobs, topk_indices = torch.topk(raw_logprobs, num_logprobs, dim=-1)
            token_logprobs = raw_logprobs.gather(-1, sampled.unsqueeze(-1))
            logprobs_tensors = (torch.cat((sampled.unsqueeze(-1), topk_indices), dim=1).to(torch.int32),
                                torch.cat((token_logprobs, topk_logprobs), dim=1))
        sampled = sampled.to(torch.int32)
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1].
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output
'''

RUNNER = '''import contextlib


class ModelRunnerOutput:                               # a dataclass without slots in vLLM
    def __init__(self, req_ids, req_id_to_index, sampled_token_ids=(), logprobs=None, num_nans_in_logits=None,
                 cudagraph_stats=None, routed_experts=None):
        self.req_ids, self.req_id_to_index = req_ids, req_id_to_index
        self.sampled_token_ids, self.logprobs = sampled_token_ids, logprobs


@contextlib.contextmanager
def copy_stream(stream):                               # torch.cuda.stream(...) in vLLM
    yield


class AsyncGPUModelRunnerOutput:
    def __init__(self, model_runner_output, sampled_token_ids, logprobs_tensors, async_output_copy_stream):
        self._model_runner_output = model_runner_output
        self._sampled_token_ids = sampled_token_ids
        self._logprobs_tensors = logprobs_tensors
        with copy_stream(async_output_copy_stream):
            self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self._routed_experts_cpu = None

    def get_output(self):
        valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
        logprobs_lists = self._logprobs_tensors_cpu

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists

        return output


class GPUModelRunner:
    def __init__(self, sampler, use_async_scheduling):
        self.sampler, self.use_async_scheduling = sampler, use_async_scheduling

    def sample_tokens(self, logits, sampling_metadata, req_ids):
        sampler_output = self.sampler(logits=logits, sampling_metadata=sampling_metadata)
        num_nans_in_logits, cudagraph_stats = {}, None
        logprobs_lists = sampler_output.logprobs_tensors
        valid_sampled_token_ids = None if self.use_async_scheduling else sampler_output.sampled_token_ids.tolist()
        with contextlib.nullcontext():
            output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
            )

        if not self.use_async_scheduling:
            return output
        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=None,
            async_output_copy_stream=None,
        )
'''

SCHEDULER = '''class Scheduler:
    def __init__(self, EngineCoreOutput):
        self.EngineCoreOutput = EngineCoreOutput

    def update_from_output(self, scheduler_output, model_runner_output):
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        outputs = []
        for req_id, request in scheduler_output.items():
            if True:
                req_index = model_runner_output.req_id_to_index[req_id]
                new_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
                new_logprobs = None
                if request.num_logprobs is not None and logprobs:
                    new_logprobs = [x[req_index] for x in logprobs]
                if True:
                    outputs.append(
                        self.EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            new_logprobs=new_logprobs,
                            num_nans_in_logits=request.num_nans_in_logits,
                        )
                    )
        return outputs
'''

ENGINE = '''from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineCoreOutput:                                # a msgspec.Struct in vLLM
    request_id: str
    new_token_ids: list[int]
    new_logprobs: object = None
    finish_reason: object = None
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None
'''

OUTPUT_PROCESSOR = '''from vllm.v1.engine.detokenizer import IncrementalDetokenizer


class FinishReason:
    STOP = 'stop'


class OutputProcessor:
    def __init__(self):
        self.request_states = {}
        self.finished = []

    def add_request(self, request_id, req_state):
        self.request_states[request_id] = req_state

    def propagate_error(self, e):
        for _, state in self.request_states.items():
            state.errors.append(e)

    def abort_requests(self, internal_req_ids):
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                pass

    def process_outputs(self, engine_core_outputs):
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            new_token_ids = engine_core_output.new_token_ids
            finish_reason = engine_core_output.finish_reason
            stop_reason = None
            if True:
                stop_string = None
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # Free completed requests.
            if finish_reason is not None:
                self.request_states.pop(req_id)
                self.finished.append((req_id, finish_reason, stop_reason))
'''
STANDINS = {'sampler': SAMPLER, 'gpu_model_runner': RUNNER, 'scheduler': SCHEDULER, 'engine': ENGINE,
            'output_processor': OUTPUT_PROCESSOR}
R2_FEED = '                    v35_stop = v35_hooks.feed(req_state.request_id, new_token_ids, engine_core_output.new_logprobs)'
R1_FEED = '                    v35_stop = v35_hooks.feed(req_state.request_id, new_token_ids)'
VLLM_IMPORTS = ('from vllm.v1.engine.detokenizer import IncrementalDetokenizer\n',
                'from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler\n')


def module(src, **names):
    """exec a (patched) stand-in; names replace the module globals the patch's import blocks set"""
    for line in VLLM_IMPORTS:
        src = src.replace(line, '', 1)
    name = f'v35_standin_{len(sys.modules)}'
    mod = sys.modules[name] = types.ModuleType(name)     # dataclasses look their module up
    exec(compile(src, name, 'exec'), mod.__dict__)
    mod.__dict__.update(names)
    return mod


def patched(key):
    return P.patch_text(key, STANDINS[key])


# ---------------------------------------------------------------- the patch itself
def test_every_file_patches_compiles_and_reverts():
    total = 0
    for key, src in STANDINS.items():
        out = P.patch_text(key, src)
        added = out.count(P.MARK.strip())
        assert added == P.n_lines(key)
        total += added
        with pytest.raises(ValueError):                  # twice is refused
            P.patch_text(key, out)
        assert P.revert_text(out) == src
    assert total == P.n_lines() == 35
    assert {k: P.n_lines(k) for k in P.FILES} == {'sampler': 7, 'gpu_model_runner': 8, 'scheduler': 2, 'engine': 1,
                                                  'output_processor': 17}


def test_a_missing_anchor_is_refused():
    with pytest.raises(ValueError):
        P.patch_text('output_processor', OUTPUT_PROCESSOR.replace('# Free completed requests.', '# Free finished requests.'))
    with pytest.raises(ValueError):
        P.patch_text('sampler', SAMPLER.replace('raw_logprobs: torch.Tensor | None = None', 'raw_logprobs = None'))
    with pytest.raises(ValueError):                      # found twice
        P.patch_text('engine', ENGINE + '\n\nclass Other:\n    num_nans_in_logits: int = 0\n')


def _tree(tmp_path, overrides=None):
    root = tmp_path / 'vllm'
    for key, (rel, _, _) in P.FILES.items():
        path = root.joinpath(*rel.split('/'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((overrides or {}).get(key, STANDINS[key]))
    return root


def _run(*args):
    r = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize('old_feed', [R2_FEED, R1_FEED])
def test_cli_upgrades_the_r1_and_r2_patches(tmp_path, old_feed):
    """the H20 environments of r1 / r2 carry their output_processor.py hook lines: --check sees them, --apply replaces
    them and patches the other four files, --revert restores every original"""
    current = patched('output_processor')
    feed = next(line for line in current.splitlines() if 'v35_hooks.feed(' in line)
    old = current.replace(feed, old_feed + P.MARK)
    assert old != current and P.revert_text(old) == OUTPUT_PROCESSOR
    root = _tree(tmp_path, {'output_processor': old})
    op = root / 'v1' / 'engine' / 'output_processor.py'
    (root / 'v1' / 'engine' / 'output_processor.py.v35-orig').write_text(OUTPUT_PROCESSOR)
    out = _run('--check', '--root', str(root))
    assert out.count('not patched') == 4 and 'patched by an earlier version' in out and '35 hook lines in total' in out
    out = _run('--apply', '--root', str(root))
    assert out.count('patched, ') == 4 and 'earlier version replaced' in out
    for key, (rel, _, _) in P.FILES.items():
        assert root.joinpath(*rel.split('/')).read_text() == patched(key)
    assert _run('--apply', '--root', str(root)).count('already patched (current)') == 5
    assert _run('--check', '--root', str(root)).count('patched (current)') == 5
    assert _run('--revert', '--root', str(root)).count('identical to the saved original') == 5
    for key, (rel, _, _) in P.FILES.items():
        assert root.joinpath(*rel.split('/')).read_text() == STANDINS[key]
    assert op.read_text() == OUTPUT_PROCESSOR


def test_cli_writes_nothing_when_one_file_does_not_match(tmp_path):
    root = _tree(tmp_path, {'scheduler': SCHEDULER.replace('num_nans_in_logits=request', 'num_nans=request')})
    r = subprocess.run([sys.executable, SCRIPT, '--apply', '--root', str(root)], capture_output=True, text=True)
    assert r.returncode != 0 and 'nothing written' in r.stderr
    assert not list(root.rglob('*.v35-orig'))
    assert (root / 'v1' / 'sample' / 'sampler.py').read_text() == SAMPLER


# ---------------------------------------------------------------- the plumbing on the patched stand-ins
class FakeHooks:
    def __init__(self, stop_after=10 ** 9):
        self.calls, self.stop_after, self.seen, self.values = [], stop_after, 0, []

    def register(self, rid):
        self.calls.append(('register', rid))

    def feed(self, rid, ids, rare):
        self.seen += len(ids)
        self.values.append((rid, list(ids), rare))
        return 'v35_repetition: X' if self.seen >= self.stop_after else None

    def finish(self, rid, reason):
        self.calls.append(('finish', rid, reason))

    def abort(self, rid):
        self.calls.append(('abort', rid))


def _state(rid):
    return types.SimpleNamespace(request_id=rid, errors=[],
                                 logprobs_processor=types.SimpleNamespace(update_from_output=lambda o: None))


def _eco(rid, ids, rare=None, finish=None):
    return types.SimpleNamespace(request_id=rid, new_token_ids=ids, finish_reason=finish, new_logprobs=None, v35_rare=rare)


def test_output_processor_routes_calls_and_intercepts():
    hooks = FakeHooks(stop_after=5)
    m = module(patched('output_processor'), v35_hooks=hooks)
    op = m.OutputProcessor()
    op.add_request('a', _state('a'))
    op.add_request('b', _state('b'))
    op.process_outputs([_eco('a', [1, 2], [-0.1, -0.2, -0.3, -0.4]), _eco('a', [3, 4])])
    assert op.finished == []
    op.process_outputs([_eco('a', [5], [-0.5, -0.6])])   # the 5th token: the hook returns a stop reason
    assert op.finished == [('a', 'stop', 'v35_repetition: X')]
    assert [v for _, _, v in hooks.values] == [[-0.1, -0.2, -0.3, -0.4], None, [-0.5, -0.6]]   # EngineCoreOutput.v35_rare
    op.abort_requests(['b'])
    assert hooks.calls == [('register', 'a'), ('register', 'b'), ('finish', 'a', 'stop'), ('abort', 'b')]


def test_an_existing_finish_reason_is_kept():
    m = module(patched('output_processor'), v35_hooks=FakeHooks(stop_after=1))
    op = m.OutputProcessor()
    op.add_request('a', _state('a'))
    op.process_outputs([_eco('a', [1], finish='length')])
    assert op.finished == [('a', 'length', None)]       # the engine already ended it: the detector does not overwrite


def test_scheduler_puts_each_requests_pair_into_its_output():
    eng = module(patched('engine'))
    sched = module(patched('scheduler')).Scheduler(eng.EngineCoreOutput)
    reqs = {'a': types.SimpleNamespace(num_logprobs=None, num_nans_in_logits=0),
            'b': types.SimpleNamespace(num_logprobs=1, num_nans_in_logits=0),
            'c': types.SimpleNamespace(num_logprobs=None, num_nans_in_logits=0)}
    mro = types.SimpleNamespace(req_id_to_index={'b': 0, 'a': 1, 'c': 2}, sampled_token_ids=[[7], [8], []],
                                logprobs=[['lp-b', 'lp-a', 'lp-c']])
    mro.v35_rare = [[-0.5, -1.5], [-0.25, -0.25], [-1.0, -2.0]]
    outs = {o.request_id: o for o in sched.update_from_output(reqs, mro)}
    assert outs['a'].v35_rare == [-0.25, -0.25] and outs['b'].v35_rare == [-0.5, -1.5]
    assert outs['c'].v35_rare is None                    # no new token (e.g. a partial prefill): no values
    assert outs['b'].new_logprobs == ['lp-b'] and outs['a'].new_logprobs is None   # client logprobs: vLLM's own path
    del mro.v35_rare                                     # V35_RARE=off: the attribute is never set
    outs = sched.update_from_output(reqs, mro)
    assert all(o.v35_rare is None for o in outs)
    assert eng.EngineCoreOutput('x', [1]).v35_rare is None   # a default field: omitted on the wire when None


def _metadata(rows, logprobs=None, bias=None, seed=None):
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return types.SimpleNamespace(max_num_logprobs=logprobs, logit_bias=bias, temperature=torch.full((rows,), 0.7),
                                 generator=gen)


def _logits(rows, vocab, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, vocab, generator=g) * 3).to(dtype)


@needs_torch
@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
def test_sampler_values_are_the_raw_distribution_and_client_logprobs_are_unchanged(dtype):
    """float32 logits are modified in place by the processors: the collector keeps a copy; bf16 ones are read in place"""
    dtype = getattr(torch, dtype)
    rows, vocab = 5, 9000
    bias = torch.zeros(rows, vocab)
    bias[:, 17] = 100.0                                  # a logit bias forces token 17: processed != raw
    original = module(SAMPLER).Sampler()
    on = module(patched('sampler'), v35_collect=collect.Collector()).Sampler()
    off = module(patched('sampler'), v35_collect=None).Sampler()
    for logprobs in (None, 2):
        outs = {}
        for name, s in (('original', original), ('on', on), ('off', off)):
            logits = _logits(rows, vocab, dtype, 3)
            raw = logits.clone()
            outs[name] = s(logits, _metadata(rows, logprobs, bias.clone(), seed=11))
            if name == 'on':
                v = outs[name].v35_rare
                ref = raw.float().log_softmax(-1)
                assert v.dtype == torch.float32 and tuple(v.shape) == (rows, 2)
                assert torch.allclose(v[:, 0], ref.max(-1).values, atol=1e-5)
                assert torch.all(outs[name].sampled_token_ids[:, 0] == 17)
                assert torch.allclose(v[:, 1], ref[:, 17], atol=1e-5)    # raw, not the biased (processed) logprob
                assert torch.all(v[:, 1] < v[:, 0] - 1)
        assert not hasattr(outs['off'], 'v35_rare') and not hasattr(outs['original'], 'v35_rare')
        for name in ('on', 'off'):                       # the client's tokens and logprobs: exactly vLLM's
            assert torch.equal(outs[name].sampled_token_ids, outs['original'].sampled_token_ids)
            if logprobs is None:
                assert outs[name].logprobs_tensors is None
            else:
                assert all(torch.equal(a, b) for a, b in zip(outs[name].logprobs_tensors, outs['original'].logprobs_tensors))


@needs_torch
def test_bonus_token_call_and_failures_give_no_values_and_do_not_touch_sampling(monkeypatch):
    col = collect.Collector()
    s = module(patched('sampler'), v35_collect=col).Sampler()
    out = s(_logits(3, 500, torch.bfloat16, 1), _metadata(3), predict_bonus_token=True)   # speculative decoding: skipped
    assert not hasattr(out, 'v35_rare')

    def boom(raw, sampled):
        raise RuntimeError('injected')
    monkeypatch.setattr(collect, 'reference', boom)
    ref = module(SAMPLER).Sampler()(_logits(3, 500, torch.bfloat16, 2), _metadata(3, seed=5))
    out = s(_logits(3, 500, torch.bfloat16, 2), _metadata(3, seed=5))
    assert out.v35_rare is None and col.errors == 1     # fail-open: no values for this step
    assert torch.equal(out.sampled_token_ids, ref.sampled_token_ids)


@needs_torch
@pytest.mark.parametrize('async_scheduling', [True, False])
def test_values_reach_the_feed_hook_through_runner_scheduler_and_output_processor(async_scheduling):
    """sampler -> model runner (async: the non-blocking copy and get_output; sync: a list) -> scheduler -> EngineCoreOutput
    -> output processor -> hooks.feed, two steps of three requests; the hook gets each request's [top1, chosen]"""
    sampler = module(patched('sampler'), v35_collect=collect.Collector()).Sampler()
    runner = module(patched('gpu_model_runner')).GPUModelRunner(sampler, async_scheduling)
    eng = module(patched('engine'))
    sched = module(patched('scheduler')).Scheduler(eng.EngineCoreOutput)
    hooks = FakeHooks()
    op = module(patched('output_processor'), v35_hooks=hooks).OutputProcessor()
    req_ids = ['r0', 'r1', 'r2']
    reqs = {r: types.SimpleNamespace(num_logprobs=None, num_nans_in_logits=0) for r in req_ids}
    for r in req_ids:
        op.add_request(r, _state(r))
    want = []
    for step in range(2):
        logits = _logits(3, 4097, torch.bfloat16, 20 + step)
        ref = logits.float().log_softmax(-1)
        out = runner.sample_tokens(logits, _metadata(3, seed=step), req_ids)
        mro = out.get_output() if async_scheduling else out
        assert isinstance(mro.v35_rare, list) and len(mro.v35_rare) == 3
        for i, r in enumerate(req_ids):
            tok = mro.sampled_token_ids[i][0]
            want.append((r, [tok], [ref[i].max().item(), ref[i, tok].item()]))
        op.process_outputs(sched.update_from_output(reqs, mro))
    assert [(r, ids) for r, ids, _ in hooks.values] == [(r, ids) for r, ids, _ in want]
    for (_, _, got), (_, _, exp) in zip(hooks.values, want):
        assert isinstance(got, list) and np.allclose(got, exp, atol=1e-5)


@needs_torch
def test_without_collection_nothing_is_carried():
    sampler = module(patched('sampler'), v35_collect=None).Sampler()
    runner = module(patched('gpu_model_runner')).GPUModelRunner(sampler, True)
    mro = runner.sample_tokens(_logits(2, 300, torch.bfloat16, 0), _metadata(2), ['a', 'b']).get_output()
    assert not hasattr(mro, 'v35_rare')


# ---------------------------------------------------------------- where the lines sit (stand-ins and real files)
def _fn(tree, cls, name):
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _target(t):
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
        return f'{t.value.id}.{t.attr}'
    return None


def _assign_line(fn, target):
    """line of the first assignment to `target` (a name or 'obj.attr') in fn"""
    lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(_target(t) == target for t in n.targets)]
    assert lines, target
    return min(lines)


def _call_line(fn, name):
    lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and name in (getattr(n.func, 'id', None),
                                                                                 getattr(n.func, 'attr', None))]
    assert lines, name
    return min(lines)


def positions(key, src):
    """the inserted lines sit where the design needs them"""
    tree = ast.parse(src)
    if key == 'sampler':
        fwd = _fn(tree, 'Sampler', 'forward')
        assert _assign_line(fwd, 'v35_raw') < _assign_line(fwd, 'logits')   # before the float32 copy and the processors
        ret = max(n.lineno for n in ast.walk(fwd) if isinstance(n, ast.Return))
        assert _assign_line(fwd, 'sampler_output') < _assign_line(fwd, 'sampler_output.v35_rare') < ret
    elif key == 'gpu_model_runner':
        init = _fn(tree, 'AsyncGPUModelRunnerOutput', '__init__')
        copy = _assign_line(init, 'self._v35_rare')
        streams = [n for n in ast.walk(init) if isinstance(n, ast.With) and isinstance(n.items[0].context_expr, ast.Call)
                   and getattr(n.items[0].context_expr.func, 'attr', None) == 'stream']
        assert any(w.lineno < copy <= w.end_lineno for w in streams)        # on the output copy stream
        assert _assign_line(init, 'self._logprobs_tensors_cpu') < copy < _call_line(init, 'record')
        get = _fn(tree, 'AsyncGPUModelRunnerOutput', 'get_output')
        assert _call_line(get, 'synchronize') < _assign_line(get, 'output.v35_rare')
        st = _fn(tree, 'GPUModelRunner', 'sample_tokens')
        assert _assign_line(st, 'output') < _assign_line(st, 'v35_rare') < _call_line(st, 'AsyncGPUModelRunnerOutput')
    elif key == 'scheduler':
        fn = _fn(tree, 'Scheduler', 'update_from_output')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and 'EngineCoreOutput' in (
            getattr(n.func, 'id', None), getattr(n.func, 'attr', None)) and any(k.arg == 'v35_rare' for k in n.keywords)]
        assert len(calls) == 1 and any(k.arg == 'new_token_ids' for k in calls[0].keywords)
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and n.lineno < calls[0].lineno <= n.end_lineno)
        assert _assign_line(fn, 'v35_rare') < loop.lineno
    elif key == 'engine':
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCoreOutput')
        fields = [n for n in cls.body if isinstance(n, ast.AnnAssign)]
        assert fields[-1].target.id == 'v35_rare' and fields[-1].value.value is None
    elif key == 'output_processor':
        fn = _fn(tree, 'OutputProcessor', 'process_outputs')
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and getattr(n.target, 'id', None) == 'engine_core_output')
        feed = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'feed'
                and getattr(n.func.value, 'id', None) == 'v35_hooks']
        assert len(feed) == 1
        arg = feed[0].args[2]
        assert isinstance(arg, ast.Attribute) and arg.attr == 'v35_rare' and arg.value.id == 'engine_core_output'


@pytest.mark.parametrize('key', ['sampler', 'scheduler', 'engine', 'output_processor'])
def test_positions_on_the_standins(key):
    positions(key, patched(key))


@pytest.mark.parametrize('key', list(P.FILES))
def test_real_vllm_file(key):
    var = P.FILES[key][1]
    if not os.environ.get(var):
        pytest.skip(f'set {var} to the vLLM file')
    src = open(os.environ[var], encoding='utf-8').read()
    base = P.revert_text(src)                            # the original, also when (an earlier version of) the patch is applied
    out = P.patch_text(key, base)
    assert P.revert_text(out) == base
    positions(key, out)
