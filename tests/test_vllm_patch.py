"""The vLLM patch (integrations/vllm_0_27/patch_output_processor.py) on a stand-in of the OutputProcessor that holds the six
anchors in vLLM 0.27.0's layout: it applies once, compiles, routes the calls (with the step's new_logprobs) as intended,
refuses to apply twice, reverts exactly, and replaces the hook lines of the r1 version (feed without logprobs).
V35_VLLM_OUTPUT_PROCESSOR=/path/to/output_processor.py also checks a real vLLM file (skipped otherwise)."""
import ast
import importlib.util
import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, '..', 'integrations', 'vllm_0_27', 'patch_output_processor.py')
spec = importlib.util.spec_from_file_location('patch_output_processor', SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

STANDIN = '''from vllm.v1.engine.detokenizer import IncrementalDetokenizer


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
                req_state.logprobs_seen.append(engine_core_output.new_logprobs)

            # Free completed requests.
            if finish_reason is not None:
                self.request_states.pop(req_id)
                self.finished.append((req_id, finish_reason, stop_reason))
'''
R1_FEED = '                    v35_stop = v35_hooks.feed(req_state.request_id, new_token_ids)'


class FakeHooks:
    def __init__(self, stop_after):
        self.calls, self.stop_after, self.seen, self.logprobs = [], stop_after, 0, []

    def register(self, rid):
        self.calls.append(('register', rid))

    def feed(self, rid, ids, logprobs):
        self.seen += len(ids)
        self.logprobs.append(logprobs)
        return 'v35_repetition: X' if self.seen >= self.stop_after else None

    def finish(self, rid, reason):
        self.calls.append(('finish', rid, reason))

    def abort(self, rid):
        self.calls.append(('abort', rid))


def _module(src, hooks):
    mod = types.ModuleType('op')
    exec(compile(src.replace('from vllm.v1.engine.detokenizer import IncrementalDetokenizer\n', '', 1), 'op', 'exec'),
         mod.__dict__)
    mod.v35_hooks = hooks                                # the patched import block set it to the real (off) hooks
    return mod


def _state(rid):
    return types.SimpleNamespace(request_id=rid, errors=[], logprobs_seen=[])


def _out(rid, ids, finish=None, logprobs=None):
    return types.SimpleNamespace(request_id=rid, new_token_ids=ids, finish_reason=finish, new_logprobs=logprobs)


def test_apply_routes_calls_and_intercepts():
    out = P.patch_text(STANDIN)
    assert out.count(P.MARK.strip()) == sum(len(lines) for _, _, lines in P.HOOKS) == 17
    hooks = FakeHooks(stop_after=5)
    m = _module(out, hooks)
    op = m.OutputProcessor()
    state = _state('a')
    op.add_request('a', state)
    op.add_request('b', _state('b'))
    op.process_outputs([_out('a', [1, 2], logprobs='lp12'), _out('a', [3, 4], logprobs='lp34')])
    assert op.finished == []
    op.process_outputs([_out('a', [5])])                 # the 5th token: the hook returns a stop reason
    assert op.finished == [('a', 'stop', 'v35_repetition: X')]
    assert hooks.logprobs == ['lp12', 'lp34', None]      # the step's new_logprobs, as the logprobs processor then gets it
    assert state.logprobs_seen == ['lp12', 'lp34', None]
    op.abort_requests(['b'])
    assert hooks.calls == [('register', 'a'), ('register', 'b'), ('finish', 'a', 'stop'), ('abort', 'b')]


def test_existing_finish_reason_is_kept():
    hooks = FakeHooks(stop_after=1)
    m = _module(P.patch_text(STANDIN), hooks)
    op = m.OutputProcessor()
    op.add_request('a', _state('a'))
    op.process_outputs([_out('a', [1], 'length')])       # the engine already ended it: the detector does not overwrite
    assert op.finished == [('a', 'length', None)]


def test_twice_refused_and_revert_is_exact():
    out = P.patch_text(STANDIN)
    with pytest.raises(ValueError):
        P.patch_text(out)
    assert P.revert_text(out) == STANDIN
    with pytest.raises(ValueError):                      # a missing anchor is refused, not guessed
        P.patch_text(STANDIN.replace('# Free completed requests.', '# Free finished requests.'))


def _run(*args):
    r = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_cli_upgrades_the_r1_patch(tmp_path):
    """the H20 environment of r1 carries the r1 hook lines: --check sees them, --apply replaces them, --revert restores"""
    path = tmp_path / 'output_processor.py'
    current = P.patch_text(STANDIN)
    feed = next(line for line in current.splitlines() if 'v35_hooks.feed(' in line)
    r1 = current.replace(feed, R1_FEED + P.MARK)
    assert r1 != current and P.revert_text(r1) == STANDIN
    path.write_text(r1)
    (tmp_path / 'output_processor.py.v35-orig').write_text(STANDIN)
    assert 'patched by an earlier version' in _run('--check', '--file', str(path))
    assert 'earlier version replaced' in _run('--apply', '--file', str(path))
    assert path.read_text() == current
    assert 'already patched (current)' in _run('--apply', '--file', str(path))
    assert 'patched (current)' in _run('--check', '--file', str(path))
    assert 'identical to the saved original' in _run('--revert', '--file', str(path))
    assert path.read_text() == STANDIN
    assert 'not patched' in _run('--check', '--file', str(path))
    assert 'patched,' in _run('--apply', '--file', str(path))
    assert (tmp_path / 'output_processor.py.v35-orig').read_text() == STANDIN


def feed_hook_in_loop(src):
    """the feed hook call sits in process_outputs' `for engine_core_output in ...` loop, before the logprobs processor reads
    engine_core_output (so new_logprobs is the current step's), and passes engine_core_output.new_logprobs"""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'process_outputs')
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and getattr(n.target, 'id', None) == 'engine_core_output')
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    feed = [c for c in calls if c.func.attr == 'feed' and getattr(c.func.value, 'id', None) == 'v35_hooks']
    assert len(feed) == 1
    arg = feed[0].args[2]
    assert isinstance(arg, ast.Attribute) and arg.attr == 'new_logprobs' and arg.value.id == 'engine_core_output'
    later = [c.lineno for c in calls if c.func.attr in ('update_from_output', 'append') and
             any(getattr(a, 'id', None) == 'engine_core_output' or getattr(a, 'attr', None) == 'new_logprobs' for a in c.args)]
    assert later and min(later) > feed[0].lineno


def test_feed_hook_position_on_the_standin():
    feed_hook_in_loop(P.patch_text(STANDIN))


@pytest.mark.skipif(not os.environ.get('V35_VLLM_OUTPUT_PROCESSOR'), reason='set V35_VLLM_OUTPUT_PROCESSOR to a vLLM file')
def test_real_vllm_file():
    src = open(os.environ['V35_VLLM_OUTPUT_PROCESSOR'], encoding='utf-8').read()
    base = P.revert_text(src)                            # the original, also when (an earlier version of) the patch is applied
    out = P.patch_text(base)
    assert P.revert_text(out) == base
    feed_hook_in_loop(out)
