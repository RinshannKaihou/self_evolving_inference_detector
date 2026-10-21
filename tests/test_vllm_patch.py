"""The vLLM patch (integrations/vllm_0_27/patch_output_processor.py) on a stand-in of the OutputProcessor that holds the six
anchors: it applies once, compiles, routes the calls as intended, refuses to apply twice and reverts exactly.
V35_VLLM_OUTPUT_PROCESSOR=/path/to/output_processor.py also checks a real vLLM file (skipped otherwise)."""
import importlib.util
import os
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'patch_output_processor', os.path.join(HERE, '..', 'integrations', 'vllm_0_27', 'patch_output_processor.py'))
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

    def process_outputs(self, outputs):
        for req_id, new_token_ids, finish_reason in outputs:
            req_state = self.request_states.get(req_id)
            stop_reason = None
            if True:
                stop_string = None
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

            # Free completed requests.
            if finish_reason is not None:
                self.request_states.pop(req_id)
                self.finished.append((req_id, finish_reason, stop_reason))
'''


class FakeHooks:
    def __init__(self, stop_after):
        self.calls, self.stop_after, self.seen = [], stop_after, 0

    def register(self, rid):
        self.calls.append(('register', rid))

    def feed(self, rid, ids):
        self.seen += len(ids)
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


def test_apply_routes_calls_and_intercepts():
    out = P.patch_text(STANDIN)
    assert out.count(P.MARK.strip()) == sum(len(lines) for _, _, lines in P.HOOKS)
    hooks = FakeHooks(stop_after=5)
    m = _module(out, hooks)
    op = m.OutputProcessor()
    state = types.SimpleNamespace(request_id='a', errors=[])
    op.add_request('a', state)
    op.add_request('b', types.SimpleNamespace(request_id='b', errors=[]))
    op.process_outputs([('a', [1, 2], None), ('a', [3, 4], None)])
    assert op.finished == []
    op.process_outputs([('a', [5], None)])               # the 5th token: the hook returns a stop reason
    assert op.finished == [('a', 'stop', 'v35_repetition: X')]
    op.abort_requests(['b'])
    assert hooks.calls == [('register', 'a'), ('register', 'b'), ('finish', 'a', 'stop'), ('abort', 'b')]


def test_existing_finish_reason_is_kept():
    hooks = FakeHooks(stop_after=1)
    m = _module(P.patch_text(STANDIN), hooks)
    op = m.OutputProcessor()
    op.add_request('a', types.SimpleNamespace(request_id='a', errors=[]))
    op.process_outputs([('a', [1], 'length')])           # the engine already ended it: the detector does not overwrite
    assert op.finished == [('a', 'length', None)]


def test_twice_refused_and_revert_is_exact():
    out = P.patch_text(STANDIN)
    with pytest.raises(ValueError):
        P.patch_text(out)
    assert P.revert_text(out) == STANDIN
    with pytest.raises(ValueError):                      # a missing anchor is refused, not guessed
        P.patch_text(STANDIN.replace('# Free completed requests.', '# Free finished requests.'))


@pytest.mark.skipif(not os.environ.get('V35_VLLM_OUTPUT_PROCESSOR'), reason='set V35_VLLM_OUTPUT_PROCESSOR to a vLLM file')
def test_real_vllm_file():
    src = open(os.environ['V35_VLLM_OUTPUT_PROCESSOR'], encoding='utf-8').read()
    out = P.patch_text(P.revert_text(src)) if P.MARK.strip() in src else P.patch_text(src)
    assert P.revert_text(out) == (P.revert_text(src) if P.MARK.strip() in src else src)
