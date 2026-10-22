#!/usr/bin/env python3
"""Insert the v35 detector hooks into vLLM 0.27.0 (r3: five files, 35 lines).

  v1/sample/sampler.py            Sampler.forward: keeps the raw logits (prepare) and, once the tokens are sampled, reduces
                                  each row to rc3's two values, (top-1 logprob, sampled token's logprob) of the raw
                                  distribution, with two Triton kernels (v35_detector/serving/collect.py). Only when
                                  V35_RARE is collect or shadow; with V35_RARE=off nothing runs. Skipped for speculative
                                  decoding's bonus-token call.
  v1/worker/gpu_model_runner.py   carries the [rows, 2] values from SamplerOutput to ModelRunnerOutput.v35_rare: with async
                                  scheduling (vLLM's default) a non-blocking copy on the output copy stream next to the
                                  sampled token ids, converted to lists in get_output(); without it, a list after the
                                  step's synchronisation
  v1/core/sched/scheduler.py      update_from_output: the request's pair goes into its EngineCoreOutput
  v1/engine/__init__.py           EngineCoreOutput gets the field v35_rare: list[float] | None (None is omitted on the wire)
  v1/engine/output_processor.py   register / feed / finish / abort of v35_detector/serving/hooks.py (m4 and rc3); feed gets
                                  the step's new token ids and EngineCoreOutput.v35_rare

Client-visible logprobs are untouched: the values never enter LogprobsTensors / new_logprobs, and a request that asks for
logprobs gets vLLM's own output. Every anchor must occur exactly once in its file (true in vLLM 0.27.0) and every patched
file must compile. Every inserted line ends with the marker '# v35-detector hook', so the patch is easy to review and
--revert removes exactly those lines. Run it with the Python of the vLLM environment:

  python patch_vllm.py --check     # anchors found once each, patched text compiles; nothing is written
  python patch_vllm.py --apply     # writes the patched files; each original is kept as <file>.v35-orig
  python patch_vllm.py --revert    # removes the hook lines
  --root DIR             patch the vllm package directory DIR instead of the installed vLLM's
  --allow-version X.Y.Z  accept another vLLM version (the anchors are still checked)
Files that carry the hook lines of an earlier version (r1, r2: output_processor.py only) are checked and patched as their
originals: --apply replaces the old hook lines.

If the v35_detector package is not importable, the patched vLLM behaves as the original.
"""
import argparse
import importlib.metadata
import importlib.util
import os
import shutil
import sys

MARK = '  # v35-detector hook'
EXPECTED_VERSION = '0.27.0'
IMPORT_COLLECT = ['try:',
                  '    from v35_detector.serving.collect import SAMPLER as v35_collect',
                  'except ImportError:',
                  '    v35_collect = None']

# file key -> (path under the vllm package, test variable, [(name, anchor text, lines inserted right after it)])
FILES = {
    'sampler': ('v1/sample/sampler.py', 'V35_VLLM_SAMPLER', [
        ('import',
         'from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler\n',
         IMPORT_COLLECT),
        ('prepare',                                       # Sampler.forward, before the logits are converted and processed
         '        num_logprobs = sampling_metadata.max_num_logprobs\n'
         '        raw_logprobs: torch.Tensor | None = None\n',
         ['        v35_raw = v35_collect.prepare(logits, predict_bonus_token) if v35_collect is not None else None']),
        ('finish',                                        # Sampler.forward, the SamplerOutput built, before it is returned
         '            sampled_token_ids=sampled.unsqueeze(-1),\n'
         '            logprobs_tensors=logprobs_tensors,\n'
         '        )\n',
         ['        if v35_raw is not None:',
          '            sampler_output.v35_rare = v35_collect.finish(v35_raw, sampled)']),
    ]),
    'gpu_model_runner': ('v1/worker/gpu_model_runner.py', 'V35_VLLM_GPU_MODEL_RUNNER', [
        ('async_copy',                                    # AsyncGPUModelRunnerOutput.__init__, on the output copy stream
         '            self._logprobs_tensors_cpu = (\n'
         '                self._logprobs_tensors.to_cpu_nonblocking()\n'
         '                if self._logprobs_tensors\n'
         '                else None\n'
         '            )\n',
         ['            self._v35_rare = getattr(model_runner_output, \'v35_rare\', None)',
          '            if self._v35_rare is not None:',
          '                model_runner_output.v35_rare = self._v35_rare.to(\'cpu\', non_blocking=True)']),
        ('async_get',                                     # AsyncGPUModelRunnerOutput.get_output, after the copy is synchronised
         '        output.sampled_token_ids = valid_sampled_token_ids\n'
         '        output.logprobs = logprobs_lists\n',
         ['        if self._v35_rare is not None:',
          '            output.v35_rare = output.v35_rare.tolist()']),
        ('output',                                        # GPUModelRunner.sample_tokens, the ModelRunnerOutput built
         '                cudagraph_stats=cudagraph_stats,\n'
         '                routed_experts=None,\n'
         '            )\n',
         ['            v35_rare = getattr(sampler_output, \'v35_rare\', None)',
          '            if v35_rare is not None:',
          '                output.v35_rare = v35_rare if self.use_async_scheduling else v35_rare.tolist()']),
    ]),
    'scheduler': ('v1/core/sched/scheduler.py', 'V35_VLLM_SCHEDULER', [
        ('read',                                          # Scheduler.update_from_output
         '        logprobs = model_runner_output.logprobs\n',
         ['        v35_rare = getattr(model_runner_output, \'v35_rare\', None)']),
        ('engine_core_output',                            # the EngineCoreOutput(...) of a request with new tokens
         '                        num_nans_in_logits=request.num_nans_in_logits,\n',
         ['                        v35_rare=v35_rare[req_index] if v35_rare is not None and new_token_ids else None,']),
    ]),
    'engine': ('v1/engine/__init__.py', 'V35_VLLM_ENGINE', [
        ('field',                                         # the last field of EngineCoreOutput
         '    num_nans_in_logits: int = 0\n',
         ['    v35_rare: list[float] | None = None']),
    ]),
    'output_processor': ('v1/engine/output_processor.py', 'V35_VLLM_OUTPUT_PROCESSOR', [
        ('import',
         'from vllm.v1.engine.detokenizer import IncrementalDetokenizer\n',
         ['try:',
          '    from v35_detector.serving import hooks as v35_hooks',
          'except ImportError:',
          '    v35_hooks = None']),
        ('register',                                      # OutputProcessor.add_request
         '        self.request_states[request_id] = req_state\n',
         ['        if v35_hooks is not None:',
          '            v35_hooks.register(request_id)']),
        ('feed',                                          # OutputProcessor.process_outputs, after the stop-string check
         '                if stop_string:\n'
         '                    finish_reason = FinishReason.STOP\n'
         '                    stop_reason = stop_string\n',
         ['                if v35_hooks is not None:',
          '                    v35_stop = v35_hooks.feed(req_state.request_id, new_token_ids, engine_core_output.v35_rare)',
          '                    if v35_stop is not None and finish_reason is None:',
          '                        finish_reason = FinishReason.STOP',
          '                        stop_reason = v35_stop']),
        ('finish',                                        # OutputProcessor.process_outputs, freeing a finished request
         '            # Free completed requests.\n'
         '            if finish_reason is not None:\n',
         ['                if v35_hooks is not None:',
          '                    v35_hooks.finish(req_state.request_id, str(finish_reason))']),
        ('abort',                                         # OutputProcessor.abort_requests
         '            req_state = self.request_states.pop(request_id, None)\n'
         '            if req_state is not None:\n',
         ['                if v35_hooks is not None:',
          '                    v35_hooks.abort(request_id)']),
        ('error',                                         # OutputProcessor.propagate_error
         '        for _, state in self.request_states.items():\n',
         ['            if v35_hooks is not None:',
          '                v35_hooks.abort(state.request_id)']),
    ]),
}


def n_lines(key=None):
    keys = [key] if key else list(FILES)
    return sum(len(lines) for k in keys for _, _, lines in FILES[k][2])


def patch_text(key, src):
    """patched source of file `key`; raises ValueError if it is already patched or an anchor is not found exactly once"""
    path, _, hooks = FILES[key]
    if MARK.strip() in src:
        raise ValueError('already patched (hook lines present); use --revert first')
    for name, anchor, _ in hooks:
        n = src.count(anchor)
        if n != 1:
            raise ValueError(f'anchor {name!r} found {n} times (expected 1): vLLM version not supported by this patch')
    for name, anchor, lines in hooks:
        src = src.replace(anchor, anchor + ''.join(line + MARK + '\n' for line in lines), 1)
    compile(src, path, 'exec')                           # indentation / syntax of the result
    return src


def revert_text(src):
    return ''.join(line for line in src.splitlines(keepends=True) if not line.rstrip('\n').endswith(MARK.strip()))


def installed_root(allow_version):
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit('vllm is not installed in this Python environment (run with the vLLM environment, or pass --root)')
    version = importlib.metadata.version('vllm')
    if version not in (EXPECTED_VERSION, allow_version):
        raise SystemExit(f'vllm {version} installed; this patch targets {EXPECTED_VERSION} (--allow-version {version} to try)')
    return list(spec.submodule_search_locations)[0], version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--apply', action='store_true')
    g.add_argument('--revert', action='store_true')
    ap.add_argument('--root', help='a vllm package directory (the one holding v1/)')
    ap.add_argument('--allow-version')
    a = ap.parse_args()
    root, version = (a.root, 'n/a (--root)') if a.root else installed_root(a.allow_version)
    plan = []                                            # (key, path, current text, original, patched)
    for key, (rel, _, hooks) in FILES.items():
        path = os.path.join(root, *rel.split('/'))
        src = open(path, encoding='utf-8').read()
        base = revert_text(src)                          # the original, also under the hook lines of an earlier version
        if a.revert:
            plan.append((key, path, src, base, None))
            continue
        try:
            out = patch_text(key, base)
        except ValueError as e:
            sys.exit(f'{path}: {e}; nothing written')
        plan.append((key, path, src, base, out))
    if a.revert:
        for key, path, src, base, _ in plan:
            if base == src:
                print(f'{path}: no hook lines, nothing to revert')
                continue
            open(path, 'w', encoding='utf-8').write(base)
            orig = path + '.v35-orig'
            same = os.path.exists(orig) and open(orig, encoding='utf-8').read() == base
            print(f'{path}: hook lines removed' + (' (identical to the saved original)' if same else ''))
        return
    for key, path, src, base, out in plan:
        state = 'not patched' if base == src else 'patched (current)' if out == src else 'patched by an earlier version'
        if a.check:
            print(f'{path} (vllm {version}): {state}; all {len(FILES[key][2])} anchors found once; the patch adds '
                  f'{n_lines(key)} lines and compiles')
            continue
        if out == src:
            print(f'{path} (vllm {version}): already patched (current), nothing to do')
            continue
        orig = path + '.v35-orig'
        if base == src:
            shutil.copy2(path, orig)
        elif not os.path.exists(orig):
            open(orig, 'w', encoding='utf-8').write(base)
        open(path, 'w', encoding='utf-8').write(out)
        print(f'{path} (vllm {version}): ' + ('patched' if base == src else 'hook lines of an earlier version replaced') +
              f', {n_lines(key)} hook lines; original kept as {orig}')
    if a.check:
        print(f'all {len(FILES)} files: {n_lines()} hook lines in total')


if __name__ == '__main__':
    main()
