#!/usr/bin/env python3
"""Insert the v35 detector hooks into vLLM's OutputProcessor (vllm/v1/engine/output_processor.py).

Six anchors, each of which must occur exactly once (true in vLLM 0.27.0, and in 0.18.0). Every inserted line ends with
the marker '# v35-detector hook', so the patch is easy to review and --revert removes exactly those lines.
Run it with the Python of the vLLM environment:

  python patch_output_processor.py --check     # anchors found once each, patched text compiles; nothing is written
  python patch_output_processor.py --apply     # writes the patched file; the original is kept as <file>.v35-orig
  python patch_output_processor.py --revert    # removes the hook lines
  --file PATH            patch this file instead of the installed vLLM's
  --allow-version X.Y.Z  accept another vLLM version (the anchors are still checked)
A file that carries the hook lines of an earlier version of this script (r1: feed without logprobs) is checked and patched
as its original: --apply replaces the old hook lines.

What the hooks do (src/v35_detector/serving/hooks.py): nothing until V35_DETECTOR or V35_RARE is set; then register / feed /
finish / abort per request. feed() gets the step's new token ids and their sample logprobs (EngineCoreOutput.new_logprobs:
a LogprobsLists [tokens, 1 + the batch's largest logprobs] with the generated token in column 0 and the top-k after it, or
None when the request did not ask for logprobs). In intercept mode feed() returns a stop reason and the request is finished
exactly like a stop string (finish_reason 'stop', stop_reason 'v35_repetition: <component>', aborted in the engine core).
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
TARGET = os.path.join('v1', 'engine', 'output_processor.py')

# (name, anchor text, lines inserted right after it; indentation is part of each line)
HOOKS = [
    ('import',
     'from vllm.v1.engine.detokenizer import IncrementalDetokenizer\n',
     ['try:',
      '    from v35_detector.serving import hooks as v35_hooks',
      'except ImportError:',
      '    v35_hooks = None']),
    ('register',                                          # OutputProcessor.add_request
     '        self.request_states[request_id] = req_state\n',
     ['        if v35_hooks is not None:',
      '            v35_hooks.register(request_id)']),
    ('feed',                                              # OutputProcessor.process_outputs, after the stop-string check
                                                          # (before the logprobs processor reads new_logprobs)
     '                if stop_string:\n'
     '                    finish_reason = FinishReason.STOP\n'
     '                    stop_reason = stop_string\n',
     ['                if v35_hooks is not None:',
      '                    v35_stop = v35_hooks.feed(req_state.request_id, new_token_ids, engine_core_output.new_logprobs)',
      '                    if v35_stop is not None and finish_reason is None:',
      '                        finish_reason = FinishReason.STOP',
      '                        stop_reason = v35_stop']),
    ('finish',                                            # OutputProcessor.process_outputs, freeing a finished request
     '            # Free completed requests.\n'
     '            if finish_reason is not None:\n',
     ['                if v35_hooks is not None:',
      '                    v35_hooks.finish(req_state.request_id, str(finish_reason))']),
    ('abort',                                             # OutputProcessor.abort_requests
     '            req_state = self.request_states.pop(request_id, None)\n'
     '            if req_state is not None:\n',
     ['                if v35_hooks is not None:',
      '                    v35_hooks.abort(request_id)']),
    ('error',                                             # OutputProcessor.propagate_error
     '        for _, state in self.request_states.items():\n',
     ['            if v35_hooks is not None:',
      '                v35_hooks.abort(state.request_id)']),
]


def patch_text(src):
    """patched source; raises ValueError if the file is already patched or an anchor is not found exactly once"""
    if MARK.strip() in src:
        raise ValueError('already patched (hook lines present); use --revert first')
    for name, anchor, _ in HOOKS:
        n = src.count(anchor)
        if n != 1:
            raise ValueError(f'anchor {name!r} found {n} times (expected 1): vLLM version not supported by this patch')
    for name, anchor, lines in HOOKS:
        src = src.replace(anchor, anchor + ''.join(line + MARK + '\n' for line in lines), 1)
    compile(src, TARGET, 'exec')                          # indentation / syntax of the result
    return src


def revert_text(src):
    return ''.join(line for line in src.splitlines(keepends=True) if not line.rstrip('\n').endswith(MARK.strip()))


def installed_target(allow_version):
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit('vllm is not installed in this Python environment (run with the vLLM environment, or pass --file)')
    version = importlib.metadata.version('vllm')
    if version not in (EXPECTED_VERSION, allow_version):
        raise SystemExit(f'vllm {version} installed; this patch targets {EXPECTED_VERSION} (--allow-version {version} to try)')
    return os.path.join(list(spec.submodule_search_locations)[0], TARGET), version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--apply', action='store_true')
    g.add_argument('--revert', action='store_true')
    ap.add_argument('--file')
    ap.add_argument('--allow-version')
    a = ap.parse_args()
    path, version = (a.file, 'n/a (--file)') if a.file else installed_target(a.allow_version)
    src = open(path, encoding='utf-8').read()
    base = revert_text(src)                              # the original, also under the hook lines of an earlier version
    orig = path + '.v35-orig'
    if a.revert:
        if base == src:
            print(f'{path}: no hook lines, nothing to revert')
            return
        open(path, 'w', encoding='utf-8').write(base)
        same = os.path.exists(orig) and open(orig, encoding='utf-8').read() == base
        print(f'{path}: hook lines removed' + (' (identical to the saved original)' if same else ''))
        return
    try:
        out = patch_text(base)
    except ValueError as e:
        sys.exit(f'{path}: {e}')
    added = out.count(MARK.strip())
    state = 'not patched' if base == src else 'patched (current)' if out == src else 'patched by an earlier version'
    if a.check:
        print(f'{path} (vllm {version}): {state}; all {len(HOOKS)} anchors found once; the patch adds {added} lines and compiles')
        return
    if out == src:
        print(f'{path} (vllm {version}): already patched (current), nothing to do')
        return
    if base == src:
        shutil.copy2(path, orig)
    elif not os.path.exists(orig):
        open(orig, 'w', encoding='utf-8').write(base)
    open(path, 'w', encoding='utf-8').write(out)
    print(f'{path} (vllm {version}): ' + ('patched' if base == src else 'hook lines of an earlier version replaced') +
          f', {added} hook lines; original kept as {orig}')


if __name__ == '__main__':
    main()
