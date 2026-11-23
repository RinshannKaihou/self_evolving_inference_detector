"""Tests for m4_1 novelty guard and tool-call span exclusion across RepDetector, RareDetector, vocab builders, and hooks."""
import json

import numpy as np
import pytest

from synth import make
from test_rare import _pool, synth as rare_synth
from test_serving import drive, events
from test_state import state_bytes
from v35_detector import (
    CONFIG_M4,
    CONFIG_M4_1,
    CONFIG_RC1T,
    CONFIG_RC2T,
    CONFIG_RC3,
    CONFIG_RC4,
    RareDetector,
    RepDetector,
)
from v35_detector import rare as R
from v35_detector import tool_spans as TS
from v35_detector.rare_vocab import load as load_rare
from v35_detector.rc2t_vocab import build as build_rc2t_bundle
from v35_detector.serving import hooks
from v35_detector.vocab import SEP, THINK_END, _byte_decoder, load as load_rep

M4 = json.load(open(CONFIG_M4))
M4_1 = json.load(open(CONFIG_M4_1))
RC3 = json.load(open(CONFIG_RC3))
RC4 = json.load(open(CONFIG_RC4))
RC1T = json.load(open(CONFIG_RC1T))
RC2T = json.load(open(CONFIG_RC2T))

_BYTE_ENC = {b: ch for ch, b in _byte_decoder().items()}


def _bl(text):
    return ''.join(_BYTE_ENC[b] for b in text.encode('utf-8'))


def _stream_rep(det, tok, size):
    st = det.start()
    res = None
    for i in range(0, len(tok), size):
        res = det.feed(st, tok[i:i + size])
        if res is not None:
            break
    res = res or det.finish(st)
    mon = det.monitor(st)
    return (res.fired_at, res.reason), (mon.fired_at, mon.reason)


def _stream_rare(det, tok, lp1, lpc, size):
    st = det.start()
    res = None
    for i in range(0, len(tok), size):
        ch = tok[i:i + size]
        if det.token_only:
            res = det.feed(st, ch)
        else:
            res = det.feed(st, ch, lp1[i:i + size], lpc[i:i + size])
        if res is not None:
            break
    res = res or det.finish(st)
    mon = det.monitor(st)
    return (res.fired_at, res.reason), (mon.fired_at, mon.reason)


# ------------------------------------------------------------------ 1. CONFIG_M4_1 novelty guard
def test_m4_1_config_diff_from_frozen_m4():
    assert M4['name'] == 'm4' and M4_1['name'] == 'm4_1'
    assert len(M4['components']) == len(M4_1['components']) == 14
    diffs = []
    for i, (c0, c1) in enumerate(zip(M4['components'], M4_1['components'])):
        if c0 != c1:
            diffs.append((i, c0, c1))
    assert len(diffs) == 1
    idx, c0, c1 = diffs[0]
    assert idx == 8
    assert c0 == {'kind': 'M', 'feat': 'wrep_W2048', 'theta': 0.99, 'S': 24, 'and': []}
    assert c1 == {'kind': 'M', 'feat': 'wrep_W2048', 'theta': 0.99, 'S': 24, 'nov': {'feat': 'N_n4_W512', 'max': 0.2}, 'and': []}


def test_m4_1_suppresses_vocab_saturation_fp_but_keeps_low_novelty_loop():
    """Long context drawing from a bounded vocabulary with fresh 4-grams (high wrep_W2048, high N_n4_W512)
    fires bare wrep_W2048>=0.99 S24 under M4, but is suppressed by M4_1's N_n4_W512<=0.2 guard."""
    T = load_rep('qwen3')
    n, ids = T['norm'], np.arange(len(T['norm']))
    # 220 distinct plain non-marker words (no sentence ends, no run chars)
    words = ids[(n > 0) & T['letter'] & ~T['marker'] & ~T['send'] & (T['run_char'] < 0) & (T['chars'] >= 3) & (T['chars'] <= 8)]
    # Pick words with distinct normalised IDs so every word is a distinct vocab item.
    _, uniq_idx = np.unique(n[words], return_index=True)
    vocab_pool = words[uniq_idx[:220]]

    # First emit all 220 words once so every word is in `seen`, then sample fresh random permutations
    # (zero repeated 4-grams -> N_n4_W512 ~ 1.0, while wrep_W2048 == 1.0 for every boundary after warmup).
    rng = np.random.default_rng(123)
    chunks = [vocab_pool] + [rng.permutation(vocab_pool) for _ in range(25)]
    sat_stream = np.concatenate(chunks)

    res_m4, _ = _stream_rep(RepDetector('qwen3', CONFIG_M4), sat_stream, 64)
    res_m4_1, _ = _stream_rep(RepDetector('qwen3', CONFIG_M4_1), sat_stream, 64)
    assert res_m4[0] > 0 and res_m4[1] == 'wrep_W2048>=0.99 S24'
    assert res_m4_1 == (-1, '')

    # Genuine low-novelty circling (only 12 words reshuffled with circling markers) still fires under M4_1.
    circling = make('glm', 'circling', 1)
    assert _stream_rep(RepDetector('glm', CONFIG_M4_1), circling, 64)[0][0] > 0


# ------------------------------------------------------------------ 2. RepDetector tool-call exclusion
@pytest.mark.parametrize('corpus,start_seq,end_seq', [
    ('qwen3', (151657,), (151658,)),
    ('glm', (154843,), (154844,)),
    ('dsv4', (128806,), (128807,)),
    ('dsv4', (128808,), (128809,)),
    ('dsv4', (128825, 8701, 4941, 12548), (128825, 8701, 4941, 12548)),
    ('dsv4', (128825, 72461, 4941, 12548), (128825, 72461, 4941, 12548)),
    ('dsv4', (128825, 10699), (128825, 10699)),
    ('dsv4', (128840, 8701, 4941, 12548), (128841, 8701, 4941, 12548)),
    ('dsv4', (128840, 72461, 4941, 12548), (128841, 72461, 4941, 12548)),
    ('dsv4', (128840, 10699), (128841, 10699)),
])
def test_rep_skips_tool_call_and_preserves_original_fired_at(corpus, start_seq, end_seq):
    loop = make(corpus, 'loop40', 1)
    normal = make(corpus, 'random', 1)[:300]
    s_arr = np.array(start_seq, np.int64)
    e_arr = np.array(end_seq, np.int64)

    # Pure tool-call turn containing a severe repetition loop -> suppressed when auto, fires when disabled.
    pure_tool = np.concatenate([s_arr, loop, e_arr])
    assert _stream_rep(RepDetector(corpus, CONFIG_M4_1), pure_tool, 64)[0] == (-1, '')
    assert _stream_rep(RepDetector(corpus, CONFIG_M4_1, tool_spans=None), pure_tool, 64)[0][0] > 0

    # Normal prefix + repetitive tool call + normal suffix -> clean.
    wrapped = np.concatenate([normal, s_arr, loop, e_arr, normal])
    assert _stream_rep(RepDetector(corpus, CONFIG_M4_1), wrapped, 37)[0] == (-1, '')

    # Normal prefix + tool call + looping suffix -> fires at exact original raw-token coordinate across all chunk sizes.
    interleaved = np.concatenate([normal, s_arr, loop, e_arr, loop])
    seg2_offset = len(normal) + len(s_arr) + len(loop) + len(e_arr)
    expected = TS.decide_rep(
        interleaved, load_rep(corpus), M4_1, THINK_END[corpus],
        tool_spans=TS.SHIPPED_TOOL_SPANS[corpus], think_pairs=TS.SHIPPED_THINK_PAIRS[corpus],
    )
    assert expected[0] > seg2_offset
    for size in (1, 2, 7, 64, 129, len(interleaved)):
        got, _ = _stream_rep(RepDetector(corpus, CONFIG_M4_1), interleaved, size)
        assert got == expected, (corpus, start_seq, size, got, expected)


def test_rep_segment_disconnect_prevents_cross_gap_bridging():
    """Tool-call exclusion must NOT delete-and-concatenate: splitting a character run across a tool call disconnects it."""
    T = load_rep('qwen3')
    ids = np.arange(len(T['norm']))
    run_tok = int(ids[(T['run_char'] >= 0) & (T['chars'] >= 8) & (T['norm'] == SEP)][0])
    ch = int(T['chars'][run_tok])
    half = (500 // ch) // 2 + 2                          # each half < 500 chars, together > 500 chars
    assert half * ch < 500 and 2 * half * ch >= 500

    stream_tok = np.concatenate([
        np.repeat(run_tok, half),
        np.array([151657, 100, 101, 151658], np.int64),
        np.repeat(run_tok, half),
    ])
    # Without bridging, neither segment reaches 500 chars.
    assert _stream_rep(RepDetector('qwen3', CONFIG_M4_1), stream_tok, 17)[0] == (-1, '')
    # Without tool skipping, the two halves + tool tokens wouldn't bridge either; verify direct concatenation WOULD fire.
    assert _stream_rep(RepDetector('qwen3', CONFIG_M4_1), np.repeat(run_tok, 2 * half), 17)[0][1] == 'run>=500'


def test_think_protection_and_literal_think_end_inside_tool_call():
    """1) Open <think> suppresses tool_call skipping so errors inside <think> are still caught.
    2) Literal </think> inside an active <tool_call> MUST NOT exit tool-call skipping."""
    loop = make('qwen3', 'loop40', 2)
    T_rare = load_rare('qwen3')
    lone = next(i for i, b in enumerate(T_rare['bytes']) if b == b'\x80')
    think_s, think_e, tc_s, tc_e = 151667, 151668, 151657, 151658

    # Case A: <think> ... <tool_call> (quoted in thought) ... loop ... </think> -> RepDetector MUST catch loop!
    quoted_in_think = np.concatenate([
        np.array([think_s, 10, 11, tc_s], np.int64),
        loop,
        np.array([think_e], np.int64),
    ])
    assert _stream_rep(RepDetector('qwen3', CONFIG_M4_1), quoted_in_think, 19)[0][0] > 0

    # RareDetector also catches bad_utf8 inside open <think> even after a quoted <tool_call>.
    tok_r, _, _ = rare_synth('qwen3', 260, 1, han_share=0.0)
    tok_r[0], tok_r[5], tok_r[100], tok_r[150] = think_s, tc_s, lone, think_e
    assert _stream_rare(RareDetector('qwen3', CONFIG_RC2T), tok_r, None, None, 13)[0] == (102, 'bad_utf8')

    # Case B: <think> ok </think> <tool_call> ... </think> (literal in Write args) ... loop & bad_utf8 ... </tool_call>
    # MUST stay skipped!
    think_end_in_tool = np.concatenate([
        np.array([think_s, 10, 11, think_e, tc_s, 20, think_e], np.int64),
        loop,
        np.array([lone, 21, tc_e], np.int64),
    ])
    assert _stream_rep(RepDetector('qwen3', CONFIG_M4_1), think_end_in_tool, 11)[0] == (-1, '')
    assert _stream_rare(RareDetector('qwen3', CONFIG_RC2T), think_end_in_tool, None, None, 11)[0] == (-1, '')


# ------------------------------------------------------------------ 3. RareDetector tool-call exclusion & seam semantics
@pytest.mark.parametrize('cfg_name,cfg', [('rc3', RC3), ('rc4', RC4), ('rc1t', RC1T), ('rc2t', RC2T)])
def test_rare_skips_tool_call_and_matches_decide_across_chunks(cfg_name, cfg):
    T = load_rare('qwen3')
    lone = next(i for i, b in enumerate(T['bytes']) if b == b'\x80')
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    han = int(_pool(T, 'han', False)[0])
    tok, lp1, lpc = rare_synth('qwen3', 500, 7, han_share=0.0)
    lp1[:], lpc[:] = -0.01, -0.01
    tc_s, tc_e = 151657, 151658

    # Pure skip: bad_utf8 and glued Han inside <tool_call> do NOT fire (no bad_utf8 fallback).
    tool_payload = np.array([tc_s, word, han, lone, word, tc_e], np.int64)
    seq_clean = np.concatenate([tok[:300], tool_payload, tok[300:]])
    lp1_clean = None if cfg_name in ('rc1t', 'rc2t') else np.full(len(seq_clean), -0.01)
    lpc_clean = None if cfg_name in ('rc1t', 'rc2t') else np.full(len(seq_clean), -0.01)

    det = RareDetector('qwen3', cfg)
    assert _stream_rare(det, seq_clean, lp1_clean, lpc_clean, 17) == ((-1, ''), (-1, ''))
    assert R.decide(seq_clean, lp1_clean, lpc_clean, 'qwen3', cfg) == (-1, '')

    # With tool_spans=None, the planted anomaly inside <tool_call> MUST fire (rc3 hits glued Han first, others bad_utf8).
    disabled_res = R.decide(seq_clean, lp1_clean, lpc_clean, 'qwen3', cfg, tool_spans=None)
    assert disabled_res[0] > 0 and disabled_res[1] in ('bad_utf8', 'fs<0.01 & caseless_after_latin & answer')

    # Seam disconnect: Latin word right before <tool_call> + Han right after </tool_call> must NOT trigger caseless_after_latin.
    seq_seam = seq_clean.copy()
    seq_seam[299] = word
    seq_seam[300 + len(tool_payload)] = han
    assert R.decide(seq_seam, lp1_clean, lpc_clean, 'qwen3', cfg, tier='monitor')[1] != 'fs<0.01 & caseless_after_latin & answer'
    assert 'caseless_after_latin' not in R.decide(seq_seam, lp1_clean, lpc_clean, 'qwen3', cfg, tier='monitor')[1]

    # Cumulative context preserved & exact original fired_at after tool_call:
    # plant bad_utf8 in segment 2 after </tool_call>.
    seq_after = seq_clean.copy()
    bad_pos = 300 + len(tool_payload) + 40
    seq_after[bad_pos] = lone
    want_strict = R.decide(seq_after, lp1_clean, lpc_clean, 'qwen3', cfg, tier='intercept')
    want_mon = R.decide(seq_after, lp1_clean, lpc_clean, 'qwen3', cfg, tier='monitor')
    assert want_strict == (bad_pos + 2, 'bad_utf8')
    for size in (1, 2, 7, 64, len(seq_after)):
        got_s, got_m = _stream_rare(det, seq_after, lp1_clean, lpc_clean, size)
        assert got_s == want_strict and got_m == want_mon, (cfg_name, size, got_s, want_strict)

    # Causality at segment boundary: bad_utf8 at the LAST token before <tool_call> fires at exact position (lag=0).
    seq_edge = seq_clean.copy()
    seq_edge[299] = lone
    want_edge = R.decide(seq_edge, lp1_clean, lpc_clean, 'qwen3', cfg)
    assert want_edge == (300, 'bad_utf8')
    for size in (1, 3, 64, len(seq_edge)):
        assert _stream_rare(det, seq_edge, lp1_clean, lpc_clean, size)[0] == want_edge


def test_rare_multitoken_dsml_logprobs_lockstep_and_trailing_prefix():
    """Multi-token DeepSeek DSML start marker buffers '｜DSML｜' (128825) with its logprobs in lockstep across chunks."""
    T = load_rare('dsv4')
    lone = next(i for i, b in enumerate(T['bytes']) if b == b'\x80')
    tok, lp1, lpc = rare_synth('dsv4', 120, 4, han_share=0.0)
    lp1[:] = -np.linspace(0.01, 0.05, len(tok))
    lpc[:] = lp1 - 0.005

    dsml_s = np.array([128825, 8701, 4941, 12548], np.int64)
    dsml_e = np.array([128825, 8701, 4941, 12548], np.int64)
    # Include overlapping 128825 right before real DSML start, bad_utf8 inside DSML, bad_utf8 after DSML, and trailing 128825.
    seq = np.concatenate([
        tok[:30],
        np.array([128825, 128825], np.int64),            # overlapping prefix before real DSML start
        dsml_s,
        np.array([lone, 10, 11], np.int64),              # skipped inside DSML!
        dsml_e,
        tok[30:50],
        np.array([lone, tok[51], 128825], np.int64),     # real bad_utf8 + trailing partial prefix
    ])
    lp1_full = -np.linspace(0.01, 0.09, len(seq))
    lpc_full = lp1_full - 0.002

    for cfg in (RC4, RC2T):
        det = RareDetector('dsv4', cfg)
        l1 = None if det.token_only else lp1_full
        lc = None if det.token_only else lpc_full
        want = R.decide(seq, l1, lc, 'dsv4', cfg)
        assert want == (len(seq) - 1, 'bad_utf8')
        for size in (1, 2, 3, 5, 64, len(seq)):
            assert _stream_rare(det, seq, l1, lc, size)[0] == want

    # Stream ending with partial prefix 128825 right after a lone byte (no DSML match): flushed at finish().
    seq_tail = np.array([tok[0], lone, 128825], np.int64)
    lp_tail = np.array([-0.01, -0.01, -0.01])
    det4 = RareDetector('dsv4', RC4)
    assert R.decide(seq_tail, lp_tail, lp_tail, 'dsv4', RC4) == (3, 'bad_utf8')
    for size in (1, 2, 3):
        assert _stream_rare(det4, seq_tail, lp_tail, lp_tail, size)[0] == (3, 'bad_utf8')


# ------------------------------------------------------------------ 4. Dynamic tokenizer auto-detection
def test_dynamic_kimi_k3_xtml_mistral_llama_and_dsml_detection(tmp_path):
    # 1. Kimi-K3 XTML multi-token <|open|>tools<|sep|> .. <|close|>tools<|sep|> and think protection
    k3_vocab = {
        '<|open|>': 0, '<|close|>': 1, '<|sep|>': 2,
        'tools': 3, 'think': 4, 'answer': 5,
        '\u2581hello': 6, '\u2581world': 7, '<0x80>': 8,
    }
    k3_spec = {
        'model': {'type': 'Unigram', 'vocab': [[k, 0.0] for k in k3_vocab]},
        'added_tokens': [
            {'id': 0, 'content': '<|open|>', 'special': False},
            {'id': 1, 'content': '<|close|>', 'special': False},
            {'id': 2, 'content': '<|sep|>', 'special': False},
        ],
        'decoder': {'type': 'Sequence', 'decoders': [
            {'type': 'Replace', 'pattern': {'String': '\u2581'}, 'content': ' '},
            {'type': 'ByteFallback'},
        ]},
    }
    k3_path = tmp_path / 'k3_tokenizer.json'
    k3_path.write_text(json.dumps(k3_spec), encoding='utf-8')
    det_k3 = RareDetector(str(k3_path), CONFIG_RC2T)
    assert det_k3.tool_spans == (((0, 3, 2), (1, 3, 2)),)
    assert det_k3.think_pairs == (((0, 4, 2), (1, 4, 2)),)

    # Non-tool <|open|>answer<|sep|> passes through; <|open|>tools<|sep|> skips bad_utf8;
    # <|open|>think<|sep|> protects quoted <|open|>tools<|sep|> from skipping bad_utf8!
    seq_tool_skip = np.array([0, 5, 2, 6, 7, 0, 3, 2, 8, 1, 3, 2, 6, 7], np.int64)
    for size in (1, 2, 5, len(seq_tool_skip)):
        assert _stream_rare(det_k3, seq_tool_skip, None, None, size)[0] == (-1, '')

    seq_in_think = np.array([0, 4, 2, 0, 3, 2, 8, 6, 1, 4, 2], np.int64)
    for size in (1, 2, 4, len(seq_in_think)):
        assert _stream_rare(det_k3, seq_in_think, None, None, size)[0] == (8, 'bad_utf8')

    # 2. Mistral open-to-end [TOOL_CALLS] and Llama <|python_tag|> .. <|eom_id|>
    mistral_spans = TS.detect_tool_spans({'[TOOL_CALLS]': 9, 'hello': 1})
    assert mistral_spans == (((9,), None),)
    m_matcher = TS.ToolSpanMatcher(mistral_spans)
    assert m_matcher.split_segments([1, 1, 9, 1, 1, 1]) == [(0, 2)]

    llama_spans = TS.detect_tool_spans({'<|python_tag|>': 128010, '<|eom_id|>': 128008})
    assert llama_spans == (((128010,), (128008,)),)

    # 3. DeepSeek DSML BPE merges detection (V3.2 function_calls, V4 tool_calls, V4.1 ' calls' -> ByteLevel 'Ġcalls')
    ds_vocab = {c: i for i, c in enumerate(list('</function_als>\u0120'))}
    base = len(ds_vocab)
    for extra in ('</', 'function', 'tool', '_calls', '\u0120calls', '｜DSML｜', '<dsml:', '</dsml:'):
        ds_vocab[extra] = base
        base += 1
    merges_str = [
        '< /',
        'f u', 'fu n', 'fun c', 'func t', 'funct i', 'functi o', 'functio n',
        't o', 'to o', 'too l',
        '_ c', '_c a', '_ca l', '_cal l', '_call s',
        '\u0120 c', '\u0120c a', '\u0120ca l', '\u0120cal l', '\u0120call s',
    ]
    # Add intermediate pieces to vocab
    for m in merges_str:
        joined = m.replace(' ', '')
        if joined not in ds_vocab:
            ds_vocab[joined] = base
            base += 1
    spans_str = TS.detect_tool_spans(ds_vocab, merges=merges_str)
    spans_list = TS.detect_tool_spans(ds_vocab, merges=[m.split(' ') for m in merges_str])
    assert spans_str == spans_list
    assert spans_str == (
        ((ds_vocab['｜DSML｜'], ds_vocab['function'], ds_vocab['_calls']),
         (ds_vocab['｜DSML｜'], ds_vocab['function'], ds_vocab['_calls'])),
        ((ds_vocab['<dsml:'], ds_vocab['function'], ds_vocab['_calls']),
         (ds_vocab['</dsml:'], ds_vocab['function'], ds_vocab['_calls'])),
        ((ds_vocab['｜DSML｜'], ds_vocab['tool'], ds_vocab['_calls']),
         (ds_vocab['｜DSML｜'], ds_vocab['tool'], ds_vocab['_calls'])),
        ((ds_vocab['<dsml:'], ds_vocab['tool'], ds_vocab['_calls']),
         (ds_vocab['</dsml:'], ds_vocab['tool'], ds_vocab['_calls'])),
        ((ds_vocab['｜DSML｜'], ds_vocab['\u0120calls']),
         (ds_vocab['｜DSML｜'], ds_vocab['\u0120calls'])),
        ((ds_vocab['<dsml:'], ds_vocab['\u0120calls']),
         (ds_vocab['</dsml:'], ds_vocab['\u0120calls'])),
    )


def test_npz_bundle_persists_tool_spans_for_rep_and_rare(tmp_path):
    tk = pytest.importorskip('tokenizers')
    alphabet = tk.pre_tokenizers.ByteLevel.alphabet()
    pieces = sorted(alphabet) + ['Ġhello', 'Ġworld', 'Ġloop']
    tokenizer = tk.Tokenizer(tk.models.BPE({c: i for i, c in enumerate(pieces)}, []))
    tokenizer.decoder = tk.decoders.ByteLevel()
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_tokens(['<think>', '</think>', '<tool_call>', '</tool_call>'])

    tok_json = tmp_path / 'tokenizer.json'
    tokenizer.save(str(tok_json))
    npz_path = tmp_path / 'bundle.npz'
    np.savez_compressed(npz_path, **build_rc2t_bundle(str(tok_json), include_m4=True))

    tc_s = tokenizer.token_to_id('<tool_call>')
    tc_e = tokenizer.token_to_id('</tool_call>')
    th_s = tokenizer.token_to_id('<think>')
    th_e = tokenizer.token_to_id('</think>')

    det_rep = RepDetector(str(npz_path), CONFIG_M4_1, think_end=th_e)
    det_rare = RareDetector(str(npz_path), CONFIG_RC2T)
    assert det_rep.tool_spans == (((tc_s,), (tc_e,)),)
    assert det_rep.think_pairs == (((th_s,), (th_e,)),)
    assert det_rare.tool_spans == (((tc_s,), (tc_e,)),)
    assert det_rare.think_pairs == (((th_s,), (th_e,)),)

    # Plain .npz without metadata or bytes gracefully disables auto tool_spans.
    plain_path = tmp_path / 'plain.npz'
    plain_data = {k: v for k, v in load_rep('qwen3').items()}
    np.savez(plain_path, **plain_data)
    det_plain = RepDetector(str(plain_path), CONFIG_M4_1, think_end=None)
    assert det_plain.tool_spans == ()


# ------------------------------------------------------------------ 5. Serving hooks defaults & state bounds
@pytest.mark.parametrize('runner', ['sync', 'async'])
def test_serving_hooks_default_m4_1_and_tool_call_exclusion(tmp_path, runner):
    # Default config is CONFIG_M4_1; default tool_call is 'exclude'; shorthands 'm4'/'m4_1' work.
    s_default = hooks.settings({'V35_DETECTOR': 'intercept', 'V35_DETECTOR_TABLE': 'qwen3',
                                'V35_DETECTOR_RUNNER': runner, 'V35_DETECTOR_EVENTS': str(tmp_path / f'd_{runner}.jsonl')})
    assert s_default['config'] == CONFIG_M4_1
    assert s_default['tool_call'] == 'exclude' and s_default['tool_spans'] == 'auto'

    s_m4 = hooks.settings({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': 'qwen3', 'V35_DETECTOR_CONFIG': 'm4'})
    assert s_m4['config'] == CONFIG_M4
    s_m4_1 = hooks.settings({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': 'qwen3', 'V35_DETECTOR_CONFIG': 'm4_1'})
    assert s_m4_1['config'] == CONFIG_M4_1
    with pytest.raises(ValueError, match='V35_DETECTOR_TOOL_CALL'):
        hooks.settings({'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': 'qwen3', 'V35_DETECTOR_TOOL_CALL': 'bad'})

    # Build stream with clean Latin text + repetitive & bad_utf8 payload inside <tool_call>...</tool_call>
    T_rare = load_rare('qwen3')
    lone = next(i for i, b in enumerate(T_rare['bytes']) if b == b'\x80')
    clean_text, _, _ = rare_synth('qwen3', 450, 1, han_share=0.0)
    loop = make('qwen3', 'loop40', 1)
    tok = np.concatenate([
        clean_text[:220],
        np.array([151657, lone], np.int64),
        loop,
        np.array([151658], np.int64),
        clean_text[220:],
    ])

    # Default V35_DETECTOR_TOOL_CALL=exclude -> ZERO repetition or rare_chars alerts!
    env_excl = {
        'V35_DETECTOR': 'intercept',
        'V35_RARE': 'shadow',
        'V35_RARE_CONFIG': 'rc2t',
        'V35_DETECTOR_TABLE': 'qwen3',
        'V35_DETECTOR_RUNNER': runner,
        'V35_DETECTOR_EVENTS': str(tmp_path / f'excl_{runner}.jsonl'),
    }
    svc = hooks.Service(hooks.settings(env_excl))
    try:
        seen, stop = drive(svc, 'r_excl', tok, None, None, np.random.default_rng(1), with_values=False,
                           sleep=0.0001 if runner == 'async' else 0.0)
    finally:
        svc.close()
    assert stop is None and seen == len(tok)
    assert not [e for e in events(svc) if e['event'] in ('alert', 'rare_alert', 'error')]

    # Opt-out V35_DETECTOR_TOOL_CALL=off -> BOTH repetition intercept and rare_chars bad_utf8 alert fire!
    env_off = dict(env_excl, V35_DETECTOR_TOOL_CALL='off', V35_DETECTOR_EVENTS=str(tmp_path / f'off_{runner}.jsonl'))
    svc_off = hooks.Service(hooks.settings(env_off))
    try:
        _, stop_off = drive(svc_off, 'r_off', tok, None, None, np.random.default_rng(1), with_values=False,
                            sleep=0.0002 if runner == 'async' else 0.0)
    finally:
        svc_off.close()
    ev_off = events(svc_off)
    assert stop_off is not None and stop_off.startswith(hooks.STOP_PREFIX)
    assert any(e['event'] == 'alert' for e in ev_off)
    assert any(e['event'] == 'rare_alert' and e['reason'] == 'bad_utf8' for e in ev_off)


def test_bounded_state_across_many_tool_calls():
    det_rep = RepDetector('qwen3', CONFIG_M4_1)
    det_rare = RareDetector('qwen3', CONFIG_RC2T)
    st_rep = det_rep.start()
    st_rare = det_rare.start()
    rng = np.random.default_rng(9)
    for _ in range(20):
        seg = rng.integers(10, 50000, 300)
        tool = np.concatenate([[151657], rng.integers(10, 50000, 200), [151658]])
        chunk = np.concatenate([seg, tool])
        det_rep.feed(st_rep, chunk)
        det_rare.feed(st_rare, chunk)
    assert state_bytes(st_rep) <= 256 * 1024
    assert state_bytes(st_rare) < 8 * 1024


@pytest.mark.parametrize('runner', ['sync', 'async'])
@pytest.mark.parametrize('dsml_seq,expected_text', [
    ((128825, 8701, 4941, 12548), '｜DSML｜function_calls'),
    ((128825, 72461, 4941, 12548), '｜DSML｜tool_calls'),
    ((128825, 10699), '｜DSML｜ calls'),
])
def test_dsv4_official_v4_and_v41_write_tool_calls_not_intercepted(tmp_path, runner, dsml_seq, expected_text):
    """Regression: legal DeepSeek V3.2 / V4 (<｜DSML｜tool_calls>) / V4.1 (<｜DSML｜ calls>) Write calls containing
    a >=500-char repeated divider must NOT be intercepted with run>=500 by default serving hooks."""
    T_rare = load_rare('dsv4')
    T_rep = load_rep('dsv4')
    decoded = b''.join(T_rare['bytes'][i] for i in dsml_seq).decode('utf-8')
    assert decoded == expected_text

    lt_id = T_rare['bytes'].index(b'<')
    lt_slash_id = T_rare['bytes'].index(b'</')
    gt_nl_id = T_rare['bytes'].index(b'>\n')
    ids = np.arange(len(T_rep['norm']))
    run_tok = int(ids[(T_rep['run_char'] == ord('=')) & (T_rep['chars'] >= 8)][0])
    n_run = 600 // int(T_rep['chars'][run_tok]) + 2
    assert n_run * int(T_rep['chars'][run_tok]) >= 500

    # <｜DSML｜{tag}>\n ... 600x '=' ... </｜DSML｜{tag}>\n
    stream_tok = np.concatenate([
        np.array([lt_id, *dsml_seq, gt_nl_id], np.int64),
        np.repeat(run_tok, n_run),
        np.array([lt_slash_id, *dsml_seq, gt_nl_id], np.int64),
    ])

    env_default = {
        'V35_DETECTOR': 'intercept',
        'V35_DETECTOR_TABLE': 'dsv4',
        'V35_DETECTOR_RUNNER': runner,
        'V35_DETECTOR_EVENTS': str(tmp_path / f'dsv4_{runner}_{dsml_seq[1]}.jsonl'),
    }
    svc = hooks.Service(hooks.settings(env_default))
    try:
        seen, stop = drive(svc, 'write_ok', stream_tok, None, None, np.random.default_rng(7), with_values=False,
                           sleep=0.0001 if runner == 'async' else 0.0)
    finally:
        svc.close()
    assert stop is None and seen == len(stream_tok)
    assert not [e for e in events(svc) if e['event'] in ('alert', 'error')]

    # With V35_DETECTOR_TOOL_CALL=off, the same Write payload fires run>=500.
    env_off = dict(env_default, V35_DETECTOR_TOOL_CALL='off',
                   V35_DETECTOR_EVENTS=str(tmp_path / f'dsv4_off_{runner}_{dsml_seq[1]}.jsonl'))
    svc_off = hooks.Service(hooks.settings(env_off))
    try:
        _, stop_off = drive(svc_off, 'write_off', stream_tok, None, None, np.random.default_rng(7), with_values=False,
                            sleep=0.0002 if runner == 'async' else 0.0)
    finally:
        svc_off.close()
    ev_off = events(svc_off)
    assert any(e['event'] == 'alert' and e['reason'] == 'run>=500' for e in ev_off)
    if runner == 'sync' and stop_off is not None:
        assert stop_off == hooks.STOP_PREFIX + 'run>=500'

