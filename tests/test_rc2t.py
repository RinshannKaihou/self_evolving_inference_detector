"""Tests for RC2T: universal Unicode-prior + stdlib codec token-only detector across tokenizer families."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from v35_detector import CONFIG_RC2T, RareDetector
from v35_detector.rare import decide, record_features
from v35_detector.rare_vocab import CI, load as load_rare
from v35_detector.rc2t_vocab import build as build_rc2t_bundle
from v35_detector.serving import hooks
from v35_detector.vocab import THINK_END, _byte_decoder

_BYTE_ENC = {b: ch for ch, b in _byte_decoder().items()}


def _bl(text):
    return ''.join(_BYTE_ENC[b] for b in text.encode('utf-8'))


def _pool(T, cls, lead):
    return np.flatnonzero((T['rcls'] == CI[cls]) & (T['lead'] == lead))


def _synth_tokens(corpus, n, seed, han_share=0.2):
    T = load_rare(corpus)
    rng = np.random.default_rng(seed)
    pools = [
        _pool(T, 'latin', True)[:3000],
        _pool(T, 'han', False)[:3000],
        _pool(T, 'space_punct', False)[:200],
        _pool(T, 'digit', False)[:50],
    ]
    kind = rng.choice(4, n, p=[0.6 - han_share / 2, han_share, 0.3 - han_share / 2, 0.1])
    tok = np.array([rng.choice(pools[k]) for k in kind], np.int64)
    if THINK_END[corpus] is not None and n > 200:
        tok[200] = THINK_END[corpus]
    return tok


def _stream(det, tok, rng):
    st = det.start()
    i = 0
    while i < len(tok):
        k = int(rng.choice([1, 3, 17, 64, 129]))
        res = det.feed(st, tok[i:i + k])
        if res is not None:
            break
        i += k
    res = det.finish(st)
    mon = det.monitor(st)
    return (res.fired_at, res.reason), (mon.fired_at, mon.reason)


def _bytes_size(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(_bytes_size(k, seen) + _bytes_size(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(_bytes_size(x, seen) for x in obj)
    if hasattr(obj, '__dict__'):
        return sys.getsizeof(obj) + _bytes_size(obj.__dict__, seen)
    return sys.getsizeof(obj)


def test_golden_alerts_and_gb_scores():
    fixture = Path(__file__).parent / 'fixtures' / 'rc2t_golden.json'
    cases = json.loads(fixture.read_text(encoding='utf-8'))['cases']
    rng = np.random.default_rng(42)
    for case in cases:
        corpus = case['corpus']
        tok = np.asarray(case['tokens'], np.int64)
        det = RareDetector(corpus, CONFIG_RC2T)
        assert not det.ref['neutral']
        assert decide(tok, None, None, corpus, CONFIG_RC2T, tier='intercept') == tuple(case['expected']['intercept'])
        assert decide(tok, None, None, corpus, CONFIG_RC2T, tier='monitor') == tuple(case['expected']['monitor'])
        strict, mon = _stream(det, tok, rng)
        assert strict == tuple(case['expected']['intercept'])
        assert mon == tuple(case['expected']['monitor'])
        F = record_features(det, tok)
        for pos, expected in case['gb_scores'].items():
            actual = F['gb'].values(np.array([int(pos)]))[0]
            assert np.isclose(actual, expected, rtol=0.0, atol=1e-12), (case['name'], pos, actual, expected)


@pytest.mark.parametrize('corpus', ['qwen3', 'glm', 'dsv4'])
def test_non_neutral_priors_and_stream_equivalence(corpus):
    det = RareDetector(corpus, CONFIG_RC2T)
    assert det.ref['neutral'] is False
    assert np.isfinite(det.ref['u']).all() and det.ref['u'].std() > 0.1
    assert np.isfinite(det.ref['cp']).all() and det.ref['cp'].std() > 0.05
    assert (det.ref['c_max'] >= 3).any()

    rng = np.random.default_rng(11)
    T = load_rare(corpus)
    word = next(int(i) for i in _pool(T, 'latin', True) if T['bytes'][i].decode().strip().isalpha() and len(T['bytes'][i]) > 4)
    rare_han = int(_pool(T, 'han', False)[0])
    fired = 0
    for seed in range(8):
        tok = _synth_tokens(corpus, 900, seed, han_share=0.05 * (seed % 3))
        if seed % 3 == 0:
            tok[500], tok[501] = word, rare_han
        strict, mon = _stream(det, tok, rng)
        assert strict == decide(tok, None, None, corpus, CONFIG_RC2T, tier='intercept')
        assert mon == decide(tok, None, None, corpus, CONFIG_RC2T, tier='monitor')
        fired += mon[0] >= 0
    assert fired > 0

    # State stays bounded over long streams
    st = det.start()
    long_tok = _synth_tokens(corpus, 10000, 99)
    for ch in np.array_split(long_tok, 80):
        det.feed(st, ch)
    assert _bytes_size(st) < 8192


def test_direct_tokenizer_json_dir_and_live_objects(tmp_path):
    tk = pytest.importorskip('tokenizers')
    alphabet = tk.pre_tokenizers.ByteLevel.alphabet()
    bl_zhong, bl_xing = _bl('中'), _bl('醒')
    pieces = sorted(alphabet) + ['Ġthe', 'Ġhello', 'Ġworld', bl_zhong, bl_xing]
    tokenizer = tk.Tokenizer(tk.models.BPE({c: i for i, c in enumerate(pieces)}, []))
    tokenizer.decoder = tk.decoders.ByteLevel()
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(['<bos>', '</think>'])

    model_dir = tmp_path / 'my_bpe_model'
    model_dir.mkdir()
    tok_json = model_dir / 'tokenizer.json'
    tokenizer.save(str(tok_json))

    think_id = tokenizer.token_to_id('</think>')
    det_json = RareDetector(str(tok_json), CONFIG_RC2T)
    det_dir = RareDetector(model_dir, CONFIG_RC2T)
    det_obj = RareDetector(tokenizer, CONFIG_RC2T)

    # Mock PreTrainedTokenizerFast wrapper
    class FastWrapper:
        backend_tokenizer = tokenizer
        all_special_ids = [tokenizer.token_to_id('<bos>'), think_id]

    det_fast = RareDetector(FastWrapper(), CONFIG_RC2T)

    # Exported .npz bundle
    npz_path = tmp_path / 'exported_rc2t.npz'
    np.savez_compressed(npz_path, **build_rc2t_bundle(str(tok_json), include_m4=True))
    det_npz = RareDetector(str(npz_path), CONFIG_RC2T)

    for det in (det_json, det_dir, det_obj, det_fast, det_npz):
        assert det.think_end == think_id
        for k in ('u', 'u_mean', 'cp', 'cn', 'c_max', 'c_mean', 'c_len', 'c_rare', 'c_nscripts'):
            np.testing.assert_array_equal(det.ref[k], det_json.ref[k])

    w_hello = tokenizer.token_to_id('Ġhello')
    w_the = tokenizer.token_to_id('Ġthe')
    dot_id = tokenizer.token_to_id('.')
    rare_cjk = tokenizer.token_to_id(bl_xing)
    seq = np.array([w_the, w_hello] * 80 + [think_id] + [w_the, w_hello] * 80 + [rare_cjk, dot_id], np.int64)
    F_json = record_features(det_json, seq)
    F_obj = record_features(det_obj, seq)
    F_npz = record_features(det_npz, seq)
    idx = np.array([len(seq) - 2])
    np.testing.assert_array_equal(F_json['gb'].values(idx), F_obj['gb'].values(idx))
    np.testing.assert_array_equal(F_json['gb'].values(idx), F_npz['gb'].values(idx))
    assert F_json['gb'].values(idx)[0] > 3.85
    assert decide(seq, None, None, tokenizer, CONFIG_RC2T)[1] == 'gb>3.85 & letter'


def test_sentencepiece_byte_fallback_and_wordpiece_and_tiktoken(tmp_path):
    # 1. SentencePiece Unigram + Metaspace + ByteFallback tokenizer.json (no `tokenizers` library needed)
    spm_vocab = [
        ['<unk>', 0.0],
        ['<s>', 0.0],
        ['</s>', 0.0],
        ['\u2581the', -1.0],
        ['\u2581answer', -1.2],
        ['\u2581is', -1.1],
        ['.', -1.5],
        ['醒', -9.0],
    ] + [[f'<0x{b:02X}>', -6.0] for b in range(256)]
    spm_spec = {
        'model': {'type': 'Unigram', 'vocab': spm_vocab, 'unk_id': 0},
        'added_tokens': [
            {'id': 0, 'content': '<unk>', 'special': True},
            {'id': 1, 'content': '<s>', 'special': True},
            {'id': 2, 'content': '</s>', 'special': True},
            {'id': len(spm_vocab), 'content': '</think>', 'special': True},
        ],
        'decoder': {
            'type': 'Sequence',
            'decoders': [
                {'type': 'Replace', 'pattern': {'String': '\u2581'}, 'content': ' '},
                {'type': 'ByteFallback'},
                {'type': 'Fuse'},
            ],
        },
    }
    spm_path = tmp_path / 'spm_tokenizer.json'
    spm_path.write_text(json.dumps(spm_spec), encoding='utf-8')
    det_spm = RareDetector(str(spm_path), CONFIG_RC2T)
    assert det_spm.think_end == len(spm_vocab)

    # Multi-byte UTF-8 via <0xE4><0xB8><0xAD> ('中') across single-token chunks must NOT trigger bad_utf8
    base_offset = 8
    zhong_bytes = [base_offset + b for b in '中'.encode('utf-8')]
    st = det_spm.start()
    for tid in [3, 4, 5] + zhong_bytes + [6]:
        det_spm.feed(st, [tid])
    assert not det_spm.finish(st).is_ill

    # Lone <0x80> continuation byte MUST trigger bad_utf8
    lone_80 = base_offset + 0x80
    assert decide([3, lone_80, 4], None, None, str(spm_path), CONFIG_RC2T) == (3, 'bad_utf8')

    # Glitch CJK after Latin in SPM triggers strict GB rule
    seq_spm = [3, 4, 5] * 60 + [len(spm_vocab)] + [3, 4, 5] * 60 + [7, 6]
    assert decide(seq_spm, None, None, str(spm_path), CONFIG_RC2T)[1] == 'gb>3.85 & letter'

    # 2. WordPiece tokenizer.json
    wp_spec = {
        'model': {'type': 'WordPiece', 'vocab': {'[PAD]': 0, '[CLS]': 1, 'hello': 2, 'world': 3, '##中': 4}},
        'added_tokens': [{'id': 0, 'content': '[PAD]', 'special': True}, {'id': 1, 'content': '[CLS]', 'special': True}],
        'decoder': {'type': 'WordPiece', 'prefix': '##', 'cleanup': True},
    }
    wp_path = tmp_path / 'wp_tokenizer.json'
    wp_path.write_text(json.dumps(wp_spec), encoding='utf-8')
    det_wp = RareDetector(str(wp_path), CONFIG_RC2T)
    assert det_wp.think_end is None
    seq_wp = [2, 3] * 60 + [4, 3]
    assert decide(seq_wp, None, None, str(wp_path), CONFIG_RC2T, tier='monitor')[1] == 'fs<0.01 & fsw<0.05 & caseless_after_latin'

    # 3. Tiktoken-style live object
    class MockTiktoken:
        mergeable_ranks = {b' hello': 0, b' world': 1, b'\x80': 2, '龘'.encode('utf-8'): 3}
        _special_tokens = {'<|endoftext|>': 4, '</think>': 5}

    tk_obj = MockTiktoken()
    det_tk = RareDetector(tk_obj, CONFIG_RC2T)
    assert det_tk.think_end == 5
    assert decide([0, 2, 1], None, None, tk_obj, CONFIG_RC2T) == (3, 'bad_utf8')


@pytest.mark.parametrize('runner', ['sync', 'async'])
def test_serving_hooks_with_rc2t_tokenizer_json_and_bundle(tmp_path, runner):
    from test_serving import drive, events
    tk = pytest.importorskip('tokenizers')
    alphabet = tk.pre_tokenizers.ByteLevel.alphabet()
    bl_xing = _bl('醒')
    pieces = sorted(alphabet) + ['Ġhello', 'Ġworld', bl_xing]
    tokenizer = tk.Tokenizer(tk.models.BPE({c: i for i, c in enumerate(pieces)}, []))
    tokenizer.decoder = tk.decoders.ByteLevel()
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(['</think>'])
    tok_json = tmp_path / 'tokenizer.json'
    tokenizer.save(str(tok_json))

    # Direct tokenizer.json serving when m4 is off
    s = hooks.settings({
        'V35_RARE': 'shadow',
        'V35_RARE_CONFIG': 'rc2t',
        'V35_DETECTOR': 'off',
        'V35_DETECTOR_TABLE': str(tok_json),
        'V35_DETECTOR_THINK_END': 'auto',
        'V35_DETECTOR_RUNNER': runner,
        'V35_DETECTOR_EVENTS': str(tmp_path / f'events_{runner}.jsonl'),
    })
    assert s['think_end'] == tokenizer.token_to_id('</think>')
    assert s['rare_neutral'] is False
    svc = hooks.Service(s)
    w_h = tokenizer.token_to_id('Ġhello')
    w_w = tokenizer.token_to_id('Ġworld')
    dot_id = tokenizer.token_to_id('.')
    rare_id = tokenizer.token_to_id(bl_xing)
    end_id = tokenizer.token_to_id('</think>')
    stream_ids = np.array([w_h, w_w] * 80 + [end_id] + [w_h, w_w] * 80 + [rare_id, dot_id], np.int64)
    try:
        drive(svc, 'req1', stream_ids, None, None, np.random.default_rng(3), with_values=False)
    finally:
        svc.close()
    alerts = [e for e in events(svc) if e['event'] == 'rare_alert']
    assert any(a['tier'] == 'strict' and a['reason'] == 'gb>3.85 & letter' and 'ref' not in a for a in alerts)
