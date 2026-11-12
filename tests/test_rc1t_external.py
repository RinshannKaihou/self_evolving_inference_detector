"""External vocabulary export and online token-only integration."""
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from v35_detector import CONFIG_RC1T, RareDetector
from v35_detector.rc1t_vocab import build
from v35_detector.vocab import TABLES
from v35_detector.serving import hooks


@pytest.fixture
def exported(tmp_path):
    tk = pytest.importorskip('tokenizers')
    # All byte values include partial UTF-8 pieces and allow arbitrary scripts.
    alphabet = tk.pre_tokenizers.ByteLevel.alphabet()
    pieces = sorted(alphabet) + ['he', 'hel', 'hell', 'hello']
    tokenizer = tk.Tokenizer(tk.models.BPE({c: i for i, c in enumerate(pieces)},
                                           [('h', 'e'), ('he', 'l'), ('hel', 'l'), ('hell', 'o')]))
    tokenizer.decoder = tk.decoders.ByteLevel()
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(['<think-end>'])
    source = tmp_path / 'tokenizer.json'
    tokenizer.save(str(source))
    bundle = tmp_path / 'rc1t.npz'
    np.savez_compressed(bundle, **build(str(source), None))
    return tokenizer, source, str(bundle)


def test_export_and_gb_identity(exported):
    tokenizer, _, path = exported
    with np.load(path) as z, np.load(TABLES + '/rc1t_ctx_qwen3.npz') as original:
        for key in original.files:
            if key.startswith('gb_'):
                np.testing.assert_array_equal(z[key], original[key])
        ids = tokenizer.encode('hello 世界').ids
        raw = b''.join(z['bflat'][z['boff'][i]:z['boff'][i+1]].tobytes() for i in ids)
        assert raw.decode() == tokenizer.decode(ids)
        end = tokenizer.token_to_id('<think-end>')
        assert z['bflat'][z['boff'][end]:z['boff'][end+1]].tobytes() == b'<think-end>'
    det = RareDetector(path, CONFIG_RC1T, think_end=None)
    assert det.ref['neutral'] and not det.ref['u'].any()
    # Partial-byte tokens concatenate into a valid Han character without alerts.
    ids = tokenizer.encode('hello 世界').ids
    st = det.start()
    for i in ids:
        det.feed(st, [i])
    det.finish(st)
    assert st.result is None
    # A lone continuation byte must fire regardless of chunk boundaries.
    byte = det.T['bytes'].index(b'\x80')
    for chunks in ([[byte]], [[byte], ids[:1]]):
        st = det.start()
        for chunk in chunks:
            det.feed(st, chunk)
        result = det.finish(st)
        assert result.reason == 'bad_utf8'


@pytest.mark.parametrize('runner', ['sync', 'async'])
@pytest.mark.parametrize('mode', ['off', 'shadow'])
def test_online_external(exported, tmp_path, runner, mode):
    from test_serving import drive, events
    _, _, path = exported
    settings = hooks.settings({'V35_RARE': 'shadow', 'V35_RARE_CONFIG': 'rc1t',
                               'V35_DETECTOR': mode, 'V35_DETECTOR_TABLE': path,
                               'V35_DETECTOR_THINK_END': 'none', 'V35_DETECTOR_RUNNER': runner,
                               'V35_DETECTOR_EVENTS': str(tmp_path / 'events.jsonl')})
    assert settings['rare_neutral']
    svc = hooks.Service(settings)
    byte = RareDetector(path, CONFIG_RC1T, think_end=None).T['bytes'].index(b'\x80')
    try:
        drive(svc, 'external', np.array([byte, byte]), None, None, np.random.default_rng(2), with_values=False)
    finally:
        svc.close()
    alert = next(e for e in events(svc) if e['event'] == 'rare_alert')
    assert alert['ref'] == 'neutral' and alert['reason'] == 'bad_utf8'


def test_boundary_and_schema(exported, tmp_path):
    _, source, path = exported
    with pytest.raises(ValueError, match='think_end'):
        RareDetector(path, CONFIG_RC1T)
    with pytest.raises(ValueError, match='boundary'):
        RareDetector(path, CONFIG_RC1T, think_end=1)
    with np.load(path) as z:
        data = dict(z)
    for change, error in (({'fcat': np.zeros(1)}, 'fcat'),
                           ({'u': np.zeros(len(data['rcls']))}, 'supplied together')):
        bad = str(tmp_path / (error.split()[0] + '.npz'))
        np.savez(bad, **dict(data, **change))
        with pytest.raises(ValueError, match=error):
            RareDetector(bad, CONFIG_RC1T, think_end=None)
    with open(source) as f:
        spec = json.load(f)
    spec['decoder'] = {'type': 'WordPiece', 'prefix': '##', 'cleanup': True}
    source.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='ByteLevel'):
        build(str(source), None)


def test_frozen_bundle_external_equivalence(tmp_path):
    from v35_detector.rare import record_features
    with np.load(TABLES + '/rc1t_ctx_qwen3.npz') as z:
        path = str(tmp_path / 'frozen.npz')
        np.savez_compressed(path, **dict(z))
    original = RareDetector('qwen3', CONFIG_RC1T)
    external = RareDetector(path, CONFIG_RC1T, think_end=original.think_end)
    with (Path(__file__).parent / 'fixtures' / 'rc1t_golden.json').open() as f:
        cases = json.load(f)['cases']
    for case in cases:
        if case['corpus'] != 'qwen3':
            continue
        tok = np.asarray(case['tokens'], np.int64)
        a, b = record_features(original, tok), record_features(external, tok)
        for pos in case['gb_scores']:
            idx = np.array([int(pos)])
            np.testing.assert_array_equal(a['gb'].values(idx), b['gb'].values(idx))
        st = external.start()
        for chunk in np.array_split(tok, 7):
            external.feed(st, chunk)
        external.finish(st)
        for result, tier in ((st.result, 'intercept'), (st.monitor, 'monitor')):
            actual = (result.fired_at, result.reason) if result else (-1, '')
            assert actual == tuple(case['expected'][tier])
    assert not external.ref['neutral']


def test_added_tokens_use_runtime_ids_and_bytelevel_decoder(exported):
    tk = pytest.importorskip('tokenizers')
    tokenizer, source, _ = exported
    tokenizer.add_tokens([tk.AddedToken('word space', rstrip=True),
                          tk.AddedToken('Ġtool', lstrip=True), tk.AddedToken('工具'),
                          tk.AddedToken('工具Ġ'), tk.AddedToken('word spaceĠ')])
    tokenizer.save(str(source))
    spec = json.loads(source.read_text())
    for token in spec['added_tokens']:
        token['id'] += 10000  # tokenizers assigns contiguous ids on load
    source.write_text(json.dumps(spec))
    runtime = tk.Tokenizer.from_file(str(source))
    table = build(str(source), None)
    for token in runtime.get_added_tokens_decoder().values():
        i = runtime.token_to_id(token.content)
        raw = table['bflat'][table['boff'][i]:table['boff'][i+1]].tobytes()
        assert raw.decode('utf-8', 'replace') == runtime.decode([i], skip_special_tokens=False)
        if token.special:
            from v35_detector.rare_vocab import CI
            assert table['rcls'][i] == CI['special']
            assert table['norm'][i] >= 0  # m4 also uses the actual special id
    assert str(table['tokenizer_sha256']) == hashlib.sha256(source.read_bytes()).hexdigest()
    assert int(table['vocab_size']) == runtime.get_vocab_size()
    assert str(table['unicode_version']) == unicodedata.unidata_version


@pytest.mark.parametrize('missing', ['rcls', 'boff', 'fcat', 'gb_depth'])
def test_missing_bundle_fields_are_actionable(exported, tmp_path, missing):
    _, _, path = exported
    with np.load(path) as z:
        table = {k: z[k] for k in z.files if k != missing}
    bad = str(tmp_path / (missing + '.npz'))
    np.savez(bad, **table)
    with pytest.raises(ValueError, match='not an RC1T bundle: missing ' + missing):
        RareDetector(bad, CONFIG_RC1T, think_end=None)


def test_non_bytelevel_piece_has_actionable_error(exported):
    _, source, _ = exported
    spec = json.loads(source.read_text())
    spec['model']['vocab']['工具'] = len(spec['model']['vocab'])
    source.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='token .*outside ByteLevel byte alphabet'):
        build(str(source), None)


@pytest.mark.parametrize('runner', ['sync', 'async'])
def test_reasoning_boundary_and_m4_alert(exported, tmp_path, runner):
    from test_serving import drive, events
    tokenizer, source, _ = exported
    end = tokenizer.token_to_id('<think-end>')
    path = str(tmp_path / 'reasoning.npz')
    np.savez_compressed(path, **build(str(source), end))
    det = RareDetector(path, CONFIG_RC1T, think_end=end)
    st = det.start()
    ids = np.array(tokenizer.encode('hello').ids + [end] + tokenizer.encode('world').ids)
    for chunk in np.array_split(ids, 3):
        det.feed(st, chunk)
    det.finish(st)
    assert st.f['answer']
    with pytest.raises(ValueError, match=f'bundle chat=False, think_end={end}; requested think_end=None'):
        RareDetector(path, CONFIG_RC1T, think_end=None)
    settings = hooks.settings({'V35_RARE': 'shadow', 'V35_RARE_CONFIG': 'rc1t',
                               'V35_DETECTOR': 'shadow', 'V35_DETECTOR_TABLE': path,
                               'V35_DETECTOR_THINK_END': str(end), 'V35_DETECTOR_RUNNER': runner,
                               'V35_DETECTOR_EVENTS': str(tmp_path / 'reasoning-events.jsonl')})
    svc = hooks.Service(settings)
    # ASCII repetition in reasoning should trigger m4, independently of RC1T.
    tok = np.array(tokenizer.encode('hello ' * 700).ids + [end] + tokenizer.encode('world').ids)
    try:
        drive(svc, 'repeat', tok, None, None, np.random.default_rng(2), with_values=False)
    finally:
        svc.close()
    assert any(e['event'] == 'alert' and e['request_id'] == 'repeat' for e in events(svc))
