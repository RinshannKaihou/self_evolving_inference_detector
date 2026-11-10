"""External vocabulary export and online token-only integration."""
import json

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
    tokenizer = tk.Tokenizer(tk.models.BPE({c: i for i, c in enumerate(sorted(alphabet))}, []))
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
    with open('tests/fixtures/rc1t_golden.json') as f:
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
