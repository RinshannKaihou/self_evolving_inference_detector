"""Tokenizer-agnostic vocabulary parser and universal Unicode-prior mapper for RC2T.

Supports out-of-the-box initialization from:
  1. Shipped corpus names ('qwen3', 'glm', 'dsv4')
  2. Exported .npz vocabulary bundles (built by rare_vocab, rc1t_vocab, or rc2t_vocab)
  3. HuggingFace fast tokenizer files ('/path/to/tokenizer.json' — ByteLevel BPE,
     SentencePiece Metaspace / ByteFallback / Unigram, WordPiece; pure Python, no Rust dep)
  4. Model directories ('/path/to/model_dir' containing tokenizer.json or loadable via AutoTokenizer)
  5. In-memory Tokenizer objects (tokenizers.Tokenizer, transformers.PreTrainedTokenizerBase,
     tiktoken.Encoding, sentencepiece.SentencePieceProcessor, or get_vocab() duck-typed objects)
"""
import argparse
import hashlib
import json
import os
import unicodedata as U
import weakref

import numpy as np

from . import vocab
from .rare_ref import K, charcat
from .rare_vocab import CI, classify, kind, load as load_rare, script
from .vocab import TABLES, THINK_END, _byte_decoder

THINK_MARKERS = ('</think>', '<|end_of_thought|>', '[/THINK]', '<|thought_end|>')
COMMON_CJK_PUNCT = set('。，、；：？！“”‘’（）【】《》—…·「」『』〈〉')

_BASE_FIELDS = {'rcls', 'first', 'last', 'lead', 'u8', 'boff', 'bflat'}
_UNI_FIELDS = {'uni_cp', 'uni_u', 'uni_prev', 'uni_next', 'uni_default_u', 'uni_default_c'}
_GB_FIELDS = {'gb_' + k for k in ('depth', 'feat', 'thr', 'cat', 'mask', 'leaf', 'base', 'inputs', 'quant')}
_PRECOMPUTED_REF = ('fcat', 'lcat', 'u', 'u_mean', 'cp', 'cn', 'c_max', 'c_mean', 'c_len', 'c_rare', 'c_nscripts')

_TIER_CACHE = {}
_MODEL_CACHE = {}
_TARGET_CACHE = {}
_REF_CACHE = {}
_OBJ_CACHE = weakref.WeakKeyDictionary()


def _char_codec_tier(ch):
    """Deterministic character rarity tier (0..4) using Python stdlib codecs & Unicode blocks."""
    o = ord(ch)
    if o < 128:
        return 0 if (ch >= ' ' or ch in '\n\r\t') else 4
    if o == 0xFFFD:
        return 4
    cat = U.category(ch)
    if cat in ('Cc', 'Cf', 'Co', 'Cs', 'Cn'):
        return 4
    s = script(ch)
    if s == 'han':
        try:
            b = ch.encode('gb2312')
            if 0xB0 <= b[0] <= 0xD7:
                return 1
            return 2
        except UnicodeEncodeError:
            pass
        try:
            b = ch.encode('big5')
            if 0xA4 <= b[0] <= 0xC6:
                return 1
            return 3
        except UnicodeEncodeError:
            pass
        try:
            ch.encode('gbk')
            return 3
        except UnicodeEncodeError:
            return 4
    if s == 'hangul':
        try:
            b = ch.encode('euc_kr')
            if len(b) == 2 and 0xB0 <= b[0] <= 0xC8:
                return 1
            return 2
        except UnicodeEncodeError:
            return 4
    if s == 'kana':
        if (0x3041 <= o <= 0x3096) or (0x30A1 <= o <= 0x30FA) or o in (0x30FC, 0x30FB):
            return 1
        return 3
    if s == 'latin':
        try:
            ch.encode('iso8859_1')
            return 1
        except UnicodeEncodeError:
            try:
                ch.encode('iso8859_2')
                return 2
            except UnicodeEncodeError:
                return 3
    if s == 'greek':
        if (0x0391 <= o <= 0x03A9 and o != 0x03A2) or (0x03B1 <= o <= 0x03C9):
            return 1
        return 3
    if s == 'cyrillic':
        if (0x0410 <= o <= 0x044F) or o in (0x0401, 0x0451):
            return 2
        return 4
    if s == 'arabic':
        if 0x0621 <= o <= 0x064A:
            return 2
        return 4
    if s == 'thai':
        try:
            ch.encode('tis_620')
            return 2
        except UnicodeEncodeError:
            return 4
    if s in ('hebrew', 'indic'):
        if (0x05D0 <= o <= 0x05EA) or (0x0904 <= o <= 0x0939):
            return 2
        return 3
    if ch in COMMON_CJK_PUNCT or (0x2000 <= o <= 0x206F) or (0x2190 <= o <= 0x22FF):
        return 1
    try:
        ch.encode('gb2312')
        return 1
    except UnicodeEncodeError:
        pass
    if s == 'emoji':
        return 2
    if cat.startswith('M'):
        return 3
    return 3


def char_codec_tier(ch):
    t = _TIER_CACHE.get(ch)
    if t is None:
        t = _TIER_CACHE[ch] = _char_codec_tier(ch)
    return t


def _validate_think_end(think_end, vocab_size):
    if think_end is None:
        return None
    if isinstance(think_end, (bool, np.bool_)) or not isinstance(think_end, (int, np.integer)):
        raise ValueError(f'think_end must be an integer token id or None, got {think_end!r}')
    tid = int(think_end)
    if not 0 <= tid < vocab_size:
        raise ValueError(f'think_end={tid} out of vocabulary range [0, {vocab_size})')
    return tid


def _detect_think_id(piece_to_id):
    for marker in THINK_MARKERS:
        if marker in piece_to_id:
            return int(piece_to_id[marker])
    return None


def _iter_steps(node):
    if not isinstance(node, dict):
        return
    if node.get('type') == 'Sequence':
        for sub in node.get('decoders') or node.get('pre_tokenizers') or ():
            yield from _iter_steps(sub)
    else:
        yield node


def _tables_from_token_bytes(chunks, special_ids, think_detected=None):
    V = len(chunks)
    rcls = np.zeros(V, np.int8)
    first = np.zeros(V, np.int8)
    last = np.zeros(V, np.int8)
    lead = np.zeros(V, bool)
    u8 = np.zeros(V, bool)
    fcat = np.zeros(V, np.int8)
    lcat = np.zeros(V, np.int8)

    for i, b in enumerate(chunks):
        lead[i] = b[:1] in (b' ', b'\n', b'\t', b'\r')
        try:
            s = b.decode('utf-8')
        except UnicodeDecodeError:
            u8[i] = True
            rcls[i] = CI['special'] if i in special_ids else CI['partial']
            s = b.decode('utf-8', 'ignore')
        else:
            rcls[i] = CI['special'] if i in special_ids else CI[classify(s)]
        first[i], last[i] = kind(s[:1]), kind(s[-1:])
        fcat[i], lcat[i] = charcat(s[:1]), charcat(s[-1:])

    boff = np.r_[0, np.cumsum([len(b) for b in chunks])].astype(np.int64)
    bflat = np.frombuffer(b''.join(chunks), np.uint8)
    return {
        'rcls': rcls,
        'first': first,
        'last': last,
        'lead': lead,
        'u8': u8,
        'boff': boff,
        'bflat': bflat,
        'fcat': fcat,
        'lcat': lcat,
        'bytes': list(chunks),
        'think_detected': think_detected,
    }


def parse_tokenizer_spec(spec, source_name='<tokenizer>', extra_special_ids=()):
    """Parse a HuggingFace tokenizer.json dict without requiring the `tokenizers` package."""
    model = spec.get('model') if isinstance(spec, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f'{source_name}: missing model section in tokenizer JSON')
    mtype = model.get('type')
    vocab_raw = model.get('vocab')

    if isinstance(vocab_raw, list):
        id_to_piece = {i: entry[0] for i, entry in enumerate(vocab_raw)}
        piece_to_id = {entry[0]: i for i, entry in enumerate(vocab_raw)}
    elif isinstance(vocab_raw, dict):
        id_to_piece = {int(i): p for p, i in vocab_raw.items()}
        piece_to_id = {p: int(i) for p, i in vocab_raw.items()}
    else:
        raise ValueError(f'{source_name}: unsupported model.vocab format')

    # Match HuggingFace tokenizers' exact runtime ID assignment for added_tokens:
    # tokens already in base vocab retain their base ID; new tokens get contiguous IDs.
    added_map = {}
    special_ids = set(int(x) for x in extra_special_ids)
    next_id = max(id_to_piece.keys(), default=-1) + 1
    for tok in spec.get('added_tokens', ()):
        content = tok['content']
        if content in piece_to_id:
            tid = piece_to_id[content]
        else:
            tid = next_id
            next_id += 1
            id_to_piece[tid] = content
            piece_to_id[content] = tid
        added_map[tid] = content
        if tok.get('special', False):
            special_ids.add(tid)

    V = max(id_to_piece.keys(), default=-1) + 1
    if len(id_to_piece) != V:
        raise ValueError(f'{source_name}: vocabulary has holes (size {len(id_to_piece)} vs max id {V - 1})')

    dec_steps = list(_iter_steps(spec.get('decoder')))
    dtypes = {d.get('type') for d in dec_steps}
    pre_steps = list(_iter_steps(spec.get('pre_tokenizer')))
    ptypes = {p.get('type') for p in pre_steps}

    is_bytelevel = ('ByteLevel' in dtypes) or (not dtypes and 'ByteLevel' in ptypes)
    is_spm = (
        ('Metaspace' in dtypes)
        or ('ByteFallback' in dtypes)
        or (mtype == 'Unigram')
        or any(d.get('type') == 'Replace' and d.get('pattern', {}).get('String') == '\u2581' for d in dec_steps)
    )
    has_byte_fallback = 'ByteFallback' in dtypes or (
        is_spm and '<0x00>' in piece_to_id and '<0xFF>' in piece_to_id
    )
    is_wordpiece = ('WordPiece' in dtypes) or (mtype == 'WordPiece')

    bd = _byte_decoder() if is_bytelevel else None
    wp_prefix = next((d.get('prefix', '##') for d in dec_steps if d.get('type') == 'WordPiece'), '##')
    spm_rep = next((d.get('replacement', '\u2581') for d in dec_steps if d.get('type') == 'Metaspace'), '\u2581')

    chunks = []
    for i in range(V):
        p = id_to_piece[i]
        if is_bytelevel:
            try:
                b = bytes(bd[c] for c in p)
            except KeyError as e:
                if i not in added_map:
                    raise ValueError(f'token {i} {p!r}: outside ByteLevel byte alphabet') from e
                b = added_map[i].encode('utf-8')
        elif is_spm:
            if i in special_ids:
                b = p.encode('utf-8')
            elif has_byte_fallback and len(p) == 6 and p.startswith('<0x') and p.endswith('>'):
                try:
                    b = bytes([int(p[3:5], 16)])
                except ValueError:
                    b = p.replace(spm_rep, ' ').encode('utf-8')
            elif i in added_map:
                b = p.encode('utf-8')
            else:
                b = p.replace(spm_rep, ' ').encode('utf-8')
        elif is_wordpiece:
            if i in special_ids or i in added_map:
                b = p.encode('utf-8')
            elif p.startswith(wp_prefix) and p != wp_prefix:
                b = p[len(wp_prefix):].encode('utf-8')
            else:
                b = (' ' + p).encode('utf-8') if p and p[0].isalnum() else p.encode('utf-8')
        else:
            b = p.encode('utf-8')
        chunks.append(b)

    return _tables_from_token_bytes(chunks, special_ids, think_detected=_detect_think_id(piece_to_id))


def parse_tokenizer_json(tokenizer_json_path, extra_special_ids=()):
    with open(tokenizer_json_path, 'rb') as f:
        raw = f.read()
    spec = json.loads(raw)
    out = parse_tokenizer_spec(spec, source_name=str(tokenizer_json_path), extra_special_ids=extra_special_ids)
    out['tokenizer_sha256'] = hashlib.sha256(raw).hexdigest()
    return out


def _read_dir_special_and_think(model_dir):
    extra_special = set()
    think_id = None
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
            for tid_str, meta in (cfg.get('added_tokens_decoder') or {}).items():
                tid = int(tid_str)
                if isinstance(meta, dict):
                    if meta.get('special', False):
                        extra_special.add(tid)
                    if think_id is None and meta.get('content') in THINK_MARKERS:
                        think_id = tid
        except Exception:
            pass
    return extra_special, think_id


def parse_tokenizer_object(tok_obj):
    """Extract exact RC2T token tables from an in-memory Tokenizer instance."""
    try:
        if tok_obj in _OBJ_CACHE:
            return _OBJ_CACHE[tok_obj]
    except TypeError:
        pass

    extra_special = set(int(x) for x in (getattr(tok_obj, 'all_special_ids', None) or ()))

    # 1. tokenizers.Tokenizer
    if hasattr(tok_obj, 'to_str') and callable(tok_obj.to_str) and hasattr(tok_obj, 'get_vocab_size'):
        spec = json.loads(tok_obj.to_str())
        res = parse_tokenizer_spec(spec, source_name=type(tok_obj).__name__, extra_special_ids=extra_special)
    # 2. transformers.PreTrainedTokenizerFast
    elif hasattr(tok_obj, 'backend_tokenizer') and hasattr(tok_obj.backend_tokenizer, 'to_str'):
        spec = json.loads(tok_obj.backend_tokenizer.to_str())
        res = parse_tokenizer_spec(spec, source_name=type(tok_obj).__name__, extra_special_ids=extra_special)
    # 3. tiktoken.Encoding (or tiktoken-backed HF tokenizer)
    elif hasattr(tok_obj, '_mergeable_ranks') or hasattr(tok_obj, 'mergeable_ranks'):
        ranks = getattr(tok_obj, 'mergeable_ranks', None) or getattr(tok_obj, '_mergeable_ranks')
        special_map = getattr(tok_obj, '_special_tokens', None) or {}
        id_to_bytes = {int(v): (k if isinstance(k, bytes) else str(k).encode('utf-8')) for k, v in ranks.items()}
        piece_to_id = {}
        for s_tok, tid in special_map.items():
            tid = int(tid)
            id_to_bytes[tid] = s_tok.encode('utf-8')
            extra_special.add(tid)
            piece_to_id[s_tok] = tid
        for k, tid in ranks.items():
            if isinstance(k, bytes):
                try:
                    piece_to_id[k.decode('utf-8')] = int(tid)
                except UnicodeDecodeError:
                    pass
        V = max(id_to_bytes.keys(), default=-1) + 1
        chunks = [id_to_bytes.get(i, b'') for i in range(V)]
        for i in range(V):
            if i not in id_to_bytes:
                extra_special.add(i)
        res = _tables_from_token_bytes(chunks, extra_special, think_detected=_detect_think_id(piece_to_id))
    # 4. SentencePieceProcessor or slow SentencePiece HF tokenizer
    elif hasattr(tok_obj, 'sp_model') or (hasattr(tok_obj, 'GetPieceSize') and hasattr(tok_obj, 'IdToPiece')):
        sp = getattr(tok_obj, 'sp_model', tok_obj)
        V_sp = int(sp.GetPieceSize())
        vocab_dict = tok_obj.get_vocab() if hasattr(tok_obj, 'get_vocab') else {}
        V = max([V_sp - 1] + [int(v) for v in vocab_dict.values()]) + 1
        id_to_piece = {int(v): k for k, v in vocab_dict.items()}
        for i in range(V_sp):
            id_to_piece.setdefault(i, sp.IdToPiece(i))
            if (hasattr(sp, 'IsControl') and sp.IsControl(i)) or (hasattr(sp, 'IsUnknown') and sp.IsUnknown(i)):
                extra_special.add(i)
        piece_to_id = {p: i for i, p in id_to_piece.items()}
        chunks = []
        for i in range(V):
            p = id_to_piece.get(i, '')
            if i in extra_special:
                b = p.encode('utf-8')
            elif len(p) == 6 and p.startswith('<0x') and p.endswith('>'):
                try:
                    b = bytes([int(p[3:5], 16)])
                except ValueError:
                    b = p.replace('\u2581', ' ').encode('utf-8')
            else:
                b = p.replace('\u2581', ' ').encode('utf-8')
            chunks.append(b)
        res = _tables_from_token_bytes(chunks, extra_special, think_detected=_detect_think_id(piece_to_id))
    # 5. Generic PreTrainedTokenizerBase / duck-typed get_vocab() object
    elif hasattr(tok_obj, 'get_vocab') and callable(tok_obj.get_vocab):
        vocab_dict = tok_obj.get_vocab()
        if not isinstance(vocab_dict, dict) or not vocab_dict:
            raise ValueError(f'{type(tok_obj).__name__}.get_vocab() returned an empty or invalid vocabulary')
        id_to_piece = {int(v): str(k) for k, v in vocab_dict.items()}
        piece_to_id = {str(k): int(v) for k, v in vocab_dict.items()}
        V = max(id_to_piece.keys()) + 1
        sample_pieces = list(piece_to_id.keys())
        has_spm = any('\u2581' in p for p in sample_pieces[:2000])
        has_bytelevel = (not has_spm) and ('\u0120' in ''.join(sample_pieces[:2000]))
        bd = _byte_decoder() if has_bytelevel else None
        chunks = []
        for i in range(V):
            if i not in id_to_piece:
                extra_special.add(i)
                chunks.append(b'')
                continue
            p = id_to_piece[i]
            if i in extra_special:
                b = p.encode('utf-8')
            elif has_bytelevel:
                try:
                    b = bytes(bd[c] for c in p)
                except KeyError:
                    b = p.encode('utf-8')
            elif has_spm:
                if len(p) == 6 and p.startswith('<0x') and p.endswith('>'):
                    try:
                        b = bytes([int(p[3:5], 16)])
                    except ValueError:
                        b = p.replace('\u2581', ' ').encode('utf-8')
                else:
                    b = p.replace('\u2581', ' ').encode('utf-8')
            else:
                b = p.encode('utf-8')
            chunks.append(b)
        res = _tables_from_token_bytes(chunks, extra_special, think_detected=_detect_think_id(piece_to_id))
    else:
        raise ValueError(
            f'unsupported tokenizer target {type(tok_obj).__name__}: expected shipped name, .npz path, '
            'tokenizer.json path, model directory, or Tokenizer object'
        )

    try:
        _OBJ_CACHE[tok_obj] = res
    except TypeError:
        pass
    return res


def _mtime_ns(path):
    return os.stat(path).st_mtime_ns


def _load_npz_target(path, think_end):
    abspath = os.path.abspath(path)
    mtime = _mtime_ns(abspath)
    ckey = ('npz', abspath, mtime)
    if ckey not in _TARGET_CACHE:
        with np.load(abspath) as z:
            missing = _BASE_FIELDS - set(z.files)
            if missing:
                raise ValueError(f'{path}: not a valid vocabulary bundle: missing {", ".join(sorted(missing))}')
            data = {k: z[k] for k in z.files}
        V = len(data['rcls'])
        for k in ('rcls', 'first', 'last', 'lead', 'u8'):
            if data[k].shape != (V,):
                raise ValueError(f'{path}: invalid {k} shape {data[k].shape}, expected {(V,)}')
        boff, bflat = data['boff'], data['bflat']
        if boff.shape != (V + 1,) or boff.dtype.kind not in 'iu':
            raise ValueError(f'{path}: invalid boff')
        if bflat.ndim != 1 or int(boff[0]) != 0 or int(boff[-1]) != len(bflat) or np.any(boff[1:] < boff[:-1]):
            raise ValueError(f'{path}: invalid boff / bflat offsets')
        for k in ('fcat', 'lcat'):
            if k in data:
                a = data[k]
                if a.shape != (V,) or a.dtype.kind not in 'iu' or np.any((a < 0) | (a >= K)):
                    raise ValueError(f'{path}: invalid {k}')
        if ('chat' in data) != ('think_end' in data):
            raise ValueError(f'{path}: chat and think_end must be supplied together')
        data['bytes'] = [bflat[boff[i]:boff[i + 1]].tobytes() for i in range(V)]
        _TARGET_CACHE[ckey] = data
    data = _TARGET_CACHE[ckey]
    V = len(data['rcls'])

    if 'chat' in data:
        chat = bool(data['chat'])
        te = int(data['think_end'])
        if think_end == 'auto':
            resolved = None if chat else _validate_think_end(te, V)
        else:
            resolved = _validate_think_end(think_end, V)
            if chat != (resolved is None) or (not chat and te != resolved):
                raise ValueError(
                    f'{path}: think/chat boundary does not match: bundle chat={chat}, '
                    f'think_end={te}; requested think_end={think_end!r}'
                )
    else:
        if think_end == 'auto':
            raise ValueError(
                f"{path}: think_end='auto' needs boundary metadata (chat / think_end) in the .npz bundle, "
                "or pass think_end=<id> / None explicitly"
            )
        resolved = _validate_think_end(think_end, V)

    return data, resolved, ckey


def resolve_target(target, think_end='auto'):
    """Resolve `target` into `(T, resolved_think_end, cache_key)`."""
    if isinstance(target, os.PathLike):
        target = os.fspath(target)

    if not isinstance(target, str):
        T = parse_tokenizer_object(target)
        V = len(T['rcls'])
        resolved = T['think_detected'] if think_end == 'auto' else _validate_think_end(think_end, V)
        return T, resolved, ('obj', id(T))

    if target in THINK_END:
        T = load_rare(target)
        V = len(T['rcls'])
        resolved = THINK_END[target] if think_end == 'auto' else _validate_think_end(think_end, V)
        return T, resolved, ('shipped', target)

    if target.endswith('.npz'):
        if not os.path.isfile(target):
            raise FileNotFoundError(f'{target}: vocabulary bundle not found')
        return _load_npz_target(target, think_end)

    if os.path.isdir(target):
        abspath = os.path.abspath(target)
        tok_json = os.path.join(abspath, 'tokenizer.json')
        extra_special, cfg_think = _read_dir_special_and_think(abspath)
        if os.path.isfile(tok_json):
            mtime = _mtime_ns(tok_json)
            ckey = ('dir_json', abspath, mtime)
            if ckey not in _TARGET_CACHE:
                T = parse_tokenizer_json(tok_json, extra_special_ids=extra_special)
                if T['think_detected'] is None and cfg_think is not None:
                    T['think_detected'] = cfg_think
                _TARGET_CACHE[ckey] = T
            T = _TARGET_CACHE[ckey]
            V = len(T['rcls'])
            resolved = T['think_detected'] if think_end == 'auto' else _validate_think_end(think_end, V)
            return T, resolved, ckey
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ValueError(
                f'{target}: no tokenizer.json found in directory and `transformers` is not installed'
            ) from e
        tok_obj = AutoTokenizer.from_pretrained(abspath, trust_remote_code=True, local_files_only=True)
        T = parse_tokenizer_object(tok_obj)
        if T['think_detected'] is None and cfg_think is not None:
            T['think_detected'] = cfg_think
        V = len(T['rcls'])
        resolved = T['think_detected'] if think_end == 'auto' else _validate_think_end(think_end, V)
        return T, resolved, ('dir_hf', abspath)

    if os.path.isfile(target) or target.endswith('.json'):
        if not os.path.isfile(target):
            raise FileNotFoundError(f'{target}: tokenizer file not found')
        abspath = os.path.abspath(target)
        mtime = _mtime_ns(abspath)
        ckey = ('json', abspath, mtime)
        if ckey not in _TARGET_CACHE:
            _TARGET_CACHE[ckey] = parse_tokenizer_json(abspath)
        T = _TARGET_CACHE[ckey]
        V = len(T['rcls'])
        resolved = T['think_detected'] if think_end == 'auto' else _validate_think_end(think_end, V)
        return T, resolved, ckey

    raise ValueError(
        f'unknown RC2T target {target!r}: expected one of {tuple(THINK_END)}, a .npz bundle path, '
        'a tokenizer.json path, a model directory, or a Tokenizer instance'
    )


def load_uni_model(model_path):
    """Load and validate `rc2t_model.npz` (cached by absolute path and mtime)."""
    abspath = os.path.abspath(model_path)
    mtime = _mtime_ns(abspath)
    mkey = (abspath, mtime)
    if mkey not in _MODEL_CACHE:
        with open(abspath, 'rb') as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()
        with np.load(abspath) as z:
            missing = (_UNI_FIELDS | _GB_FIELDS) - set(z.files)
            if missing:
                raise ValueError(f'{model_path}: not an RC2T model bundle: missing {", ".join(sorted(missing))}')
            spec = {k: z[k] for k in z.files}
        M = len(spec['uni_cp'])
        if spec['uni_u'].shape != (M,) or spec['uni_prev'].shape != (M, K) or spec['uni_next'].shape != (M, K):
            raise ValueError(f'{model_path}: invalid Unicode prior table shapes')
        spec['sha256'] = sha
        spec['ch_to_idx'] = {chr(int(cp_val)): idx for idx, cp_val in enumerate(spec['uni_cp'].tolist())}
        _MODEL_CACHE[mkey] = spec
    return _MODEL_CACHE[mkey]


def compute_uni_ref(T, model_spec):
    """Map universal Unicode priors and stdlib codec rarity tiers onto vocabulary `T`."""
    V = len(T['rcls'])
    if all(k in T for k in _PRECOMPUTED_REF) and str(T.get('uni_model_sha256', '')) == model_spec['sha256']:
        for k in ('u', 'u_mean', 'c_max', 'c_mean', 'c_len', 'c_rare', 'c_nscripts'):
            if T[k].shape != (V,) or not np.isfinite(T[k]).all():
                raise ValueError(f'invalid precomputed {k} in bundle')
        for k in ('cp', 'cn'):
            if T[k].shape != (V, K) or not np.isfinite(T[k]).all():
                raise ValueError(f'invalid precomputed {k} in bundle')
        fcat_i64 = T['fcat'].astype(np.int64)
        lcat_i64 = T['lcat'].astype(np.int64)
        return {
            'fcat': fcat_i64,
            'lcat': lcat_i64,
            'lcat_s': np.r_[lcat_i64, 0],
            'fcat_s': np.r_[fcat_i64, 0],
            'u': T['u'].astype(np.float32, copy=False),
            'u_mean': T['u_mean'].astype(np.float32, copy=False),
            'cp': T['cp'].astype(np.float32, copy=False),
            'cn': T['cn'].astype(np.float32, copy=False),
            'c_max': T['c_max'].astype(np.float32, copy=False),
            'c_mean': T['c_mean'].astype(np.float32, copy=False),
            'c_len': T['c_len'].astype(np.float32, copy=False),
            'c_rare': T['c_rare'].astype(np.float32, copy=False),
            'c_nscripts': T['c_nscripts'].astype(np.float32, copy=False),
            'neutral': False,
        }

    def_u = float(model_spec['uni_default_u'])
    def_c = float(model_spec['uni_default_c'])
    ch_to_idx = model_spec['ch_to_idx']
    u_arr = model_spec['uni_u']
    pv_arr = model_spec['uni_prev']
    nx_arr = model_spec['uni_next']

    has_cats = ('fcat' in T) and ('lcat' in T)
    fcat = T['fcat'].astype(np.int64) if has_cats else np.zeros(V, np.int64)
    lcat = T['lcat'].astype(np.int64) if has_cats else np.zeros(V, np.int64)

    u_tok = np.full(V, def_u, np.float32)
    u_mean_tok = np.full(V, def_u, np.float32)
    cp_tok = np.full((V, K), def_c, np.float32)
    cn_tok = np.full((V, K), def_c, np.float32)
    c_max = np.zeros(V, np.float32)
    c_mean = np.zeros(V, np.float32)
    c_len = np.zeros(V, np.float32)
    c_rare = np.zeros(V, np.float32)
    c_nscripts = np.zeros(V, np.float32)

    rcls_arr = T['rcls']
    u8_arr = T['u8']
    bytes_list = T['bytes']
    sp_code = CI['special']

    for i in range(V):
        b = bytes_list[i]
        s = b.decode('utf-8', 'ignore')
        if not has_cats:
            fcat[i] = charcat(s[:1])
            lcat[i] = charcat(s[-1:])
        if rcls_arr[i] == sp_code:
            continue
        if u8_arr[i]:
            c_max[i] = 3.5
            c_mean[i] = 3.5
            continue
        if not s:
            continue
        us = [float(u_arr[ch_to_idx[ch]]) if ch in ch_to_idx else def_u for ch in s]
        u_tok[i] = min(us)
        u_mean_tok[i] = sum(us) / len(us)
        i0 = ch_to_idx.get(s[0])
        if i0 is not None:
            cp_tok[i] = pv_arr[i0]
        i1 = ch_to_idx.get(s[-1])
        if i1 is not None:
            cn_tok[i] = nx_arr[i1]
        tiers = [char_codec_tier(ch) for ch in s]
        mx = max(tiers)
        c_max[i] = mx
        c_mean[i] = sum(tiers) / len(tiers)
        c_len[i] = len(s)
        c_rare[i] = float(mx >= 3)
        c_nscripts[i] = len({script(ch) for ch in s} - {'space_punct'})

    return {
        'fcat': fcat,
        'lcat': lcat,
        'lcat_s': np.r_[lcat, 0],
        'fcat_s': np.r_[fcat, 0],
        'u': u_tok,
        'u_mean': u_mean_tok,
        'cp': cp_tok,
        'cn': cn_tok,
        'c_max': c_max,
        'c_mean': c_mean,
        'c_len': c_len,
        'c_rare': c_rare,
        'c_nscripts': c_nscripts,
        'neutral': False,
    }


def load_uni_ref(cache_key, T, model_path):
    model_spec = load_uni_model(model_path)
    rkey = (cache_key, model_spec['sha256'])
    if cache_key is not None and rkey in _REF_CACHE:
        return _REF_CACHE[rkey]
    ref = compute_uni_ref(T, model_spec)
    if cache_key is not None:
        _REF_CACHE[rkey] = ref
    return ref


def build(target, think_end='auto', model_path=None, include_m4=False):
    """Build a self-contained precomputed .npz vocabulary bundle for fast startup (optional)."""
    if model_path is None:
        model_path = os.path.join(TABLES, 'rc2t_model.npz')
    T, resolved_te, _ = resolve_target(target, think_end)
    model_spec = load_uni_model(model_path)
    ref = compute_uni_ref(T, model_spec)
    V = len(T['rcls'])
    out = {
        'rcls': T['rcls'],
        'first': T['first'],
        'last': T['last'],
        'lead': T['lead'],
        'u8': T['u8'],
        'boff': T['boff'],
        'bflat': T['bflat'],
        'fcat': ref['fcat'].astype(np.int8),
        'lcat': ref['lcat'].astype(np.int8),
        'u': ref['u'],
        'u_mean': ref['u_mean'],
        'cp': ref['cp'],
        'cn': ref['cn'],
        'c_max': ref['c_max'],
        'c_mean': ref['c_mean'],
        'c_len': ref['c_len'],
        'c_rare': ref['c_rare'],
        'c_nscripts': ref['c_nscripts'],
        'vocab_size': np.array(V),
        'unicode_version': np.array(U.unidata_version),
        'uni_model_sha256': np.array(model_spec['sha256']),
        'chat': np.array(resolved_te is None),
        'think_end': np.array(-1 if resolved_te is None else int(resolved_te)),
    }
    if 'tokenizer_sha256' in T:
        out['tokenizer_sha256'] = np.array(T['tokenizer_sha256'])
    if include_m4:
        tok_json = target
        if isinstance(tok_json, os.PathLike):
            tok_json = os.fspath(tok_json)
        if isinstance(tok_json, str) and os.path.isdir(tok_json):
            tok_json = os.path.join(tok_json, 'tokenizer.json')
        if not (isinstance(tok_json, str) and os.path.isfile(tok_json)):
            raise ValueError('include_m4=True requires a tokenizer.json path or model directory containing tokenizer.json')
        out.update(vocab.build(tok_json))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizer', required=True, help='tokenizer.json path or model directory')
    ap.add_argument('--out', required=True, help='output .npz path')
    ap.add_argument('--think-end', default='auto', help='auto (default), none for chat, or reasoning-end token id')
    ap.add_argument('--include-m4', action='store_true', help='also export m4 repetition columns (ByteLevel BPE)')
    a = ap.parse_args()
    te = 'auto' if a.think_end == 'auto' else None if a.think_end == 'none' else int(a.think_end)
    np.savez_compressed(a.out, **build(a.tokenizer, think_end=te, include_m4=a.include_m4))
    print(a.out)


if __name__ == '__main__':
    main()
