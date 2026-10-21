#!/usr/bin/env python3
"""Per-tokenizer lookup tables (built once offline; online the detector only indexes them by token id).

Every token is decoded on its own from its byte-level BPE bytes (all three tokenizers are GPT-2 byte-level BPE). Columns:
  norm      int32  normalised id: the token's characters lower-cased, keeping str.isalnum ones only (digits keep their
                   value: '12' == ' 12'); a token that becomes empty -> SEP (-1: whitespace / punctuation, never enters the
                   normalised stream); a partial UTF-8 byte sequence or a special token -> its own opaque id
  letter    bool   the normalised form has a letter (any script); partial-byte tokens count as letters
  chars     int16  decoded characters (partial bytes: 1)
  ws        int16  whitespace characters
  run_char  int32  code point if the token is one non-whitespace character repeated (e.g. '----'), else -1
  marker    bool   the normalised form is a circling / hedging marker of reasoning (MARKERS: wait, hmm, alternatively, ...)
  send      bool   the decoded text holds a sentence end: . ? ! 。 ？ ！ or a newline
Same construction as the previous repo's severe_detector/vocab.py, except digits are not merged into one NUM id.

Shipped tables: tables/{qwen3,glm,dsv4}.npz (byte copies of the evaluated ones, MANIFEST.json). A new tokenizer:
  python3 -m v35_detector.vocab --tokenizer /path/to/tokenizer.json --out my_model.npz
then RepDetector('/path/to/my_model.npz', config, think_end=<'</think>' id, or None in chat mode>).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SEP = -1
TABLES = os.path.join(HERE, 'tables')
# '</think>' token id of the corpus's generation mode: the answer starts there. DeepSeek ran without a
# reasoning block (chat mode): its '</think>' tokens are answer text, not a boundary -> None.
THINK_END = {'qwen3': 151668, 'glm': 154842, 'dsv4': None}
# words a no-progress reasoning loop keeps returning to (normalised form: lower-case, alphanumeric only); chosen before
# calibration from the v3.5 rule's description of such loops ("Hmm ...", "Wait, ...") and their Chinese counterparts
MARKERS = {'wait', 'hmm', 'alternatively', 'actually', 'maybe', 'perhaps', 'but', 'however', 'reconsider', 'again', 'no',
           'ok', 'okay', 'let', '不过', '或者', '等等', '嗯', '但是', '可能', '再', '重新', '还是', '对吗'}


def _byte_decoder():
    """inverse of GPT-2 bytes_to_unicode (byte-level BPE alphabet -> byte)"""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def build(tokenizer_json):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokenizer_json)
    V = tk.get_vocab_size()
    special = {a['id'] for a in json.load(open(tokenizer_json)).get('added_tokens', []) if a.get('special')}
    bd = _byte_decoder()
    norm = np.full(V, SEP, np.int32)
    letter = np.zeros(V, bool)
    chars = np.zeros(V, np.int16)
    ws = np.zeros(V, np.int16)
    run_char = np.full(V, -1, np.int32)
    marker = np.zeros(V, bool)
    send = np.zeros(V, bool)
    ids = {}                                            # normalised string -> id (0 unused)
    for i in range(V):
        piece = tk.id_to_token(i)
        if piece is None:
            continue
        if i in special:
            norm[i] = ids.setdefault(('special', i), len(ids) + 1)
            chars[i] = len(piece)
            continue
        try:
            s = bytes(bd[c] for c in piece).decode('utf-8')
        except (KeyError, UnicodeDecodeError):
            norm[i] = ids.setdefault(('bytes', piece), len(ids) + 1)   # partial UTF-8: opaque, same bytes = same id
            letter[i] = True
            chars[i] = 1
            continue
        chars[i] = min(len(s), 32767)
        send[i] = any(c in '.?!。？！\n' for c in s)
        ws[i] = min(sum(c.isspace() for c in s), 32767)
        if s and not s[0].isspace() and s == s[0] * len(s):
            run_char[i] = ord(s[0])
        key = ''.join(c for c in s.lower() if c.isalnum())
        if key:
            norm[i] = ids.setdefault(key, len(ids) + 1)
            letter[i] = any(c.isalpha() for c in key)
            marker[i] = key in MARKERS
    return {'norm': norm, 'letter': letter, 'chars': chars, 'ws': ws, 'run_char': run_char, 'marker': marker, 'send': send}


_CACHE = {}


def load(corpus):
    """tables of a shipped tokenizer (qwen3 | glm | dsv4) or of a table file built by main() (a path ending in .npz)"""
    if corpus not in _CACHE:
        path = corpus if corpus.endswith('.npz') else os.path.join(TABLES, f'{corpus}.npz')
        z = np.load(path)
        _CACHE[corpus] = {k: z[k] for k in z.files}
    return _CACHE[corpus]


def main():
    ap = argparse.ArgumentParser(description='build the lookup table of one byte-level BPE tokenizer')
    ap.add_argument('--tokenizer', required=True, help='tokenizer.json of the served model')
    ap.add_argument('--out', required=True, help='output .npz')
    a = ap.parse_args()
    t = build(a.tokenizer)
    np.savez_compressed(a.out, **t)
    n = t['norm']
    print(f'{a.out}: vocab {n.size}, separators {int((n == SEP).sum())}, distinct normalised ids {len(np.unique(n[n > 0]))}, '
          f"run_char tokens {int((t['run_char'] >= 0).sum())}", file=sys.stderr)


if __name__ == '__main__':
    main()
