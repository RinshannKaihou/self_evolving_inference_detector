"""Build an external RC1T bundle for a GPT-2 byte-level BPE tokenizer.

The frozen GB is reused; absent reference counts select neutral zeros. This
exports features, not a newly trained or evaluated detector.
"""
import argparse
import json
import os

import numpy as np

from . import rare_vocab, vocab
from .rare_ref import charcat


def build(tokenizer_json, think_end):
    with open(tokenizer_json) as f:
        spec = json.load(f)
    # Reject other decoder semantics rather than silently treating pieces as bytes.
    if (spec.get('model', {}).get('type') != 'BPE'
            or spec.get('decoder', {}).get('type') != 'ByteLevel'):
        raise ValueError('RC1T export supports BPE with a ByteLevel decoder only')
    for token in spec.get('added_tokens', []):
        if not token.get('special') and (token.get('lstrip') or token.get('rstrip')):
            raise ValueError('added tokens with whitespace stripping are unsupported')
    tables = rare_vocab.build(tokenizer_json)
    V = len(tables['rcls'])
    if think_end is not None and not 0 <= think_end < V:
        raise ValueError('think_end must be a vocabulary token id or None')
    bd = vocab._byte_decoder()
    special = {a['id'] for a in spec.get('added_tokens', []) if a.get('special')}
    # Added tokens are literal strings, even when their characters belong to the
    # GPT-2 byte alphabet. Ordinary vocabulary pieces must use that alphabet.
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokenizer_json)
    added = {a['id']: a['content'] for a in spec.get('added_tokens', [])}
    chunks = []
    for i in range(V):
        piece = tk.id_to_token(i)
        if piece is None:
            raise ValueError(f'vocabulary has a hole at token {i}')
        b = added[i].encode('utf-8') if i in added else bytes(bd[c] for c in piece)
        chunks.append(b)
        s = b.decode('utf-8', 'ignore')
        tables['lead'][i] = b[:1] in (b' ', b'\n', b'\t', b'\r')
        tables['u8'][i] = b.decode('utf-8', 'replace') != s
        tables['rcls'][i] = (rare_vocab.CI['special'] if i in special else
                             rare_vocab.CI['partial'] if tables['u8'][i] else rare_vocab.CI[rare_vocab.classify(s)])
        tables['first'][i], tables['last'][i] = rare_vocab.kind(s[:1]), rare_vocab.kind(s[-1:])
    tables.update(boff=np.r_[0, np.cumsum([len(b) for b in chunks])].astype(np.int64),
                  bflat=np.frombuffer(b''.join(chunks), np.uint8),
                  fcat=np.array([charcat(b.decode('utf-8', 'ignore')[:1]) for b in chunks], np.int8),
                  lcat=np.array([charcat(b.decode('utf-8', 'ignore')[-1:]) for b in chunks], np.int8),
                  chat=np.array(think_end is None), think_end=np.array(-1 if think_end is None else think_end))
    # Include the existing repetition columns for the shared online table option.
    tables.update(vocab.build(tokenizer_json))
    with np.load(os.path.join(vocab.TABLES, 'rc1t_ctx_qwen3.npz')) as z:
        tables.update({k: z[k] for k in z.files if k.startswith('gb_')})
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--think-end', required=True, help='none for chat, otherwise the reasoning-end token id')
    a = ap.parse_args()
    think_end = None if a.think_end == 'none' else int(a.think_end)
    np.savez_compressed(a.out, **build(a.tokenizer, think_end))
    print(a.out)


if __name__ == '__main__':
    main()
