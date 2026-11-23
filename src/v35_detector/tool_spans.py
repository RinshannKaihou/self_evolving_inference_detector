"""Streaming and whole-sequence tool-call span exclusion for RepDetector and RareDetector.

Tool-call payloads (JSON arguments, file patches, code diffs, shell commands) are a major source of
false positives for both repetition and rare_chars detectors. Excluding tool-call spans operates purely
on token IDs so it works inside vLLM's OutputProcessor before any OpenAI ToolParser runs:

  * Supports single-token markers (Qwen3 / GLM `<tool_call>`..`</tool_call>`, DeepSeek
    `<｜tool▁calls▁begin｜>`..`<｜tool▁calls▁end｜>`, Kimi-K2, Mistral `[TOOL_CALLS]`, Llama
    `<|python_tag|>`) and multi-token markers (Kimi-K3 `<|open|>tools<|sep|>`..`<|close|>tools<|sep|>`,
    DeepSeek DSML `<｜DSML｜function_calls>`..`</｜DSML｜function_calls>`).
  * Protects explicit open `<think>`..`</think>` blocks from suppressing normal reasoning text when a
    model quotes a tool-call marker inside its thinking block.
  * Never exits an active tool-call span on `</think>` (models calling Write/Edit tools frequently emit
    literal `</think>` strings inside tool arguments).
  * Segment-break semantics (never naive delete-and-concatenate): entering a tool-call span emits a
    `('break',)` event so downstream detectors flush the preceding segment and disconnect cross-gap
    n-gram / neighbor state, while preserving exact 1-based original raw-token `fired_at` coordinates.
  * Chunk-size invariant and bounded O(max_marker_len) per-request state.
"""
import json

import numpy as np

# Verified token IDs for shipped vocabularies (qwen3.npz, glm.npz, dsv4.npz).
SHIPPED_TOOL_SPANS = {
    'qwen3': (
        ((151657,), (151658,)),                                      # <tool_call> .. </tool_call>
    ),
    'glm': (
        ((154843,), (154844,)),                                      # <tool_call> .. </tool_call>
    ),
    'dsv4': (
        ((128806,), (128807,)),                                      # <｜tool▁calls▁begin｜> .. <｜tool▁calls▁end｜>
        ((128808,), (128809,)),                                      # <｜tool▁call▁begin｜> .. <｜tool▁call▁end｜>
        ((128825, 8701, 4941, 12548),                                # V3.2: ｜DSML｜function_calls
         (128825, 8701, 4941, 12548)),
        ((128840, 8701, 4941, 12548),                                # <dsml:function_calls .. </dsml:function_calls
         (128841, 8701, 4941, 12548)),
        ((128825, 72461, 4941, 12548),                               # V4: ｜DSML｜tool_calls
         (128825, 72461, 4941, 12548)),
        ((128840, 72461, 4941, 12548),                               # <dsml:tool_calls .. </dsml:tool_calls
         (128841, 72461, 4941, 12548)),
        ((128825, 10699),                                            # V4.1: ｜DSML｜ calls (leading space)
         (128825, 10699)),
        ((128840, 10699),                                            # <dsml: calls .. </dsml: calls
         (128841, 10699)),
    ),
}

SHIPPED_THINK_PAIRS = {
    'qwen3': (((151667,), (151668,)),),                              # <think> .. </think>
    'glm': (((154841,), (154842,)),),                                # <think> .. </think>
    'dsv4': (((128821,), (128822,)),),                               # <think> .. </think>
}

SINGLE_TOOL_PAIRS = (
    ('<tool_call>', '</tool_call>'),
    ('<｜tool▁calls▁begin｜>', '<｜tool▁calls▁end｜>'),
    ('<｜tool▁call▁begin｜>', '<｜tool▁call▁end｜>'),
    ('<|tool_calls_section_begin|>', '<|tool_calls_section_end|>'),
    ('<|tool_call_begin|>', '<|tool_call_end|>'),
    ('<|tool_call|>', '<|/tool_call|>'),
    ('<|call|>', '<|/call|>'),
    ('<function_call>', '</function_call>'),
    ('<|plugin|>', '<|/plugin|>'),
    ('<|action_start|>', '<|action_end|>'),
)

# Open-to-end markers: (start_piece, optional_end_pieces_in_priority_order)
OPEN_TOOL_MARKERS = (
    ('[TOOL_CALLS]', ('[/TOOL_CALLS]',)),
    ('<|python_tag|>', ('<|eom_id|>',)),
)

SINGLE_THINK_PAIRS = (
    ('<think>', '</think>'),
    ('<|thought_start|>', '<|thought_end|>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'),
    ('[THINK]', '[/THINK]'),
    ('<thought>', '</thought>'),
)

# Official DeepSeek DSML container tag names across V3.2 ('function_calls'), V4 ('tool_calls'), and V4.1 (' calls').
_DSML_TAGS = ('function_calls', 'tool_calls', ' calls')
_DSML_PARTS = ('function_calls', 'tool_calls', ' calls', '\u0120calls', '\u2581calls')
_DSML_SUBS = frozenset(
    s[i:j] for s in _DSML_PARTS for i in range(len(s)) for j in range(i + 1, len(s) + 1)
)


def _lookup_exact(piece_to_id, text, bytes_to_id=None):
    if text in piece_to_id:
        return int(piece_to_id[text])
    if bytes_to_id is not None:
        raw = text.encode('utf-8')
        if raw in bytes_to_id:
            return int(bytes_to_id[raw])
    return None


def _build_dsml_merge_ranks(merges):
    ranks = {}
    for rank, item in enumerate(merges):
        if isinstance(item, str):
            parts = item.split(' ')
            if len(parts) != 2:
                continue
            a, b = parts
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = item[0], item[1]
        else:
            continue
        if a + b in _DSML_SUBS:
            ranks[(a, b)] = rank
    return ranks


def _bpe_encode_ascii(text, piece_to_id, merge_ranks):
    if ' ' in text:
        if '\u0120' in piece_to_id:
            text = text.replace(' ', '\u0120')
        elif '\u2581' in piece_to_id:
            text = text.replace(' ', '\u2581')
    syms = list(text)
    if any(ch not in piece_to_id for ch in syms):
        return None
    while len(syms) > 1:
        best_rank = None
        best_pair = None
        for i in range(len(syms) - 1):
            pair = (syms[i], syms[i + 1])
            rank = merge_ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_pair = pair
        if best_pair is None:
            break
        merged = []
        i = 0
        n = len(syms)
        pa, pb = best_pair
        pab = pa + pb
        while i < n:
            if i + 1 < n and syms[i] == pa and syms[i + 1] == pb:
                merged.append(pab)
                i += 2
            else:
                merged.append(syms[i])
                i += 1
        syms = merged
    try:
        return tuple(int(piece_to_id[s]) for s in syms)
    except KeyError:
        return None


def _encode_via_fn(encode_fn, text):
    try:
        res = encode_fn(text)
    except Exception:
        return None
    if hasattr(res, 'ids'):
        ids = res.ids
    else:
        ids = res
    if not ids:
        return None
    return tuple(int(x) for x in ids)


def detect_tool_spans(piece_to_id, chunks=None, merges=None, encode_fn=None):
    """Auto-detect tool-call (start_seq, end_seq_or_None) spans from a tokenizer vocabulary."""
    bytes_to_id = None
    if chunks is not None:
        bytes_to_id = {}
        for i, b in enumerate(chunks):
            bytes_to_id.setdefault(b, i)

    found = []
    seen_starts = set()

    def add(start_seq, end_seq):
        if start_seq and start_seq not in seen_starts:
            seen_starts.add(start_seq)
            found.append((tuple(start_seq), tuple(end_seq) if end_seq is not None else None))

    for s_tok, e_tok in SINGLE_TOOL_PAIRS:
        sid = _lookup_exact(piece_to_id, s_tok, bytes_to_id)
        eid = _lookup_exact(piece_to_id, e_tok, bytes_to_id)
        if sid is not None and eid is not None:
            add((sid,), (eid,))

    for s_tok, end_cands in OPEN_TOOL_MARKERS:
        sid = _lookup_exact(piece_to_id, s_tok, bytes_to_id)
        if sid is None:
            continue
        eid = None
        for ec in end_cands:
            eid = _lookup_exact(piece_to_id, ec, bytes_to_id)
            if eid is not None:
                break
        add((sid,), (eid,) if eid is not None else None)

    # Kimi-K3 XTML multi-token markers: <|open|>tools<|sep|> .. <|close|>tools<|sep|>
    id_open = _lookup_exact(piece_to_id, '<|open|>', bytes_to_id)
    id_close = _lookup_exact(piece_to_id, '<|close|>', bytes_to_id)
    id_sep = _lookup_exact(piece_to_id, '<|sep|>', bytes_to_id)
    if id_open is not None and id_close is not None and id_sep is not None:
        id_tools = _lookup_exact(piece_to_id, 'tools', bytes_to_id)
        if id_tools is not None:
            add((id_open, id_tools, id_sep), (id_close, id_tools, id_sep))

    # DeepSeek DSML multi-token markers: omit outer '<' / '</' / '>' because BPE merges them with adjacent
    # whitespace/newlines/tags (' <', '></', '>\n', '><'). Matching '｜DSML｜' + tag and '<dsml:' / '</dsml:' + tag
    # covers V3.2 ('function_calls'), V4 ('tool_calls'), and V4.1 (' calls') across all BPE boundary contexts.
    id_dsml = _lookup_exact(piece_to_id, '｜DSML｜', bytes_to_id)
    id_dsml_open = _lookup_exact(piece_to_id, '<dsml:', bytes_to_id)
    id_dsml_close = _lookup_exact(piece_to_id, '</dsml:', bytes_to_id)
    if id_dsml is not None or (id_dsml_open is not None and id_dsml_close is not None):
        if encode_fn is not None:
            for tag in _DSML_TAGS:
                if id_dsml is not None:
                    seq = _encode_via_fn(encode_fn, '｜DSML｜' + tag)
                    if seq and seq[0] == id_dsml and len(seq) >= 2:
                        add(seq, seq)
                if id_dsml_open is not None and id_dsml_close is not None:
                    s_seq = _encode_via_fn(encode_fn, '<dsml:' + tag)
                    e_seq = _encode_via_fn(encode_fn, '</dsml:' + tag)
                    if s_seq and e_seq and s_seq[0] == id_dsml_open and e_seq[0] == id_dsml_close:
                        add(s_seq, e_seq)
        elif merges is not None:
            ranks = _build_dsml_merge_ranks(merges)
            for tag in _DSML_TAGS:
                tail = _bpe_encode_ascii(tag, piece_to_id, ranks)
                if tail is None:
                    continue
                if id_dsml is not None:
                    seq = (id_dsml,) + tail
                    add(seq, seq)
                if id_dsml_open is not None and id_dsml_close is not None:
                    add((id_dsml_open,) + tail, (id_dsml_close,) + tail)

    return tuple(found)


def detect_think_pairs(piece_to_id, chunks=None):
    """Auto-detect explicit (think_start_seq, think_end_seq) pairs for open-think protection."""
    bytes_to_id = None
    if chunks is not None:
        bytes_to_id = {}
        for i, b in enumerate(chunks):
            bytes_to_id.setdefault(b, i)

    found = []
    seen_starts = set()

    def add(start_seq, end_seq):
        if start_seq and end_seq and start_seq not in seen_starts:
            seen_starts.add(start_seq)
            found.append((tuple(start_seq), tuple(end_seq)))

    for s_tok, e_tok in SINGLE_THINK_PAIRS:
        sid = _lookup_exact(piece_to_id, s_tok, bytes_to_id)
        eid = _lookup_exact(piece_to_id, e_tok, bytes_to_id)
        if sid is not None and eid is not None:
            add((sid,), (eid,))

    # Kimi-K3 XTML multi-token thinking markers: <|open|>think<|sep|> .. <|close|>think<|sep|>
    id_open = _lookup_exact(piece_to_id, '<|open|>', bytes_to_id)
    id_close = _lookup_exact(piece_to_id, '<|close|>', bytes_to_id)
    id_sep = _lookup_exact(piece_to_id, '<|sep|>', bytes_to_id)
    if id_open is not None and id_close is not None and id_sep is not None:
        id_think = _lookup_exact(piece_to_id, 'think', bytes_to_id)
        if id_think is not None:
            add((id_open, id_think, id_sep), (id_close, id_think, id_sep))

    return tuple(found)


def serialize_spans(spans):
    """Serialize normalized spans to JSON string for compact .npz metadata storage."""
    return json.dumps([[list(s), None if e is None else list(e)] for s, e in spans])


def deserialize_spans(raw):
    """Deserialize JSON metadata from .npz back to tuple of (start_tuple, end_tuple_or_None)."""
    if raw is None:
        return ()
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    if not raw:
        return ()
    items = json.loads(raw)
    out = []
    for s, e in items:
        out.append((tuple(int(x) for x in s), None if e is None else tuple(int(x) for x in e)))
    return tuple(out)


def _normalize_seq(seq, label, vocab_size=None):
    if isinstance(seq, (bool, np.bool_)):
        raise ValueError(f'{label} must be a token id or non-empty sequence of token ids, got {seq!r}')
    if isinstance(seq, (int, np.integer)):
        seq = (int(seq),)
    elif isinstance(seq, (list, tuple, np.ndarray)):
        out = []
        for x in seq:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
                raise ValueError(f'{label} must contain integer token ids, got {x!r}')
            out.append(int(x))
        seq = tuple(out)
    else:
        raise ValueError(f'{label} must be a token id or sequence of token ids, got {seq!r}')
    if not seq:
        raise ValueError(f'{label} must be non-empty')
    for tid in seq:
        if tid < 0 or (vocab_size is not None and tid >= vocab_size):
            raise ValueError(f'{label} token id {tid} out of range')
    return seq


def normalize_tool_spans(tool_spans, vocab_size=None):
    """Validate and normalize explicit tool_spans into tuple[(tuple[int,...], tuple[int,...]|None)]."""
    if tool_spans is None:
        return ()
    if isinstance(tool_spans, str):
        raise ValueError(f'tool_spans must be "auto", None, or a sequence of (start, end) pairs, got {tool_spans!r}')
    out = []
    seen = set()
    for item in tool_spans:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f'each tool_span must be a (start, end) pair, got {item!r}')
        s_raw, e_raw = item
        s = _normalize_seq(s_raw, 'tool_span start', vocab_size)
        e = None if e_raw is None else _normalize_seq(e_raw, 'tool_span end', vocab_size)
        if s not in seen:
            seen.add(s)
            out.append((s, e))
    return tuple(out)


def normalize_think_pairs(think_pairs, vocab_size=None):
    """Validate and normalize explicit think_pairs into tuple[(tuple[int,...], tuple[int,...])]."""
    if think_pairs is None:
        return ()
    if isinstance(think_pairs, str):
        raise ValueError(f'think_pairs must be "auto", None, or a sequence of (start, end) pairs, got {think_pairs!r}')
    out = []
    seen = set()
    for item in think_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f'each think_pair must be a (start, end) pair, got {item!r}')
        s = _normalize_seq(item[0], 'think_pair start', vocab_size)
        e = _normalize_seq(item[1], 'think_pair end', vocab_size)
        if s not in seen:
            seen.add(s)
            out.append((s, e))
    return tuple(out)


def detect_from_table(T):
    """Extract (tool_spans, think_pairs) from a loaded table dict (or .npz contents)."""
    if not isinstance(T, dict):
        return (), ()
    spans = None
    thinks = None
    if 'tool_spans' in T:
        spans = normalize_tool_spans(T['tool_spans'])
    elif 'tool_spans_json' in T:
        spans = normalize_tool_spans(deserialize_spans(T['tool_spans_json']))
    if 'think_pairs' in T:
        thinks = normalize_think_pairs(T['think_pairs'])
    elif 'think_pairs_json' in T:
        thinks = normalize_think_pairs(deserialize_spans(T['think_pairs_json']))

    if spans is not None and thinks is not None:
        return spans, thinks

    # Fallback: reconstruct exact UTF-8 piece map from decoded token bytes if present (e.g., legacy .npz).
    chunks = T.get('bytes')
    if chunks is None and 'boff' in T and 'bflat' in T:
        boff, bflat = T['boff'], T['bflat']
        chunks = [bflat[int(boff[i]):int(boff[i + 1])].tobytes() for i in range(len(boff) - 1)]
    if chunks is not None:
        piece_to_id = {}
        for i, b in enumerate(chunks):
            try:
                s = b.decode('utf-8')
            except UnicodeDecodeError:
                continue
            piece_to_id.setdefault(s, i)
        if spans is None:
            spans = detect_tool_spans(piece_to_id, chunks=chunks)
        if thinks is None:
            thinks = detect_think_pairs(piece_to_id, chunks=chunks)

    return spans or (), thinks or ()


def resolve_tool_spans(corpus, T, tool_spans='auto', think_pairs='auto', vocab_size=None):
    """Resolve (tool_spans, think_pairs) for RepDetector or RareDetector."""
    if tool_spans is None:
        return (), ()

    shipped_spans = SHIPPED_TOOL_SPANS.get(corpus) if isinstance(corpus, str) else None
    shipped_thinks = SHIPPED_THINK_PAIRS.get(corpus) if isinstance(corpus, str) else None

    if tool_spans == 'auto':
        if shipped_spans is not None:
            spans = shipped_spans
        else:
            spans, _ = detect_from_table(T)
    else:
        spans = normalize_tool_spans(tool_spans, vocab_size=vocab_size)

    if not spans:
        return (), ()

    if think_pairs is None:
        thinks = ()
    elif think_pairs == 'auto':
        if shipped_thinks is not None:
            thinks = shipped_thinks
        else:
            _, thinks = detect_from_table(T)
    else:
        thinks = normalize_think_pairs(think_pairs, vocab_size=vocab_size)

    return spans, thinks


class ToolSpanState:
    """Tiny per-request streaming state for ToolSpanMatcher (no reference to matcher or tables)."""
    __slots__ = (
        'raw_pos', 'in_tool', 'active_tool_end', 'tool_tail',
        'in_think', 'active_think_end', 'think_tail',
        'buf', 'buf_start', 'buf_lp1', 'buf_lpc',
    )

    def __init__(self):
        self.raw_pos = 0
        self.in_tool = False
        self.active_tool_end = None
        self.tool_tail = []
        self.in_think = False
        self.active_think_end = None
        self.think_tail = []
        self.buf = []
        self.buf_start = 0
        self.buf_lp1 = None
        self.buf_lpc = None


class ToolSpanMatcher:
    """Immutable, chunk-size-invariant streaming filter excluding tool-call spans."""

    def __init__(self, tool_spans, think_pairs=(), vocab_size=None):
        self.tool_spans = normalize_tool_spans(tool_spans, vocab_size=vocab_size)
        self.think_pairs = normalize_think_pairs(think_pairs, vocab_size=vocab_size)
        self._start_to_end = {s: e for s, e in self.tool_spans}
        self._start_prefixes = frozenset(
            s[:k] for s in self._start_to_end for k in range(1, len(s))
        )
        self._max_start_len = max((len(s) for s in self._start_to_end), default=0)

        self._think_start_to_end = {s: e for s, e in self.think_pairs}
        self._think_start_prefixes = frozenset(
            s[:k] for s in self._think_start_to_end for k in range(1, len(s))
        )
        self._max_think_start_len = max((len(s) for s in self._think_start_to_end), default=0)

        triggers = {s[0] for s in self._start_to_end} | {s[0] for s in self._think_start_to_end}
        self._outside_triggers = frozenset(triggers)
        if vocab_size is not None and vocab_size > 0 and triggers:
            lut = np.zeros(int(vocab_size), dtype=bool)
            for tid in triggers:
                if 0 <= tid < vocab_size:
                    lut[tid] = True
            self._outside_lut = lut
        else:
            self._outside_lut = None

    @staticmethod
    def start():
        return ToolSpanState()

    def _has_outside_trigger(self, tok):
        if not self._outside_triggers:
            return False
        lut = self._outside_lut
        if lut is not None:
            V = len(lut)
            if len(tok) and int(tok.min()) >= 0 and int(tok.max()) < V:
                return bool(lut[tok].any())
        triggers = self._outside_triggers
        return any(x in triggers for x in tok.tolist())

    def _update_think(self, tst, tid):
        if not self._think_start_to_end:
            return
        if tst.in_think:
            end = tst.active_think_end
            if len(end) == 1:
                if tid == end[0]:
                    tst.in_think = False
                    tst.active_think_end = None
            else:
                tst.think_tail.append(tid)
                while tst.think_tail:
                    t = tuple(tst.think_tail)
                    if t == end:
                        tst.in_think = False
                        tst.active_think_end = None
                        tst.think_tail.clear()
                        break
                    if len(t) < len(end) and end[:len(t)] == t:
                        break
                    tst.think_tail.pop(0)
            return

        if self._max_think_start_len == 1:
            end = self._think_start_to_end.get((tid,))
            if end is not None:
                tst.in_think = True
                tst.active_think_end = end
        else:
            tst.think_tail.append(tid)
            while tst.think_tail:
                t = tuple(tst.think_tail)
                end = self._think_start_to_end.get(t)
                if end is not None:
                    tst.in_think = True
                    tst.active_think_end = end
                    tst.think_tail.clear()
                    break
                if t in self._think_start_prefixes:
                    break
                tst.think_tail.pop(0)

    def _step_tool_end(self, tst, tid):
        end = tst.active_tool_end
        if end is None:
            return
        if len(end) == 1:
            if tid == end[0]:
                tst.in_tool = False
                tst.active_tool_end = None
            return
        tst.tool_tail.append(tid)
        while tst.tool_tail:
            t = tuple(tst.tool_tail)
            if t == end:
                tst.in_tool = False
                tst.active_tool_end = None
                tst.tool_tail.clear()
                break
            if len(t) < len(end) and end[:len(t)] == t:
                break
            tst.tool_tail.pop(0)

    def feed(self, tst, tokens, top1_lp=None, chosen_lp=None):
        """Filter a chunk of raw tokens (and optional lockstep logprobs).

        Returns a list of events:
          * ('data', orig_offset, tok_arr, lp1_arr_or_None, lpc_arr_or_None)
          * ('break',)   -- emitted when entering a tool-call span
        """
        tok = np.asarray(tokens, np.int64)
        n = len(tok)
        if not n:
            return []
        has_lp = top1_lp is not None
        lp1 = np.asarray(top1_lp, np.float64) if has_lp else None
        lpc = np.asarray(chosen_lp, np.float64) if has_lp else None

        # Fast paths avoiding Python token-by-token iteration on normal / inside-tool / inside-think chunks.
        if not tst.in_tool and not tst.in_think and not tst.buf and not tst.think_tail:
            if not self._has_outside_trigger(tok):
                start = tst.raw_pos
                tst.raw_pos += n
                return [('data', start, tok, lp1, lpc)]
        elif tst.in_tool:
            end = tst.active_tool_end
            if end is None:
                tst.raw_pos += n
                return []
            if len(end) == 1 and not bool((tok == end[0]).any()):
                tst.raw_pos += n
                return []
            if len(end) > 1 and not tst.tool_tail and not bool((tok == end[0]).any()):
                tst.raw_pos += n
                return []
        elif tst.in_think:
            end = tst.active_think_end
            if len(end) == 1 and not bool((tok == end[0]).any()):
                start = tst.raw_pos
                tst.raw_pos += n
                return [('data', start, tok, lp1, lpc)]
            if len(end) > 1 and not tst.think_tail and not bool((tok == end[0]).any()):
                start = tst.raw_pos
                tst.raw_pos += n
                return [('data', start, tok, lp1, lpc)]

        events = []
        out_start = 0
        out_tok = []
        out_lp1 = [] if has_lp else None
        out_lpc = [] if has_lp else None

        def flush_out():
            if out_tok:
                events.append((
                    'data',
                    out_start,
                    np.asarray(out_tok, np.int64),
                    np.asarray(out_lp1, np.float64) if has_lp else None,
                    np.asarray(out_lpc, np.float64) if has_lp else None,
                ))
                out_tok.clear()
                if has_lp:
                    out_lp1.clear()
                    out_lpc.clear()

        def emit_one(pos, tid, v1, vc):
            nonlocal out_start
            if not out_tok:
                out_start = pos
            out_tok.append(tid)
            if has_lp:
                out_lp1.append(v1)
                out_lpc.append(vc)
            self._update_think(tst, tid)

        tok_list = tok.tolist()
        lp1_list = lp1.tolist() if has_lp else None
        lpc_list = lpc.tolist() if has_lp else None
        start_to_end = self._start_to_end
        start_prefixes = self._start_prefixes
        pos0 = tst.raw_pos

        for idx, tid in enumerate(tok_list):
            pos = pos0 + idx
            v1 = lp1_list[idx] if has_lp else None
            vc = lpc_list[idx] if has_lp else None

            if tst.in_tool:
                self._step_tool_end(tst, tid)
                continue

            if tst.in_think:
                emit_one(pos, tid, v1, vc)
                continue

            if not tst.buf:
                tst.buf_start = pos
            tst.buf.append(tid)
            if has_lp:
                if tst.buf_lp1 is None:
                    tst.buf_lp1 = []
                    tst.buf_lpc = []
                tst.buf_lp1.append(v1)
                tst.buf_lpc.append(vc)

            while tst.buf:
                if tst.in_think:
                    # A popped prefix token entered an explicit <think> block: flush buffered tokens until </think>.
                    b0 = tst.buf.pop(0)
                    bpos = tst.buf_start
                    tst.buf_start += 1
                    bv1 = tst.buf_lp1.pop(0) if has_lp else None
                    bvc = tst.buf_lpc.pop(0) if has_lp else None
                    emit_one(bpos, b0, bv1, bvc)
                    continue

                key = tuple(tst.buf)
                if key in start_to_end:
                    tst.in_tool = True
                    tst.active_tool_end = start_to_end[key]
                    tst.tool_tail.clear()
                    tst.think_tail.clear()
                    tst.buf.clear()
                    if tst.buf_lp1 is not None:
                        tst.buf_lp1.clear()
                        tst.buf_lpc.clear()
                    flush_out()
                    events.append(('break',))
                    break
                if key in start_prefixes:
                    break
                b0 = tst.buf.pop(0)
                bpos = tst.buf_start
                tst.buf_start += 1
                bv1 = tst.buf_lp1.pop(0) if has_lp else None
                bvc = tst.buf_lpc.pop(0) if has_lp else None
                emit_one(bpos, b0, bv1, bvc)

        flush_out()
        tst.raw_pos = pos0 + n
        return events

    def finish(self, tst, has_lp=False):
        """Flush any incomplete start-marker prefix buffered at end-of-stream."""
        if not tst.buf:
            return []
        start = tst.buf_start
        tok_arr = np.asarray(tst.buf, np.int64)
        lp1_arr = np.asarray(tst.buf_lp1, np.float64) if (has_lp and tst.buf_lp1 is not None) else None
        lpc_arr = np.asarray(tst.buf_lpc, np.float64) if (has_lp and tst.buf_lpc is not None) else None
        tst.buf.clear()
        if tst.buf_lp1 is not None:
            tst.buf_lp1.clear()
            tst.buf_lpc.clear()
        return [('data', start, tok_arr, lp1_arr, lpc_arr)]

    def split_segments(self, tokens):
        """Whole-sequence non-tool segments as [(start, end), ...], guaranteed identical to streaming."""
        tok = np.asarray(tokens, np.int64)
        if not len(tok):
            return []
        tst = self.start()
        events = self.feed(tst, tok) + self.finish(tst)
        segs = []
        cur_s = cur_e = None
        for ev in events:
            if ev[0] == 'break':
                if cur_s is not None:
                    segs.append((cur_s, cur_e))
                    cur_s = cur_e = None
            else:
                _, off, arr, _, _ = ev
                end = off + len(arr)
                if cur_s is None:
                    cur_s, cur_e = off, end
                else:
                    cur_e = end
        if cur_s is not None:
            segs.append((cur_s, cur_e))
        return segs


def decide_rep(tok, T, config, think_end=None, tier='intercept', tool_spans=(), think_pairs=()):
    """Whole-record RepDetector reference with tool-call span exclusion (offline.py stays frozen)."""
    from .offline import decide as offline_decide

    tok = np.asarray(tok, np.int64)
    spans = normalize_tool_spans(tool_spans)
    if not spans:
        return offline_decide(tok, T, config, think_end=think_end, tier=tier)
    matcher = ToolSpanMatcher(spans, normalize_think_pairs(think_pairs), vocab_size=len(T['norm']))
    segs = matcher.split_segments(tok)
    answer_seen = False
    for s, e in segs:
        seg = tok[s:e]
        seg_te = None if answer_seen else think_end
        at, reason = offline_decide(seg, T, config, think_end=seg_te, tier=tier)
        if at >= 0:
            return s + at, reason
        if not answer_seen and think_end is not None and bool((seg == think_end).any()):
            answer_seen = True
    return -1, ''
