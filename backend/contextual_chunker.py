"""
contextual_chunker.py

Simplified contextual chunker for Docling-style technical document
elements, for LightRAG / GraphRAG / BM25 / hybrid search / embeddings /
entity extraction.

Element shape (each item in `elements`):
    {
        "type": "TITLE" | "HEADING" | "SECTION_HEADER" | "PARAGRAPH"
              | "LIST_ITEM" | "CAPTION" | "TABLE",
        "text": str,                                   # optional for TABLE
        "table_data": {"columns": [...], "rows": [[...], ...]}  # for TABLE
    }
(type matching is case-insensitive)

Output: a list of dicts, in order:
    {
        "index": int,
        "type": "section" | "table",
        "text": str,             # "\n\n".join(complete_text)
        "complete_text": list,   # the un-joined pieces, in order
        "data": list | None,     # structured rows, only for "table" chunks
    }
"""

import re

SMALL_PARA_CHARS = 300   # a paragraph this short can be folded into a table's intro
MAX_INTRO_PARAS = 2      # attach at most this many trailing paragraphs to a table

_TABLE_NUM_PREFIX_RE = re.compile(r"^\s*table\s+[\w.\-]+\s*[:.\-]?\s*", re.IGNORECASE)


def _clean_caption(text):
    """'Table 4: Pump Characteristics' -> 'Pump Characteristics'."""
    cleaned = _TABLE_NUM_PREFIX_RE.sub("", text or "").strip()
    return cleaned or text.strip()


def _extract_table(el):
    """Pull (columns, rows) out of a TABLE element. Accepts a dict under
    table_data/table/data with columns+rows, or a bare list-of-lists
    where the first row is the header."""
    raw = el.get("table_data") or el.get("table") or el.get("data") or {}
    if isinstance(raw, dict):
        columns = list(raw.get("columns") or raw.get("headers") or [])
        rows = [list(r) for r in (raw.get("rows") or [])]
    elif isinstance(raw, list) and raw:
        columns = list(raw[0])
        rows = [list(r) for r in raw[1:]]
    else:
        columns, rows = [], []
    return columns, rows


def _format_table_block(columns, rows, caption, part=None, total_parts=None):
    """Build the TABLE:/COLUMNS:/ROWS: block as a single text piece. This
    internal format is never split with blank lines."""
    name = _clean_caption(caption) if caption else None
    title_line = f"TABLE: {name}" if name else "TABLE"
    if total_parts and total_parts > 1:
        title_line += f" (part {part + 1}/{total_parts})"

    lines = [title_line]
    if columns:
        lines.append(f"COLUMNS: {' | '.join(str(c) for c in columns)}")
    row_strs = [" | ".join(str(c) for c in row) for row in rows]
    lines.append(f"ROWS: {' '.join(row_strs)}")
    return "\n".join(lines)


def _split_oversized(text, limit, overlap):
    """Split a single paragraph that's longer than `limit` into pieces,
    breaking on whitespace (never mid-word) and carrying `overlap`
    characters of context into the next piece."""
    pieces = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + limit, n)
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        next_start = end - overlap if end - overlap > start else end
        # snap forward to the next word boundary so the overlap doesn't start mid-word
        space = text.find(" ", next_start, end)
        if space != -1:
            next_start = space + 1
        start = next_start
    return pieces


def chunk_elements(elements, target_size=1200, overlap=150, max_table_rows=30):
    chunks = []

    ctx = {"title": None, "section": None, "subsection": None}
    current_parts = []   # body pieces (paragraphs/list items) under the live heading
    current_size = 0
    pending_caption = None

    def header():
        lines = []
        if ctx["title"]:
            lines.append(f"TITLE: {ctx['title']}")
        if ctx["section"]:
            lines.append(f"SECTION: {ctx['section']}")
        if ctx["subsection"]:
            lines.append(f"SUBSECTION: {ctx['subsection']}")
        return "\n".join(lines)

    def flush():
        nonlocal current_parts, current_size
        if not current_parts:
            return
        h = header()
        pieces = ([h] if h else []) + current_parts
        chunks.append({"type": "section", "text": "\n\n".join(pieces), "complete_text": pieces, "data": None})
        current_parts, current_size = [], 0

    n = len(elements)
    i = 0
    while i < n:
        el = elements[i]
        el_type = (el.get("type") or "").upper()
        text = (el.get("text") or "").strip()

        # --- headings: update context, flush whatever belonged to the old one ---
        if el_type in ("TITLE", "SECTION_HEADER", "HEADING"):
            flush()
            if el_type == "TITLE":
                ctx["title"], ctx["section"], ctx["subsection"] = text, None, None
            elif el_type == "SECTION_HEADER":
                ctx["section"], ctx["subsection"] = text, None
            else:
                ctx["subsection"] = text
            pending_caption = None
            i += 1
            continue

        # --- captions: hold onto one that's immediately followed by a table ---
        if el_type == "CAPTION":
            nxt = elements[i + 1] if i + 1 < n else None
            if nxt and (nxt.get("type") or "").upper() == "TABLE":
                pending_caption = text
            elif text:
                current_parts.append(text)
                current_size += len(text)
            i += 1
            continue

        # --- tables: standalone chunk(s), with context + caption + schema ---
        if el_type == "TABLE":
            # fold up to MAX_INTRO_PARAS short trailing paragraphs into the table's intro
            intro = []
            while current_parts and len(intro) < MAX_INTRO_PARAS and len(current_parts[-1]) <= SMALL_PARA_CHARS:
                popped = current_parts.pop()
                current_size -= len(popped)
                intro.insert(0, popped)
            flush()

            caption = pending_caption
            pending_caption = None
            consumed_next_caption = False
            if not caption and i + 1 < n and (elements[i + 1].get("type") or "").upper() == "CAPTION":
                caption = (elements[i + 1].get("text") or "").strip()
                consumed_next_caption = True

            columns, rows = _extract_table(el)
            h = header()

            row_groups = [rows[j:j + max_table_rows] for j in range(0, len(rows), max_table_rows)] or [rows]
            multi = len(row_groups) > 1

            for part_idx, group in enumerate(row_groups):
                table_block = _format_table_block(columns, group, caption, part_idx, len(row_groups) if multi else None)
                pieces = ([h] if h else []) + (intro if part_idx == 0 else []) + [table_block]
                data = [dict(zip(columns, r)) for r in group] if columns else [list(r) for r in group]
                chunks.append({"type": "table", "text": "\n\n".join(pieces), "complete_text": pieces, "data": data})

            i += 2 if consumed_next_caption else 1
            continue

        # --- paragraphs and list items: accumulate, splitting on size only ---
        if el_type in ("PARAGRAPH", "LIST_ITEM"):
            if not text:
                i += 1
                continue
            if el_type == "LIST_ITEM":
                text = f"- {text}"

            if len(text) > target_size:
                flush()
                for piece in _split_oversized(text, target_size, overlap):
                    h = header()
                    pieces = ([h] if h else []) + [piece]
                    chunks.append({"type": "section", "text": "\n\n".join(pieces), "complete_text": pieces, "data": None})
                i += 1
                continue

            if current_parts and current_size + len(text) > target_size:
                tail = current_parts[-1][-overlap:].strip() if overlap > 0 else ""
                flush()
                current_parts = [tail] if tail else []
                current_size = len(tail)

            current_parts.append(text)
            current_size += len(text)
            i += 1
            continue

        # --- anything unrecognized degrades to plain text ---
        if text:
            current_parts.append(text)
            current_size += len(text)
        i += 1

    flush()

    return [
        {
            "index": idx,
            "type": c["type"],
            "text": c["text"],
            "complete_text": c["complete_text"],
            "data": c.get("data"),
        }
        for idx, c in enumerate(chunks)
    ]