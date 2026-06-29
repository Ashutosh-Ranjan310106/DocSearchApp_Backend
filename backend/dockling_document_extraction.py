"""
Hierarchical, context-aware document extraction and chunking.

Improvements in this revision
──────────────────────────────
  • Single DocumentConverter instance (module-level singleton) so model
    weights are loaded exactly once across all files in a session.

  • PDF title promotion: the very first heading seen in a PDF (which
    Docling usually calls SECTION_HEADER with .level == 1) is promoted
    to level 0 (document root) when no TITLE element has been seen yet,
    matching DOCX behaviour where "TITLE" comes through explicitly.

  • Heading-level guard: _resolve_heading_level now uses docling's
    reported .level properly for SECTION_HEADER (1-based → 0-based
    internally, so level-1 section headers sit at depth 1, not depth 2).

  • Section flush no longer unconditionally drops current_parts[0].
    It only strips the leading heading line when the chunk was opened by
    a Heading element (tracked via `_current_parts_has_heading` flag).

  • Table NER prose: each table chunk gets a one-line natural-language
    summary prepended to `text` so spaCy / graph extractors see entities
    without having to parse the key=value schema format.

  • Empty display_text guard: _with_markdown_breadcrumb returns only
    the trail when body is empty, which is valid but now explicitly
    handled so the viewer always gets something sensible.

  • _resolve_heading_level: treats docling's 1-based .level as
    0-based internally (level 1 → depth 1, not depth 2).
"""

import re
import tempfile
import uuid
from pathlib import Path

from bs4 import BeautifulSoup
import fitz
from docling.document_converter import DocumentConverter


# ---------------------------------------------------------
# MODULE-LEVEL SINGLETON  (avoids reloading model weights)
# ---------------------------------------------------------

_CONVERTER: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = DocumentConverter()
    return _CONVERTER


# ---------------------------------------------------------
# HTML TABLE -> RECORDS
# ---------------------------------------------------------

def html_table_to_records(html: str):
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    if len(rows) < 2:
        return []
    headers = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"])]
    records = []
    for row in rows[1:]:
        values = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if not values:
            continue
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        records.append({header: values[i] for i, header in enumerate(headers)})
    return records


# ---------------------------------------------------------
# TABLE TEXT REPRESENTATIONS
# ---------------------------------------------------------

def table_to_text(df, header_columns=None):
    """
    Compact LLM/embedding-facing representation.
    Prefixed with a one-line prose summary for NER / graph extraction:
      'Table with columns: Col1, Col2, Col3 (N rows)'
    so entity extractors see the column names as natural text.
    """
    columns = header_columns if header_columns is not None else list(df.columns)
    n_rows = len(df)

    summary = f"Table with columns: {', '.join(columns)} ({n_rows} row{'s' if n_rows != 1 else ''})"

    lines = [summary, "", "TABLE SCHEMA:", ", ".join(columns), "", "TABLE DATA:"]
    for _, row in df.iterrows():
        pairs = [f"{col}={str(val).strip()}" for col, val in row.items() if str(val).strip()]
        lines.append("; ".join(pairs))

    return "\n".join(lines)


def _escape_md_cell(val) -> str:
    val = str(val).strip()
    val = val.replace("|", "\\|").replace("\n", " ")
    return val if val else " "


def table_to_markdown(df, header_columns=None) -> str:
    """Human-facing markdown table for display_text."""
    columns = header_columns if header_columns is not None else list(df.columns)
    if not columns:
        return ""

    # compute column widths for alignment
    col_widths = [max(3, len(str(c))) for c in columns]
    for _, row in df.iterrows():
        for j, col in enumerate(columns):
            col_widths[j] = max(col_widths[j], len(_escape_md_cell(row.get(col, ""))))

    def fmt(cells):
        return "| " + " | ".join(
            _escape_md_cell(cells[j] if j < len(cells) else "").ljust(col_widths[j])
            for j in range(len(columns))
        ) + " |"

    header_row = fmt(columns)
    sep_row = "| " + " | ".join("-" * col_widths[j] for j in range(len(columns))) + " |"
    body_rows = [fmt([row.get(col, "") for col in columns]) for _, row in df.iterrows()]

    return "\n".join([header_row, sep_row, *body_rows])


def markdown_breadcrumb(path) -> str:
    """
    Render hierarchy path as markdown headings.
    level 0 → h1, level 1 → h2, etc. (clamped to h1–h6).
    """
    if not path:
        return ""
    lines = []
    for node in path:
        level = node.get("level", 0)
        md_level = max(1, min(6, level + 1))
        lines.append(f"{'#' * md_level} {node.get('text', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------
# HEADING LEVEL HELPERS
# ---------------------------------------------------------

_HEADING_LEVEL_BY_LABEL = {
    "TITLE":          0,
    "SECTION_HEADER": 1,   # will be overridden by .level when available
    "HEADING":        1,
}


def _resolve_heading_level(item, label: str) -> int:
    """
    Map a Docling item to an internal 0-based heading depth.

    Docling SECTION_HEADER .level is 1-based (1 = top-level section).
    We convert: internal_depth = docling_level   (so level-1 → depth 1,
    which nests correctly under a TITLE at depth 0).

    TITLE is always depth 0 (document root).
    HEADING without .level → depth 1 (same as top-level section).
    """
    if label == "TITLE":
        return 0

    level = getattr(item, "level", None)
    if isinstance(level, int) and level >= 1:
        return level          # 1-based == 0-based depth for non-TITLE

    return _HEADING_LEVEL_BY_LABEL.get(label, 1)


class HeadingStack:
    """
    Maintains the current ancestor chain.
    Pushing level N pops everything at depth >= N first (siblings replace,
    children nest).
    """

    def __init__(self):
        self._stack = []   # list of (depth, text, node_id)
        self._has_title = False   # tracks whether a depth-0 heading was seen

    def push(self, depth: int, text: str) -> str:
        while self._stack and self._stack[-1][0] >= depth:
            self._stack.pop()
        node_id = str(uuid.uuid4())[:8]
        self._stack.append((depth, text, node_id))
        if depth == 0:
            self._has_title = True
        return node_id

    def promote_first_to_title(self):
        """
        Call this when the document is a PDF and no TITLE element has been
        emitted yet but we are about to push the very first heading.
        Resets _has_title so the next push() at any depth is treated as
        the document root (depth 0).

        Callers should only invoke this once per document.
        """
        self._has_title = True   # mark as handled; push will do the rest

    @property
    def needs_title_promotion(self) -> bool:
        return not self._has_title and not self._stack

    @property
    def breadcrumb(self) -> str:
        return " > ".join(t for _, t, _ in self._stack)

    @property
    def path(self):
        return [{"level": d, "text": t, "node_id": nid} for d, t, nid in self._stack]

    @property
    def current_section_id(self):
        return self._stack[-1][2] if self._stack else None

    @property
    def current_level(self) -> int:
        return self._stack[-1][0] if self._stack else -1


# ---------------------------------------------------------
# DOCLING EXTRACTION (hierarchy-aware)
# ---------------------------------------------------------

def _process_doc(doc, output: list, heading_stack: HeadingStack, is_pdf: bool):
    """
    Walk one Docling Document and append structured elements to `output`.
    Mutates `heading_stack` in place (carries state across PDF page batches).
    """
    for item, _depth in doc.iterate_items():
        label = item.label.name
        text = (getattr(item, "text", "") or "").strip()

        # ── HEADINGS ──────────────────────────────────────────────────────
        if label in {"TITLE", "SECTION_HEADER", "HEADING"}:
            if not text:
                continue

            h_depth = _resolve_heading_level(item, label)

            # PDF title promotion: first heading ever seen in a PDF document
            # that is NOT already labelled TITLE gets promoted to depth 0.
            if is_pdf and label != "TITLE" and heading_stack.needs_title_promotion:
                h_depth = 0

            node_id = heading_stack.push(h_depth, text)

            output.append({
                "type":       "Heading",
                "text":       text,
                "level":      h_depth,
                "node_id":    node_id,
                "breadcrumb": heading_stack.breadcrumb,
                "path":       heading_stack.path,
            })
            continue

        # ── TABLES ────────────────────────────────────────────────────────
        if label == "TABLE":
            table_data, columns, table_text = [], [], ""
            try:
                df = item.export_to_dataframe(doc)
                table_data = df.fillna("").astype(str).to_dict(orient="records")
                columns    = list(df.columns)
                table_text = table_to_text(df)
            except Exception:
                try:
                    table_text = item.export_to_markdown(doc)
                except Exception:
                    table_text = ""

            output.append({
                "type":       "Table",
                "breadcrumb": heading_stack.breadcrumb,
                "path":       heading_stack.path,
                "section_id": heading_stack.current_section_id,
                "level":      heading_stack.current_level,
                "text":       table_text,
                "data":       table_data,
                "columns":    columns,
            })
            continue

        # ── NORMAL TEXT ───────────────────────────────────────────────────
        if text:
            output.append({
                "type":       label,
                "breadcrumb": heading_stack.breadcrumb,
                "path":       heading_stack.path,
                "section_id": heading_stack.current_section_id,
                "level":      heading_stack.current_level,
                "text":       text,
            })

    return heading_stack


def extract_with_dockling(filename: str, content: bytes, batch_size: int = 2):
    suffix   = Path(filename).suffix.lower()
    converter = _get_converter()          # reuse singleton

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    output        = []
    heading_stack = HeadingStack()
    is_pdf        = suffix == ".pdf"

    if is_pdf:
        pdf         = fitz.open(tmp_path)
        total_pages = len(pdf)
        pdf.close()

        for start in range(1, total_pages + 1, batch_size):
            end = min(start + batch_size - 1, total_pages)
            print(f"Processing PDF pages {start}-{end}")
            result        = converter.convert(tmp_path, page_range=(start, end))
            heading_stack = _process_doc(result.document, output, heading_stack, is_pdf=True)
    else:
        print(f"Processing {suffix} as a single document")
        result        = converter.convert(tmp_path)
        heading_stack = _process_doc(result.document, output, heading_stack, is_pdf=False)

    return output


# ---------------------------------------------------------
# SENTENCE-AWARE SPLITTING
# ---------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_sentences(text: str):
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _split_long_text(text: str, target_size: int, overlap: int):
    sentences = _split_sentences(text)
    if not sentences:
        pieces, start = [], 0
        while start < len(text):
            end = start + target_size
            pieces.append(text[start:end])
            start += max(target_size - overlap, 1)
        return pieces

    pieces, current, current_len = [], [], 0
    for sent in sentences:
        if current_len + len(sent) > target_size and current:
            pieces.append(" ".join(current))
            carry, carry_len = [], 0
            for s in reversed(current):
                if carry_len + len(s) > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s)
            current, current_len = carry + [sent], sum(len(s) for s in carry) + len(sent)
        else:
            current.append(sent)
            current_len += len(sent)
    if current:
        pieces.append(" ".join(current))
    return pieces


# ---------------------------------------------------------
# CONTEXT INJECTION
# ---------------------------------------------------------

def _with_breadcrumb(breadcrumb: str | None, text: str) -> str:
    """LLM/embedding-facing: compact [Context: ...] tag."""
    if breadcrumb:
        return f"[Context: {breadcrumb}]\n\n{text}"
    return text


def _with_markdown_breadcrumb(path, body_markdown: str) -> str:
    """Human-facing: real markdown heading trail."""
    trail = markdown_breadcrumb(path)
    if trail and body_markdown.strip():
        return f"{trail}\n\n{body_markdown}"
    if trail:
        return trail
    return body_markdown


# ---------------------------------------------------------
# LARGE TABLE SPLITTING
# ---------------------------------------------------------

def _table_ner_summary(columns: list, data: list) -> str:
    """
    One-line prose summary prepended to table `text` so spaCy NER and
    graph extractors see entity-bearing tokens in natural language, not
    just schema format.

    Example:
      'Table with columns: Project, Hull No, Equipment (6 rows).
       Sample values — Project: 50,000 DWT Bulk Carrier; Hull No: HN-2026-045.'
    """
    n = len(data)
    col_str = ", ".join(columns)
    summary = f"Table with columns: {col_str} ({n} row{'s' if n != 1 else ''})."

    # add a sample of real values from row 0 so NER sees actual entities
    if data:
        row0 = data[0]
        samples = "; ".join(
            f"{k}: {v}" for k, v in list(row0.items())[:5] if str(v).strip()
        )
        if samples:
            summary += f" Sample values — {samples}."

    return summary


def _split_large_table(el: dict, target_size: int) -> list:
    data      = el.get("data") or []
    columns   = el.get("columns") or (list(data[0].keys()) if data else [])
    breadcrumb = el.get("breadcrumb")
    path       = el.get("path")

    base_meta = {
        "breadcrumb": breadcrumb,
        "path":       path,
        "section_id": el.get("section_id"),
        "level":      el.get("level"),
    }

    if not data:
        raw_text = el.get("text", "")
        return [{
            **base_meta,
            "type":              "table",
            "text":              _with_breadcrumb(breadcrumb, raw_text),
            "display_text":      _with_markdown_breadcrumb(path, raw_text),
            "data":              data,
            "table_part":        1,
            "table_parts_total": 1,
        }]

    import pandas as pd

    full_df   = pd.DataFrame(data)
    full_text = table_to_text(full_df, header_columns=columns)

    if len(full_text) <= target_size:
        full_markdown = table_to_markdown(full_df, header_columns=columns)
        ner_summary   = _table_ner_summary(columns, data)
        return [{
            **base_meta,
            "type":              "table",
            "text":              _with_breadcrumb(breadcrumb, f"{ner_summary}\n\n{full_text}"),
            "display_text":      _with_markdown_breadcrumb(path, full_markdown),
            "data":              data,
            "table_part":        1,
            "table_parts_total": 1,
        }]

    # row-batch split
    header_overhead  = len("TABLE SCHEMA:\n" + ", ".join(columns) + "\n\nTABLE DATA:\n")
    avg_row_len      = max(1, (len(full_text) - header_overhead) // max(1, len(data)))
    rows_per_chunk   = max(1, (target_size - header_overhead) // avg_row_len)
    row_batches      = [data[i:i + rows_per_chunk] for i in range(0, len(data), rows_per_chunk)]
    total_parts      = len(row_batches)
    ner_summary      = _table_ner_summary(columns, data)   # same summary on every part

    chunks = []
    for i, batch in enumerate(row_batches):
        batch_df      = pd.DataFrame(batch)
        batch_text    = table_to_text(batch_df, header_columns=columns)
        batch_md      = table_to_markdown(batch_df, header_columns=columns)
        marker        = f"(table part {i + 1} of {total_parts})"
        md_marker     = f"*Table part {i + 1} of {total_parts}*"

        chunks.append({
            **base_meta,
            "type":              "table",
            "text":              _with_breadcrumb(breadcrumb, f"{ner_summary}\n\n{batch_text}"),
            "display_text":      _with_markdown_breadcrumb(path, f"{md_marker}\n\n{batch_md}"),
            "data":              batch,
            "table_part":        i + 1,
            "table_parts_total": total_parts,
        })

    return chunks


# ---------------------------------------------------------
# HIERARCHICAL CHUNKING
# ---------------------------------------------------------

def chunk_elements(elements: list, target_size: int = 1200, overlap: int = 150) -> list:
    """
    Build chunks while respecting document hierarchy.

    Each chunk carries:
      text          – LLM/embedding/BM25 facing. [Context: breadcrumb] prefix,
                      tables as prose summary + key=value rows.
      display_text  – human-facing. Markdown heading trail, aligned markdown tables.
      data          – structured row records (table chunks only).
      breadcrumb    – " > " joined heading path string.
      hierarchy_path – list of {level, text, node_id} dicts.
      chunk_id, prev_chunk_id, next_chunk_id, section_id, index, level.
    """
    raw_chunks = []

    current_parts             = []
    current_size              = 0
    current_breadcrumb        = None
    current_path              = None
    current_section_id        = None
    current_level             = None
    _current_opened_by_heading = False    # True when current_parts[0] is a Heading text

    def flush():
        nonlocal current_parts, current_size, _current_opened_by_heading
        if not current_parts:
            return

        body = "\n\n".join(current_parts)

        # For display_text: if this chunk was opened by a Heading element,
        # current_parts[0] is that heading's text — the markdown breadcrumb
        # already contains it as the final heading level, so skip it to avoid
        # duplication. Otherwise keep all parts.
        if _current_opened_by_heading and len(current_parts) > 1:
            display_body = "\n\n".join(current_parts[1:]).strip()
        elif _current_opened_by_heading:
            display_body = ""   # heading with no body content yet — trail is enough
        else:
            display_body = body

        raw_chunks.append({
            "type":         "section",
            "breadcrumb":   current_breadcrumb,
            "path":         current_path,
            "section_id":   current_section_id,
            "level":        current_level,
            "text":         _with_breadcrumb(current_breadcrumb, body),
            "display_text": _with_markdown_breadcrumb(current_path, display_body),
        })
        current_parts              = []
        current_size               = 0
        _current_opened_by_heading = False

    for el in elements:
        text    = (el.get("text") or "").strip()
        el_type = el.get("type", "")

        if not text:
            continue

        # ── HEADING: flush current section, start new ────────────────────
        if el_type == "Heading":
            flush()
            current_breadcrumb         = el.get("breadcrumb")
            current_path               = el.get("path")
            current_section_id         = el.get("node_id")
            current_level              = el.get("level")
            current_parts              = [text]
            current_size               = len(text)
            _current_opened_by_heading = True
            continue

        # ── TABLE: flush then emit own chunk(s) ──────────────────────────
        if el_type == "Table":
            flush()
            raw_chunks.extend(_split_large_table(el, target_size))
            continue

        # ── SECTION BOUNDARY GUARD ────────────────────────────────────────
        # Defensive: if content arrives tagged to a different section than
        # what we're accumulating, flush first.
        if (
            current_parts
            and el.get("section_id") is not None
            and el.get("section_id") != current_section_id
            and el.get("breadcrumb") != current_breadcrumb
        ):
            flush()
            current_breadcrumb         = el.get("breadcrumb")
            current_path               = el.get("path")
            current_section_id         = el.get("section_id")
            current_level              = el.get("level")
            _current_opened_by_heading = False

        # ── LARGE PARAGRAPH: sentence-aware split ────────────────────────
        if len(text) > target_size * 1.5:
            flush()
            pieces = _split_long_text(text, target_size, overlap)
            total  = len(pieces)
            for i, piece in enumerate(pieces):
                marker   = f"(continued {i + 1}/{total})" if total > 1 else ""
                body     = f"{marker}\n{piece}".strip() if marker else piece
                md_marker = f"*(continued {i + 1}/{total})*\n\n" if total > 1 else ""
                raw_chunks.append({
                    "type":         "paragraph",
                    "breadcrumb":   el.get("breadcrumb"),
                    "path":         el.get("path"),
                    "section_id":   el.get("section_id"),
                    "level":        el.get("level"),
                    "text":         _with_breadcrumb(el.get("breadcrumb"), body),
                    "display_text": _with_markdown_breadcrumb(el.get("path"), f"{md_marker}{piece}"),
                })
            continue

        # ── NORMAL ACCUMULATION ───────────────────────────────────────────
        if current_size + len(text) > target_size:
            flush()
            current_breadcrumb         = el.get("breadcrumb")
            current_path               = el.get("path")
            current_section_id         = el.get("section_id")
            current_level              = el.get("level")
            current_parts              = [text]
            current_size               = len(text)
            _current_opened_by_heading = False
        else:
            current_parts.append(text)
            current_size += len(text)

    flush()

    # ── LINKAGE PASS ─────────────────────────────────────────────────────
    for c in raw_chunks:
        c["chunk_id"] = str(uuid.uuid4())

    final = []
    for i, c in enumerate(raw_chunks):
        final.append({
            "index":             i,
            "chunk_id":          c["chunk_id"],
            "type":              c["type"],
            "breadcrumb":        c.get("breadcrumb"),
            "hierarchy_path":    c.get("path"),
            "section_id":        c.get("section_id"),
            "level":             c.get("level"),
            "text":              c["text"],
            "display_text":      c.get("display_text", c["text"]),
            "data":              c.get("data"),
            "table_part":        c.get("table_part"),
            "table_parts_total": c.get("table_parts_total"),
            "prev_chunk_id":     raw_chunks[i - 1]["chunk_id"] if i > 0 else None,
            "next_chunk_id":     raw_chunks[i + 1]["chunk_id"] if i < len(raw_chunks) - 1 else None,
        })

    return final