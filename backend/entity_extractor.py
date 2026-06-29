"""
entity_extractor.py

Uses spaCy's medium English model (en_core_web_md) for both NER and
relation extraction. GLiNER / remote API calls have been removed entirely.

Install the model if needed:
    python -m spacy download en_core_web_md
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import spacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy medium model — used for NER, dependency parsing, and co-occurrence
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_md")

# ---------------------------------------------------------------------------
# Entity labels we care about (mapped from spaCy's built-in label set)
# ---------------------------------------------------------------------------
RELEVANT_SPACY_LABELS = {
    "ORG", "PRODUCT", "GPE", "LOC", "FACILITY",
    "PERSON", "NORP", "EVENT", "WORK_OF_ART", "LAW",
    "LANGUAGE", "DATE", "TIME", "QUANTITY", "CARDINAL",
}

SPECIAL_COLUMNS = {
    "Acceptance Criteria": "acceptance_criteria",
    "Actual Result":       "actual_result",
    "Status":              "status",
    "Manufacturer":        "manufacturer",
    "Location":            "location",
    "Equipment":           "equipment",
    "Tag Number":          "tag_number",
    "Serial Number":       "serial_number",
}

# Cell values longer than this are treated as free-text paragraphs.
LONG_TEXT_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _starts_with_digit(text: str) -> bool:
    """Return True if the first non-whitespace character is a digit."""
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].isdigit()


def add_node(maybe_nodes, name, entity_type, chunk_key, file_path, timestamp):
    if not name:
        return
    maybe_nodes[name].append({
        "entity_name": name,
        "entity_type": entity_type,
        "description": name,
        "source_id":   chunk_key,
        "file_path":   file_path,
        "timestamp":   timestamp,
    })


def add_edge(
    maybe_edges, src, tgt, relation, chunk_key, file_path, timestamp,
    description=None,
):
    if not src or not tgt or src == tgt:
        return
    edge_key = (src, tgt)
    maybe_edges[edge_key].append({
        "src_id":      src,
        "tgt_id":      tgt,
        "weight":      1.0,
        "description": description if description is not None else relation,
        "keywords":    relation,
        "source_id":   chunk_key,
        "file_path":   file_path,
        "timestamp":   timestamp,
    })


def _col_to_relation(col: str) -> str:
    return SPECIAL_COLUMNS.get(col, col.strip().lower().replace(" ", "_"))


def _is_numeric_only(value: str) -> bool:
    return not any(ch.isalpha() for ch in value)


def _extract_entities_spacy(text: str) -> list[dict[str, Any]]:
    """
    Run spaCy NER on *text* and return a list of entity dicts:
        [{"text": "...", "label": "..."}, ...]

    Entities whose text starts with a digit are excluded.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    seen: set[str] = set()
    entities: list[dict[str, Any]] = []

    for ent in doc.ents:
        name = ent.text.strip()
        if not name:
            continue
        if _starts_with_digit(name):
            logger.debug(f"[NER] Skipping digit-start entity: {name!r}")
            continue
        if ent.label_ not in RELEVANT_SPACY_LABELS:
            continue
        if name in seen:
            continue
        seen.add(name)
        entities.append({"text": name, "label": ent.label_})

    return entities


def _handle_long_text_cell(
    maybe_nodes, maybe_edges, primary_entity,
    col, value, chunk_key, file_path, timestamp,
):
    col_relation = _col_to_relation(col)
    raw_entities = _extract_entities_spacy(value)

    if raw_entities:
        relation = f"mentioned_in_{col_relation}"
        for ent in raw_entities:
            ent_name = ent["text"]
            add_node(maybe_nodes, ent_name, ent["label"],
                     chunk_key, file_path, timestamp)
            add_edge(maybe_edges, primary_entity, ent_name, relation,
                     chunk_key, file_path, timestamp)
        logger.debug(
            f"[TABLE][LONG] col={col!r} → "
            f"{len(raw_entities)} spaCy entities via {relation!r}"
        )
    else:
        fallback_entity = col.strip()
        relation = f"has_{col_relation}"
        add_node(maybe_nodes, fallback_entity, col,
                 chunk_key, file_path, timestamp)
        add_edge(maybe_edges, primary_entity, fallback_entity, relation,
                 chunk_key, file_path, timestamp, description=value)
        logger.debug(
            f"[TABLE][LONG] col={col!r} → "
            f"no spaCy entities; fallback node={fallback_entity!r} "
            f"via {relation!r}"
        )


# ===========================================================================
# Main extraction function
# ===========================================================================

def extract_rule_entities(
    text, chunk_key, file_path, timestamp, table_data=None
):
    maybe_nodes: dict = defaultdict(list)
    maybe_edges: dict = defaultdict(list)
    logger.info(f"[ENTITY] Processing chunk={chunk_key}")

    # =====================================================
    # TABLE EXTRACTION
    # =====================================================
    table_entity_count   = 0
    table_relation_count = 0

    if table_data:
        logger.info(f"[TABLE] Found {len(table_data)} rows")

        for row in table_data:
            if not isinstance(row, dict):
                continue

            # First non-empty cell becomes the primary entity
            primary_entity = None
            for key, value in row.items():
                if value is None:
                    continue
                value_str = str(value).strip()
                if value_str and not _starts_with_digit(value_str):
                    primary_entity = value_str
                    break

            if not primary_entity:
                continue

            add_node(maybe_nodes, primary_entity, "TableEntity",
                     chunk_key, file_path, timestamp)
            table_entity_count += 1

            for col, value in row.items():
                if value is None:
                    continue
                value_str = str(value).strip()
                if not value_str or value_str == primary_entity:
                    continue

                # Skip purely numeric cells
                if _is_numeric_only(value_str):
                    logger.debug(
                        f"[TABLE] Skipping numeric-only cell "
                        f"col={col!r} value={value_str!r}"
                    )
                    continue

                # Skip cells whose value starts with a digit
                if _starts_with_digit(value_str):
                    logger.debug(
                        f"[TABLE] Skipping digit-start cell "
                        f"col={col!r} value={value_str!r}"
                    )
                    continue

                if len(value_str) <= LONG_TEXT_THRESHOLD:
                    add_node(maybe_nodes, value_str, col,
                             chunk_key, file_path, timestamp)
                    table_entity_count += 1
                    add_edge(maybe_edges, primary_entity, value_str,
                             _col_to_relation(col),
                             chunk_key, file_path, timestamp)
                    table_relation_count += 1
                else:
                    before_n = len(maybe_nodes)
                    before_e = len(maybe_edges)
                    _handle_long_text_cell(
                        maybe_nodes, maybe_edges, primary_entity,
                        col, value_str, chunk_key, file_path, timestamp,
                    )
                    table_entity_count   += len(maybe_nodes) - before_n
                    table_relation_count += len(maybe_edges) - before_e

    logger.info(
        f"[TABLE] Extracted "
        f"{table_entity_count} entities, "
        f"{table_relation_count} relations"
    )

    # =====================================================
    # SPACY NER  (free-text)
    # =====================================================
    entities = _extract_entities_spacy(text)
    logger.info(f"[SPACY NER] Found {len(entities)} entities")

    entity_lookup: dict = {}
    for ent in entities:
        entity_name = ent["text"]
        entity_lookup[entity_name] = ent
        add_node(maybe_nodes, entity_name, ent["label"],
                 chunk_key, file_path, timestamp)

    # =====================================================
    # SPACY DEPENDENCY RELATIONS
    # =====================================================
    doc = nlp(text)
    relation_count = 0

    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        sent_entities = [
            e for e in entity_lookup if e.lower() in sent_text_lower
        ]
        if len(sent_entities) < 2:
            continue

        for token in sent:
            if token.pos_ != "VERB":
                continue
            subjects = [
                c.text for c in token.children
                if c.dep_ in ("nsubj", "nsubjpass")
            ]
            objects = [
                c.text for c in token.children
                if c.dep_ in ("dobj", "obj", "attr", "pobj")
            ]
            for src in subjects:
                for tgt in objects:
                    if src not in maybe_nodes or tgt not in maybe_nodes:
                        continue
                    add_edge(maybe_edges, src, tgt, token.lemma_,
                             chunk_key, file_path, timestamp)
                    relation_count += 1

    # =====================================================
    # CO-OCCURRENCE RELATIONS
    # =====================================================
    cooccur_count = 0

    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        sent_entities = [
            e for e in entity_lookup if e.lower() in sent_text_lower
        ]
        if len(sent_entities) < 2:
            continue
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                add_edge(
                    maybe_edges,
                    sent_entities[i], sent_entities[j],
                    "associated_with",
                    chunk_key, file_path, timestamp,
                )
                cooccur_count += 1

    logger.info(
        f"[RELATIONS] Dependency={relation_count} "
        f"CoOccurrence={cooccur_count}"
    )
    logger.info(
        f"[FINAL] Nodes={len(maybe_nodes)} Edges={len(maybe_edges)}"
    )

    return dict(maybe_nodes), dict(maybe_edges)