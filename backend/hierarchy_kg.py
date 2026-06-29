"""
hierarchy_kg.py

Injects heading-hierarchy nodes and edges into the nodes/edges dicts
produced by extract_rule_entities(), BEFORE convert_nodes/convert_edges
are called.

Graph structure built for a chunk with hierarchy_path:
  [L0: "FAT Report"] → [L1: "Main Switchboard"] → [L2: "Test Results"]
                                                          ↓
                                                   [chunk entities]

Edge types added
────────────────
  HAS_SUBSECTION   parent heading  → child heading   (every adjacent level pair)
  BELONGS_TO       chunk entity    → immediate parent heading
  PART_OF          chunk entity    → every ancestor heading (skips immediate parent)

All injected nodes/edges use the same dict shape as extract_rule_entities()
output so convert_nodes() / convert_edges() require zero changes.
"""

from __future__ import annotations
from typing import Dict, List, Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _heading_entity_name(node: Dict) -> str:
    """Stable, unique entity name for a heading node."""
    return node["text"].strip()


def _heading_node(
    node: Dict,
    source_id: str,
    file_path: str,
) -> Dict:
    """Build a node record for a heading, matching convert_nodes() input shape."""
    name = _heading_entity_name(node)
    level = node.get("level", 0)
    type_map = {0: "DOCUMENT", 1: "SECTION", 2: "SUBSECTION"}
    entity_type = type_map.get(level, "SUBSECTION")

    return {
        "entity_name": name,
        "entity_type": entity_type,
        "description": f"Document heading at level {level}: {name}",
        "source_id":   source_id,
        "file_path":   file_path,
    }


def _edge(
    src: str,
    tgt: str,
    description: str,
    keywords: str,
    source_id: str,
    file_path: str,
    weight: float = 1.0,
) -> Dict:
    """Build an edge record matching convert_edges() input shape."""
    return {
        "src_id":      src,
        "tgt_id":      tgt,
        "description": description,
        "keywords":    keywords,
        "weight":      weight,
        "source_id":   source_id,
        "file_path":   file_path,
    }


# ── main injection function ───────────────────────────────────────────────────

def inject_hierarchy_edges(
    nodes: Dict[str, List[Dict]],
    edges: Dict[str, List[Dict]],
    hierarchy_path: List[Dict] | None,
    chunk_id: str,
    file_path: str,
) -> tuple[Dict, Dict]:
    """
    Mutates `nodes` and `edges` in-place to add heading hierarchy structure,
    then returns them.

    Parameters
    ──────────
    nodes          : entity dict from extract_rule_entities()
    edges          : edge dict from extract_rule_entities()
    hierarchy_path : chunk["hierarchy_path"] —
                     list of {level, text, node_id} from HeadingStack
    chunk_id       : chunk_id of the current chunk (used as source_id)
    file_path      : original filename

    Returns
    ───────
    (nodes, edges) — same objects, mutated in-place
    """
    if not hierarchy_path:
        return nodes, edges

    # ── 1. Add a node for every heading in the path ───────────────────────────
    heading_names: List[str] = []
    for h in hierarchy_path:
        name = _heading_entity_name(h)
        heading_names.append(name)

        if name not in nodes:
            nodes[name] = [_heading_node(h, chunk_id, file_path)]

    # ── 2. HAS_SUBSECTION edges between adjacent heading levels ───────────────
    #   L0 → L1, L1 → L2, L2 → L3, ...
    for i in range(len(heading_names) - 1):
        parent_name = heading_names[i]
        child_name  = heading_names[i + 1]
        edge_key    = f"{parent_name}::{child_name}"

        if edge_key not in edges:
            edges[edge_key] = []

        # avoid duplicate edges for the same parent→child pair
        existing_tgts = {e["tgt_id"] for e in edges[edge_key]}
        if child_name not in existing_tgts:
            edges[edge_key].append(_edge(
                src         = parent_name,
                tgt         = child_name,
                description = f'"{child_name}" is a subsection of "{parent_name}"',
                keywords    = "subsection, hierarchy, structure",
                source_id   = chunk_id,
                file_path   = file_path,
                weight      = 1.2,   # slightly higher — structural edges are reliable
            ))

    # ── 3. BELONGS_TO / PART_OF: chunk entities → heading ancestors ───────────
    #   immediate parent → BELONGS_TO (weight 1.0)
    #   all ancestors    → PART_OF    (weight 0.6, decreasing with distance)
    immediate_parent = heading_names[-1] if heading_names else None
    ancestors        = heading_names[:-1]   # everything above immediate parent

    # entity names that are NOT headings (i.e. the real content entities)
    heading_name_set = set(heading_names)
    content_entities = [n for n in nodes if n not in heading_name_set]

    for entity_name in content_entities:
        # BELONGS_TO → immediate section heading
        if immediate_parent:
            edge_key = f"{entity_name}::{immediate_parent}"
            if edge_key not in edges:
                edges[edge_key] = []

            existing_tgts = {e["tgt_id"] for e in edges[edge_key]}
            if immediate_parent not in existing_tgts:
                edges[edge_key].append(_edge(
                    src         = entity_name,
                    tgt         = immediate_parent,
                    description = f'"{entity_name}" belongs to section "{immediate_parent}"',
                    keywords    = "belongs_to, section, context",
                    source_id   = chunk_id,
                    file_path   = file_path,
                    weight      = 1.0,
                ))

        # PART_OF → all ancestor headings (with distance-based weight decay)
        for depth, ancestor in enumerate(reversed(ancestors)):
            weight   = max(0.3, 0.8 - depth * 0.15)
            edge_key = f"{entity_name}::{ancestor}"
            if edge_key not in edges:
                edges[edge_key] = []

            existing_tgts = {e["tgt_id"] for e in edges[edge_key]}
            if ancestor not in existing_tgts:
                edges[edge_key].append(_edge(
                    src         = entity_name,
                    tgt         = ancestor,
                    description = f'"{entity_name}" is part of document section "{ancestor}"',
                    keywords    = "part_of, ancestor, document_structure",
                    source_id   = chunk_id,
                    file_path   = file_path,
                    weight      = weight,
                ))

    return nodes, edges