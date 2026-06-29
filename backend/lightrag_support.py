def convert_nodes(nodes):
    entities = []

    for entity_name, entity_records in nodes.items():

        entity = entity_records[0]

        entities.append({
            "entity_name": entity["entity_name"],
            "entity_type": entity["entity_type"],
            "description": entity.get("description", ""),
            "source_id": entity["source_id"],
            "file_path": entity["file_path"]
        })

    return entities

def convert_edges(edges):
    relationships = []

    for _, edge_records in edges.items():

        for edge in edge_records:

            relationships.append({
                "src_id": edge["src_id"],
                "tgt_id": edge["tgt_id"],
                "description": edge.get("description", ""),
                "keywords": edge.get("keywords", ""),
                "weight": edge.get("weight", 1.0),
                "source_id": edge["source_id"],
                "file_path": edge["file_path"]
            })

    return relationships

def build_custom_kg(
    chunk_text,
    chunk_id,
    file_name,
    nodes,
    edges
):

    return {
        "chunks": [
            {
                "content": chunk_text,
                "source_id": chunk_id,
                "file_path": file_name
            }
        ],
        "entities": convert_nodes(nodes),
        "relationships": convert_edges(edges)
    }