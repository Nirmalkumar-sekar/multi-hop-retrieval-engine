def resolve_aliases(entities, alias_map):
    resolved = []

    for entity in entities:
        normalized = entity["normalized"]
        resolved_name = alias_map.get(normalized, normalized)

        resolved.append({
            **entity,
            "normalized": resolved_name
        })

    return resolved

def update_alias_map(entities, alias_map):
    concept_names = [
        entity["normalized"]
        for entity in entities
        if entity["label"] == "CONCEPT"
    ]

    if concept_names:
        alias_map["paper"] = concept_names[-1]

    return alias_map
