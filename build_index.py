import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingest import split_sentences
from src.entities import extract_entities
from src.relations import extract_relation
from src.aliases import resolve_aliases, update_alias_map
from src.graph_store import KnowledgeGraph
from src.vector_store import VectorStore

document_path = Path("data/documents/sample_documents.txt")
text = document_path.read_text(encoding="utf-8")

passages = split_sentences("sample_documents", text)
passage_map = {passage.passage_id: passage for passage in passages}

graph = KnowledgeGraph()
vector_store = VectorStore()
alias_map = {}

for passage in passages:
    entities = extract_entities(passage.text)
    entities = resolve_aliases(entities, alias_map)
    relation = extract_relation(passage.text, entities)

    for entity in entities:
        graph.add_entity(entity["normalized"], entity["label"])

    if relation:
        graph.add_relation(
            relation["subject"],
            relation["relation"],
            relation["object"],
            passage.passage_id,
            relation["confidence"]
        )

    alias_map = update_alias_map(entities, alias_map)

vector_store.add_passages(passages)

print(f"Indexed {len(passages)} passages")
print(f"Graph nodes: {graph.graph.number_of_nodes()}")
print(f"Graph edges: {graph.graph.number_of_edges()}")

print("\nGRAPH EDGES:")
for start, end, key, data in graph.graph.edges(keys=True, data=True):
    print(start, "--", data["relation"], "-->", end, "|", data["passage_id"])
