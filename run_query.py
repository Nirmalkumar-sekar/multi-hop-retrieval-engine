import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingest import split_sentences
from src.entities import extract_entities
from src.relations import extract_relation
from src.aliases import resolve_aliases, update_alias_map
from src.graph_store import KnowledgeGraph
from src.vector_store import VectorStore
from src.query_engine import QueryEngine

query = " ".join(sys.argv[1:]) or "Which product was launched by the company acquired by the organization that employed Employee X?"

text = Path("data/documents/sample_documents.txt").read_text(encoding="utf-8")
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

engine = QueryEngine(graph, vector_store, passage_map)
result = engine.answer(query)

print("STATUS:", result.status)
print("ANSWER:", result.answer)
print("CONFIDENCE:", round(result.confidence, 4))

print("\nRELATIONSHIP CHAIN:")
for edge in result.path:
    print(f"{edge['subject']} --[{edge['relation']}]--> {edge['object']}")

print("\nSUPPORTING EVIDENCE:")

if result.evidence and isinstance(result.evidence[0], dict):
    for item in result.evidence:
        passage = passage_map.get(item["passage_id"])

        if passage:
            print(f"[{passage.passage_id}] {passage.text}")
else:
    for passage in result.evidence:
        print(f"[{passage.passage_id}] {passage.text}")
