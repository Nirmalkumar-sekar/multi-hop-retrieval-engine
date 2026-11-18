import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingest import split_sentences
from src.entities import extract_entities
from src.relations import extract_relation
from src.aliases import resolve_aliases, update_alias_map
from src.graph_store import KnowledgeGraph
from src.vector_store import VectorStore
from src.query_engine import QueryEngine

def build_engine():
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

    return QueryEngine(graph, vector_store, passage_map), passage_map

def normalize_answer(answer):
    return answer.strip().lower() if answer else ""

def evaluate():
    engine, passage_map = build_engine()

    dataset_path = Path("data/evaluation_dataset.json")
    test_cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    total = len(test_cases)
    correct_answers = 0
    evidence_recall_total = 0.0
    evidence_precision_total = 0.0
    reciprocal_ranks = []
    latencies = []

    print("EVALUATION RESULTS")
    print("=" * 60)

    for index, case in enumerate(test_cases, start=1):
        query = case["query"]
        expected_answer = normalize_answer(case["expected_answer"])
        expected_passages = set(case["expected_passages"])

        start_time = time.time()
        result = engine.answer(query)
        latency = time.time() - start_time

        predicted_answer = normalize_answer(result.answer)
        is_correct = predicted_answer == expected_answer

        retrieved_passage_ids = set()

        for item in result.evidence:
            if isinstance(item, dict):
                retrieved_passage_ids.add(item["passage_id"])
            else:
                retrieved_passage_ids.add(item.passage_id)

        relevant_retrieved = retrieved_passage_ids.intersection(expected_passages)

        evidence_recall = (
            len(relevant_retrieved) / len(expected_passages)
            if expected_passages else 0.0
        )

        evidence_precision = (
            len(relevant_retrieved) / len(retrieved_passage_ids)
            if retrieved_passage_ids else 0.0
        )

        reciprocal_rank = 1.0 if is_correct else 0.0

        correct_answers += int(is_correct)
        evidence_recall_total += evidence_recall
        evidence_precision_total += evidence_precision
        reciprocal_ranks.append(reciprocal_rank)
        latencies.append(latency)

        print(f"\nQuery {index}: {query}")
        print(f"Expected answer: {expected_answer}")
        print(f"Predicted answer: {predicted_answer}")
        print(f"Answer correct: {is_correct}")
        print(f"Status: {result.status}")
        print(f"Evidence recall: {evidence_recall:.2f}")
        print(f"Evidence precision: {evidence_precision:.2f}")
        print(f"Latency: {latency:.3f} seconds")

        print("Relationship chain:")
        for edge in result.path:
            print(
                f"  {edge['subject']} --[{edge['relation']}]--> {edge['object']}"
            )

    answer_accuracy = correct_answers / total
    mean_evidence_recall = evidence_recall_total / total
    mean_evidence_precision = evidence_precision_total / total
    mrr = sum(reciprocal_ranks) / total
    average_latency = sum(latencies) / total

    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    print(f"Total queries: {total}")
    print(f"Answer accuracy: {answer_accuracy:.2f}")
    print(f"Mean evidence recall: {mean_evidence_recall:.2f}")
    print(f"Mean evidence precision: {mean_evidence_precision:.2f}")
    print(f"Mean reciprocal rank (MRR): {mrr:.2f}")
    print(f"Average latency: {average_latency:.3f} seconds")

if __name__ == "__main__":
    evaluate()
