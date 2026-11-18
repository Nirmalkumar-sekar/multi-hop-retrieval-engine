import re
from src.entities import extract_entities, normalize_entity

ANSWER_TYPE_HINTS = {
    "which product": "PRODUCT",
    "which organization": "ORG",
    "which company": "ORG",
    "which university": "ORG",
    "which institute": "ORG",
    "which person": "PERSON",
    "who": "PERSON"
}

REQUIRED_RELATIONS = {
    "developed": "DEVELOPED",
    "created": "DEVELOPED",
    "built": "DEVELOPED",
    "launched": "LAUNCHED",
    "released": "LAUNCHED",
    "acquired": "ACQUIRED",
    "employed": "EMPLOYED_BY",
    "cited": "CITED",
    "referenced": "CITED",
    "collaborated": "COLLABORATED_WITH",
    "published": "PUBLISHED"
}

ANSWER_ROLE_PATTERNS = [
    (r"which (organization|company|university|institute|person) (developed|created|built|acquired|published|cited|referenced|collaborated)", "subject"),
    (r"which (product|technology|system) (was )?(launched|released|developed|created|built)", "object"),
    (r"which (product|technology|system) (did|was)", "object")
]

class RetrievalResult:
    def __init__(self, answer, confidence, evidence, path, status):
        self.answer = answer
        self.confidence = confidence
        self.evidence = evidence
        self.path = path
        self.status = status

class QueryEngine:
    def __init__(self, graph, vector_store, passages):
        self.graph = graph
        self.vector_store = vector_store
        self.passages = passages

    def expected_answer_type(self, query):
        query_lower = query.lower()

        for phrase, entity_type in ANSWER_TYPE_HINTS.items():
            if phrase in query_lower:
                return entity_type

        return None

    def required_relations(self, query):
        query_lower = query.lower()
        required = []

        for phrase, relation in REQUIRED_RELATIONS.items():
            if phrase in query_lower and relation not in required:
                required.append(relation)

        return required

    def answer_role(self, query):
        query_lower = query.lower()

        for pattern, role in ANSWER_ROLE_PATTERNS:
            if re.search(pattern, query_lower):
                return role

        return "terminal"

    def link_entities(self, query):
        graph_nodes = list(self.graph.graph.nodes)
        linked = []

        for entity in extract_entities(query):
            normalized = normalize_entity(entity["text"])

            if normalized in graph_nodes and normalized not in linked:
                linked.append(normalized)

        query_lower = query.lower()

        for node in graph_nodes:
            if node in query_lower and node not in linked:
                linked.append(node)

        return linked

    def path_edges(self, path):
        edges = []

        for start, end in zip(path, path[1:]):
            edge_data = self.graph.graph.get_edge_data(start, end)

            if edge_data:
                first_edge = next(iter(edge_data.values()))
                edges.append({
                    "subject": start,
                    "relation": first_edge["relation"],
                    "object": end,
                    "passage_id": first_edge["passage_id"],
                    "confidence": first_edge["confidence"]
                })

        return edges

    def select_answer(self, edges, query, answer_role):
        required_relations = self.required_relations(query)

        # Prefer the last edge that satisfies an explicitly requested relation.
        matching_edges = [
            edge for edge in edges
            if edge["relation"] in required_relations
        ]

        if not matching_edges:
            matching_edges = edges

        edge = matching_edges[-1]

        if answer_role == "subject":
            return edge["subject"]
        if answer_role == "object":
            return edge["object"]

        return edges[-1]["object"]

    def rank_paths(self, paths, query):
        query_results = self.vector_store.search(query, top_k=20)
        query_score = query_results[0]["score"] if query_results else 0.0

        expected_type = self.expected_answer_type(query)
        required_relations = self.required_relations(query)
        answer_role = self.answer_role(query)

        ranked = []

        for path in paths:
            edges = self.path_edges(path)

            if not edges:
                continue

            path_relations = {edge["relation"] for edge in edges}

            if required_relations and not set(required_relations).issubset(path_relations):
                continue

            answer = self.select_answer(edges, query, answer_role)
            answer_type = self.graph.graph.nodes[answer].get("type")

            if expected_type and answer_type != expected_type:
                continue

            passage_ids = [edge["passage_id"] for edge in edges]
            evidence = [
                self.passages[passage_id]
                for passage_id in passage_ids
                if passage_id in self.passages
            ]

            score = (
                0.40 * query_score
                + 0.20 * sum(edge["confidence"] for edge in edges) / len(edges)
                - 0.10 * len(path)
                + 0.15 * len(evidence) / len(edges)
                + 0.25
            )

            ranked.append({
                "answer": answer,
                "score": score,
                "relations": edges,
                "passages": evidence
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def answer(self, query, max_hops=4):
        linked_entities = self.link_entities(query)

        if not linked_entities:
            return RetrievalResult(
                answer=None,
                confidence=0.0,
                evidence=self.vector_store.search(query, top_k=5),
                path=[],
                status="AMBIGUOUS_QUERY"
            )

        paths = []

        for source in linked_entities:
            for target in self.graph.graph.nodes:
                if target != source:
                    paths.extend(
                        self.graph.paths(source, target, max_hops)
                    )

        ranked_paths = self.rank_paths(paths, query)

        if not ranked_paths:
            return RetrievalResult(
                answer=None,
                confidence=0.0,
                evidence=self.vector_store.search(query, top_k=5),
                path=[],
                status="INSUFFICIENT_EVIDENCE"
            )

        best = ranked_paths[0]

        return RetrievalResult(
            answer=best["answer"],
            confidence=best["score"],
            evidence=best["passages"],
            path=best["relations"],
            status="ANSWERED"
        )
