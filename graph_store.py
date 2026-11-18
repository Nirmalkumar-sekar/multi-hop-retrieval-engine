import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity_id: str, entity_type: str):
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id, type=entity_type)

    def add_relation(self, subject, relation, object_, passage_id, confidence):
        self.graph.add_edge(
            subject,
            object_,
            key=f"{subject}->{relation}->{object_}->{passage_id}",
            relation=relation,
            passage_id=passage_id,
            confidence=confidence
        )

    def paths(self, source, target, max_hops=4):
        if source not in self.graph or target not in self.graph:
            return []

        return list(nx.all_simple_paths(
            self.graph,
            source,
            target,
            cutoff=max_hops
        ))
