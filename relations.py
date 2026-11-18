import re
import spacy

nlp = spacy.load("en_core_web_sm")

RELATION_PATTERNS = [
    (r"\b(joined|works at|employed by|is an employee of)\b", "EMPLOYED_BY"),
    (r"\b(acquired|bought|merged with)\b", "ACQUIRED"),
    (r"\b(launched|released|introduced)\b", "LAUNCHED"),
    (r"\b(published|authored|wrote)\b", "PUBLISHED"),
    (r"\b(cited|referenced)\b", "CITED"),
    (r"\b(collaborated with|partnered with)\b", "COLLABORATED_WITH"),
    (r"\b(developed|created|built)\b", "DEVELOPED")
]

NON_OBJECT_LABELS = {
    "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "CARDINAL", "ORDINAL"
}

def find_entity_for_token(token, entities):
    token_start = token.idx
    token_end = token.idx + len(token.text)

    for entity in entities:
        if entity["start"] <= token_start and entity["end"] >= token_end:
            return entity

    return None

def dependency_relation(sentence, entities, relation_name):
    doc = nlp(sentence)

    trigger = None

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in {
            "join", "acquire", "launch", "publish",
            "cite", "collaborate", "develop", "create", "build"
        }:
            trigger = token
            break

    if trigger is None:
        return None

    subject = None
    object_ = None

    for child in trigger.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
            candidate = find_entity_for_token(child, entities)

            if candidate and candidate["label"] not in NON_OBJECT_LABELS:
                subject = candidate

        if child.dep_ in {"dobj", "obj", "pobj", "attr", "oprd"}:
            candidate = find_entity_for_token(child, entities)

            if candidate and candidate["label"] not in NON_OBJECT_LABELS:
                object_ = candidate

        # Passive pattern: "The paper was cited by ABC University."
        if child.dep_ == "agent":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    candidate = find_entity_for_token(grandchild, entities)

                    if candidate and candidate["label"] not in NON_OBJECT_LABELS:
                        object_ = candidate

    if subject and object_:
        return {
            "relation": relation_name,
            "subject": subject["normalized"],
            "object": object_["normalized"],
            "confidence": 0.85
        }

    return None

def fallback_relation(sentence, entities, relation_name):
    verb_match = re.search(
        r"\b(joined|acquired|launched|published|cited|collaborated with|developed|created|built)\b",
        sentence.lower()
    )

    if not verb_match:
        return None

    verb_start = verb_match.start()

    before = [
        entity for entity in entities
        if entity["end"] <= verb_start
        and entity["label"] not in NON_OBJECT_LABELS
    ]

    after = [
        entity for entity in entities
        if entity["start"] >= verb_match.end()
        and entity["label"] not in NON_OBJECT_LABELS
    ]

    if not before or not after:
        return None

    return {
        "relation": relation_name,
        "subject": before[-1]["normalized"],
        "object": after[0]["normalized"],
        "confidence": 0.70
    }

def extract_relation(sentence: str, entities):
    if len(entities) < 2:
        return None

    lowered = sentence.lower()
    relation_name = None

    for pattern, relation in RELATION_PATTERNS:
        if re.search(pattern, lowered):
            relation_name = relation
            break

    if relation_name is None:
        return None

    result = dependency_relation(sentence, entities, relation_name)

    if result:
        return result

    return fallback_relation(sentence, entities, relation_name)
