import re
import spacy

nlp = spacy.load("en_core_web_sm")

ENTITY_LABELS = {
    "PERSON", "ORG", "PRODUCT", "GPE", "DATE",
    "WORK_OF_ART", "EVENT", "FAC", "LAW", "LANGUAGE"
}

NON_ENTITY_LABELS = {
    "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "CARDINAL", "ORDINAL"
}

CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][a-zA-Z]*\.?\s+)+[A-Z][a-zA-Z]*\b"
)

IMPORTANT_NOUNS = {
    "paper", "research", "project", "system",
    "product", "technology", "network", "networks"
}

def normalize_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text

def infer_label(text: str) -> str:
    lowered = text.lower()

    if any(word in lowered for word in [
        "company", "university", "institute",
        "research team", "organization"
    ]):
        return "ORG"

    if any(word in lowered for word in [
        "product", "system", "platform", "software"
    ]):
        return "PRODUCT"

    if any(word in lowered for word in [
        "employee", "dr.", "dr ", "mr.", "ms.", "prof."
    ]):
        return "PERSON"

    if any(word in lowered for word in [
        "paper", "research", "project", "technology"
    ]):
        return "CONCEPT"

    return "ENTITY"

def extract_entities(sentence: str):
    doc = nlp(sentence)
    candidates = []

    for ent in doc.ents:
        candidates.append({
            "text": ent.text,
            "normalized": normalize_entity(ent.text),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    for match in CAPITALIZED_PHRASE.finditer(sentence):
        text = match.group(0).strip()
        start = match.start()
        end = match.end()

        candidates.append({
            "text": text,
            "normalized": normalize_entity(text),
            "label": infer_label(text),
            "start": start,
            "end": end
        })

    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        root = chunk.root.text.lower()

        if root not in IMPORTANT_NOUNS:
            continue

        candidates.append({
            "text": text,
            "normalized": normalize_entity(text),
            "label": infer_label(text),
            "start": chunk.start_char,
            "end": chunk.end_char
        })

    # Keep the longest candidate for each normalized name.
    unique = {}

    for entity in candidates:
        normalized = entity["normalized"]

        if normalized not in unique:
            unique[normalized] = entity
        elif entity["end"] - entity["start"] > unique[normalized]["end"] - unique[normalized]["start"]:
            unique[normalized] = entity

    return list(unique.values())
