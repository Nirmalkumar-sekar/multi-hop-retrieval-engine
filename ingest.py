import hashlib
import re
import spacy
from dataclasses import dataclass

sentence_nlp = spacy.blank("en")
sentence_nlp.add_pipe("sentencizer")

@dataclass
class Passage:
    passage_id: str
    document_id: str
    text: str
    paragraph_id: int
    sentence_id: int

def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

def split_sentences(document_id: str, text: str):
    doc = sentence_nlp(text)
    passages = []

    for index, sentence in enumerate(doc.sents):
        cleaned = sentence.text.strip()

        if cleaned:
            passages.append(Passage(
                passage_id=f"{document_id}:{index}",
                document_id=document_id,
                text=cleaned,
                paragraph_id=0,
                sentence_id=index
            ))

    return passages
