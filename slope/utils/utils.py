from typing import List

import spacy

nlp = spacy.load('en_core_web_sm')


def tokenized_spacy_doc(tokenized: List[str]) -> spacy.tokens.doc.Doc:
    """
    Allows for tokenized strings to be parsed and turned into SpaCy docs.
    """
    doc = spacy.tokens.doc.Doc(nlp.vocab, words=tokenized)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    return doc
