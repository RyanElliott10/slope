from typing import List

import spacy

nlp = spacy.load('en_trf_bertbaseuncased_lg')


def tokenize(s: List[str]) -> str:
    doc = spacy.tokens.doc.Doc(nlp.vocab, words=s)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    return doc


class TextProcessor(object):
    pass


if __name__ == '__main__':
    doc = tokenize(
        ["This", "is", "a", "gift", "that", ",", "if", "we", "'re", "being", "honest", ",", "many", "dads", "want",
         "to", "own", "for", "themselves", "."])
    print(doc._.trf_alignment)  # Alignment between spaCy tokens and wordpieces
    print(doc._.trf_word_pieces_)  # String values of the wordpieces
    print(doc.tensor.shape)
    for tok in doc:
        print(tok.vector.shape, tok)
