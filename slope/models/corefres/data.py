import torch
from transformers import BertModel, BertTokenizer

PRETRAINED_WEIGHTS = 'bert-base-uncased'


class DataLoader(object):
    """
    Coreference resolution data loading class. Accepts raw data from Parsing classes.
    """

    def __init__(self, data=''):
        self.bert_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
        self.bert_model = BertModel.from_pretrained(PRETRAINED_WEIGHTS)

    @classmethod
    def from_preco(cls, raw_data):
        """
        Parses raw PreCo data into the common format.
        """
        return cls()

    @classmethod
    def from_conll(cls, raw_data):
        """
        Parses raw CoNLL data into the common format.
        """
        return cls()

    def _encode(self, doc: str) -> torch.Tensor:
        return self.bert_tokenizer.encode(doc, add_special_tokens=True, return_tensors='pt')

    def tokenize(self, doc: str) -> torch.Tensor:
        """
        Accepts a doc (non-tokenized string) and returns the word encodings.
        """
        # TODO make this more efficient by batching the encodings and predictions
        input_ids = self._encode(doc)
        with torch.no_grad():
            last_hidden_states = self.bert_model(input_ids)[0]  # Models outputs are now tuples

        return last_hidden_states


if __name__ == '__main__':
    d = DataLoader.from_preco({})

    for i in range(100):
        embeddings = d.tokenize('Hello, this is just a string minding it\'s own business')
        print(embeddings.shape)
