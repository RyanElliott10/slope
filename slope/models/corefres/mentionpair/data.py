# -*- coding: utf-8 -*-
# Data modeling for Mention Pair coreference model. Per the model proposed by Kevin Clark in
# Neural Coreference Resolution
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import pprint
from itertools import combinations, product
from typing import List, Tuple

import spacy
import torch
from progress.bar import IncrementalBar

from slope.utils.embeddings.w2v_embeddings import Word2VecEmbedding
from slope.utils.nnmodel import NNModel
from slope.utils.preco_parser import PreCoFileType, PreCoParser
from slope.utils.utils import tokenized_spacy_doc

MentionIndices = List[int]
RawMentionClusters = List[List[MentionIndices]]

nlp = spacy.load('en_core_web_sm')
pp = pprint.PrettyPrinter()


class MentionPair(NNModel):
    """
    Uses word2vec rather than BERT to get word embeddings. Less than state-of-the-art model.
    """

    def __init__(self, i_indices: MentionIndices, j_indices: MentionIndices, doc: List[List[str]], iscoreferent: bool, id: str):
        self.i_indices = i_indices
        self.j_indices = j_indices
        self.doc = doc
        self.iscoreferent = iscoreferent
        self.id = id
        self._sent_embeds = None

    @property
    def sent_indices(self) -> Tuple[int, int]:
        return (self.i_indices[0], self.j_indices[0])

    @property
    def sents(self) -> Tuple[List[str], List[str]]:
        return (self.doc[self.i_indices[0]], self.doc[self.j_indices[0]])

    @property
    def sent_embeds(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self._sent_embeds:
            return self._sent_embeds

        w2v = Word2VecEmbedding.inst()
        self._sent_embeds = (w2v.embed_sent(self.sents[0]), w2v.embed_sent(self.sents[1]))
        return self._sent_embeds

    @property
    def nn_format(self) -> torch.Tensor:
        """
        Returns a PyTorch tensor containing input ready for a neural network. Might consider
        turning this into __call__
        """
        i_embeds, j_embeds = self.sent_embeds
        ifirst_embed, ilast_embed, iavg, dep = MentionPair._mention_features(
            i_embeds, self.i_indices, self.sents[0])
        jfirst_embed, jlast_embed, javg, dep = MentionPair._mention_features(
            j_embeds, self.j_indices, self.sents[1])

        # MentionPair._mention_features(None, self.i_indices, self.sents[0])
        # MentionPair._mention_features(None, self.j_indices, self.sents[1])

    def _mention_relation_features(self):
        pass

    @staticmethod
    def _mention_features(embeds: List[torch.Tensor], indices: MentionIndices, sent: List[str]):
        """
        Return type ommited for brevity.
        """
        first_embed = embeds[indices[1]]
        last_embed = embeds[indices[2]-1]
        avg = Word2VecEmbedding.avg_embed(embeds, indices[1], indices[2])

        doc = tokenized_spacy_doc(sent)
        dep = None
        for chunk in doc.noun_chunks:
            if chunk.start == indices[1] and chunk.end == indices[2]:
                dep = chunk.root.dep_
                break
            elif chunk.start <= indices[1] and chunk.end >= indices[2]:
                dep = chunk.root.dep_

        return first_embed, last_embed, avg, dep

    @staticmethod
    def _context_features(embeds: List[torch.Tensor], indices: MentionIndices):
        pass

    def __str__(self) -> str:
        return f'{self.i_indices} | {self.j_indices} {self.iscoreferent} : {self.id} {self.sents}'

    def __repr__(self) -> str:
        return str(self)


class MentionPairDataLoader(object):
    """
    Splits data from PreCoParser into mention pairs (if training), extracts necessary features, and
    allows iteration.
    """

    def __init__(self, filetype: PreCoFileType, singletons: bool):
        parser = PreCoParser(filetype, singletons=singletons)
        self.parsed_data = parser.data()
        self.data = []
        self.preprocess()

    def preprocess(self):
        """
        Converts parsed datga into training data.
        """
        np_data = self.parsed_data.to_numpy()
        print(len(np_data))

        self.data: List[MentionPair] = []
        bar = IncrementalBar('Creating MentionPairs for training...', max=len(np_data))

        for id, dp in enumerate(np_data):
            self.data.extend(self._build_pairs(dp[2], dp[1], id))
            bar.next()
        print()

        self.batch()

    def _build_pairs(self, clusters: RawMentionClusters, doc: List[List[str]], id: int) -> List[MentionPair]:
        """
        Iterates through all mention clusters for a given datapoint/document and constructs a
        combinatory matrix (of types) to produce true training data.
        """
        combs: List[MentionPair] = []
        for i, value in enumerate(clusters):
            for j, sec in enumerate(clusters[i:]):
                if j == 0:
                    # The "value" itself; coreferents
                    combs.extend([MentionPair(*el, doc, True, MentionPair.make_id(id, i))
                                  for el in list(combinations(value, 2))])
                else:
                    combs.extend([MentionPair(*el, doc, False, MentionPair.make_id(id, i))
                                  for el in list(product(value, sec))])
        return combs

    def batch(self):
        """
        Converts self.data into true training data (i.e. xtrain, ytrain, xtest, ytest)
        """
        self.data[0].nn_format

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        pass
