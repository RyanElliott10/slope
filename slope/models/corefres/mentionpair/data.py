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

import torch
from progress.bar import IncrementalBar

from slope.utils.embeddings.w2v_embeddings import Word2VecEmbedding
from slope.utils.nnmodel import NNModel
from slope.utils.preco_parser import PreCoFileType, PreCoParser

MentionIndices = List[int]
RawMentionClusters = List[List[MentionIndices]]

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
        ifirst_embed, ilast_embed, iavg = MentionPair._mention_features(i_embeds, self.i_indices)
        jfirst_embed, jlast_embed, javg = MentionPair._mention_features(j_embeds, self.j_indices)
        print(ifirst_embed, ilast_embed, iavg)
        print(jfirst_embed, jlast_embed, javg)

    def _mention_relation_features(self):
        pass

    @staticmethod
    # def _mention_features(embeds: List[torch.Tensor],
    #                       indices: MentionIndices) -> torch.Tensor, torch.Tensor, torch.Tensor:
    def _mention_features(embeds: List[torch.Tensor],
                          indices: MentionIndices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_embed = embeds[indices[1]]
        last_embed = embeds[indices[2]]
        avg = Word2VecEmbedding.avg_embed(embeds)
        return first_embed, last_embed, avg

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
        self.data = None
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
        for i, value in enumerate(clusters[:2]):
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
        # self.data[0].nn_format
        [el.nn_format for el in self.data]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        pass
