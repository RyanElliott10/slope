from itertools import combinations, product
import pprint
from typing import List, Tuple

import numpy as np

from slope.utils.preco_parser import PreCoFileType, PreCoParser

MentionIndices = List[int]
RawMentionClusters = List[List[MentionIndices]]

pp = pprint.PrettyPrinter()


class MentionPair(object):
    def __init__(self, i_indices: MentionIndices, j_indices: MentionIndices, iscoreferent: bool, id: str):
        self.i_indices = i_indices
        self.j_indices = j_indices
        self.iscoreferent = iscoreferent
        self.id = id

    @property
    def sent_indices(self) -> Tuple[int, int]:
        return (self.i_indices[0], self.j_indices[0])

    def __str__(self) -> str:
        return f'{self.i_indices} | {self.j_indices} {self.iscoreferent} : {self.id}'

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def make_id(major: int, minor: int) -> str:
        return f'{major}.{minor}'


class MentionPairDataLoader(object):
    '''
    Splits data from PreCoParser into mention pairs (if training), extracts necessary features, and
    allows iteration.
    '''

    def __init__(self, filetype: PreCoFileType, singletons: bool):
        parser = PreCoParser(filetype, singletons=singletons)
        self.parsed_data = parser.data()
        self.data = self.preprocess()

    def preprocess(self) -> List[MentionPair]:
        '''
        Converts parsed datga into training data.
        '''
        np_data = self.parsed_data.to_numpy()

        combs: List[MentionPair] = []
        for id, dp in enumerate(np_data[1:2]):
            pp.pprint(dp[2])
            combs.extend(self._build_pairs(dp[2], id))

        return combs

    def _build_pairs(self, clusters: RawMentionClusters, id: int) -> List[MentionPair]:
        '''
        Iterates through all mention clusters for a given datapoint/document and constructs a
        combinatory matrix (of types) to produce true training data.
        '''
        combs: List[MentionPair] = []
        for i, value in enumerate(clusters[:2]):
            for j, sec in enumerate(clusters[i:]):
                if j == 0:
                    # The "value" itself; coreferents
                    combs.extend([MentionPair(*el, True, MentionPair.make_id(id, i))
                                  for el in list(combinations(value, 2))])
                else:
                    combs.extend([MentionPair(*el, False, MentionPair.make_id(id, i))
                                  for el in list(product(value, sec))])
        return combs

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        pass
