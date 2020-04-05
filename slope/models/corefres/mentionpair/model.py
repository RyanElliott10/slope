# -*- coding: utf-8 -*-
# Neural network for Mention Pair coreference model.
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import torch.functional as F
import torch.nn as nn

from slope.models.corefres.mentionpair.data import MentionPairDataLoader
from slope.utils.preco_parser import PreCoFileType


class MentionPairResolver(nn.Module):
    def __init__(self, filetype: PreCoFileType, singletons: bool = True):
        self.train_batch = MentionPairDataLoader(filetype, singletons)
        print(len(self.train_batch))

    def forward(self):
        pass


if __name__ == '__main__':
    MentionPairResolver(PreCoFileType.DEV, singletons=False)
