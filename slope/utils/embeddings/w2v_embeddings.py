# -*- coding: utf-8 -*-
# Word2Vec embedding utils
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
from typing import List

import numpy as np
import torch
from gensim.models import KeyedVectors
from progress.bar import IncrementalBar

EMBEDDING_DIM = 300


class Word2VecEmbedding:
    """
    Singleton to avoid loading the model mutliple times. Utilizes Word2Vec to generate 300D word
    vectors.
    """
    instance = None

    def __init__(self, model_path: str = '', is_tokenized: bool = False):
        self.embedding_model = self._load_model(model_path, is_tokenized)

    @classmethod
    def inst(cls):
        if cls.instance is None:
            cls.instance = Word2VecEmbedding()
        return cls.instance

    def _load_model(self, model_path: str, is_tokenized: bool):
        # TODO convert relative filepaths into absolute filepaths
        if model_path is not '' and os.path.exists(model_path):
            print(' *\tLoading word embedding model')
            model = KeyedVectors.load(model_path)
            word_vectors = model
        else:
            print(' *\tLoading Google News word vectors...')
            model = KeyedVectors.load_word2vec_format(
                '.././data/GoogleNews-vectors-negative300.bin.gz', binary=True)
            word_vectors = model

        return word_vectors

    def _estimate_embedding(self, surrounding_words: List[str], unknown: str) -> torch.tensor:
        """
        Used to determine an estimated word embedding for a given word. Takes
        average embeddings of the surrounding words.
        """
        # TODO find alternative. Production models using word2vec should not rely on estimations.
        summations = None
        valid_num = 0

        for surr in surrounding_words:
            if surr in self.embedding_model:
                valid_num += 1
                arr = np.asarray(self.embedding_model[surr])
                summations = arr if summations is None else np.add(summations, arr)

        return torch.zeros(EMBEDDING_DIM) if summations is None else torch.tensor(summations / valid_num)

    def _embed_wrd(self, token: str, tokenized: List[str], index: int) -> torch.Tensor:
        if token in self.embedding_model:
            return torch.tensor(self.embedding_model[token])

        # OOB checking
        if index < 3:
            index = 3
        elif index > len(tokenized) - 3:
            index = len(tokenized) - 4

        return self._estimate_embedding(tokenized[index-3:index+3], token)

    def embed_sent(self, tokenized: List[str], verbose: bool = False) -> List[torch.Tensor]:
        """
        Gets embeddings for the input sentences.
        """
        embeddings: List[torch.Tensor] = []
        if verbose:
            bar = IncrementalBar('Generating word embeddings...', max=len(tokenized))
        for i, token in enumerate(tokenized):
            embedding = self._embed_wrd(token, tokenized, i)
            if embedding is not None:
                embeddings.append(embedding)
            if verbose:
                bar.next()

        return embeddings

    @staticmethod
    def avg_embed(embeddings: List[torch.Tensor], start: int, end: int) -> torch.Tensor:
        if len(embeddings) == 0 or len(embeddings) < end:
            return torch.zeros(EMBEDDING_DIM)
        cummu = embeddings[start]
        for embed in embeddings[start:end]:
            cummu += embed
        return cummu / end - start
