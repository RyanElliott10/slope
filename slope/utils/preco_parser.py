from enum import Enum

import pandas as pd

from slope.utils.decorators import setter


class PreCoFileType(Enum):
    DEV = 'dev.json'
    TRAIN = 'train.json'


class PrecoParser(object):
    '''
    Parses data from PreCo formatted files. Allows filtration of mentions on the number of referents
    are associated with a given mention.
    '''

    def __init__(self, file_type: PreCoFileType, singletons: bool = True):
        self.file_type = file_type
        self.singletons = singletons
        self.df = pd.read_json(self.filepath, lines=True, encoding='ascii')

    def data(self) -> pd.DataFrame:
        if not self.singletons:
            self._filter_singleton()
        return self.df

    def _filter_singleton(self):
        filtered = list()
        for mentions in self.df.mention_clusters:
            filtered.append([clusters for clusters in mentions if len(clusters) > 1])

        self.df.mention_clusters = filtered

    def debug_ents(self, num: int = None, show_sent: bool = False):
        ''' Prints clusters of `num` datapoints. '''
        sents = self.df.sentences
        mention_clusters = self.df.mention_clusters
        for i, sent in enumerate(sents[:num]):
            if show_sent:
                print(sent)
            print('********\tClusters\t********\n')
            ent = list()
            for cluster in mention_clusters[i]:
                ent = [' '.join(sent[sent_idx][start:end]) for sent_idx, start, end in cluster]
                print(ent)

    @setter
    def file_type(self, f: PreCoFileType):
        vars(self)['file_type'] = f.value
        self.df = pd.read_json(self.filepath, lines=True, encoding='ascii')

    @property
    def filepath(self) -> str:
        return f'../data/preco/{self.file_type}'


if __name__ == '__main__':
    parser = PrecoParser(PreCoFileType.DEV, singletons=False)
    data = parser.data()
    print(data.columns)
    parser.debug_ents(num=1, show_sent=False)
