from enum import Enum

import pandas as pd

from slope.utils.setter import setter


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

    def data(self):
        return self.df

    @setter
    def file_type(self, f: PreCoFileType):
        vars(self)['file_type'] = f.value
        self.df = pd.read_json(self.filepath, lines=True, encoding='ascii')

    @property
    def filepath(self):
        return f'../data/preco/{self.file_type}'


if __name__ == '__main__':
    parser = PrecoParser(PreCoFileType.DEV)
    print(parser.data().shape)
    parser.file_type = PreCoFileType.TRAIN
    print(parser.data().shape)
