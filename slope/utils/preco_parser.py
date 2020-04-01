from enum import Enum

import pandas as pd


class PreCoFileType(Enum):
    DEV = 'dev.json'
    TRAIN = 'train.json'


class PrecoParser(object):
    def __init__(self, file_type: PreCoFileType, singletons: bool = True):
        self.file_type = file_type.value
        self.singletons = singletons

    def data(self):
        df = pd.read_json(self.filepath, lines=True, encoding='ascii')
        print(df.head())

    @property
    def filepath(self):
        return f'../data/preco/{self.file_type}'


if __name__ == '__main__':
    parser = PrecoParser(PreCoFileType.DEV)
    parser.data()
