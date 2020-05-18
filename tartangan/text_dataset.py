from collections import Counter
import os

import numpy as np
import pandas as pd
from PIL import Image
import smart_open
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import tqdm


class TextDataset(Dataset):
    def __init__(self, df_docs, doc_len=128, transform=None, column='summary',
                 tokenizer='basic_english'):
        super().__init__()
        self.df_docs = df_docs
        self.doc_len = doc_len
        self.column = column
        self.transform = transform
        self.tokenizer = get_tokenizer(tokenizer)
        self.build_vocab()  # put this somewhere else?

    def build_vocab(self):
        tokenized = self.df_docs[self.column].apply(self.tokenizer)
        frequencies = Counter()
        tokenized.apply(frequencies.update)
        self.vocab = Vocab(frequencies, specials=['<unk>', '<pad>'])
        self.doc_indexes = tokenized.map(lambda x: list(map(self.vocab.stoi.get, x)))

    def __getitem__(self, idx):
        indexes = self.doc_indexes.iloc[idx]
        if len(indexes) < self.doc_len:
            indexes = np.pad(indexes, (0, self.doc_len - len(indexes)), 'constant')
        else:
            indexes = np.array(indexes[:self.doc_len])
        return indexes

    def __len__(self):
        return len(self.df_docs)

    @classmethod
    def from_path(cls, path, **cls_kwargs):
        infile = smart_open.open(path, 'rb')
        docs = pd.read_pickle(infile, compression=None)
        return cls(docs, **cls_kwargs)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Create image data from a folder.')
    p.add_argument('source', help='Root path of images')
    p.add_argument('destination', help='Output location of dataset')
    p.add_argument('--resize', help='Width/height of saved images', default=64, type=int)
    p.add_argument('--trunc', default=None, type=int, help='Take only first N samples')
    args = p.parse_args()

    print(f'preparing data from "{args.source}"')
    data = ImageBytesDataset.prepare_data_from_path(
        args.source, transform=transform, trunc=args.trunc
    )
    print(f'saving dataset to "{args.destination}"')
    with smart_open.open(args.destination, 'wb') as outfile:
        np.savez_compressed(outfile, images=data)
