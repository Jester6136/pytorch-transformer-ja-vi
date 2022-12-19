import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext.legacy.data import Field
from underthesea import word_tokenize
import spacy

def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """
    spacy_jp = spacy.load('ja_core_news_sm')

    def tokenize_vi(self,text):
        return word_tokenize(text,format='')

    def tokenize_jp(self, text):
        return [tok.text for tok in self.spacy_jp.tokenizer(text)]

    # include lengths of the source sentences to use pack pad sequence
    source = Field(tokenize=tokenize_jp,
                    lower=True,
                    batch_first=True)

    target = Field(tokenize=tokenize_vi,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data, source, target)

    print(f'Build vocabulary using torchtext . . .')

    source.build_vocab(train_data, max_size=config.kor_vocab)
    target.build_vocab(train_data, max_size=config.eng_vocab)

    print(f'Unique tokens in Source vocabulary: {len(source.vocab)}')
    print(f'Unique tokens in Target vocabulary: {len(target.vocab)}')

    print(f'Most commonly used Source words are as follows:')
    print(source.vocab.freqs.most_common(20))

    print(f'Most commonly used Target words are as follows:')
    print(target.vocab.freqs.most_common(20))

    with open('pickles/source.pickle', 'wb') as kor_file:
        pickle.dump(source, kor_file)

    with open('pickles/target.pickle', 'wb') as eng_file:
        pickle.dump(target, eng_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--source_vocab', type=int, default=55000)
    parser.add_argument('--target_vocab', type=int, default=30000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
