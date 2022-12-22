import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext.legacy.data import Field
from tokenize import Tokenizer
tokenizer =  Tokenizer()
def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """

    # include lengths of the source sentences to use pack pad sequence
    kor = Field(tokenize=tokenizer.tokenize_kr,
                    lower=True,
                    batch_first=True)

    eng = Field(tokenize='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data, kor, eng)

    print(f'Build vocabulary using torchtext . . .')

    kor.build_vocab(train_data, max_size=config.kor_vocab)
    eng.build_vocab(train_data, max_size=config.eng_vocab)

    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')

    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    with open('pickles/kor.pickle', 'wb') as kor_file:
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file:
        pickle.dump(eng, eng_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--kor_vocab', type=int, default=55000)
    parser.add_argument('--eng_vocab', type=int, default=30000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
