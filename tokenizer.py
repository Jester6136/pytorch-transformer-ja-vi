from underthesea import word_tokenize
import spacy

class Tokenizer:
    def __init__(self):
        self.spacy_jp = spacy.load('ja_core_news_sm')

    def tokenize_vi(self,text):
        return word_tokenize(text,format='')

    def tokenize_jp(self, text):
        return [tok.text for tok in self.spacy_jp.tokenizer(text)]