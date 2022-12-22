from underthesea import word_tokenize
import spacy

class Tokenizer:
    def __init__(self):
        self.spacy_jp = spacy.load('ja_core_news_sm')
        self.spacy_kr = spacy.load('ko_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_vi(self,text):
        return word_tokenize(text,format='')

    def tokenize_en(self,text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def tokenize_jp(self, text):
        return [tok.text for tok in self.spacy_jp.tokenizer(text)]

    def tokenize_kr(self, text):
        return [tok.text for tok in self.spacy_kr.tokenizer(text)]
