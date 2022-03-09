import re

import nltk
from nltk import word_tokenize
from pre_process.models import Tokenizer


class NLTKTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, sent):
        tokens = NLTKTokenizer.parse(sent)
        return nltk.pos_tag(tokens)

    @staticmethod
    def parse(sent):
        # sent = re.sub(r'/', ' / ', sent)
        # sent = re.sub(r']', '] ', sent)
        sent = re.sub(r'\s{2,}', ' ', sent)

        tokens = word_tokenize(sent)
        return tokens
