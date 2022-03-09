import re

import nltk
from nltk import word_tokenize
from pre_process.models import Tokenizer
from transformers import BertTokenizer


def token_in_tag(token, tags):
    for tag in tags:
        if token in tag:
            return tag
    return None


class CustomBertTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, sent):
        tokens, tag_tokens = CustomBertTokenizer.parse(sent)
        res = []
        for tok in tokens:
            if token_in_tag(tok, tag_tokens):
                res.append(token_in_tag(tok, tag_tokens))
            else:
                res.append((tok, 'NN'))
        return res

    @staticmethod
    def parse(sent):
        # sent = re.sub(r'/', ' / ', sent)
        # sent = re.sub(r']', '] ', sent)
        sent = re.sub(r'\s{2,}', ' ', sent)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(sent)
        tag_tokens = nltk.pos_tag(word_tokenize(sent))
        return tokens, tag_tokens

