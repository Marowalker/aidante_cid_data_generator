import constants
import pre_process
from data_managers import CDRDataManager as data_manager
from feature_engineering.deptree.parsers import SpacyParser
from pre_process import opt as pre_opt
from readers import BioCreativeReader
import os
# from bert_embedding import BertEmbedding
# import mxnet as mx
# from collections import defaultdict
# import pickle


print('Start')
pre_config = {
    pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
    pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
}
parser = SpacyParser()
input_path = "data/cdr_data"
output_path = "data/pickle"

# generate data for vocab files

datasets = ['train', 'dev', 'test']
sentence_list = []

for dataset in datasets:
    print('Process dataset: ' + dataset)
    reader = BioCreativeReader(os.path.join(input_path, "cdr_" + dataset + ".txt"))
    raw_documents = reader.read()
    raw_entities = reader.read_entity()
    raw_relations = reader.read_relation()

    title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

    # Pre-process
    title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
    abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)
    documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)
    # documents = data_manager.merge_documents_without_titles(title_doc_objs, abs_doc_objs)

    for doc in documents:
        for sent in doc.sentences:
            sentence_list.append(sent.content)


print("Number of sentences: ", len(sentence_list))

with open("data/all_sentences.txt", "w") as f:
    for sent in sentence_list:
        f.write(sent)
        f.write('\n')

# ctx = mx.gpu(0)
# bert = BertEmbedding(ctx=ctx)
# result = bert(sentence_list)
#
# # print(result[0])
# # print(result[1])
#
# emb_dict = defaultdict()
#
# for tup in result:
#     toks = tup[0]
#     embs = tup[1]
#     for i in range(len(toks)):
#         emb_dict[toks[i]] = embs[i]
#
# with open('bert_embedding.pkl', 'wb') as f:
#     pickle.dump(emb_dict, f)