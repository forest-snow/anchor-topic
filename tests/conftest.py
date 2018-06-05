import pytest
import anchor
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy

MAX_VOCAB = 50
MAX_DF = 6
K = 5
THRESHOLD = 0.1
SEED = 1 





def get_data(file_):
    path_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_dir, file_), 'r') as f:
        docs = f.read().splitlines()
    cv = CountVectorizer(stop_words=None, max_features=MAX_VOCAB,
        max_df=MAX_DF)
    doc_word = cv.fit_transform(doc for doc in docs)
    doc_word_sparse = scipy.sparse.coo_matrix(doc_word)
    word_doc_sparse = doc_word_sparse.T
    word_doc = word_doc_sparse.tocsc()
    vocab = cv.get_feature_names()
    return word_doc, vocab


@pytest.fixture(scope='session')
def toy_model():
    word_doc, vocab = get_data('toy_data_en.txt')
    model = anchor.MonoModel(word_doc, K, THRESHOLD, SEED, vocab)
    return model

@pytest.fixture(scope='session')
def toy_dict():
    toy_dict = []
    path_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_dir, 'toy_dict.txt'), 'r') as f:
        entries = f.read().splitlines()
    for entry in entries:
        w1, w2 = entry.split('\t')
        toy_dict.append([int(w1), int(w2)])
    return toy_dict


@pytest.fixture(scope='session')
def toy_multimodel(toy_dict):
    word_doc_en, vocab_en = get_data('toy_data_en.txt')
    word_doc_zh, vocab_zh = get_data('toy_data_zh.txt')
    model = anchor.MultiModel(word_doc_en, word_doc_zh, K, THRESHOLD, THRESHOLD, SEED, toy_dict, vocab_en, vocab_zh)
    return model





