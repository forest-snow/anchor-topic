import pytest
import anchor.topics
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy

MAX_VOCAB1 = 50
MAX_VOCAB2 = 45
MAX_DF = 10
K = 5
THRESHOLD = 0.1
SEED = 1 
TOP = 5





def get_data(file_, max_vocab):
    path_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_dir, file_), 'r') as f:
        docs = f.read().splitlines()
    cv = CountVectorizer(stop_words=None, max_features=max_vocab,
        max_df=MAX_DF)
    doc_word = cv.fit_transform(doc for doc in docs)
    doc_word_sparse = scipy.sparse.coo_matrix(doc_word)
    word_doc_sparse = doc_word_sparse.T
    word_doc = word_doc_sparse.tocsc()
    vocab = cv.get_feature_names()
    return word_doc, vocab


@pytest.fixture(scope='session')
def toy_doc_en():
    word_doc, vocab = get_data('toy_data_en.txt', MAX_VOCAB1)
    return word_doc

@pytest.fixture(scope='session')
def toy_doc_zh():
    word_doc, vocab = get_data('toy_data_zh.txt', MAX_VOCAB2)
    return word_doc


@pytest.fixture(scope='session')
def toy_dict():
    toy_dict = []
    path_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_dir, 'toy_dict.txt'), 'r') as f:
        entries = f.read().splitlines()
    for entry in entries:
        w1, w2 = entry.split('\t')
        toy_dict.append([int(w2), int(w1)])
    return toy_dict

@pytest.fixture(scope='session')
def toy_model(toy_doc_en):
    A, Q, anchors = anchor.topics.model_topics(toy_doc_en, K, THRESHOLD)
    return A, Q, anchors

@pytest.fixture(scope='session')
def toy_multimodel(toy_doc_en, toy_doc_zh, toy_dict):
    A1, A2, Q1, Q2, anchors1, anchors2 = \
    anchor.topics.model_multi_topics(toy_doc_en, toy_doc_zh, K, THRESHOLD, THRESHOLD, SEED, toy_dict)
    return A1, A2, Q1, Q2, anchors1, anchors2




