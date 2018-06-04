import pytest
import anchor
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy

MAX_VOCAB = 50
MAX_DF = 10
K = 5
THRESHOLD = 0.1
SEED = 1 

@pytest.fixture(scope='session')
def toy_model():
    path_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_dir,'toy_data.txt'), 'r') as f:
        docs = f.read().splitlines()
    cv = CountVectorizer(stop_words='english', max_features=MAX_VOCAB,
        max_df=MAX_DF)
    doc_word = cv.fit_transform(doc for doc in docs)
    doc_word_sparse = scipy.sparse.coo_matrix(doc_word)
    word_doc_sparse = doc_word_sparse.T
    word_doc = word_doc_sparse.tocsc()
    vocab = cv.get_feature_names()
    model = anchor.MonoModel(word_doc, K, THRESHOLD, SEED, vocab)
    return model