from . import search, cooccur, recover

import numpy 
import scipy.sparse
import multiprocessing.pool


def flatten_list(lst):
    flat_lst = [item for sublist in lst for item in sublist]
    return flat_lst  

class MonoModel:
    def __init__(self, M, k, threshold, seed, vocab=None):
        assert type(M) == scipy.sparse.csc_matrix, \
            'word-document matrix must be scipy.sparse.csc_matrix'
        self.word_doc = M
        self.cooccur = cooccur.computeQ(M)
        self.n_topics = k
        self.seed = seed
        self.anchors = None
        self.word_topic = None
        self.doc_threshold = int(M.shape[1] * threshold) 
        self.vocab = vocab

    def identify_candidates(self):
        n_words = self.word_doc.shape[0]
        candidates = []

        def add_candidate(candidates, w):
            # number of documents that word w occurs in 
            docs_per_word = self.word_doc[w,:].count_nonzero() 
            if docs_per_word >= self.doc_threshold:
                candidates.append(w)

        worker = lambda w: add_candidate(candidates, w)
        chunksize = 5000
        with multiprocessing.pool.ThreadPool() as pool:
            pool.map(worker, range(n_words), chunksize)
        return numpy.array(candidates)

    def find_anchors(self):
        candidates = self.identify_candidates()
        anchors = search.greedy_anchors(self, candidates)
        self.anchors = [[w] for w in anchors]

    def update_topics(self, anchors=None):
        # first update: single anchor word for each topic
        if anchors is None:
            anchors = flatten_list(self.anchors)
            self.word_topic = recover.computeA(self.cooccur, anchors)

        # later updates: multiword anchors are allowed
        else:
            self.anchors = anchors
            self.n_topics = len(anchors)
            n_words = self.cooccur.shape[0]
            Q = cooccur.augmentQ(self.cooccur, self.anchors)
            psuedo_anchors = range(n_words, n_words + self.n_topics)
            self.word_topic = recover.computeA(Q, psuedo_anchors)[:n_words]

    def print_anchors(self):
        vocab = self.vocab
        for topic in self.anchors:
            for word in topic:
                print(vocab[word])

        


