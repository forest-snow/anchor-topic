from . import search, cooccur, recover

import numpy 
import scipy.sparse
import multiprocessing.pool


def flatten_list(lst):
    flat_lst = [item for sublist in lst for item in sublist]
    return flat_lst

def convert_2dlist(lst, index):
    new_lst = []
    for row in lst:
        new_row = []
        for entry in row:
            new_row.append(index[entry])
        new_lst.append(new_row)
    return new_lst

def print_2dlist(lst):
    for row in lst:
        string = ' '.join(row)
        print(string)

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

    def passes_threshold(self, w):
        docs_per_word = self.word_doc[w, :].count_nonzero()
        return docs_per_word >= self.doc_threshold

    def identify_candidates(self):
        n_words = self.word_doc.shape[0]
        candidates = []

        def add_candidate(candidates, w):
            # number of documents that word w occurs in 
            if self.passes_threshold(w):
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

    def get_anchors(self, word=False, show=False):
        if not word:
            return self.anchors 
        else:
            anchors = convert_2dlist(self.anchors, self.vocab)
            if show:
                print('Printing anchors')
                print_2dlist(anchors)
                print('\n')
            return anchors

    def get_top_topic_words(self, n, word=False, show=False):
        """Return top [n] words for every topic with information about probability
        distribution provided by [self.word_topic]
        """  
        assert n <= self.word_topic.shape[0], \
            'Number of words requested greater than model\'s number of words'
        assert self.word_topic is not None, \
            'Word-topic is None. Model may not have been built.'
        topic_words = self.word_topic.T

        # sort words based on probabilities
        sort_words = numpy.argsort(topic_words, axis=1)
        # reverse so that the higher probabilities come first
        rev_words = numpy.flip(sort_words, axis=1)
        # retrieve top n words
        top_words = rev_words[:,:n]

        if not word:
            return top_words
        else:
            top_words = convert_2dlist(top_words, self.vocab)
            if show:
                print('Printing top n words')
                print_2dlist(top_words)
                print('\n')
            return top_words

        
class MultiModel:
    def __init__(self, M1, M2, k, threshold1, threshold2, \
        seed, dictionary, vocab1=None, vocab2=None):
        self.model1 = MonoModel(M1, k, threshold1, seed, vocab1)
        self.model2 = MonoModel(M2, k, threshold2, seed, vocab2)
        self.dictionary = dictionary
        self.seed = seed 
        self.n_topics = k

    def identify_candidates(self):
        candidates = []
        for w1, w2 in self.dictionary:
            if self.model1.passes_threshold(w1) and \
            self.model2.passes_threshold(w2):
                candidates.append([w1, w2])
        return numpy.array(candidates)

    def find_anchors(self):
        candidates = self.identify_candidates()
        anchors1, anchors2 = search.greedy_linked_anchors(self, candidates)
        self.model1.anchors = [[w] for w in anchors1]
        self.model2.anchors = [[w] for w in anchors2] 

    def update_topics(self, anchors1=None, anchors2=None):
        self.model1.update_topics(anchors1)
        self.model2.update_topics(anchors2)
        
    def get_anchors(self, word=False, show=False):
        anchors1 = self.model1.get_anchors(word, show)
        anchors2 = self.model2.get_anchors(word, show)
        return anchors1, anchors2

    def get_top_topic_words(self, n, word=False, show=False):
        top_words1 = self.model1.get_top_topic_words(n, word, show)
        top_words2 = self.model2.get_top_topic_words(n, word, show)
        return top_words1, top_words2

