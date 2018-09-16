from . import search, cooccur, recover

import numpy 
import scipy.sparse
import multiprocessing.pool

# Functions for preprocessing lists #

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

# Helper functions to find candidates #


def identify_candidates(M, doc_threshold):
    """Identify anchor candidates given word-document matrix [M].
    Candidates must appear in at least [doc_threshold] number of documents.
    """
    n_words = M.shape[0]
    candidates = []

    def add_candidate(candidates, w):
        docs_per_word = M[w, :].count_nonzero()        
        if docs_per_word >= doc_threshold:
            candidates.append(w)

    worker = lambda w: add_candidate(candidates, w)
    chunksize = 5000
    with multiprocessing.pool.ThreadPool() as pool:
        pool.map(worker, range(n_words), chunksize)
    return numpy.array(candidates)

def identify_linked_candidates(M1, M2, dictionary, doc_threshold1, doc_threshold2):
    """Identify anchor candidates in two languages given first word-document 
    matrix [M1], second word-document matrix [M2], and [dictionary] which maps
    features from [M1] to features to [M2] (does not need to be 1-to-1).

    Returns numpy array [candidates] of linked anchor words for each language.
    """
    candidates = []
    for w1, w2 in dictionary:
        docs_per_word1 = M1[w1, :].count_nonzero()
        docs_per_word2 = M2[w2, :].count_nonzero()
        if docs_per_word1 >= doc_threshold1 and docs_per_word2 >= doc_threshold2:
            candidates.append([w1, w2])
    return numpy.array(candidates)

# Functions for anchor-based topic modeling 

def model_topics(M, k, threshold, seed=1):
    """Model [k] topics of corpus using anchoring algorithm (Arora et al., 2013). 
    Corpus represented as word-document matrix [M] of type scipy.sparse.csc_matrix. 
    [Threshold] is minimum percentage of document occurrences for word to be anchor candidate. 

    Returns word-topic matrix [A], word-coocurrence matrix [Q], and 
    int list list [anchors]. These can be used to further update model.
    """

    Q = cooccur.computeQ(M)
    doc_threshold = int(M.shape[1] * threshold)

    # identify candidates
    candidates = identify_candidates(M, doc_threshold)

    # find anchors
    anchors = search.greedy_anchors(Q, k, candidates, seed)

    # recover topics
    A = recover.computeA(Q, anchors)
    anchors = [[w] for w in anchors]

    return A, Q, anchors


def model_multi_topics(M1, M2, k, threshold1, threshold2, \
    dictionary, seed=1):    
    """Model [k] topics of corpus using multi-anchoring algorithm (Yuan et al., 2018). 
    Each corpus represented as word-document matrix [M] of type scipy.sparse.csc_matrix. 
    [Threshold] is minimum percentage of document occurrences for word to be anchor candidate. 
    [Dictionary] maps features from [M1] to features to [M2] (does not need to be 1-to-1) in 
    list or array format.

    For each corpus, returns word-topic matrix [A], word-coocurrence matrix [Q], 
    and int list list [anchors]. These can be used to further update model.
    """
    Q1 = cooccur.computeQ(M1)
    Q2 = cooccur.computeQ(M2)
    doc_threshold1 = int(M1.shape[1] * threshold1)
    doc_threshold2 = int(M2.shape[1] * threshold2)
    candidates = identify_linked_candidates(M1, M2, dictionary, doc_threshold1, doc_threshold2)
    anchors1, anchors2 = search.greedy_linked_anchors(Q1, Q2, k, candidates, seed)
    A1 = recover.computeA(Q1, anchors1)
    A2 = recover.computeA(Q2, anchors2)
    anchors1 = [[w] for w in anchors1]
    anchors2 = [[w] for w in anchors2]

    return A1, A2, Q1, Q2, anchors1, anchors2


def update_topics(Q, anchors):
    """Update topics given [anchors] and word co-occurrence matrix [Q].
    For each topic, multiple anchors can be given (Lund et al., 2017). 
    [Anchors] in the form of int list list.
    
    Returns updated word-topic matrix [A].
    """
    n_topics = len(anchors)
    n_words = Q.shape[0]
    Q_aug = cooccur.augmentQ(Q, anchors)
    pseudo_anchors = range(n_words, n_words + n_topics)
    A = recover.computeA(Q_aug, pseudo_anchors)[:n_words]
    return A

