import scipy.sparse
import numpy
import math
import scipy.stats

def computeQ(word_doc, epsilon=1e-15):
    # print('\ncomputing Q')
    M = scipy.sparse.csc_matrix(word_doc.copy(), dtype=float)
    n_words, n_docs = M.shape

    # word_probs is sum of probabilities of word occurring in all documents 
    word_probs = numpy.zeros(n_words)


    # M.indptr.size-1 is number of columns of M
    # M.data is VD X 1 array of all data
    for doc in range(M.indptr.size-1):
        doc_start = M.indptr[doc]
        doc_end = M.indptr[doc + 1]

        # find word indices for nonzero entries
        word_indices = M.indices[doc_start: doc_end]

        # get count of words in document j and compute norm
        words_per_doc = numpy.sum(M.data[doc_start: doc_end])
        norm = words_per_doc * (words_per_doc - 1)

        if norm == 0: 
            norm = 1

        word_probs[word_indices] += M.data[doc_start: doc_end]/norm
        M.data[doc_start: doc_end] = M.data[doc_start: doc_end]/numpy.sqrt(norm)

    

    Q = (M * M.T)/n_docs
    Q = Q.todense()
    Q = numpy.array(Q, copy=False)

    word_probs = word_probs/n_docs
    Q = Q - numpy.diag(word_probs)

    # handle precision errors
    Q[(-epsilon < Q) & (Q < epsilon)] = 0

    return Q

def augmentQ(Q, anchor_facets, epsilon=1e-10):
    psuedoQ = []
    for facet in anchor_facets:
        psuedoword = scipy.stats.hmean(Q[facet, :] + epsilon, axis=0)
        psuedoQ.append(psuedoword)

    Q_new = numpy.vstack((Q, numpy.array(psuedoQ)))
    return Q_new