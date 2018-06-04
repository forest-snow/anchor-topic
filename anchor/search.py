import numpy 
import scipy.sparse
import math
from . import projection

def row_normalize(Q):
    Q_new = Q.copy()
    row_sums = Q_new.sum(1)
    for i in range(Q_new.shape[0]):
        # numpy.seterr(divide='ignore', invalid='ignore')
        if row_sums[i] != 0:
            Q_new[i, :] = Q_new[i, :]/float(row_sums[i])
    return Q_new

def basis_vector(Q, i):
    return Q[i]/numpy.sqrt(numpy.dot(Q[i], Q[i].T))

def greedy_anchors(model, candidates, project_dim=1000):
    Q_bar = row_normalize(model.cooccur)
    Q_red = projection.random_projection(Q_bar, project_dim, model.seed)

    anchors = numpy.zeros(model.n_topics, dtype=numpy.int)
    dim = Q_red.shape[1]
    basis = numpy.zeros((model.n_topics - 1, dim))

    # find p1 with farthest distance from origin
    max_dist = 0
    for w in candidates:
        dist = numpy.dot(Q_red[w], Q_red[w].T)
        if dist > max_dist:
            max_dist = dist
            anchors[0] = w

    # let p1 be origin of both coordinate systems
    for w in candidates:
        Q_red[w] = Q_red[w] - Q_red[anchors[0]]

    # find farthest point from p1 
    max_dist = 0
    for w in candidates:
        dist = numpy.dot(Q_red[w], Q_red[w].T)
        if dist > max_dist:
            max_dist = dist
            anchors[1] = w

    basis[0] = basis_vector(Q_red, anchors[1])



    # stabilized gram-schmidt which finds new anchor words to expand our subspace
    for j in range(1, model.n_topics - 1):
        max_dist = 0
        for w in candidates:
            Q_red[w] = Q_red[w] - numpy.dot(Q_red[w], basis[j-1]) * basis[j-1]
            
            dist = numpy.dot(Q_red[w], Q_red[w].T)
            if dist > max_dist:
                max_dist = dist
                anchors[j+1] = w

        basis[j] = basis_vector(Q_red, anchors[j+1])

    return anchors
