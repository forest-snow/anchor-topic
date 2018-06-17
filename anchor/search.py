import numpy 
import scipy.sparse
import math
from . import projection

def row_normalize(Q):
    # return matrix [Q] where each row is normalized 
    # returns a copy of [Q]
    Q_new = Q.copy()
    row_sums = Q_new.sum(1)
    for i in range(Q_new.shape[0]):
        # numpy.seterr(divide='ignore', invalid='ignore')
        if row_sums[i] != 0:
            Q_new[i, :] = Q_new[i, :]/float(row_sums[i])
    return Q_new

def basis_vector(Q, i):
    return Q[i]/numpy.sqrt(numpy.dot(Q[i], Q[i].T))

def greedy_anchors(Q, k, candidates, seed, project_dim=1000):
    Q_bar = row_normalize(Q)
    Q_red = projection.random_projection(Q_bar, project_dim, seed)

    anchors = numpy.zeros(k, dtype=numpy.int)
    dim = Q_red.shape[1]
    basis = numpy.zeros((k - 1, dim))

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
    for j in range(1, k - 1):
        max_dist = 0
        for w in candidates:
            Q_red[w] = Q_red[w] - numpy.dot(Q_red[w], basis[j-1]) * basis[j-1]
            
            dist = numpy.dot(Q_red[w], Q_red[w].T)
            if dist > max_dist:
                max_dist = dist
                anchors[j+1] = w

        basis[j] = basis_vector(Q_red, anchors[j+1])

    return anchors

def min_distance(Q1, Q2, w1, w2):
    return min(numpy.dot(Q1[w1], Q1[w1].T), numpy.dot(Q2[w2], Q2[w2].T))


def greedy_linked_anchors(Q1, Q2, k, candidates, seed, distance=min_distance, project_dim=1000):
    Q1_bar = row_normalize(Q1)
    Q2_bar = row_normalize(Q2)
    Q1_red = projection.random_projection(Q1_bar, project_dim, seed)
    Q2_red = projection.random_projection(Q2_bar, project_dim, seed)

    n_anchors = min(k, len(candidates))

    anchors1 = numpy.zeros(k, dtype=numpy.int)
    anchors2 = numpy.zeros(k, dtype=numpy.int)

    if n_anchors == 1:
        return candidates[0]

    dim = Q1_red.shape[1]
    basis1 = numpy.zeros((n_anchors - 1, dim))
    basis2 = numpy.zeros((n_anchors - 1, dim))

    # find p1 with farthest distance from origin
    max_dist = 0
    for w1, w2 in candidates:
        dist = distance(Q1_red, Q2_red, w1, w2)
        if dist > max_dist:
            max_dist = dist
            anchors1[0] = w1
            anchors2[0] = w2


    # let p1 be origin of both coordinate systems
    for w1, w2 in candidates:
        Q1_red[w1] = Q1_red[w1] - Q1_red[anchors1[0]]
        Q2_red[w2] = Q2_red[w2] - Q2_red[anchors2[0]]

    # find farthest point from p1 
    max_dist = 0
    for w1, w2 in candidates:
        dist = distance(Q1_red, Q2_red, w1, w2)
        if dist > max_dist:
            max_dist = dist
            anchors1[1] = w1
            anchors2[1] = w2

    basis1[0] = basis_vector(Q1_red, anchors1[1])
    basis2[0] = basis_vector(Q2_red, anchors2[1])



    # stabilized gram-schmidt which finds new anchor words to expand our subspace
    for j in range(1, n_anchors - 1):
        max_dist = 0
        for w1, w2 in candidates:
            Q1_red[w1] = Q1_red[w1] - numpy.dot(Q1_red[w1], basis1[j-1]) * basis1[j-1]
            Q2_red[w2] = Q2_red[w2] - numpy.dot(Q2_red[w2], basis2[j-1]) * basis2[j-1]
            
            dist = distance(Q1_red, Q2_red, w1, w2)
            

            if dist > max_dist:
                max_dist = dist
                anchors1[j+1] = w1
                anchors2[j+1] = w2

        basis1[j] = basis_vector(Q1_red, anchors1[j+1])
        basis2[j] = basis_vector(Q2_red, anchors2[j+1])

    return anchors1, anchors2
