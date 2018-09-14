import scipy.sparse
import numpy 
import random
import multiprocessing.pool
from numba import jit

@jit(nopython=True)
def logsum_exp(y):
    """Computes the sum of y in log space"""
    ymax = y.max()
    return ymax + numpy.log((numpy.exp(y - ymax)).sum())

@jit(nopython=True)
def exponentiated_gradient(Y, X, XX, epsilon, k):
    """Solves an exponentied gradient problem with L2 divergence"""
    _C1 = 1e-4
    _C2 = .75

    XY = numpy.dot(X, Y)
    YY = numpy.dot(Y, Y)

    alpha = numpy.ones(X.shape[0]) / X.shape[0]
    old_alpha = numpy.copy(alpha)
    log_alpha = numpy.log(alpha)
    old_log_alpha = numpy.copy(log_alpha)

    AXX = numpy.dot(alpha, XX)
    AXY = numpy.dot(alpha, XY)
    AXXA = numpy.dot(AXX, alpha.transpose())

    grad = 2 * (AXX - XY)
    old_grad = numpy.copy(grad)

    new_obj = AXXA - 2 * AXY + YY

    # Initialize book keeping
    stepsize = 1
    decreased = False
    convergence = numpy.inf

    while convergence >= epsilon:
        old_obj = new_obj
        old_alpha = numpy.copy(alpha)
        old_log_alpha = numpy.copy(log_alpha)
        if new_obj == 0 or stepsize == 0:
            break

        # Add the gradient and renormalize in logspace, then exponentiate
        log_alpha -= stepsize * grad
        log_alpha -= logsum_exp(log_alpha)
        alpha = numpy.exp(log_alpha)

        # Precompute quantities needed for adaptive stepsize
        AXX = numpy.dot(alpha, XX)
        AXY = numpy.dot(alpha, XY)
        AXXA = numpy.dot(AXX, alpha.transpose())

        # See if stepsize should decrease
        old_obj, new_obj = new_obj, AXXA - 2 * AXY + YY
        offset = _C1 * stepsize * numpy.dot(grad, alpha - old_alpha)
        new_obj_threshold = old_obj + offset
        if new_obj >= new_obj_threshold:
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        # compute the new gradient
        old_grad, grad = grad, 2 * (AXX - XY)

        # See if stepsize should increase
        if numpy.dot(grad, alpha - old_alpha) < _C2 * numpy.dot(old_grad, alpha - old_alpha) and not decreased:
            stepsize *= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        # Update book keeping
        decreased = False
        convergence = numpy.dot(alpha, grad - grad.min())

    if numpy.isnan(alpha).any():
        alpha = numpy.ones(X.shape[0]) / X.shape[0]
    return alpha

def computeA(cooccur, anchors, parallelism=True, epsilon=2e-7):
    Q = cooccur.copy()
    v = Q.shape[0]
    k = len(anchors)
    Q_anchors = Q[anchors, :]

    # compute normalized anchors X and precompute X*X.T
    X = Q_anchors / Q_anchors.sum(axis=1)[:, numpy.newaxis]
    XX = numpy.dot(X, X.transpose())

    # store normalization constants
    P_w = numpy.diag(Q.sum(axis=1))
    for word in range(v):
        if numpy.isnan(P_w[word, word]):
            P_w[word, word] = 1e-16

    # Normalize rows of Q to get Q_prime
    for word in range(v):
        if Q[word, :].sum() != 0:
            Q[word, :] = Q[word, :] / Q[word, :].sum()

    # Represent each word as a convex combination of anchors.
    
    if parallelism:
        worker = lambda word: exponentiated_gradient(Q[word], X, XX, epsilon, k)
        chunksize =  5000
        with multiprocessing.pool.ThreadPool() as pool:
            C_matrix = pool.map(worker, range(v), chunksize)
        C = numpy.array(C_matrix)

    else:
        C = numpy.zeros((v, k))
        for word in range(v):
            C[word] = exponentiated_gradient(Q[word], X, XX, epsilon)

    # Use Bayes rule to compute topic matrix
    A = numpy.dot(P_w, C)
    
    # Normalize columns
    for k in range(k):
        A[:, k] = A[:, k] / A[:, k].sum()

    return numpy.array(A)



