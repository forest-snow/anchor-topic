import scipy.sparse
import scipy.stats
import numpy 
import random
import multiprocessing.pool
from numba import jit

def logsum_exp(y):
    m = y.max()
    return m + numpy.log((numpy.exp(y - m)).sum())

def KL(p,q):
    return scipy.stats.entropy(p, q)        

def KLSolveExpGrad(y,x,eps, alpha=None):
    c1 = 10**(-4)
    c2 = 0.9
    it = 1 
    
    y = numpy.clip(y, 0, 1)
    x = numpy.clip(x, 0, 1)

    (k, n) = x.shape
    mask = list(numpy.nonzero(y)[0])

    y = y[mask]
    x = x[:, mask]

    x += 10**(-9)
    x /= x.sum(axis=1)[:, numpy.newaxis]

    if alpha == None:
        alpha = numpy.ones(k)/k

    old_alpha = numpy.copy(alpha)
    log_alpha = numpy.log(alpha)
    old_log_alpha = numpy.copy(log_alpha)
    proj = numpy.dot(alpha,x)
    old_proj = numpy.copy(proj)

    new_obj = KL(y, proj)
    y_over_proj = y/proj
    grad = -numpy.dot(x, y_over_proj.transpose())
    old_grad = numpy.copy(grad)

    stepsize = 1
    decreasing = False
    repeat = False
    gap = float('inf')

    while 1:
        eta = stepsize
        old_obj = new_obj
        old_alpha = numpy.copy(alpha)
        old_log_alpha = numpy.copy(log_alpha)

        old_proj = numpy.copy(proj)

        it += 1
        #take a step
        log_alpha -= eta*grad

        #normalize
        log_alpha -= logsum_exp(log_alpha)

        #compute new objective
        alpha = numpy.exp(log_alpha)
        proj = numpy.dot(alpha,x)
        new_obj = KL(y, proj)
        if new_obj < eps:
            break

        grad_dot_deltaAlpha = numpy.dot(grad, alpha - old_alpha)
        assert (grad_dot_deltaAlpha <= 10**(-9))
        if not new_obj <= old_obj + c1*stepsize*grad_dot_deltaAlpha: #sufficient decrease
            stepsize /= 2.0 #reduce stepsize
            if stepsize < 10**(-6):
                break
            alpha = old_alpha 
            log_alpha = old_log_alpha
            proj = old_proj
            new_obj = old_obj
            repeat = True
            decreasing = True
            continue

        
        #compute the new gradient
        old_grad = numpy.copy(grad)
        y_over_proj = y/proj
        grad = -numpy.dot(x, y_over_proj)

        if not numpy.dot(grad, alpha - old_alpha) >= c2*grad_dot_deltaAlpha and not decreasing: #curvature
            stepsize *= 2.0 #increase stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            proj = old_proj
            new_obj = old_obj
            repeat = True
            continue

        decreasing= False
        lam = numpy.copy(grad)
        lam -= lam.min()
        
        gap = numpy.dot(alpha, lam)
        convergence = gap
        if (convergence < eps):
            break

    return alpha

def fast_recover(y, X, w, anchors, XXT, initial_stepsize, epsilon):
    k = len(anchors)
    alpha = numpy.zeros(k)
    gap = None
    if w in anchors:
        alpha[anchors.index(w)] = 1
        it = -1
        dist = 0
        stepsize = 0

    else: 
        alpha = KLSolveExpGrad(y, X, epsilon)

    if numpy.isnan(alpha).any():
        alpha = ones(k)/k

    return alpha
        



def computeA(cooccur, anchors, initial_stepsize=1, epsilon=2e-7):
    Q = cooccur.copy()
    v = Q.shape[0]
    k = len(anchors)
    A = numpy.zeros((v, k))

    # store normalization constants
    P_w = numpy.diag(Q.sum(axis=1))
    for w in range(v):
        if numpy.isnan(P_w[w, w]):
            P_w[w, w] = 1e-16    

    # Normalize rows of Q
    for w in range(v):
        if Q[w, :].sum() != 0:
            Q[w, :] = Q[w, :] / Q[w, :].sum()

    X = Q[anchors]
    XXT = numpy.dot(X, X.transpose())

    initial_stepsize = 1

    C = numpy.zeros((v, k))
    for w in range(v):
        y = Q[w, :]
        alpha = fast_recover(y, X, w, anchors, XXT, initial_stepsize, epsilon)
        C[w, :] = alpha 

    # Use Bayes rule to compute topic matrix
    A = numpy.dot(P_w, C)
    
    # Normalize columns
    for i in range(k):
        A[:, i] = A[:, i] / A[:, i].sum()


    return numpy.array(A)



