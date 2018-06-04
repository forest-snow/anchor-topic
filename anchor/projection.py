import numpy

def random_projection(A, new_dim, seed=0):
    # Project rows of [A] into lower dimension [new_dim]
    if A.shape[1] <= new_dim:
        return A
    state = numpy.random.RandomState(seed)
    R = state.choice([-1, 0, 0, 0, 0, 1], (new_dim, A.shape[1])) * numpy.sqrt(3)
    A_new = numpy.dot(R, A.T).T
    return A_new