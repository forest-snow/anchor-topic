import numpy
import anchor.topics

MAX_VOCAB1 = 50
MAX_VOCAB2 = 45
MAX_DF = 10
K = 5
THRESHOLD = 0.1
SEED = 1 
TOP = 5




def test_model_topics(toy_model):
    A, Q, anchors = toy_model
    assert Q.shape == (MAX_VOCAB1, MAX_VOCAB1)
    assert len(anchors) == K
    assert A.shape == (MAX_VOCAB1, K) 
    assert numpy.all(numpy.isfinite(A))

def test_update_topics(toy_model):
    A, Q, anchors = toy_model
    new_anchors = [[1,3,4],[15,6,2],[11],[13,14],[29,45]]
    A_new = anchor.topics.update_topics(Q, new_anchors)
    assert A.shape == (MAX_VOCAB1, K) 
    assert numpy.all(numpy.isfinite(A))

def test_multimodel_topics(toy_multimodel, toy_dict):
    A1, A2, Q1, Q2, anchors1, anchors2 = toy_multimodel

    assert Q1.shape == (MAX_VOCAB1, MAX_VOCAB1)
    assert len(anchors1) == K
    assert A1.shape == (MAX_VOCAB1, K) 
    assert numpy.all(numpy.isfinite(A1))

    assert Q2.shape == (MAX_VOCAB2, MAX_VOCAB2)
    assert len(anchors2) == K
    assert A2.shape == (MAX_VOCAB2, K) 
    assert numpy.all(numpy.isfinite(A2))

    def check_in_dict(anchors, entries):
        def flatten(l):
            return [item for sublist in l for item in sublist]
        for anchor in flatten(anchors):
            if anchor not in entries:
                return False
        return True

    assert check_in_dict(anchors1, [entry[0] for entry in toy_dict])
    assert check_in_dict(anchors2, [entry[1] for entry in toy_dict])


def test_update_multitopics(toy_multimodel):
    A1, A2, Q1, Q2, anchors1, anchors2 = toy_multimodel
    new_anchors = [[1,3,4],[15,6,2],[11],[13,14],[29,42]]
    A_new1 = anchor.topics.update_topics(Q1, new_anchors)
    A_new2 = anchor.topics.update_topics(Q2, new_anchors)

    assert A_new1.shape == (MAX_VOCAB1, K) 
    assert numpy.all(numpy.isfinite(A_new1))
    assert A_new2.shape == (MAX_VOCAB2, K) 
    assert numpy.all(numpy.isfinite(A_new2))

       
