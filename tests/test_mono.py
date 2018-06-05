MAX_VOCAB = 50
MAX_DF = 10
K = 5
THRESHOLD = 0.1
SEED = 1 


def test_init(toy_model):
    assert toy_model.n_topics == K
    assert toy_model.seed == SEED
    assert toy_model.anchors is None
    assert toy_model.word_topic is None
    assert toy_model.doc_threshold == 1

def test_find_anchors(toy_model):
    toy_model.find_anchors()
    assert len(toy_model.anchors) == toy_model.n_topics
    toy_model.print_anchors()

def test_update_topics(toy_model):
    toy_model.update_topics()
    assert toy_model.word_topic.shape == (MAX_VOCAB, K)
    print(toy_model.word_topic)

def test_tandem_update(toy_model):
    anchors = [[1,3,4],[15,6,2],[11],[13,14],[29,45]]
    toy_model.update_topics(anchors)
    assert toy_model.word_topic.shape == (MAX_VOCAB, K)
    print(toy_model.word_topic)