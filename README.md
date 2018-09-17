# anchor-topic

This package supports implementation of anchor-based topic modeling and variants of the anchoring algorithm in Python 3. 

If you use this package for academic research, [please cite the relevant papers.](http://github.com/forest-snow#publications)

## Installation

Install the package through terminal with this command: 
```sh
pip install anchor-topic 
```
Dependencies (Numpy, Scipy, Numba) will be installed as well.

## Models

To build a topic model using the code, you must include this import statement:
```sh
import anchor_topic.topics
```

### Preprocessing 
Anchoring algorithm takes in word-document matrix _M_ as input (_M(i,j) =_ frequency of word _i_ in document _j_).  As with other topic models, corpus should be preprocessed to improve quality of model.  The word-document matrix _M_ should be of type ```scipy.sparse.csc_matrix```.

### Anchoring
To build an anchor-based topic model for monolingual corpus, use the following function: 
```sh
A, Q, anchors = anchor_topic.topics.model_topics(M, k, threshold)
```
Inputs: 
- _M_, word-document matrix
- _k_, is number of topics
- _threshold_, minimum percentage of document occurrences for word to be considered as an anchor candidate

Outputs:
- _A_, word-topic matrix
- _Q_, word-cooccurrence matrix
- _anchors_, 2D list of anchor words for each topic  

### Multilingual anchoring
To build an anchor-based topic model for comparable corpora, use the following function: 
```sh
A1, A2, Q1, Q2, anchors1, anchors2 = anchor_topic.topics.model_multi_topics(M1, M2, k, threshold1, threshold2, dictionary)
```
_dictionary_ should be a text file where each line is a tab-separated dictionary entry.
```sh
hello  你好
goodbye 再見
```

### Updating topics
To support topic model interactivity, users can choose their own anchors.  First, topic model should be built from anchoring algorithm to get initial anchors and word-cooccurrence matrix _Q_.  Then, use the following function to update topics: 
```sh 
A = update_topics(Q, anchors)
```
For each topic, user may pick one or more anchors.  Make sure _anchors_ is a 2d list of type ```int``` where each number represents the word's index in _Q_.

## Publications

If you use this package for academic research, please cite the relevant paper(s) as follows:
```sh
@inproceedings{yuan2018mtanchor,
  title={Multilingual Anchoring: Interactive Topic Modeling and Alignment Across Languages},
  author={Yuan, Michelle and Van Durme, Benjamin and Boyd-Graber, Jordan},
  booktitle={Advances in neural information processing systems},
  year={2018}
}

@inproceedings{lund2017tandem,
  title={Tandem anchoring: A multiword anchor approach for interactive topic modeling},
  author={Lund, Jeffrey and Cook, Connor and Seppi, Kevin and Boyd-Graber, Jordan},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={896--905},
  year={2017}
}

@inproceedings{arora2013practical,
  title={A practical algorithm for topic modeling with provable guarantees},
  author={Arora, Sanjeev and Ge, Rong and Halpern, Yonatan and Mimno, David and Moitra, Ankur and Sontag, David and Wu, Yichen and Zhu, Michael},
  booktitle={International Conference on Machine Learning},
  pages={280--288},
  year={2013}
}
```
## License
Copyright (C) 2018, Michelle Yuan

Licensed under the terms of the MIT License. A full copy of the license can be found in LICENSE.txt.

