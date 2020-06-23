import numpy as np
from sklearn.metrics import jaccard_score,hamming_loss as jaccard_similarity_score
from sklearn.metrics import hamming_loss
print (hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[0.0, 1.0], [1.0, 1.0]])))
print (hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [1.0, 1.0]])))
print (hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [0.0, 1.0]])))
print (jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[0.0, 1.0], [1.0, 1.0]])))
print (jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [1.0, 1.0]])))
print (jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [0.0, 1.0]])))
