import numpy as np

class Classifier:
    def __init__(self, similiarity_func, cluster_centers, verbose=False):
        self.similarity_func = similiarity_func
        self.cluster_centers = cluster_centers

        self.verbose = verbose

    def classify(self, X):
        similarities = [self.similarity_func(X, cluster_center) for cluster_center in self.cluster_centers]
        return np.argmax(similarities) + 1 # label from 1

    def classify_mul(self, Xs):
        # TODO improve the performance by using parallel
        return map(self.classify, Xs)