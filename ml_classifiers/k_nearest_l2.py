import numpy as np
from collections import Counter


def euclidean(l1, l2):
    return np.sqrt(np.sum(np.square(l1-l2), axis=1))


class NearestNeighbour:
    def __init__(self, k=2):
        self.k = k

    def train(self, features, labels):
        # Nearest Neighbour memorise the whole data
        self.features = features
        self.labels = labels

    def predict(self, new_features):
        sample_number = new_features.shape[0]
        k_predictions = []
        common = []
        final_pred = np.zeros(sample_number, dtype=self.labels.dtype)

        for i in range(sample_number):
            difference = euclidean(self.features, new_features[i, :])
            k_index = np.argsort(difference)[:self.k]
            k_predictions = [self.labels[i][0] for i in k_index]
            common.append(Counter(k_predictions).most_common(1))

        for i, each in enumerate(common):
            final_pred[i] = (each[0][0])

        return final_pred.reshape(sample_number, 1)


# Driver code
if __name__ == '__main__':
    X = np.array([[1, 2, 3],
                 [2, 5, 6],
                 [2, 5, 6],
                 [1, 2, 3]])
    y = np.array([1, 2, 2, 1]).reshape((4, 1))
    X_new = np.array([[1, 2, 3],
                      [1, 2, 3]])
    cat = NearestNeighbour()
    cat.train(X, y)
    preds = cat.predict(X_new)
    print(preds)
