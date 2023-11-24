import numpy as np


def manhattan(l1, l2):
    return np.sum(np.abs(l1-l2), axis=1)


class NearestNeighbour:
    def __init__(self) -> None:
        pass

    def train(self, features, labels):
        # Nearest Neighbour memorise the whole data
        self.features = features
        self.labels = labels

    def predict(self, new_features):
        sample_number = new_features.shape[0]
        predictions = np.zeros(sample_number, dtype=self.labels.dtype)

        for i in range(sample_number):
            # Calculating the L1 distance or Manhattan distance
            difference = manhattan(self.features, new_features[i, :])
            print(difference)
            min_index = np.argmin(difference)
            predictions[i] = self.labels[min_index]

        return predictions


# Driver code
if __name__ == '__main__':
    X = np.array([[1, 2, 3],
                 [2, 5, 6],
                 [2, 5, 6],
                 [1, 2, 3]])
    y = np.array([1, 2, 2, 1]).reshape((4, 1))
    X_new = np.array([[1, 2, 3],
                      [2, 5, 6]])
    cat = NearestNeighbour()
    cat.train(X, y)
    preds = cat.predict(X_new)
    print(preds)
