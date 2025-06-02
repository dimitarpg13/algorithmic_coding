import numpy as np

class KNearestNeighbours:
    def __init__(self, k, metric='euclidean'):
        self._k = k # number of nearest neighbors
        self._metric = metric # the metric defining proximity

    def fit(self, data, labels):
        self._data = data
        self._labels = labels

    def predict(self, x):
        pass


if __name__ == '__main__':
    ### TEST CASES ###

    X = np.array([[5, 1, 0], [1,5,0], [8,1,0], [0, 4, 0]])
    y = np.array([[1], [0], [1], [0]])
    k = 3
    target = np.array([[7, 1, 0]])

    knn = KNearestNeighbours(k)
    knn.fit(X, y)
    assert knn.predict(target) == 1

    X = np.array([[100, 100, 100], [1,5,0], [101,101,101], [0, 4, 0]])
    y = np.array([["cat"], ["dog"], ["cat"], ["dog"]])
    k = 3
    target = np.array([[99, 99, 99]])

    knn = KNearestNeighbours(k)
    knn.fit(X, y)
    assert knn.predict(target) == "cat"

    np.random.seed(42)
    mean_1 = np.array([10,10,10,10,10])
    mean_2 = np.array([5,5,5,5,5])

    cov = np.array(
        [
            [10, 1, 1, 1, 1],
            [1, 10, 1, 1, 1],
            [1, 1, 10, 1, 1],
            [1, 1, 1, 10, 1],
            [1, 1, 1, 1, 10]
        ]
    )
    X_1 = np.random.multivariate_normal(mean_1, cov, 1000)
    X_2 = np.random.multivariate_normal(mean_2, cov, 1000)

    X = np.vstack([X_1, X_2])
    y = np.vstack(
        [["class_1"] for _ in range(1000)] + [['class_2'] for _ in range(1000)]
    )

    k = 10
    knn = KNearestNeighbours(k)
    knn.fit(X, y)

target_2 = np.array([[12, 12, 12, 12, 12]])
assert knn.predict(target_2) == "class_1"

target_1 = np.array([[0, 0, 0, 0, 0]])
assert knn.predict(target_1) == "class_2"
