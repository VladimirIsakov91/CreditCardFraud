import numpy
from sklearn.metrics.pairwise import euclidean_distances
import pandas

numpy.set_printoptions(suppress=True)


class SMOTE:

    def __init__(self, k, N):
        """"
        k - max number of k-nearest neighbours
        N - oversampling size (100%, 200%, 300%)
        """
        self._k = k
        self._N = N

    @property
    def k(self):
        return self._k

    @property
    def N(self):
        return self._N

    def simulate(self, dataset, labels, oversample_target=None):

        data = self._knn(dataset, labels, oversample_target)

        for origin, knn, cl in data:
            sample_idx = self._sample_neighbours()
            simulated = self._simulate(origin=origin,
                                       knn=knn,
                                       sample_idx=sample_idx)

            cl = numpy.array([cl for _ in range(simulated.shape[0])], dtype=numpy.int32)
            yield simulated, cl

    def _knn(self, dataset, labels, oversample_target):

        if not oversample_target:

            classes = numpy.unique(labels).astype(dtype=numpy.int32)

        else:

            classes = numpy.array(oversample_target, dtype=numpy.int32)

        for cl in classes:

            cl_group = numpy.where(labels == cl)[0]
            origin = dataset[cl_group]
            dist = euclidean_distances(origin, origin)
            idx = numpy.argsort(dist, axis=1)
            knn = origin[idx, :]
            knn = knn[:, 1:self._k+1, :]

            yield (origin, knn, cl)

    def _sample_neighbours(self):
        idx = numpy.random.choice(a=self._k,
                                  size=self._N)
        return idx

    def _simulate(self, origin, knn, sample_idx):

        stacked = [origin for _ in range(self._N)]
        stacked = numpy.stack(stacked, axis=1)
        diff = knn[:, sample_idx, :] - stacked
        rand = numpy.random.random_sample(stacked.shape).astype(dtype=numpy.float32)
        simulated = stacked + rand * diff
        simulated = simulated.reshape(-1, origin.shape[1])
        return simulated


if __name__ == '__main__':

    pass

