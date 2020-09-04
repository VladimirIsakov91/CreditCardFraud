from sklearn.feature_selection import chi2, f_regression, f_classif
from itertools import combinations
from operator import itemgetter
import pandas
import numpy


class Hypot:

    def __init__(self):
        pass

    def as_numpy(self, x, y):
        x = x.to_numpy().reshape(-1, 1)
        y = y.to_numpy()
        return x, y

    def _chi2(self, x, y):
        x, y = self.as_numpy(x, y)
        return chi2(x, y)

    def _f_regression(self, x, y):
        x, y = self.as_numpy(x, y)
        return f_regression(x, y)

    def _f_classif(self, x, y):
        x, y = self.as_numpy(x, y)
        return f_classif(x, y)

    def _get_pairs(self, keys):
        return combinations(keys, r=2)

    @staticmethod
    def _make_mapping(f1, f2, dic, value):

        if f1 not in dic:
            dic[f1] = {f1: 1.}
            if f2 not in dic[f1]:
                dic[f1][f2] = value
        else:
            dic[f1][f2] = value

        return dic

    @staticmethod
    def _map2numpy(mapping, order):

        arr = []
        for entry in order:
            arr.append([mapping[entry][f] for f in order])
        arr = numpy.array(arr, dtype=numpy.float32)

        return arr

    @staticmethod
    def _numpy2pandas(arr, columns, index=None):
        return pandas.DataFrame(data=arr, columns=columns, index=index)

    def test(self, data, mapping):

        features = list(mapping.keys())
        feature_pairs = self._get_pairs(features)
        p_vals = {}

        for pair in feature_pairs:
            f1, f2 = pair
            if mapping[f1] == mapping[f2] == 'cont':
                _, p = self._f_regression(data[f1], data[f2])
            elif mapping[f1] == mapping[f2] == 'cat':
                _, p = self._chi2(data[f1], data[f2])
            elif mapping[f1] != mapping[f2]:
                if mapping[f1] == 'cont' and mapping[f2] == 'cat':
                    _, p = self._f_classif(data[f1], data[f2])
                elif mapping[f1] == 'cat' and mapping[f2] == 'cont':
                    _, p = self._f_classif(data[f2], data[f1])
            else:
                p = None

            p = round(p[0], 5)

            p_vals = self._make_mapping(f1=f1, f2=f2, dic=p_vals, value=p)
            p_vals = self._make_mapping(f1=f2, f2=f1, dic=p_vals, value=p)

        arr = self._map2numpy(mapping=p_vals, order=features)
        df = self._numpy2pandas(arr=arr, columns=features, index=features)

        return df
