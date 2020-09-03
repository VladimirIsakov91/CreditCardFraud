from sklearn.feature_selection import chi2, f_regression, f_classif
from itertools import combinations
from operator import itemgetter


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

    def test(self, data, mapping):

        features = list(mapping.keys())
        feature_pairs = self._get_pairs(features)
        p_vals = []
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

            p_vals.append((f1 + '_' + f2, p[0]))

        p_vals = sorted(p_vals, key=itemgetter(1), reverse=True)
        return p_vals
