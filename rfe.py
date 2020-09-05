from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import pandas
import numpy
import pickle

numpy.set_printoptions(suppress=True)

JOBS = 10
SEED = 0

types = {f'V{i}': 'float32' for i in range(1, 29)}
types['Amount'] = 'float32'

X = pandas.read_csv('./data/features.csv', header=0, dtype=types)
y = pandas.read_csv('./data/target.csv', header=0, dtype={'Class': 'int32'})

rf = RandomForestClassifier(random_state=SEED)

selector = RFECV(estimator=rf,
                 step=1,
                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                 n_jobs=JOBS,
                 verbose=10,
                 scoring='precision',
                 min_features_to_select=1
                 )

filename = './data/rfe_precision.pkl'
rfe = selector.fit(X.to_numpy(), y.to_numpy().reshape(-1, ))
pickle.dump(obj=rfe, file=open(filename, 'wb'))





