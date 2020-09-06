from scipy.stats import ttest_ind, pearsonr, spearmanr, mannwhitneyu, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import pandas
import numpy

numpy.set_printoptions(suppress=True)

JOBS = 10
SEED = 0

types = {'V1': 'float32', 'V3': 'float32', 'V4': 'float32', 'V6': 'float32',
         'V7': 'float32', 'V9': 'float32', 'V10': 'float32', 'V11': 'float32',
         'V12': 'float32', 'V14': 'float32', 'V16': 'float32', 'V17': 'float32',
         'V18': 'float32', 'V20': 'float32', 'V21': 'float32', 'V26': 'float32'}

X_ = pandas.read_csv('./data/important_features.csv', header=0, dtype=types)
y = pandas.read_csv('./data/target.csv', header=0, dtype={'Class': 'int32'})

rf = RandomForestClassifier(random_state=SEED)

params = {'n_estimators': randint(1, 201),
          'max_depth': randint(2, 201),
          'min_samples_split': randint(2, 201),
          'min_samples_leaf': randint(1, 201),
          'max_features': randint(2, 17)
          }

search = RandomizedSearchCV(estimator=rf,
                            param_distributions=params,
                            n_iter=200,
                            n_jobs=JOBS,
                            cv=StratifiedKFold(random_state=SEED, shuffle=True, n_splits=5),
                            verbose=10,
                            random_state=SEED,
                            scoring={'AP': 'average_precision', 'Recall': 'recall', 'Precision': 'precision'},
                            refit=False,
                            return_train_score=True,
                            error_score='raise'
                            )

search.fit(X_.to_numpy(), y.to_numpy().reshape(-1, ))

res = pandas.DataFrame(search.cv_results_)
res.to_csv('./data/tune_results.csv', index=False, header=True)
