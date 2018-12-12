import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.utils

import hccEncoding.EncoderForClassification as hcc_class
import hccEncoding.EncoderForRegression as hcc_regr


class _HccEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self._X = X
        self._y = y
        self.target_type = sklearn.utils.multiclass.type_of_target(y)
        self._uses_pandas = isinstance(X, pd.DataFrame)
        return self

    def _process_X_y(self, X):
        if X.shape[1] != 1:
            raise ValueError("HccBayesEncoder only works for a single feature")

        if self._uses_pandas:
            y_series = pd.Series(self._y, index=self._X.index, name='target')
            train = pd.concat([self._X, y_series], axis=1)
            test = X
            feature = X.columns[0]
        else:
            y_series = pd.Series(self._y, name='target')
            X_frame = pd.DataFrame(self._X, columns=['feature'])
            train = pd.concat([X_frame, y_series], axis=1)
            test = pd.DataFrame(X, columns=['feature'])
            feature = 'feature'

        return train, test, feature


class HccBayesEncoder(_HccEncoder):

    def __init__(self, k=5, f=1, noise=0.01, clf_type='binary-clf'):
        self.k = k
        self.f = f
        self.noise = noise
        self.clf_type = clf_type

    def transform(self, X, y=None):
        train, test, feature = self._process_X_y(X)

        if self.clf_type == 'regression':
            train_transformed, test_transformed = hcc_regr.BayesEncoding(
                train, test, 'target', feature, drop_origin_feature=True,
                k=self.k, f=self.f, noise=self.noise)
        else:
            train_transformed, test_transformed = hcc_class.BayesEncoding(
                train, test, 'target', feature, drop_origin_feature=True,
                k=self.k, f=self.f, noise=self.noise)

        if self.clf_type == 'binary-clf':
            transformed = test_transformed.iloc[:, [1]]
            transformed.columns = [feature]
            if self._uses_pandas:
                return transformed
            else:
                return np.asarray(transformed)
        return test_transformed


class HccLOOEncoder(_HccEncoder):

    def __init__(self, noise=0.01, clf_type='binary-clf'):
        self.noise = noise
        self.clf_type = clf_type

    def transform(self, X, y=None):
        train, test, feature = self._process_X_y(X)

        if self.clf_type == 'regression':
            train_transformed, test_transformed = hcc_class.LOOEncoding(
                train, test, 'target', feature, drop_origin_feature=True,
                noise=self.noise)
        else:
            train_transformed, test_transformed = hcc_regr.LOOEncoding(
                train, test, 'target', feature, drop_origin_feature=True,
                noise=self.noise)

        if self.clf_type == 'binary-clf':
            test_transformed.columns = [feature]
            if self._uses_pandas:
                return test_transformed
            else:
                return np.asarray(test_transformed)
        return test_transformed
