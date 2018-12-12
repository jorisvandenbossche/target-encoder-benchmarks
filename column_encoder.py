import numpy as np
import warnings

import category_encoders as cat_enc

# from fastText import load_model

from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils import check_array
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
    LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline

from dirty_cat import SimilarityEncoder, TargetEncoder
import category_encoders
import hccencoders

# from gap_factorization import OnlineGammaPoissonFactorization
# from gap_factorization import OnlineGammaPoissonFactorization2


class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 encoder_name,
                 reduction_method=None,
                 ngram_type='sim2',
                 ngram_range=(2, 4),
                 categories='auto',
                 dtype=np.float64,
                 handle_unknown='ignore',
                 clf_type=None,
                 n_components=None):
        self.ngram_range = ngram_range
        self.encoder_name = encoder_name
        self.categories = categories
        self.dtype = dtype
        self.clf_type = clf_type
        self.handle_unknown = handle_unknown
        self.ngram_type = ngram_type
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.encoders_dict = {
            'OneHotEncoder': OneHotEncoder(handle_unknown='ignore'),
            'OneHotEncoderDense': OneHotEncoder(
                handle_unknown='ignore', sparse=False),
            'SimilarityEncoder': SimilarityEncoder(
                ngram_range=self.ngram_range, random_state=10),
            'ngrams_hot_vectorizer': [],
            'NgramsCountVectorizer': CountVectorizer(
                analyzer='char', ngram_range=self.ngram_range),
            'NgramsTfIdfVectorizer': TfidfVectorizer(
                analyzer='char', ngram_range=self.ngram_range,
                smooth_idf=False),
            'TargetEncoder': TargetEncoder(
                clf_type=self.clf_type, handle_unknown='ignore'),
            'TargetEncoder-dirty_cat': TargetEncoder(
                clf_type=self.clf_type, handle_unknown='ignore'),
            'TargetEncoder-category_encoders':
                category_encoders.TargetEncoder(),
            'TargetEncoder-hcc-bayes':
                hccencoders.HccBayesEncoder(clf_type=self.clf_type),
            'TargetEncoder-hcc-loo':
                hccencoders.HccLOOEncoder(clf_type=self.clf_type),
            # 'MDVEncoder': MDVEncoder(self.clf_type),
            'BackwardDifferenceEncoder': cat_enc.BackwardDifferenceEncoder(),
            'BinaryEncoder': cat_enc.BinaryEncoder(),
            'HashingEncoder': cat_enc.HashingEncoder(),
            'HelmertEncoder': cat_enc.HelmertEncoder(),
            'SumEncoder': cat_enc.SumEncoder(),
            'PolynomialEncoder': cat_enc.PolynomialEncoder(),
            'BaseNEncoder': cat_enc.BaseNEncoder(),
            'LeaveOneOutEncoder': cat_enc.LeaveOneOutEncoder(),
            'NgramsLDA': Pipeline([
                ('ngrams_count',
                 CountVectorizer(
                     analyzer='char', ngram_range=self.ngram_range)),
                ('LDA', LatentDirichletAllocation(
                    n_components=self.n_components, learning_method='batch'),)
                ]),
            # 'NgramsMultinomialMixture':
            #     NgramsMultinomialMixture(
            #         n_topics=self.n_components, max_iters=10),
            # 'AdHocNgramsMultinomialMixture':
            #     AdHocNgramsMultinomialMixture(n_iters=0),
            # 'AdHocIndependentPDF': AdHocIndependentPDF(),
            # 'GammaPoissonFactorization':
            #     GammaPoissonFactorization(
            #         n_topics=self.n_components),
            # 'OnlineGammaPoissonFactorization3':
            #     OnlineGammaPoissonFactorization(
            #         n_topics=self.n_components, rescale_W=True, r=.7,
            #         tol=1E-4, random_state=18, init='k-means++'),
            # 'MinHashEncoder': MinHashEncoder(
            #     n_components=self.n_components),
            # 'PretrainedFastText':
            #     PretrainedFastText(n_components=self.n_components),
            # 'PretrainedFastText2':
            #     PretrainedFastText(n_components=self.n_components),
            None: FunctionTransformer(None, validate=True),
            }
        self.list_1D_array_methods = [
            'NgramsCountVectorizer',
            'ngrams_hot_vectorizer',
            'NgramsLDA',
            'NgramsMultinomialMixture',
            'NgramsMultinomialMixtureKMeans2',
            'AdHocNgramsMultinomialMixture',
            'AdHocIndependentPDF',
            'GammaPoissonFactorization',
            'OnlineGammaPoissonFactorization',
            'OnlineGammaPoissonFactorization2',
            'OnlineGammaPoissonFactorization3',
            'MinHashEncoder',
            'MinMeanMinHashEncoder',
            ]

    def _get_most_frequent(self, X):
        unqX, count = np.unique(X, return_counts=True)
        if self.n_components <= len(unqX):
            warnings.warn(
                'Dimensionality reduction will not be applied because' +
                'the encoding dimension is smaller than the required' +
                'dimensionality: %d instead of %d' %
                (X.shape[1], self.n_components))
            return unqX.ravel()
        else:
            count_sort_ind = np.argsort(-count)
            most_frequent_cats = unqX[count_sort_ind][:self.n_components]
            return np.sort(most_frequent_cats)

    def fit(self, X, y=None):
        assert X.values.ndim == 1
        X = X.values
        if self.encoder_name not in self.encoders_dict:
            template = ("Encoder %s has not been implemented yet")
            raise ValueError(template % self.encoder_name)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoder_name == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        # if self.reduction_method == 'MostFrequentCategories':
            # unq_cats = self._get_most_frequent(X)
            # _X = []
            # for x in X:
            #     if x in unq_cats:
            #         _X.append(x)
            # X = np.array(_X)
            # del _X

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")
        self.le = LabelEncoder()

        if self.categories == 'auto':
            self.le.fit(X,)
        else:
            if self.handle_unknown == 'error':
                valid_mask = np.in1d(X, self.categories)
                if not np.all(valid_mask):
                    msg = ("Found unknown categories during fit")
                    raise ValueError(msg)
            self.le.classes_ = np.array(self.categories)

        self.categories_ = self.le.classes_

        n_samples = X.shape[0]
        try:
            self.n_features = X.shape[1]
        except IndexError:
            self.n_features = 1

        if self.encoder_name in self.list_1D_array_methods:
            assert self.n_features == 1
            X = X.reshape(-1)
        else:
            X = X.reshape(n_samples, self.n_features)

        if self.n_features > 1:
            raise ValueError("Encoder does not support more than one feature.")

        self.encoder = self.encoders_dict[self.encoder_name]

        if self.reduction_method == 'most_frequent':
            assert 'SimilarityEncoder' in self.encoder_name
            assert self.n_features == 1
            if len(np.unique(X)) <= self.n_components:
                warnings.warn(
                    'Dimensionality reduction will not be applied because ' +
                    'the encoding dimension is smaller than the required ' +
                    'dimensionality: %d instead of %d' %
                    (len(np.unique(X)), self.n_components))
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
            else:
                self.encoder.categories = 'most_frequent'
                self.encoder.n_prototypes = self.n_components
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
        elif self.reduction_method == 'k-means':
            assert 'SimilarityEncoder' in self.encoder_name
            assert self.n_features == 1
            if len(np.unique(X)) <= self.n_components:
                warnings.warn(
                    'Dimensionality reduction will not be applied because ' +
                    'the encoding dimension is smaller than the required ' +
                    'dimensionality: %d instead of %d' %
                    (len(np.unique(X)), self.n_components))
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
            else:
                self.encoder.categories = 'k-means'
                self.encoder.n_prototypes = self.n_components
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
        else:
            self.pipeline = Pipeline([
                ('encoder', self.encoder),
                ('dimension_reduction',
                 DimensionalityReduction(method_name=self.reduction_method,
                                         n_components=self.n_components))
                ])
        # for MostFrequentCategories, change the fit method to consider only
        # the selected categories
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        if self.encoder_name in self.list_1D_array_methods:
            pass
        else:
            X = X.values.reshape(n_samples, self.n_features)
        Xout = self.pipeline.transform(X)
        # if Xout.ndim == 1:
        #     Xout.reshape(-1, 1)
        return Xout


class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self, method_name=None, n_components=None,
                 column_names=None):
        self.method_name = method_name
        self.n_components = n_components
        self.methods_dict = {
            None: FunctionTransformer(None, accept_sparse=True, validate=True),
            'GaussianRandomProjection': GaussianRandomProjection(
                n_components=self.n_components, random_state=35),
            'most_frequent': 0,
            'k-means': 0,
            'PCA': PCA(n_components=self.n_components, random_state=87)
            }

    def fit(self, X, y=None):
        if self.method_name not in self.methods_dict:
            template = ("Dimensionality reduction method '%s' has not been "
                        "implemented yet")
            raise ValueError(template % self.method_name)

        self.method = self.methods_dict[self.method_name]
        if self.n_components is not None:
            if self.method_name is not None:
                if X.shape[1] <= self.n_components:
                    self.method = self.methods_dict[None]
                    warnings.warn(
                        'Dimensionality reduction will not be applied ' +
                        'because the encoding dimension is smaller than ' +
                        'the required dimensionality: %d instead of %d' %
                        (X.shape[1], self.n_components))

        self.method.fit(X)
        self.n_features = 1
        return self

    def transform(self, X):
        Xout = self.method.transform(X)
        if Xout.ndim == 1:
            return Xout.reshape(-1, 1)
        else:
            return Xout
