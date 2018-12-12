import os
import numpy as np
import warnings

import category_encoders as cat_enc

from scipy.special import polygamma, logsumexp, kl_div
from scipy import sparse

from fastText import load_model

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils import check_array
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, \
    HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
    LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.utils import murmurhash3_32
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms

from dirty_cat import SimilarityEncoder, TargetEncoder


CE_HOME = os.environ.get('CE_HOME')
from gap_factorization import OnlineGammaPoissonFactorization
from gap_factorization import OnlineGammaPoissonFactorization2


class PretrainedFastText(BaseEstimator, TransformerMixin):
    """
    Category embedding using a fastText pretrained model in english
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.ft_model = load_model(
            os.path.join(
                CE_HOME, 'data', 'fastText', 'crawl-300d-2M-subword.bin'))
        return self

    def transform(self, X):
        X_out = np.zeros((len(X), 300))
        for i, x in enumerate(X.ravel()):
            if x.find('\n') != -1:
                x = ' '.join(x.split('\n'))
            X_out[i, :] = self.ft_model.get_sentence_vector(x)
        return X_out


class MinHashEncoder(BaseEstimator, TransformerMixin):
    """
    minhash method applied to ngram decomposition of strings
    """

    def __init__(self, n_components, ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.n_components = n_components

    def get_unique_ngrams(self, string, ngram_range):
        """
        Return a list of different n-grams in a string
        """
        spaces = ' '  # * (n // 2 + n % 2)
        string = spaces + " ".join(string.lower().split()) + spaces
        ngram_list = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            string_list = [string[i:] for i in range(n)]
            ngram_list += list(set(zip(*string_list)))
        return ngram_list

    def minhash(self, string, n_components, ngram_range):
        min_hashes = np.ones(n_components) * np.infty
        grams = self.get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = self.get_unique_ngrams(' Na ', self.ngram_range)
        for gram in grams:
            hash_array = np.array([
                murmurhash3_32(''.join(gram), seed=d, positive=True)
                for d in range(n_components)])
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes/(2**31-1)

    def fit(self, X, y=None):

        self.hash_dict = {}
        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.minhash(
                    x, n_components=self.n_components,
                    ngram_range=self.ngram_range)
        return self

    def transform(self, X):

        X_out = np.zeros((len(X), self.n_components))

        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.minhash(
                    x, n_components=self.n_components,
                    ngram_range=self.ngram_range)

        for i, x in enumerate(X):
            X_out[i, :] = self.hash_dict[x]

        return X_out


class OnlineGammaPoissonFactorization(BaseEstimator, TransformerMixin):
    """
    Online Non-negative Matrix Factorization by minimizing the
    Kullback-Leibler divergence.


    Parameters
    ----------

    n_topics: int, default=10
        Number of topics of the matrix Factorization.

    batch_size: int, default=100

    gamma_shape_prior: float, default=1.1
        Shape parameter for the Gamma prior distribution.

    gamma_scale_prior: float, default=1.0
        Shape parameter for the Gamma prior distribution.

    r: float, default=1
        Weight parameter for the update of the W matrix

    hashing: boolean, default=False
        If true, HashingVectorizer is used instead of CountVectorizer.

    hashing_n_features: int, default=2**10
        Number of features for the HashingVectorizer. Only relevant if
        hashing=True.

    tol: float, default=1E-3
        Tolerance for the convergence of the matrix W

    mix_iter: int, default=2

    max_iter: int, default=10

    ngram_range: tuple, default=(2, 4)

    init: str, default 'k-means++'
        Initialization method of the W matrix.

    rescale_W: bool, default=true
        Whether or not the W matrix is rescaled at each iteration.

    random_state: default=None

    Attributes
    ----------

    References
    ----------
    """

    def __init__(self, n_topics=10, batch_size=512, gamma_shape_prior=1.1,
                 gamma_scale_prior=1.0, r=.7, hashing=False,
                 hashing_n_features=2**10, init='k-means++', rescale_W=True,
                 tol=1E-4, min_iter=3, max_iter=10, ngram_range=(2, 4),
                 random_state=None):

        self.ngram_range = ngram_range
        self.n_topics = n_topics
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.r = r
        self.batch_size = batch_size
        self.tol = tol
        self.rescale_W = rescale_W
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.random_state = check_random_state(random_state)

        if self.hashing:
            self.ngrams_count = HashingVectorizer(
                 analyzer='char', ngram_range=self.ngram_range,
                 n_features=self.hashing_n_features,
                 norm=None, alternate_sign=False)
        else:
            self.ngrams_count = CountVectorizer(
                 analyzer='char', ngram_range=self.ngram_range)

    def _rescale_W(self, W, A, B):
        s = W.sum(axis=1, keepdims=True)
        W /= s
        A /= s
        return W, A, B

    def _rescale_H(self, V, H):
        epsilon = 1e-10  # in case of a document having length=0
        doc_length = np.maximum(epsilon, V.sum(axis=1).A)
        H_length = H.sum(axis=1, keepdims=True)
        factors = doc_length / H_length
        H_out = factors * H
        return H_out

    # @profile
    def _e_step(self, Vt, W, Ht, max_iter=20, epsilon=1E-4):
        WT1 = np.sum(W, axis=1) + 1 / self.gamma_scale_prior
        const = (self.gamma_shape_prior - 1) / WT1
        Ht_out = np.empty(Ht.shape)
        epsilon2 = epsilon**2
        for vt, ht, ht_out in zip(Vt, Ht, Ht_out):
            vt_ = vt.data
            idx = vt.indices
            W_ = W[:, idx] / WT1.reshape(-1, 1)
            norm2 = 1
            iter = 0
            while norm2 > epsilon2:
                iter += 1
                if iter > max_iter:
                    break
                htW = np.dot(ht, W_)
                ht_out_ = ht * np.dot(W_, vt_/htW) + const
                ht_out[:] = ht_out_
                aux = ht - ht_out_
                norm2 = np.dot(aux, aux) / np.dot(ht, ht)
                ht = ht_out_
        return Ht_out

    # @profile
    def _m_step(self, Vt, W, A, B, Ht):
        # print(np.mean(A), np.mean(B))
        A *= self.rho
        A += W * (
            Vt.multiply(np.dot(Ht, W)**-1).transpose().dot(Ht)).transpose()
        B *= self.rho
        B += Ht.sum(axis=0).reshape(-1, 1)
        W = A / B
        return W, A, B

    def fit(self, X, y=None):
        """Fit the OnlineGammaPoissonFactorization to X.

        Parameters
        ----------
        X : string array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature
        Returns
        -------
        self
        """
        assert X.ndim == 1
        unqX = np.unique(X)
        V = self.ngrams_count.fit_transform(X)
        self.vocabulary = self.ngrams_count.get_feature_names()
        self.n_samples, self.n_vocab = V.shape

        unqV = self.ngrams_count.transform(unqX)

        # H = self.random_state.gamma(
        #     shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
        #     size=(len(unqX), self.n_topics))
        H = np.ones((len(unqX), self.n_topics))
        H = self._rescale_H(unqV, H)
        del unqV

        if self.init == 'k-means++':
            W = _k_init(
                V, self.n_topics, row_norms(V, squared=True),
                random_state=self.random_state,
                n_local_trials=None) + .1
        elif self.init == 'random':
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
                size=(self.n_topics, self.n_vocab))
        else:
            raise AttributeError(
                'Initialization method %s does not exist.' % self.init)

        self.rho = self.r**(self.batch_size / self.n_samples)

        W /= W.sum(axis=1, keepdims=True)
        factor = 1E-10
        A = np.ones((self.n_topics, self.n_vocab)) * factor
        # A = W.copy() * factor
        B = np.ones((self.n_topics, self.n_vocab)) * factor
        # B = np.ones((self.n_topics, self.n_vocab))
        # W, A, B = self._rescale_W(W, A, B)

        H_dict = {x: h for x, h in zip(unqX, H)}
        del H, unqX

        def get_H_from_X(X, H_dict):
            H_out = np.empty((X.shape[0], self.n_topics))
            for x, h_out in zip(X, H_out):
                h_out[:] = H_dict[x]
            return H_out

        n_batch = (self.n_samples-1) // self.batch_size + 1

        for iter in range(self.max_iter):
            for i in range(n_batch):
                W_last = W
                idx = range(
                    i*self.batch_size,
                    np.minimum(self.n_samples, (i+1)*self.batch_size))
                Xt = X[idx]
                Ht = get_H_from_X(Xt, H_dict)
                Vt = V[idx, :]
                Ht = self._e_step(Vt, W_last, Ht)
                W, A, B = self._m_step(Vt, W_last, A, B, Ht)
                if self.rescale_W:
                    W, A, B = self._rescale_W(W, A, B)

                if i == n_batch-1:
                    W_change = np.linalg.norm(
                        W - W_last) / np.linalg.norm(W_last)

                for x, h in zip(Xt, Ht):
                    H_dict[x] = h

            # ridx = self.random_state.choice(
            #     range(self.n_samples), size=1000, replace=True)
            # kl_divergence = kl_div(
            #     V[ridx, :].A, np.dot(get_H_from_X(X[ridx], H_dict), W)
            #     ).sum() / self.n_samples
            # kl_divergence = kl_div(
            #     V[:, :].A, np.dot(get_H_from_X(X[:], H_dict), W)
            #     ).sum() / self.n_samples
            # print('iter %d; W change:%.5f; V-WH kl divergence:%.5f' %
            #       (iter, W_change, kl_divergence))

            if (W_change < self.tol) and (iter >= self.min_iter - 1):
                break

        self.W = W
        self.H_dict = H_dict
        return self

    def transform(self, X):
        """Transform X using the trained matrix W.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_new : 2-d array, shape [n_samples, n_features_new]
            Transformed input.

        """
        unknown_X = np.unique([x for x in X if x not in self.H_dict])

        V = self.ngrams_count.transform(unknown_X)
        # H = np.random.gamma(
        #     shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
        #     size=(V.shape[0], self.n_topics))
        H = np.ones((V.shape[0], self.n_topics))
        H = self._rescale_H(V, H)

        n_batch = (V.shape[0]-1) // self.batch_size + 1
        for i in range(n_batch):
            idx = range(
                i*self.batch_size,
                np.minimum(V.shape[0], (i+1)*self.batch_size))
            Ht = H[idx, :]
            Vt = V[idx, :]
            H[idx, :] = self._e_step(Vt, self.W, Ht, max_iter=100)
        del V

        for x, h in zip(unknown_X, H):
            self.H_dict[x] = h
        del unknown_X, H

        H_out = np.empty((X.shape[0], self.n_topics))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict[x]
        return H_out


class AdHocIndependentPDF(BaseEstimator, TransformerMixin):
    def __init__(self, fisher_kernel=True, dtype=np.float64,
                 ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.count_vectorizer = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.fisher_kernel = fisher_kernel
        self.dtype = dtype

    def fit(self, X, y=None):
        self.cats, self.count = np.unique(X, return_counts=True)
        self.pD = (self.count_vectorizer.fit_transform(self.cats) > 0)
        self.theta = self.count / sum(self.count)
        self.n_features, self.n_vocab = self.pD.shape
        return self

    def transform(self, X):
        unqX = np.unique(X)
        pX = (self.count_vectorizer.transform(unqX) > 0)
        d = len(self.cats)
        encoder_dict = {}
        for i, px in enumerate(pX):
            beta = np.ones((1, self.n_vocab))
            for j, pd in enumerate(self.pD):
                beta -= (px != pd) * self.theta[j]
            inv_beta = 1 / beta
            inv_beta_trans = inv_beta.transpose()
            sum_inv_beta = inv_beta.sum()
            fisher_vector = np.ones((1, d)) * sum_inv_beta
            for j, pd in enumerate(self.pD):
                fisher_vector[0, j] -= (px != pd).dot(inv_beta_trans)
            encoder_dict[unqX[i]] = fisher_vector
        Xout = np.zeros((X.shape[0], d))
        for i, x in enumerate(X):
            Xout[i, :] = encoder_dict[x]
        return np.nan_to_num(Xout).astype(self.dtype)


class GammaPoissonFactorization(BaseEstimator, TransformerMixin):
    """
    Gamma-Poisson factorization model (Canny 2004)
    """

    def __init__(self, n_topics=10, max_iters=100, fisher_kernel=False,
                 gamma_shape_prior=1.1, gamma_scale_prior=1.0, tol=.001,
                 ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.ngrams_count = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.n_topics = n_topics  # parameter k
        self.max_iters = max_iters
        self.fisher_kernel = fisher_kernel
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.tol = tol

    def _mean_change(self, X_, X_last):
        scaled_diff = np.array(abs(X_-X_last))/np.array(X_last.sum(axis=0))
        mean_change = scaled_diff.sum(axis=0).mean()
        return mean_change

    def _rescale_Lambda(self, Lambda):
        factors = 1 / Lambda.sum(axis=0)
        Lambda_out = factors.reshape(1, -1) * Lambda
        return Lambda_out

    def _rescale_X(self, F, X):
        epsilon = 1e-10  # in case of a document having length=0
        doc_length = np.maximum(epsilon, np.array(F.sum(axis=0))).reshape(-1)
        X_length = X.sum(axis=0)
        factors = doc_length / X_length
        X_out = factors.reshape(1, -1) * X
        return X_out

    def _e_step(self, F, L, X):
        X_out = np.zeros((self.n_topics, F.shape[1]))
        aux3 = L.sum(axis=0) + 1 / self.gamma_scale_prior
        aux4 = (self.gamma_shape_prior - 1) / X
        cooF = sparse.coo_matrix(F)
        cooY_data = np.dot(L, X)[cooF.row, cooF.col]
        aux0 = cooF.data / cooY_data
        for i in range(self.n_topics):
            aux1 = sparse.coo_matrix(
                (aux0 * L[cooF.row, i], (cooF.row, cooF.col)),
                shape=cooF.shape)
            aux2 = aux1.sum(axis=0).A.ravel() + aux4[i, :]
            X_out[i, :] = X[i, :] * aux2 / aux3[i]
        return X_out

    def _m_step(self, F, L, X):
        L_out = np.zeros((self.n_vocab, self.n_topics))
        cooF = sparse.coo_matrix(F)
        cooY_data = np.dot(L, X)[cooF.row, cooF.col]
        aux2 = X.sum(axis=1)
        aux0 = cooF.data / cooY_data
        for j in range(self.n_topics):
            aux1 = sparse.coo_matrix(
                (aux0 * X[j, cooF.col], (cooF.row, cooF.col)), shape=cooF.shape
                                     ).sum(axis=1).A.ravel()
            L_out[:, j] = L[:, j] * aux1 / aux2[j]
        return L_out

    def fit(self, X, y=None):
        D = self.ngrams_count.fit_transform(X)
        self.vocabulary = self.ngrams_count.get_feature_names()
        self.n_samples, self.n_vocab = D.shape
        F = D.transpose()

        np.random.seed(seed=14)
        self.X_init = np.random.gamma(
            shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
            size=(self.n_topics, self.n_samples))
        self.X_init = self._rescale_X(F, self.X_init)
        np.random.seed(seed=15)
        Lambda_init = np.random.gamma(shape=1, scale=1,
                                      size=(self.n_vocab, self.n_topics))
        self.Lambda_init = self._rescale_Lambda(Lambda_init)

        X_ = self.X_init.copy()
        Lambda = self.Lambda_init.copy()

        for i in range(self.max_iters):
            Lambda_last = Lambda
            for q in range(1):
                X_ = self._e_step(F, Lambda, X_)
            Lambda = self._m_step(F, Lambda, X_)
            L_change = (
                np.linalg.norm(Lambda - Lambda_last) / np.linalg.norm(Lambda))
            rand_idx = np.random.choice(
                range(self.n_samples), size=np.minimum(1000, self.n_sample),
                replace=False)
            kl_divergence = kl_div(
                F[:, rand_idx].A, np.dot(Lambda, X_[:, rand_idx])
                ).sum() / F.shape[1]
            print('iter %d; Lambda-change: %.5f; kl_div: %.3f' %
                  (i, L_change, kl_divergence))
            if L_change < self.tol:
                break
        print('final fit iter: %d' % i)
        self.Lambda = Lambda
        self.X_ = X_
        return self

    def transform(self, X):
        D = self.ngrams_count.transform(X)
        F = D.transpose()
        X_ = np.random.gamma(
            shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
            size=(self.n_topics, D.shape[0]))
        X_ = self._rescale_X(F, X_)
        for i in range(self.max_iters):
            X_last = X_
            X_ = self._e_step(F, self.Lambda, X_)
            mean_change = self._mean_change(X_, X_last)
            if mean_change < self.tol:
                break
        # if normalize=False:
        #     X_ = X_ / X_.sum(axis=0).reshape(1, -1)
        return X_.transpose()


class NgramsMultinomialMixture(BaseEstimator, TransformerMixin):
    """
    Fisher kernel w/r to the mixture of unigrams model (Nigam, 2000).
    """
    # TODO: add stop_criterion; implement k-means for count-vector;
    # implement version with poisson distribution; add online_method

    def __init__(self, n_topics=10, max_iters=100, fisher_kernel=True,
                 beta_init_type=None, max_mean_change_tol=1e-5,
                 ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.ngrams_count = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.n_topics = n_topics  # parameter k
        self.max_iters = max_iters
        self.fisher_kernel = fisher_kernel
        self.beta_init_type = beta_init_type
        self.max_mean_change_tol = max_mean_change_tol

    def _get_most_frequent(self, X):
        unqX, count = np.unique(X, return_counts=True)
        # assert self.n_topics <= len(unqX)
        count_sort_ind = np.argsort(-count)
        most_frequent_cats = unqX[count_sort_ind][:self.n_topics]
        count_most_frequent = count[count_sort_ind][:self.n_topics]
        return most_frequent_cats, count_most_frequent

    def _max_mean_change(self, last_beta, beta):
        max_mean_change = max(abs((last_beta - beta)).sum(axis=1))
        return max_mean_change

    def _e_step(self, D, unqD, X, unqX, theta, beta):
        log_doc_topic_posterior_dict = {}
        log_fisher_kernel_dict = {}
        for m, d in enumerate(unqD):
            log_P_z_theta = np.log(theta)
            log_beta = np.log(beta)
            log_P_d_zbeta = np.array(
                [d.dot(log_beta[i, :])[0] - 1 for i in range(self.n_topics)])
            log_P_dz_thetabeta = log_P_d_zbeta + log_P_z_theta
            log_doc_topic_posterior_dict[unqX[m]] = (
                log_P_dz_thetabeta - logsumexp(log_P_dz_thetabeta))
            log_fisher_kernel_dict[unqX[m]] = (
                log_P_d_zbeta - logsumexp(log_P_dz_thetabeta))

        log_doc_topic_posterior = np.zeros((D.shape[0], self.n_topics))
        log_fisher_kernel = np.zeros((D.shape[0], self.n_topics))
        for m, x in enumerate(X):
            log_doc_topic_posterior[m, :] = log_doc_topic_posterior_dict[x]
            log_fisher_kernel[m, :] = log_fisher_kernel_dict[x]
        return np.exp(log_doc_topic_posterior), np.exp(log_fisher_kernel)

    def _m_step(self, D, _doc_topic_posterior):
        aux = np.dot(_doc_topic_posterior.transpose(), D.toarray())
        beta = np.divide(1 + aux,
                         np.sum(aux, axis=1).reshape(-1, 1) + self.n_vocab)
        theta = ((1 + np.sum(_doc_topic_posterior, axis=0).reshape(-1)) /
                 (self.n_topics + self.n_samples))
        return theta, beta

    def fit(self, X, y=None):
        unqX = np.unique(X)
        unqD = self.ngrams_count.fit_transform(unqX)
        D = self.ngrams_count.transform(X)
        self.vocabulary = self.ngrams_count.get_feature_names()
        self.n_samples, self.n_vocab = D.shape
        prototype_cats, protoype_counts = self._get_most_frequent(X)
        self.theta_prior = protoype_counts / self.n_topics
        protoD = self.ngrams_count.transform(prototype_cats).toarray() + 1e-5
        if self.beta_init_type == 'most-frequent-categories':
            self.beta_prior = protoD / protoD.sum(axis=1).reshape(-1, 1)
        if self.beta_init_type == 'constant':
            self.beta_prior = (np.ones(protoD.shape) /
                               protoD.sum(axis=1).reshape(-1, 1))
        if self.beta_init_type == 'random':
            np.random.seed(seed=42)
            aux = np.random.uniform(0, 1, protoD.shape) + 1e-5
            self.beta_prior = aux / protoD.sum(axis=1).reshape(-1, 1)

        theta, beta = self.theta_prior, self.beta_prior
        _last_beta = np.zeros((self.n_topics, self.n_vocab))
        for i in range(self.max_iters):
            for i in range(0):
                print(i)
            _doc_topic_posterior, _ = self._e_step(D, unqD, X, unqX,
                                                   theta, beta)
            theta, beta = self._m_step(D, _doc_topic_posterior)
            max_mean_change = self._max_mean_change(_last_beta, beta)
            if max_mean_change < self.max_mean_change_tol:
                print('final n_iters: %d' % i)
                print(max_mean_change)
                break
            _last_beta = beta
        self.theta, self.beta = theta, beta
        return self

    def transform(self, X):
        unqX = np.unique(X)
        unqD = self.ngrams_count.transform(unqX)
        D = self.ngrams_count.transform(X)
        if type(self.fisher_kernel) is not bool:
            raise TypeError('fisher_kernel parameter must be boolean.')
        if self.fisher_kernel is True:
            _, Xout = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        if self.fisher_kernel is False:
            Xout, _ = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        return Xout


class AdHocNgramsMultinomialMixture(BaseEstimator, TransformerMixin):
    """
    Fisher kernel w/r to the mixture of unigrams model (Nigam, 2000).
    The dimensionality of the embedding is set to the number of unique
    categories in the training set and the count vector matrix is give
    as initial gues for the parameter beta.
    """

    def __init__(self, n_iters=10, fisher_kernel=True, ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.ngrams_count = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.n_iters = n_iters
        self.fisher_kernel = fisher_kernel

    def _e_step(self, D, unqD, X, unqX, theta, beta):
        doc_topic_posterior_dict = {}
        fisher_kernel_dict = {}
        for m, d in enumerate(unqD):
            P_z_theta = theta
            beta = beta
            P_d_zbeta = np.array(
                [float(d.dot(beta[i, :].transpose()).toarray()) - 1
                 for i in range(self.n_topics)])
            P_dz_thetabeta = P_d_zbeta * P_z_theta
            doc_topic_posterior_dict[unqX[m]] = (
                P_dz_thetabeta / P_dz_thetabeta.sum(axis=0))
            fisher_kernel_dict[unqX[m]] = (
                P_d_zbeta / P_dz_thetabeta.sum(axis=0))

        doc_topic_posterior = np.zeros((D.shape[0], self.n_topics))
        fisher_kernel = np.zeros((D.shape[0], self.n_topics))
        for m, x in enumerate(X):
            doc_topic_posterior[m, :] = doc_topic_posterior_dict[x]
            fisher_kernel[m, :] = fisher_kernel_dict[x]
        return doc_topic_posterior, fisher_kernel

    def _m_step(self, D, _doc_topic_posterior):
        aux = np.dot(_doc_topic_posterior.transpose(), D.toarray())
        beta = np.divide(1 + aux,
                         np.sum(aux, axis=1).reshape(-1, 1) + self.n_vocab)
        theta = ((1 + np.sum(_doc_topic_posterior, axis=0).reshape(-1)) /
                 (self.n_topics + self.n_samples))
        return theta, beta

    def fit(self, X, y=None):
        unqX, self.theta_prior = np.unique(X, return_counts=True)
        self.theta_prior = self.theta_prior/self.theta_prior.sum()
        self.n_topics = len(unqX)
        unqD = self.ngrams_count.fit_transform(unqX)
        D = self.ngrams_count.transform(X)
        self.n_samples, self.n_vocab = D.shape
        self.beta_prior = sparse.csr_matrix(unqD.multiply(1/unqD.sum(axis=1)))
        theta, beta = self.theta_prior, self.beta_prior
        for i in range(self.n_iters):
            _doc_topic_posterior, _ = self._e_step(D, unqD, X, unqX,
                                                   theta, beta)
            theta, beta = self._m_step(D, _doc_topic_posterior)
        self.theta, self.beta = theta, beta
        return self

    def transform(self, X):
        unqX = np.unique(X)
        D = self.ngrams_count.transform(X)
        unqD = self.ngrams_count.transform(unqX)
        if type(self.fisher_kernel) is not bool:
            raise TypeError('fisher_kernel parameter must be boolean.')
        if self.fisher_kernel is True:
            _, Xout = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        if self.fisher_kernel is False:
            Xout, _ = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        return Xout


class MDVEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, clf_type):
        self.clf_type = clf_type

    def fit(self, X, y=None):
        if self.clf_type in ['regression']:
            pass
        if self.clf_type in ['binary-clf', 'multiclass-clf']:
            self.classes_ = np.unique(y)
            self.categories_ = np.unique(X)
            self.class_dict = {c: (y == c) for c in self.classes_}
            self.Exy = {x: [] for x in self.categories_}
            X_dict = {x: (X == x) for x in self.categories_}
            for x in self.categories_:
                for j, c in enumerate(self.classes_):
                    aux1 = X_dict[x]
                    aux2 = self.class_dict[c]
                    self.Exy[x].append(np.mean(aux1[aux2]))
        return self

    def transform(self, X):
        if self.clf_type in ['regression']:
            pass
        if self.clf_type in ['binary-clf', 'multiclass-clf']:
            Xout = np.zeros((len(X), len(self.classes_)))
            for i, x in enumerate(X):
                if x in self.Exy:
                    Xout[i, :] = self.Exy[x]
                else:
                    Xout[i, :] = 0
            return Xout


def test_MDVEncoder():
    X_train = np.array(
        ['hola', 'oi', 'bonjour', 'hola', 'oi', 'hola', 'oi', 'oi', 'hola'])
    y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    X_test = np.array(['hola', 'bonjour', 'hola', 'oi', 'hello'])
    ans = np.array([[2/4, 2/5],
                    [0, 1/5],
                    [2/4, 2/5],
                    [2/4, 2/5],
                    [0, 0]])
    encoder = MDVEncoder(clf_type='binary-clf')
    encoder.fit(X_train, y_train)
    assert np.array_equal(encoder.transform(X_test), ans)


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
            'MDVEncoder': MDVEncoder(self.clf_type),
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
            'NgramsMultinomialMixture':
                NgramsMultinomialMixture(
                    n_topics=self.n_components, max_iters=10),
            'AdHocNgramsMultinomialMixture':
                AdHocNgramsMultinomialMixture(n_iters=0),
            'AdHocIndependentPDF': AdHocIndependentPDF(),
            'GammaPoissonFactorization':
                GammaPoissonFactorization(
                    n_topics=self.n_components),
            'OnlineGammaPoissonFactorization3':
                OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, rescale_W=True, r=.7,
                    tol=1E-4, random_state=18, init='k-means++'),
            'MinHashEncoder': MinHashEncoder(
                n_components=self.n_components),
            'PretrainedFastText':
                PretrainedFastText(n_components=self.n_components),
            'PretrainedFastText2':
                PretrainedFastText(n_components=self.n_components),
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
