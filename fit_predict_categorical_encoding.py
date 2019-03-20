import os
import socket
import time
import datetime
import glob
import warnings

import numpy as np

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from joblib import Parallel, delayed

import dirty_cat
import category_encoders
import hccencoders

# local imports
from datasets import get_dataset, get_data_folder
from constants import sample_seed, shuffle_seed, clf_seed
from utils import read_json, write_json


warnings.filterwarnings('ignore', category=ConvergenceWarning)


def array2list(d):
    """
    For a dictionary d, it transforms tuple/array elements to list elements
    """
    if type(d) is dict:
        for k in d:
            if type(d[k]) is dict:
                d[k] = array2list(d[k])
            elif type(d[k]) in [tuple, np.ndarray]:
                d[k] = list(d[k])
            elif type(d[k]) is list:
                for i, j in enumerate(d[k]):
                    d[k][i] = array2list(d[k][i])
    return d


def verify_if_exists(results_path, results_dict):
    results_dict = array2list(results_dict)
    files = glob.glob(os.path.join(results_path, '*.json'))
    # files = [os.path.join(results_path, 'drago2_20170925151218997989.json')]
    for file_ in files:
        data = read_json(file_)
        params_dict = {k: data[k] for k in data
                       if k not in ['results']}
        if params_dict == results_dict:
            return True
    return False


def instantiate_score_metric(clf_type):
    if clf_type == 'regression':
        score_metric = metrics.r2_score
        score_name = 'r2'
    if clf_type == 'binary-clf':
        score_metric = metrics.average_precision_score
        score_name = 'av-prec'
    if clf_type == 'multiclass-clf':
        score_metric = metrics.accuracy_score
        score_name = 'accuracy'
    return score_metric, score_name


def instantiate_estimators(clf_type, classifiers, clf_seed,
                           y=None, **kw):

    score_metric, _ = instantiate_score_metric(clf_type)
    param_grid_LGBM = {
        'learning_rate': [0.1, .05, .5], 'num_leaves': [7, 15, 31]}
    param_grid_XGB = {
        'learning_rate': [0.1, .05, .3], 'max_depth': [3, 6, 9]}
    if clf_type in ['binary-clf']:
        print(('Fraction by class: True: %0.2f; False: %0.2f'
               % (list(y).count(True) / len(y),
                  list(y).count(False) / len(y))))
        cw = 'balanced'
        clfs = {
            'L2RegularizedLinearModel':
                linear_model.LogisticRegressionCV(
                    class_weight=cw, max_iter=100, solver='sag',
                    penalty='l2', n_jobs=1, cv=3, multi_class='ovr'),
            # linear_model.RidgeClassifierCV(
            #     class_weight=cw, cv=5),
            'GradientBoosting':
                ensemble.GradientBoostingClassifier(n_estimators=100),
            'LGBM':
                GridSearchCV(
                    estimator=LGBMClassifier(
                        n_estimators=100, n_jobs=1, is_unbalance=True),
                    param_grid=param_grid_LGBM, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'XGB':
                GridSearchCV(
                    estimator=XGBClassifier(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_XGB, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'KNN': neighbors.KNeighborsClassifier(n_neighbors=5),
            # ensemble.RandomForestClassifier(
            #     n_estimators=100, class_weight=cw)
            }

    elif clf_type in ['multiclass-clf']:
        print('fraction of the most frequent class:',
              max([list(y).count(x) for x in set(list(y))]) / len(list(y)))
        clfs = {
            'L2RegularizedLinearModel':
                linear_model.LogisticRegressionCV(
                    penalty='l2', n_jobs=1, cv=3, multi_class='ovr',
                    solver='sag', max_iter=100),
            # linear_model.RidgeClassifierCV(cv=3),
            'GradientBoosting':
                ensemble.GradientBoostingClassifier(n_estimators=100),
            # ensemble.RandomForestClassifier(
            #     n_estimators=100),
            'LGBM':
                GridSearchCV(
                    estimator=LGBMClassifier(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_LGBM, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'XGB':
                GridSearchCV(
                    estimator=XGBClassifier(
                        n_estimators=100, n_jobs=1, objective='multi:softmax',
                        num_class=len(np.unique(y))),
                    param_grid=param_grid_XGB, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'KNN': neighbors.KNeighborsClassifier(n_neighbors=5),
            }
    elif clf_type in ['regression']:
        clfs = {
            'L2RegularizedLinearModel':
                linear_model.RidgeCV(cv=3),
            'GradientBoosting':
                ensemble.GradientBoostingRegressor(n_estimators=100),
            # ensemble.RandomForestRegressor(
            #     n_estimators=100)
            'LGBM':
                GridSearchCV(
                    estimator=LGBMRegressor(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_LGBM, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'XGB':
                GridSearchCV(
                    estimator=XGBRegressor(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_XGB, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'KNN': neighbors.KNeighborsRegressor(n_neighbors=5),
            }
    else:
        raise ValueError("{} not recognized".format(clf_type))

    clfs = [clfs[clf] for clf in classifiers]
    for clf in clfs:
        try:
            if 'random_state' in clf.estimator.get_params():
                clf.estimator.set_params(random_state=clf_seed)
        except AttributeError:
            if 'random_state' in clf.get_params():
                clf.set_params(random_state=clf_seed)
    return clfs


def select_cross_val(clf_type, n_splits, test_size):
    if clf_type in ['regression']:
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                          random_state=shuffle_seed)
    if clf_type in ['binary-clf', 'multiclass-clf']:
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                    random_state=shuffle_seed)
    return cv


def choose_nrows(dataset_name):
    if dataset_name in [
            'docs_payments',
            'traffic_violations',
            'federal_election',
            'medical_charge',
            'beer_reviews',
            'road_safety',
            'public_procurement',
            'crime_data',
            'met_objects',
            'drug_directory',
            'consumer_complaints',
            'intrusion_detection',
            'kickstarter_projects',
            'building_permits',
            'wine_reviews',
            'firefighter_interventions',
            ]:
        n_rows = 100000
    else:
        n_rows = -1
    return n_rows


def get_column_action(col_action, xcols, encoder, clf_type):

    encoders_dict = {
        'OneHotEncoder':
            OneHotEncoder(handle_unknown='ignore'),
        'OneHotEncoderDense':
            OneHotEncoder(handle_unknown='ignore', sparse=False),
        'TargetEncoder':
            dirty_cat.TargetEncoder(
                clf_type=clf_type, handle_unknown='ignore'),
        'TargetEncoder-dirty_cat':
            dirty_cat.TargetEncoder(
                clf_type=clf_type, handle_unknown='ignore'),
        'TargetEncoder-category_encoders':
            category_encoders.TargetEncoder(),
        'LeaveOneOutEncoder-category_encoders':
            category_encoders.LeaveOneOutEncoder(),
        'TargetEncoder-hcc-bayes':
            hccencoders.HccBayesEncoder(clf_type=clf_type),
        'TargetEncoder-hcc-loo':
            hccencoders.HccLOOEncoder(clf_type=clf_type),
        'Numerical': 'passthrough',
        'Delete': 'drop',
        None: FunctionTransformer(None, validate=True),
    }

    column_action = {}

    for col in xcols:
        enc_name = col_action[col]
        if enc_name == 'Special':
            enc = encoders_dict[encoder]
        else:
            enc = encoders_dict[enc_name]
        column_action[col] = enc

    return column_action


def fit_predict_fold(data, scaler, column_action, clf, encoder,
                     fold, n_splits, train_index, test_index):
    """
    fits and predicts a X with y given multiple parameters.
    """
    # Use ColumnTransformer to combine the features
    transformer = ColumnTransformer([(col, column_action[col], [col])
                                     for col in data.xcols])

    # training

    y = data.df[data.ycol].values
    data_train = data.df.iloc[train_index, :]
    y_train = y[train_index]

    start_encoding = time.time()
    X_train = transformer.fit_transform(data_train[data.xcols], y_train)
    X_train = scaler.fit_transform(X_train, y_train)
    encoding_time = time.time() - start_encoding

    start_training = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_training

    score_metric, score_name = instantiate_score_metric(data.clf_type)

    if data.clf_type in ['regression', 'multiclass-clf']:
        y_pred_train = clf.predict(X_train)
    elif data.clf_type == 'binary-clf':
        try:
            y_pred_train = clf.predict_proba(X_train)[:, 1]
        except AttributeError:
            y_pred_train = clf.decision_function(X_train)

    score_train = score_metric(y_train, y_pred_train)

    train_shape = X_train.shape
    del X_train

    # testing

    data_test = data.df.iloc[test_index, :]
    y_test = y[test_index]
    X_test = transformer.transform(data_test)
    X_test = scaler.transform(X_test)

    if data.clf_type in ['regression', 'multiclass-clf']:
        y_pred = clf.predict(X_test)
    elif data.clf_type == 'binary-clf':
        try:
            y_pred = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred = clf.decision_function(X_test)

    score = score_metric(y_test, y_pred)

    print('%s (%d/%d), ' % (data.name, fold, n_splits),
          'encoder: %s, ' % encoder,
          'n_samp: %d, ' % train_shape[0],
          'n_feat: %d, ' % train_shape[1],
          '%s: %.4f, ' % (score_name, score),
          '%s-train: %.4f, ' % (score_name, score_train),
          'enc-time: %.0f s.' % encoding_time,
          'train-time: %.0f s.' % training_time)
    results = [fold, y_train.shape[0], X_test.shape[1],
               score, score_train, encoding_time, training_time]
    return results


def fit_predict_categorical_encoding(datasets, str_preprocess, encoders,
                                     classifiers, test_size, n_splits, n_jobs,
                                     results_path, model_path=None):
    '''
    Learning with dirty categorical variables.
    '''
    path = get_data_folder()
    results_path = os.path.join(path, results_path)
    model_path = os.path.join(path, model_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for dataset in datasets:
        n_rows = choose_nrows(dataset_name=dataset)
        for encoder in encoders:
            print('Dataset: %s' % dataset)
            data = get_dataset(dataset).get_df()
            data.preprocess(n_rows=n_rows, str_preprocess=str_preprocess)
            print('Data shape: %d, %d' % data.df.shape)

            cv = select_cross_val(data.clf_type, n_splits, test_size)
            scaler = preprocessing.StandardScaler(with_mean=False)

            # Define classifiers
            clfs = instantiate_estimators(
                data.clf_type, classifiers, clf_seed,
                y=data.df.loc[:, data.ycol].values,
                model_path=model_path)

            for i, clf in enumerate(clfs):
                # import pdb; pdb.set_trace()
                # print(
                #     '{}: {} \n{}: {} \n{}: {} \n{}: {} \n{}: {}'.format(
                #         'Prediction column', data.ycol,
                #         'Task type', str(data.clf_type),
                #         'Classifier', clf,
                #         'Encoder', encoder))

                try:
                    try:
                        clf2 = clf.estimator
                    except AttributeError:
                        clf2 = clf
                    clf_name = clf2.__class__.__name__
                    results_dict = {
                        'dataset': data.name,
                        'n_splits': n_splits,
                        'test_size': test_size,
                        'n_rows': n_rows,
                        'encoder': encoder,
                        'str_preprocess': str_preprocess,
                        'clf': [classifiers[i], clf_name, clf2.get_params()],
                        'ShuffleSplit': [cv.__class__.__name__],
                        'scaler': [scaler.__class__.__name__,
                                   scaler.get_params()],
                        'sample_seed': sample_seed,
                        'shuffleseed': shuffle_seed,
                        'col_action': data.col_action,
                        'clf_type': data.clf_type,
                        }

                    if verify_if_exists(results_path, results_dict):
                        print('Prediction already exists.\n')
                        continue

                    start = time.time()

                    column_action = get_column_action(
                        data.col_action, data.xcols, encoder,
                        data.clf_type)

                    pred = Parallel(n_jobs=n_jobs)(
                        delayed(fit_predict_fold)(
                            data, scaler, column_action, clf, encoder,
                            fold, cv.n_splits, train_index, test_index)
                        for fold, (train_index, test_index)
                        in enumerate(
                            cv.split(data.df, data.df[data.ycol].values)))
                    pred = np.array(pred)
                    results = {'fold': list(pred[:, 0]),
                               'n_train_samples': list(pred[:, 1]),
                               'n_train_features': list(pred[:, 2]),
                               'score': list(pred[:, 3]),
                               'train_score': list(pred[:, 4]),
                               'encoding_time': list(pred[:, 5]),
                               'training_time': list(pred[:, 6])}
                    results_dict['results'] = results

                    # Saving results
                    pc_name = socket.gethostname()
                    now = ''.join([c for c in str(datetime.datetime.now())
                                   if c.isdigit()])
                    filename = ('%s_%s_%s_%s_%s.json' %
                                   (pc_name, data.name, classifiers[i],
                                    encoder, now))
                    results_file = os.path.join(results_path, filename)
                    results_dict = array2list(results_dict)
                    write_json(results_dict, results_file)
                    print('prediction time: %.1f s.' % (time.time() - start))
                    print('Saving results to: %s\n' % results_file)
                except:  # noqa
                    print('Prediction failed.\n')
