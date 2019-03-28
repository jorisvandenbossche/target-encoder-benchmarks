import os

from fit_predict_categorical_encoding import fit_predict_categorical_encoding
from datasets import DATASETS as ALL_DATASETS


# Parameters ##################################################################

datasets = [
    'adult',
    'medical_charge',
    'employee_salaries',
    'random-n=100-zipf=2',
    # 'house_prices'
]

datsets = ALL_DATASETS

# Number of jobs for the folds (each model itself is run with n_jobs=1)
n_jobs = 20
# Number of times a shuffled train/test split is done
n_splits = 20
test_size = 1./3
str_preprocess = True
n_components = None
results_path = os.path.join('results', 'target_encoder_benchmarks')

classifiers = [
    # 'GradientBoosting',
    'L2RegularizedLinearModel',
    # 'LGBM',
    'XGB',
    ]


encoders = [
    'Delete',
    'OneHotEncoderDense',
    'TargetEncoder-shrinkage=bayes',
    'TargetEncoder-shrinkage=bayes_1',
    'TargetEncoder-shrinkage=exp_1_1',
    'TargetEncoder-shrinkage=exp_5_1',
    # 'TargetEncoder-dirty_cat',
    # 'TargetEncoder-category_encoders',
    'LeaveOneOutEncoder-category_encoders',
    'TargetEncoder-hcc-bayes',
    'TargetEncoder-hcc-loo',
    ]


fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path)
