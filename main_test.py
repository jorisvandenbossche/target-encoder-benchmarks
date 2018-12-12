import os

from fit_predict_categorical_encoding import fit_predict_categorical_encoding


# Parameters ##################################################################

datasets = [
    'adult',
    'medical_charge',
    'employee_salaries',
    #'house_prices'
]

n_jobs = 20
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
    'OneHotEncoderDense',
    'TargetEncoder-dirty_cat',
    'TargetEncoder-category_encoders',
    'TargetEncoder-hcc-bayes',
    'TargetEncoder-hcc-loo',
    ]

reduction_methods = [None]

fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 reduction_methods=reduction_methods,
                                 n_components=n_components,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path,
                                 model_path='')
