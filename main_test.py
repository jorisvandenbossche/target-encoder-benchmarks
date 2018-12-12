import os
os.environ['CE_HOME'] = '.'

from fit_predict_categorical_encoding import fit_predict_categorical_encoding


# Parameters ##################################################################

datasets = [
    'house_prices',
    'adult',
    #'indultos_espana',
    #'dating_profiles',
    #'intrusion_detection',
    #'cacao_flavors',
    'california_housing',
    #'house_sales',
]

n_jobs = 20
n_splits = 20
test_size = 1./3
str_preprocess = True
n_components = None
# results_path = os.path.join('results', 'jmlr2018_3')
results = 'results'

classifiers = [
    # 'GradientBoosting',
    'L2RegularizedLinearModel',
    # 'LGBM',
    'XGB',
    ]


encoders = [
    'OneHotEncoderDense',
    'TargetEncoder',
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
