import os
import datetime
import warnings
from constants import sample_seed

import numpy as np
import pandas as pd


BENCHMARK_HOME = os.environ.get('BENCHMARK_HOME', '.')


def get_dataset(name):
    DATASET_CLASSES = {
        'employee_salaries': EmployeeSalariesDataset,
        'medical_charge': MedicalChargeDataset,
        'adult': AdultDataset,
        'traffic_violations': TrafficViolationsDataset,
        'random-n=100-zipf=2': RandomDataset,
    }
    return DATASET_CLASSES[name]()


def preprocess_data(df, cols):
    def string_normalize(s):
        res = str(s).lower()
        res = ' ' + res + ' '
        # res = ''.join(str(c) for c in res if ((c.isalnum()) or (c == ' ')))
        # if res is '':
        #     res = ' '
        return res
    for col in cols:
        print('Preprocessing column: %s' % col)
        df[col] = [string_normalize(str(r)) for r in df[col]]
    return df


def get_data_folder():
    # hostname = socket.gethostname()
    # if hostname in ['drago', 'drago2', 'drago3']:
    #     data_folder = '/storage/store/work/pcerda/data'
    # elif hostname in ['paradox', 'paradigm', 'parametric', 'parabolic']:
    #     data_folder = '/storage/local/pcerda/data'
    # else:
    #     data_folder = os.path.join(CE_HOME, 'data')
    data_folder = os.path.join(BENCHMARK_HOME, 'data')
    return data_folder


def create_folder(path, folder):
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))
    return


def check_nan_percentage(df, col_name):
    threshold = .15
    missing_fraction = df[col_name].isnull().sum()/df.shape[0]
    if missing_fraction > threshold:
        warnings.warn(
            "Fraction of missing values for column '%s' "
            "(%.3f) is higher than %.3f"
            % (col_name, missing_fraction, threshold))


class Dataset:

    def __init__(self):
        self._set_paths()
        self._get_df()

    def preprocess(self, n_rows=-1, str_preprocess=True,
                   clf_type='regression'):
        self.col_action = {k: v for k, v in self.col_action.items()
                           if v != 'Delete'}
        self.xcols = [key for key in self.col_action
                      if self.col_action[key] is not 'y']
        self.ycol = [key for key in self.col_action
                     if self.col_action[key] is 'y'][0]
        for col in self.col_action:
            check_nan_percentage(self.df, col)
            if self.col_action[col] in ['OneHotEncoderDense', 'Special']:
                self.df = self.df.fillna(value={col: 'na'})
        self.df = self.df.dropna(
            axis=0, subset=[c for c in self.xcols if self.col_action[c]
                            is not 'Delete'] + [self.ycol])

        if n_rows == -1:
            self.df = self.df.sample(
                frac=1, random_state=sample_seed).reset_index(drop=True)
        else:
            self.df = self.df.sample(
                n=n_rows, random_state=sample_seed).reset_index(drop=True)
        if str_preprocess:
            self.df = preprocess_data(
                self.df, [key for key in self.col_action
                          if self.col_action[key] == 'Special'])
        return

    def get_df(self, preprocess_df=True):
        #######################################################################
        # if preprocess_df:
        #     self.preprocess()
        self._get_df()
        self.df = self.df[list(self.col_action)]
        # why not but not coherent with the rest --> self.preprocess
        return self


class AdultDataset(Dataset):
    '''Source: https://archive.ics.uci.edu/ml/datasets/adult'''

    name = 'adult'

    clf_type = 'binary-clf'

    col_action = {
        'age': 'Numerical',
        'workclass': 'OneHotEncoderDense',
        'fnlwgt': 'Delete',
        'education': 'OneHotEncoderDense',
        'education-num': 'Numerical',
        'marital-status': 'OneHotEncoderDense',
        'occupation': 'Special',
        'relationship': 'OneHotEncoderDense',
        'race': 'OneHotEncoderDense',
        'sex': 'OneHotEncoderDense',
        'capital-gain': 'Numerical',
        'capital-loss': 'Numerical',
        'hours-per-week': 'Numerical',
        'native-country': 'OneHotEncoderDense',
        'income': 'y'
    }

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'adult_dataset')
        create_folder(data_path, 'output/results')
        data_file = os.path.join(data_path, 'raw', 'adult.data')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)

        header = ['age', 'workclass', 'fnlwgt', 'education',
                  'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country',
                  'income']
        df = pd.read_csv(self.file, names=header)
        df = df[df['occupation'] != ' ?']
        df = df.reset_index()
        df['income'] = (df['income'] == ' >50K')
        self.df = df


class MedicalChargeDataset(Dataset):

    name = 'medical_charge'

    clf_type = 'regression'

    col_action = {
        'DRG Definition': 'Special',
        'Provider Id': 'Delete',
        'Provider Name': 'Delete',
        'Provider Street Address': 'Delete',
        'Provider City': 'Delete',
        'Provider State': 'OneHotEncoderDense',
        'Provider Zip Code': 'Delete',
        'Hospital Referral Region (HRR) Description': 'Delete',
        'Total Discharges': 'Delete',
        'Average Covered Charges': 'Numerical',
        'Average Total Payments': 'y',
        'Average Medicare Payments': 'Delete'
    }
    # col_action = {
    #     'State': 'OneHotEncoderDense',
    #     'Total population': 'Delete',
    #     'Median age': 'Delete',
    #     '% BachelorsDeg or higher': 'Delete',
    #     'Unemployment rate': 'Delete',
    #     'Per capita income': 'Delete',
    #     'Total households': 'Delete',
    #     'Average household size': 'Delete',
    #     '% Owner occupied housing': 'Delete',
    #     '% Renter occupied housing': 'Delete',
    #     '% Vacant housing': 'Delete',
    #     'Median home value': 'Delete',
    #     'Population growth 2010 to 2015 annual': 'Delete',
    #     'House hold growth 2010 to 2015 annual': 'Delete',
    #     'Per capita income growth 2010 to 2015 annual': 'Delete',
    #     '2012 state winner': 'Delete',
    #     'Medical procedure': 'Special',  # description
    #     'Total Discharges': 'Delete',
    #     'Average Covered Charges': 'Numerical',
    #     'Average Total Payments': 'y'}

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'medical_charge')
        create_folder(data_path, 'output/results')
        data_file = os.path.join(
            data_path, 'raw',
            'Medicare_Provider_Charge_Inpatient_DRG100_FY2011.csv')

        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)


class EmployeeSalariesDataset(Dataset):
    '''Source: https://catalog.data.gov/dataset/employee-salaries-2016'''

    name = 'employee_salaries'

    clf_type = 'regression'

    col_action = {
        'Full Name': 'Delete',
        'Gender': 'OneHotEncoderDense',
        'Current Annual Salary': 'y',
        '2016 Gross Pay Received': 'Delete',
        '2016 Overtime Pay': 'Delete',
        'Department': 'Delete',
        'Department Name': 'OneHotEncoderDense',
        'Division': 'OneHotEncoderDense',
        'Assignment Category': 'OneHotEncoderDense-1',
        'Employee Position Title': 'Special',
        'Underfilled Job Title': 'Delete',
        'Year First Hired': 'Numerical'
    }

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'employee_salaries')
        create_folder(data_path, 'output/results')
        data_file = os.path.join(data_path, 'raw', 'rows.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file)
        # df['Current Annual Salary'] = [float(s[1:]) for s
        #                                in df['Current Annual Salary']]
        df['Year First Hired'] = [datetime.datetime.strptime(
            d, '%m/%d/%Y').year for d
                                    in df['Date First Hired']]
        self.df = df


def float_to_int(col, index):
    c = []
    for elt in col:
        try:
            c.append(int(elt))
        except ValueError as e:
            c.append(np.nan)
    return pd.Series(c, dtype=np.object, index=index)


class TrafficViolationsDataset(Dataset):

    name = 'traffic_violations'

    clf_type = "multiclass-clf"

    col_action = {
        "Accident": "Delete",
        "Agency": "Delete",
        "Alcohol": "OneHotEncoderDense-1",
        "Arrest Type": "OneHotEncoderDense",
        "Article": "Delete",
        "Belts": "OneHotEncoderDense-1",
        "Charge": "Delete",
        "Color": "Delete",
        "Commercial License": "OneHotEncoderDense-1",
        "Commercial Vehicle": "OneHotEncoderDense-1",
        "Contributed To Accident": "Delete",
        "DL State": "Delete",
        "Date Of Stop": "Delete",
        "Description": "Special",
        "Driver City": "Delete",
        "Driver State": "Delete",
        "Fatal": "OneHotEncoderDense-1",
        "Gender": "OneHotEncoderDense",
        "Geolocation": "Delete",
        "HAZMAT": "OneHotEncoderDense",
        "Latitude": "Delete",
        "Location": "Delete",
        "Longitude": "Delete",
        "Make": "Delete",
        "Model": "Delete",
        "Personal Injury": "Delete",
        "Property Damage": "OneHotEncoderDense-1",
        "Race": "OneHotEncoderDense",
        "State": "Delete",
        "SubAgency": "Delete",
        "Time Of Stop": "Delete",
        "VehicleType": "Delete",
        "Violation Type": "y",
        "Work Zone": "OneHotEncoderDense-1",
        "Year": "Numerical"
    }

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'traffic_violations')
        create_folder(data_path, 'output/results')
        data_file = os.path.join(data_path, 'raw', 'rows.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file)
        df['Year'] = float_to_int(df['Year'], df.index)
        clean = ['Make', 'Model']
        for c in clean:
            arr = []
            for elt in df[c]:
                if elt == 'NONE':
                    arr.append(np.nan)
                else:
                    arr.append(elt)
            df[c] = pd.Series(arr, dtype=np.object, index=df.index)

        for c in df:
            arr = []
            for elt in df[c]:
                if isinstance(elt, str) and '\n' in elt:
                    elt = elt.replace('\n', '')
                arr.append(elt)
            df[c] = pd.Series(arr, dtype=df[c].dtype, index=df.index)

        # df['VehicleType'] = df['VehicleType'].astype('category')
        # df['Arrest Type'] = df['Arrest Type'].astype('category')
        # df['Race'] = df['Race'].astype('category')
        # df['Violation Type'] = df['Violation Type'].astype('category')

        self.df = df


class RandomDataset(Dataset):

    name = 'random-n=100-zipf=2'

    clf_type = 'binary-clf'

    def __init__(self):
        pass

    def get_df(self):

        X, y = make_classification_categorical(
            n_samples=100000, n_categorical_features=3,
            n_categories=[100, 10, 3], zipf_param=[2.0, 0.5, 0.1],
            alpha=[.5, .8, 2.0], class_sep=0.5, random_state=42)

        num_columns = ['num{}'.format(i) for i in range(20)]
        cat_columns = ['cat1', 'cat2', 'cat3']
        df = pd.DataFrame(X, columns=num_columns + cat_columns)
        df[cat_columns] = df[cat_columns].astype(int)
        df['target'] = y
        self.df = df

        self.col_action = {
            'cat1': 'Special',
            'cat2': 'OneHotEncoderDense',
            'cat3': 'OneHotEncoderDense',
            'target': 'y'}
        self.col_action.update({key: 'Numerical' for key in num_columns})

        return self


###############################################################################
# # Generate random categorical data
###############################################################################


def bounded_zipf(N, s):
    import scipy.stats as stats

    x = np.arange(1, N+1)
    weights = x ** (-s)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    return bounded_zipf


def generate_categorical_given_y(y, n_categories=10, zipf_param=1.0, alpha=0.5,
                                 random_state=None):
    """
    Approach:
    - Construct frequency distribution for the categories
    - Sample p-values (per category) for each target class
    - For each target class:
        - Combine the frequency distribution with p-values per category
          to create frequency distribution for this target class.
        - Sample from this set of categories (with replacement)
          to fill X corresponding to this target class.
    """
    from scipy.stats import dirichlet
    from sklearn.utils import check_random_state

    generator = check_random_state(random_state)

    n_samples = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    # frequency distribution for the categories
    dist = bounded_zipf(N=n_categories, s=zipf_param)
    categories = np.arange(1, n_categories + 1)
    categorical_distribution = dist.pmf(categories)

    # generate p value (probability for each target class) for each category
    dist_dirichlet = dirichlet([alpha] * n_classes)
    ps = dist_dirichlet.rvs(n_categories, random_state=generator)

    X = np.empty(n_samples, dtype='int')

    for i in range(n_classes):
        mask_i = y == classes[i]
        counts_i = ps[:, i] * categorical_distribution * n_samples
        vals_i = np.repeat(categories, counts_i.round().astype('int64'))
        X[mask_i] = generator.choice(vals_i, mask_i.sum())

    return X


def make_classification_categorical(
        n_samples, n_categorical_features,
        n_categories=10, zipf_param=1.0, alpha=0.5, random_state=None,
        **make_classification_kwargs):
    from sklearn.datasets import make_classification
    from sklearn.utils import check_random_state

    generator = check_random_state(random_state)
    X, y = make_classification(n_samples=n_samples, random_state=generator,
                               **make_classification_kwargs)

    feats = [X]

    for i in range(n_categorical_features):
        n_cats_ = (n_categories[i] if isinstance(n_categories, list) 
                   else n_categories)
        zipf_param_ = (zipf_param[i] if isinstance(zipf_param, list)
                       else zipf_param)
        alpha_ = alpha[i] if isinstance(alpha, list) else alpha

        cat = generate_categorical_given_y(
            y, n_categories=n_cats_, zipf_param=zipf_param_, alpha=alpha_)
        feats.append(np.atleast_2d(cat).T)

    X = np.concatenate(feats, axis=1)

    return X, y
