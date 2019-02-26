import os
import pandas as pd
import datetime
import warnings
from constants import sample_seed


BENCHMARK_HOME = os.environ.get('BENCHMARK_HOME', '.')


def get_dataset(name):
    DATASET_CLASSES = {
        'employee_salaries': EmployeeSalariesDataset,
        'medical_charge': MedicalChargeDataset,
        'adult': AdultDataset,
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
