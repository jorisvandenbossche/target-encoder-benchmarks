import os
import glob
import json
import datetime
import warnings
from constants import sample_seed

import numpy as np
import pandas as pd

# import datasets.src


BENCHMARK_HOME = os.environ.get('BENCHMARK_HOME', '.')
DATA_HOME = os.environ.get('BENCHMARK_DATA_HOME', '.')


def _populate():

    global DATASET_CLASSES
    DATASET_CLASSES = {
        'adult': AdultDataset,
        'employee_salaries': EmployeeSalariesDataset,
        'medical_charge': MedicalChargeDataset,
        'journal_influence': JournalInfluenceDataset,
        'met_objects': MetObjectsDataset,
        'colleges': CollegesDataset,
        'beer_reviews': BeerReviewsDataset,
        # 'midwest_survey': MidwestSurveyDataset,
        'traffic_violations': TrafficViolationsDataset,
        'crime_data': CrimeDataDataset,
        'public_procurement': PublicProcurementDataset,
        'intrusion_detection': IntrusionDetectionDataset,
        'emobank': EmobankDataset,
        'text_emotion': TextEmotionDataset,
        #
        # 'indultos_espana': IndultosEspanaDataset,
        'open_payments': OpenPaymentsDataset,
        'road_safety': RoadSafetyDataset,
        'consumer_complaints': ConsumerComplaintsDataset,
        'product_relevance': ProductRelevanceDataset,
        'federal_election': FederalElectionDataset,
        'drug_directory': DrugDirectoryDataset,
        # 'french_companies': FrenchCompaniesDataset,
        'dating_profiles': DatingProfilesDataset,
        'cacao_flavors': CacaoFlavorsDataset,
        'wine_reviews': WineReviewsDataset,
        'house_prices': HousePricesDataset,
        'kickstarter_projects': KickstarterProjectsDataset,
        'building_permits': BuildingPermitsDataset,
        'california_housing': CaliforniaHousingDataset,
        'house_sales': HouseSalesDataset,
        'vancouver_employee': VancouverEmployeeDataset,
        'firefighter_interventions': FirefighterInterventionsDataset,
        #
        'random-n=100-zipf=2': RandomDataset,
    }

    global DATASETS
    DATASETS = sorted(DATASET_CLASSES.keys())


dataset_config_file = os.path.join(BENCHMARK_HOME, 'datasets_config.json')
with open(dataset_config_file, 'r') as f:
    DATASET_CONFIG = json.load(f)


def get_dataset(name):
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
    data_folder = os.path.join(DATA_HOME, 'data')
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
        self.special_column = [key for key in self.col_action
                               if self.col_action[key] == 'Special'][0]
        for col in self.col_action:
            check_nan_percentage(self.df, col)
            if self.col_action[col] in ['OneHotEncoderDense', 'OneHotEncoder',
                                        'Special', 'OneHotEncoderDense-1']:
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
        data_path = os.path.join(get_data_folder(), 'adult')
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
        data_file = os.path.join(
            data_path, 'raw', 'Employee_Salaries_-_2016.csv')
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


class JournalInfluenceDataset(Dataset):

    name = 'journal_influence'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'journal_influence')
        data_file = os.path.join(
            data_path, 'raw', 'estimated-article-influence-scores-2015.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file)
        df.drop(["Unnamed: 0"], 1, inplace=True)
        cols = ['citation_count_sum', 'paper_count_sum']
        for c in cols:
            df[c] = float_to_int(df[c], df.index)
        # df['journal_name'] = df['journal_name'].astype('category')
        self.df = df


class MetObjectsDataset(Dataset):

    name = 'met_objects'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'met_objects')
        data_file = os.path.join(
            data_path, 'raw', 'MetObjects.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file, encoding='utf-8')
        # cat_cols = ['Department', 'Dynasty', 'State']
        clean = ['Geography Type', 'State', 'Classification', 'Artist Role',
                 'Artist Prefix', 'Artist Display Bio',
                 'Artist Suffix', 'Geography Type']

        period = []
        for c in df:
            arr = []
            for elt in df[c]:
                if isinstance(elt, str) and '\r\n' in elt:
                    elt = elt.replace('\r\n', '')
                if isinstance(elt, str) and '\u3000' in elt:
                    elt = elt.replace('\u3000', ' ')
                if isinstance(elt, str) and '\x1e' in elt:
                    elt = elt.replace('\x1e', '')
                arr.append(elt)
            df[c] = pd.Series(arr, dtype=df[c].dtype, index=df.index)

        for c in df['Period']:
            if type(c) is str:
                period.append(c)
            else:
                period.append(np.nan)
        df['Period'] = pd.Series(period, dtype=np.object, index=df.index)

        for c in clean:
            tab = []
            for elt in df[c]:
                if elt == '|' or elt == '||' or elt == '(none assigned)':
                    tab.append(np.nan)
                else:
                    tab.append(elt)
            df[c] = pd.Series(tab, dtype=np.object, index=df.index)

        # for c in cat_cols:
        #     df[c] = df[c].astype('category')')
        self.df = df


class CollegesDataset(Dataset):

    name = 'colleges'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'colleges')
        data_file = os.path.join(
            data_path, 'raw', 'Colleges.txt')
        self.file = data_file
        self.path = data_path

    def _get_df(self):

        def _clean_cols(cols, df):
            for c in cols:
                tab = []
                if 'Predominant' in c:
                    for elt in df[c]:
                        if isinstance(elt, str) and 'None' in elt:
                            tab.append(np.nan)
                        else:
                            tab.append(elt)
                    df[c] = pd.Series(tab, dtype=np.object, index=df.index)
                elif 'Mean Earnings' in c or 'Median Earnings' in c:
                    for elt in df[c]:
                        if isinstance(elt, str) and 'PrivacySuppressed' in elt:
                            tab.append(np.nan)
                        elif isinstance(elt, str):
                            tab.append(int(elt))
                        else:
                            tab.append(elt)
                    df[c] = pd.Series(tab, dtype=np.object, index=df.index)
                elif df[c].dtype == float:
                    df[c] = float_to_int(df[c], df.index)

            return df

        df = pd.read_csv(self.file, sep='\t', encoding='latin1',
                         index_col='UNITID')
        df.drop(["Unnamed: 0"], 1, inplace=True)
        df['State'] = df['State'].astype(str)
        cols = ['Undergrad Size', 'Predominant Degree',
                'Average Cost Academic Year', 'Average Cost Program Year',
                'Tuition (Instate)', 'Tuition (Out of state)',
                'Spend per student', 'Faculty Salary',
                'Mean Earnings 6 years', 'Median Earnings 6 years',
                'Mean Earnings 10 years', 'Median Earnings 10 years']
        df = _clean_cols(cols, df)

        # cats = ['State', 'Predominant Degree', 'Highest Degree', 'Ownership',
        #         'Region', 'ZIP']
        # for c in cats:
        #     df[c] = df[c].astype('category')
        self.df = df


class BeerReviewsDataset(Dataset):

    name = 'beer_reviews'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'beer_reviews')
        data_file = os.path.join(
            data_path, 'raw', '"beer_reviews.csv"')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file)
        for c in df:
            arr = []
            for elt in df[c]:
                if isinstance(elt, str) and '\xa0' in elt:
                    elt = elt.replace('\xa0', ' ')
                arr.append(elt)
            df[c] = pd.Series(arr, dtype=df[c].dtype, index=df.index)
        self.df = df


# class MidwestSurveyDataset(Dataset):

#     name = 'midwest_survey'

#     def _set_paths(self):
#         data_path = os.path.join(get_data_folder(), 'midwest_survey')
#         data_file = os.path.join(
#             data_path, 'raw', 'MIDWEST.csv')
#         self.file = data_file
#         self.path = data_path

#     def _get_df(self):
#         df = pd.read_csv(self.file)

#         self.df = df


class TrafficViolationsDataset(Dataset):

    name = 'traffic_violations'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'traffic_violations')
        data_file = os.path.join(data_path, 'raw', 'Traffic_Violations.csv')
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


class CrimeDataDataset(Dataset):

    name = 'crime_data'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'crime_data')
        data_file = os.path.join(
            data_path, 'raw', 'Crime_Data_from_2010_to_Present.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file)
        cols = ['Area Name', 'Victim Sex', 'Victim Descent',
                'Premise Description',
                'Weapon Description', 'Status Description',
                'Crime Code Description']
        df['Victim Age'] = float_to_int(df['Victim Age'], df.index)
        df['Premise Code'] = float_to_int(df['Premise Code'], df.index)
        df['Weapon Used Code'] = float_to_int(df['Weapon Used Code'], df.index)
        df['Crime Code 1'] = float_to_int(df['Crime Code 1'], df.index)
        df['Crime Code 2'] = float_to_int(df['Crime Code 2'], df.index)
        df['Crime Code 3'] = float_to_int(df['Crime Code 3'], df.index)
        df['Crime Code 4'] = float_to_int(df['Crime Code 4'], df.index)
        for c in cols:
            if df[c].dtype == float:
                df[c] = float_to_int(df[c], df.index)
            df[c] = df[c].astype('category')
        self.df = df


class PublicProcurementDataset(Dataset):

    name = 'public_procurement'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'public_procurement')
        data_file = os.path.join(
            data_path, 'raw', 'TED_CAN_2015.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file)
        col = 'AWARD_VALUE_EURO'
        new_col = 'LOG_AWARD_VALUE_EURO'
        df[new_col] = df[col].abs()
        # Predicting the log of the ammount
        df[new_col] = df[new_col].apply(np.log)
        df = df[df[new_col] > 0]

        for c in df:
            arr = []
            for elt in df[c]:
                if isinstance(elt, str) and '\xa0' in elt:
                    elt = elt.replace('\xa0', ' ')
                arr.append(elt)
            df[c] = pd.Series(arr, dtype=df[c].dtype, index=df.index)
        self.df = df


class IntrusionDetectionDataset(Dataset):

    name = 'intrusion_detection'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'intrusion_detection')
        data_file = os.path.join(
            data_path, 'raw', 'kddcup.data_10_percent')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        col_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'attack_type']
        df = pd.read_csv(self.file, header=None, names=col_names)
        for c in df:
            arr = []
            for elt in df[c]:
                if isinstance(elt, str) and '\xa0' in elt:
                    elt = elt.replace('\xa0', ' ')
                arr.append(elt)
            df[c] = pd.Series(arr, dtype=df[c].dtype, index=df.index)
        self.df = df


class EmobankDataset(Dataset):

    name = 'emobank'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'emobank')
        data_file = os.path.join(
            data_path, 'raw', 'emobank.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file, index_col='id')
        self.df = df


class TextEmotionDataset(Dataset):

    name = 'text_emotion'

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'text_emotion')
        data_file = os.path.join(
            data_path, 'raw', 'text_emotion.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df = pd.read_csv(self.file, index_col='tweet_id')
        self.df = df


def float_to_int(col, index):
    c = []
    for elt in col:
        try:
            c.append(int(elt))
        except ValueError:
            c.append(np.nan)
    return pd.Series(c, dtype=np.object, index=index)


# -----------------------------------------------------------------------------


# class IndultosEspanaDataset(Dataset):

#     name = "indultos_espana"

#     def _set_paths(self):
#         data_path = os.path.join(get_data_folder(),
#                                  'bigml', 'Indultos_en_Espana_1996-2013')
#         data_file = os.path.join(data_path, 'raw',
#                                  'Indultos_en_Espana_1996-2013.csv')
#         self.file = data_file
#         self.path = data_path

#     def _get_df(self):
#         self.df = pd.read_csv(self.file)


class OpenPaymentsDataset(Dataset):

    name = "open_payments"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'docs_payments')
        data_file = os.path.join(data_path, 'output', 'DfD.h5')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        # Variable names in Dollars for Docs dataset ######################
        pi_specialty = ['Physician_Specialty']
        drug_nm = ['Name_of_Associated_Covered_Drug_or_Biological1']
        dev_nm = ['Name_of_Associated_Covered_Device_or_Medical_Supply1']
        corp = ['Applicable_Manufacturer_or_Applicable_GPO_Making_' +
                'Payment_Name']
        amount = ['Total_Amount_of_Payment_USDollars']
        dispute = ['Dispute_Status_for_Publication']
        ###################################################################

        if os.path.exists(self.file):
            df = pd.read_hdf(self.file)
            # print('Loading DataFrame from:\n\t%s' % self.file)
        else:
            hdf_files = glob.glob(os.path.join(self.path, 'hdf', '*.h5'))
            hdf_files_ = []
            for file_ in hdf_files:
                if 'RSRCH_PGYR2013' in file_:
                    hdf_files_.append(file_)
                if 'GNRL_PGYR2013' in file_:
                    hdf_files_.append(file_)

            dfd_cols = pi_specialty + drug_nm + dev_nm + corp + amount + \
                dispute
            df_dfd = pd.DataFrame(columns=dfd_cols)
            for hdf_file in hdf_files_:
                if 'RSRCH' in hdf_file:
                    with pd.HDFStore(hdf_file) as hdf:
                        for key in hdf.keys():
                            df = pd.read_hdf(hdf_file, key)
                            df = df[dfd_cols]
                            df['status'] = 'allowed'
                            df = df.drop_duplicates(keep='first')
                            df_dfd = pd.concat([df_dfd, df],
                                               ignore_index=True)
                            print('size: %d, %d' % tuple(df_dfd.shape))
            unique_vals = {}
            for col in df_dfd.columns:
                unique_vals[col] = set(list(df_dfd[col].unique()))

            for hdf_file in hdf_files_:
                if 'GNRL' in hdf_file:
                    with pd.HDFStore(hdf_file) as hdf:
                        for key in hdf.keys():
                            df = pd.read_hdf(hdf_file, key)
                            df = df[dfd_cols]
                            df['status'] = 'disallowed'
                            df = df.drop_duplicates(keep='first')
                            df_dfd = pd.concat([df_dfd, df],
                                               ignore_index=True)
                            print('size: %d, %d' % tuple(df_dfd.shape))
            df_dfd = df_dfd.drop_duplicates(keep='first')
            df_dfd.to_hdf(self.file, 't1')
            df = df_dfd
        df['status'] = (df['status'] == 'allowed')
        self.df = df
        # print_unique_values(df)
        self.col_action = {
            pi_specialty[0]: 'Delete',
            drug_nm[0]: 'Delete',
            dev_nm[0]: 'Delete',
            corp[0]: 'Special',
            amount[0]: 'Numerical',
            dispute[0]: 'OneHotEncoderDense-1',
            'status': 'y'}
        self.dirtiness_type = {
            corp[0]: 'Synonyms; Overlap'
            }
        self.clf_type = 'binary'


class RoadSafetyDataset(Dataset):
    '''Source: https://data.gov.uk/dataset/road-accidents-safety-data
    '''

    name = "road_safety"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'road_safety')
        data_file = [os.path.join(data_path, 'raw', '2015_Make_Model.csv'),
                     os.path.join(data_path, 'raw', 'Accidents_2015.csv'),
                     os.path.join(data_path, 'raw', 'Casualties_2015.csv'),
                     os.path.join(data_path, 'raw', 'Vehicles_2015.csv')]
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        files = self.file
        for filename in files:
            if filename.split('/')[-1] == '2015_Make_Model.csv':
                df_mod = pd.read_csv(filename, low_memory=False)
                df_mod['Vehicle_Reference'] = (df_mod['Vehicle_Reference']
                                               .map(str))
                df_mod['Vehicle_Index'] = (df_mod['Accident_Index'] +
                                           df_mod['Vehicle_Reference'])
                df_mod = df_mod.set_index('Vehicle_Index')
                df_mod = df_mod.dropna(axis=0, how='any', subset=['make'])
        # for filename in files:
        #     if filename.split('/')[-1] == 'Accidents_2015.csv':
        #        df_acc = pd.read_csv(filename).set_index('Accident_Index')
        for filename in files:
            if filename.split('/')[-1] == 'Vehicles_2015.csv':
                df_veh = pd.read_csv(filename)
                df_veh['Vehicle_Reference'] = (df_veh['Vehicle_Reference']
                                               .map(str))
                df_veh['Vehicle_Index'] = (df_veh['Accident_Index'] +
                                           df_veh['Vehicle_Reference'])
                df_veh = df_veh.set_index('Vehicle_Index')
        for filename in files:
            if filename.split('/')[-1] == 'Casualties_2015.csv':
                df_cas = pd.read_csv(filename)
                df_cas['Vehicle_Reference'] = (df_cas['Vehicle_Reference']
                                               .map(str))
                df_cas['Vehicle_Index'] = (df_cas['Accident_Index'] +
                                           df_cas['Vehicle_Reference'])
                df_cas = df_cas.set_index('Vehicle_Index')

        df = df_cas.join(df_mod, how='left', lsuffix='_cas',
                         rsuffix='_model')
        df = df.dropna(axis=0, how='any', subset=['make'])
        df = df[df['Sex_of_Driver'] != 3]
        df = df[df['Sex_of_Driver'] != -1]
        df['Sex_of_Driver'] = df['Sex_of_Driver'] - 1
        self.df = df
        # col_action = {'Casualty_Severity': 'y',
        #               'Casualty_Class': 'Numerical',
        #               'make': 'OneHotEncoderDense',
        #               'model': 'Special'}
        self.file = self.file[0]


class ConsumerComplaintsDataset(Dataset):
    '''Source: https://catalog.data.gov/dataset/
                       consumer-complaint-database
               Documentation: https://cfpb.github.io/api/ccdb//fields.html'''

    name = "consumer_complaints"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'consumer_complaints')
        data_file = os.path.join(data_path, 'raw',
                                 'Consumer_Complaints.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)
        self.df = self.df.dropna(
            axis=0, how='any', subset=['Consumer disputed?'])
        self.df.loc[:, 'Consumer disputed?'] = (
            self.df['Consumer disputed?'] == 'Yes')


class ProductRelevanceDataset(Dataset):

    name = "product_relevance"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'product_relevance')
        data_file = os.path.join(data_path, 'raw', 'train.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, encoding='latin1')


class FederalElectionDataset(Dataset):
    '''Source: https://classic.fec.gov/finance/disclosure/
                       ftpdet.shtml#a2011_2012'''

    name = "federal_election"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'federal_election')
        data_file = os.path.join(data_path, 'raw', 'itcont.txt')
        self.data_dict_file = os.path.join(data_path, 'data_dict.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        df_dict = pd.read_csv(self.data_dict_file)
        self.df = pd.read_csv(self.file, sep='|', encoding='latin1',
                              header=None, names=df_dict['Column Name'])
        # Some donations are negative
        self.df['TRANSACTION_AMT'] = self.df['TRANSACTION_AMT'].abs()
        # Predicting the log of the donation
        self.df['TRANSACTION_AMT'] = self.df[
            'TRANSACTION_AMT'].apply(np.log)
        self.df = self.df[self.df['TRANSACTION_AMT'] > 0]


class DrugDirectoryDataset(Dataset):
    '''Source:
            https://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm'''

    name = "drug_directory"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'drug_directory')
        data_file = os.path.join(data_path, 'raw', 'product.txt')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, sep='\t', encoding='latin1')


# class FrenchCompaniesDataset(Dataset):

#     name = "french_companies"

#     def _set_paths(self):
#         data_path = os.path.join(get_data_folder(), 'french_companies')
#         data_file = [os.path.join(data_path, 'raw',
#                                   'datasets', 'chiffres-cles-2017.csv'),
#                      os.path.join(data_path, 'raw',
#                                   'datasets', 'avis_attribution_2017.csv')]
#         self.file = data_file
#         self.path = data_path

#     def _get_df(self):
#         pass


class DatingProfilesDataset(Dataset):

    name = "dating_profiles"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'dating_profiles')
        data_file = os.path.join(data_path, 'raw', 'profiles.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)


class CacaoFlavorsDataset(Dataset):
    '''Source: https://www.kaggle.com/rtatman/chocolate-bar-ratings/'''

    name = "cacao_flavors"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'cacao_flavors')
        data_file = os.path.join(data_path, 'raw', 'flavors_of_cacao.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)
        self.df['Cocoa\nPercent'] = self.df[
            'Cocoa\nPercent'].astype(str).str[:-1].astype(float)


class WineReviewsDataset(Dataset):
    '''Source: https://www.kaggle.com/zynicide/wine-reviews/home'''

    name = "wine_reviews"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'wine_reviews')
        data_file = os.path.join(
            data_path, 'raw', 'winemag-data_first150k.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)


class HousePricesDataset(Dataset):
    '''Source: https://www.kaggle.com/c/
            house-prices-advanced-regression-techniques/data'''

    name = "house_prices"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'house_prices')
        data_file = os.path.join(data_path, 'raw', 'train.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, index_col=0)
        # Identifies the type of dwelling involved in the sale.
        MSSubClass = {
            20:	'1-STORY 1946 & NEWER ALL STYLES',
            30:	'1-STORY 1945 & OLDER',
            40:	'1-STORY W/FINISHED ATTIC ALL AGES',
            45:	'1-1/2 STORY - UNFINISHED ALL AGES',
            50:	'1-1/2 STORY FINISHED ALL AGES',
            60:	'2-STORY 1946 & NEWER',
            70:	'2-STORY 1945 & OLDER',
            75:	'2-1/2 STORY ALL AGES',
            80:	'SPLIT OR MULTI-LEVEL',
            85:	'SPLIT FOYER',
            90:	'DUPLEX - ALL STYLES AND AGES',
            120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
            150: '1-1/2 STORY PUD - ALL AGES',
            160: '2-STORY PUD - 1946 & NEWER',
            180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
            190: '2 FAMILY CONVERSION - ALL STYLES AND AGES',
            }
        for key, value in MSSubClass.items():
            self.df.replace({'MSSubClass': key}, value, inplace=True)


class KickstarterProjectsDataset(Dataset):
    '''Source: https://www.kaggle.com/kemical/kickstarter-projects'''

    name = "kickstarter_projects"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'kickstarter_projects')
        data_file = os.path.join(
            data_path, 'raw', 'ks-projects-201612.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, encoding='latin1', index_col=0)
        self.df = self.df[self.df['state '].isin(['failed', 'successful'])]
        self.df['state '] = (self.df['state '] == 'successful')
        self.df['usd pledged '] = (
            self.df['usd pledged '].astype(float) + 1E-10).apply(np.log)


class BuildingPermitsDataset(Dataset):
    '''Source:
            https://www.kaggle.com/chicago/chicago-building-permits'''

    name = "building_permits"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'building_permits')
        data_file = os.path.join(data_path, 'raw', 'building-permits.csv.zip')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)
        self.df.columns = self.df.columns.str.strip()
        self.df['ESTIMATED_COST'] = (
            self.df['ESTIMATED_COST'].astype(float) + 1E-10).apply(np.log)


class CaliforniaHousingDataset(Dataset):
    '''Source:
    https://github.com/ageron/handson-ml/tree/master/datasets/housing
    '''

    name = "california_housing"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'california_housing')
        data_file = os.path.join(data_path, 'raw', 'housing.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file)


class HouseSalesDataset(Dataset):
    '''Source: https://www.kaggle.com/harlfoxem/housesalesprediction'''

    name = "house_sales"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'house_sales')
        data_file = os.path.join(data_path, 'raw', 'kc_house_data.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, index_col=0)


class VancouverEmployeeDataset(Dataset):
    '''Source: https://data.vancouver.ca/datacatalogue/
    employeeRemunerationExpensesOver75k.htm

    Remuneration and Expenses for Employees Earning over $75,000
    '''

    name = "vancouver_employee"

    def _set_paths(self):
        data_path = os.path.join(get_data_folder(), 'vancouver_employee')
        data_file = os.path.join(
            data_path, 'raw',
            '2017StaffRemunerationOver75KWithExpenses.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, header=3)
        self.df['Remuneration'] = self.df[
            'Remuneration'].apply(
                lambda x: np.log(float(''.join(str(x).split(',')))))


class FirefighterInterventionsDataset(Dataset):
    '''Source:
    https://www.data.gouv.fr/fr/datasets/interventions-des-pompiers/
    '''

    name = "firefighter_interventions"

    def _set_paths(self):
        data_path = os.path.join(
            get_data_folder(), 'firefighter_interventions')
        data_file = os.path.join(
            data_path, 'raw', 'interventions-hebdo-2010-2017.csv')
        self.file = data_file
        self.path = data_path

    def _get_df(self):
        self.df = pd.read_csv(self.file, sep=';')


# -----------------------------------------------------------------------------


class RandomDataset(Dataset):

    name = 'random-n=100-zipf=2'

    clf_type = 'binary-clf'
    col_action = None

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


_populate()

for name in DATASETS:
    if not hasattr(DATASET_CLASSES[name], 'col_action'):
        DATASET_CLASSES[name].col_action = DATASET_CONFIG[name]["col_action"]
        DATASET_CLASSES[name].clf_type = DATASET_CONFIG[name]["clf_type"]
