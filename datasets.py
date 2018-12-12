import os
import pandas as pd
import numpy as np
import glob
import datetime
import socket
import warnings
from sklearn.preprocessing import LabelEncoder
from constants import sample_seed


CE_HOME = os.environ.get('CE_HOME')


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
    hostname = socket.gethostname()
    if hostname in ['drago', 'drago2', 'drago3']:
        data_folder = '/storage/store/work/pcerda/data'
    elif hostname in ['paradox', 'paradigm', 'parametric', 'parabolic']:
        data_folder = '/storage/local/pcerda/data'
    else:
        data_folder = os.path.join(CE_HOME, 'data')
    return data_folder


def create_folder(path, folder):
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))
    return


def print_unique_values(df):
    for col in df.columns:
        print(col, df[col].unique().shape)
        print(df[col].unique())
        print('\n')


def check_nan_percentage(df, col_name):
    threshold = .15
    missing_fraction = df[col_name].isnull().sum()/df.shape[0]
    if missing_fraction > threshold:
        warnings.warn(
            "Fraction of missing values for column '%s' "
            "(%.3f) is higher than %.3f"
            % (col_name, missing_fraction, threshold))


class Data:
    def __init__(self, name):
        self.name = name
        self.configs = None
        self.xcols, self.ycol = None, None

        ''' Given the dataset name, return the respective dataframe as well as
        the the action for each column.'''

        if name == 'adult':
            '''Source: https://archive.ics.uci.edu/ml/datasets/adult'''
            data_path = os.path.join(get_data_folder(), 'adult_dataset')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'adult.data')

        if name == 'beer_reviews':
            '''Source: BigML'''
            data_path = os.path.join(get_data_folder(), 'bigml/beer_reviews/')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'beer_reviews.csv')

        if name == 'midwest_survey':
            '''FiveThirtyEight Midwest Survey
            Original source: https://github.com/fivethirtyeight/data/tree/
                             master/region-survey
            Source: BigML'''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/FiveThirtyEight_Midwest_Survey')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'FiveThirtyEight_Midwest_Survey.csv')

        if name == 'indultos_espana':
            '''Source: '''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/Indultos_en_Espana_1996-2013')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Indultos_en_Espana_1996-2013.csv')

        if name == 'docs_payments':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'docs_payments')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'output', 'DfD.h5')

        if name == 'medical_charge':
            '''Source: BigML'''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/MedicalProviderChargeInpatient')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'MedicalProviderChargeInpatient.csv')

        if name == 'road_safety':
            '''Source: https://data.gov.uk/dataset/road-accidents-safety-data
            '''
            data_path = os.path.join(get_data_folder(), 'road_safety')
            create_folder(data_path, 'output/results')
            data_file = [os.path.join(data_path, 'raw', '2015_Make_Model.csv'),
                         os.path.join(data_path, 'raw', 'Accidents_2015.csv'),
                         os.path.join(data_path, 'raw', 'Casualties_2015.csv'),
                         os.path.join(data_path, 'raw', 'Vehicles_2015.csv')]

        if name == 'consumer_complaints':
            '''Source: https://catalog.data.gov/dataset/
                       consumer-complaint-database
               Documentation: https://cfpb.github.io/api/ccdb//fields.html'''
            data_path = os.path.join(get_data_folder(), 'consumer_complaints')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Consumer_Complaints.csv')

        if name == 'traffic_violations':
            '''Source: https://catalog.data.gov/dataset/
                       traffic-violations-56dda
               Source2: https://data.montgomerycountymd.gov/Public-Safety/
                        Traffic-Violations/4mse-ku6q'''
            data_path = os.path.join(get_data_folder(), 'traffic_violations')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Traffic_Violations.csv')

        if name == 'crime_data':
            '''Source:
               https://catalog.data.gov/dataset/crime-data-from-2010-to-present
               Source2:
               https://data.lacity.org/A-Safe-City/
               Crime-Data-from-2010-to-Present/y8tr-7khq
            '''

            data_path = os.path.join(get_data_folder(), 'crime_data')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Crime_Data_from_2010_to_Present.csv')

        if name == 'employee_salaries':
            '''Source: https://catalog.data.gov/dataset/
                       employee-salaries-2016'''
            data_path = os.path.join(get_data_folder(), 'employee_salaries')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Employee_Salaries_-_2016.csv')

        if name == 'product_relevance':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'product_relevance')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'train.csv')

        if name == 'federal_election':
            '''Source: https://classic.fec.gov/finance/disclosure/
                       ftpdet.shtml#a2011_2012'''
            data_path = os.path.join(get_data_folder(), 'federal_election')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'itcont.txt')
            self.data_dict_file = os.path.join(data_path, 'data_dict.csv')

        if name == 'public_procurement':
            '''Source: https://data.europa.eu/euodp/en/data/dataset/ted-csv'''
            data_path = os.path.join(get_data_folder(), 'public_procurement')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'TED_CAN_2015.csv')

        if name == 'drug_directory':
            '''Source:
            https://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm'''
            data_path = os.path.join(get_data_folder(), 'drug_directory')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'product.txt')

        if name == 'french_companies':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'french_companies')
            create_folder(data_path, 'output/results')
            data_file = [os.path.join(data_path, 'raw',
                                      'datasets', 'chiffres-cles-2017.csv'),
                         os.path.join(data_path, 'raw',
                                      'datasets', 'avis_attribution_2017.csv')]

        if name == 'journal_influence':
            '''Source: https://github.com/FlourishOA/Data/blob/master/
            estimated-article-influence-scores-2015.csv'''
            data_path = os.path.join(get_data_folder(), 'journal_influence')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw',
                'estimated-article-influence-scores-2015.csv')

        if name == 'met_objects':
            '''Source: https://github.com/metmuseum/openaccess'''
            data_path = os.path.join(get_data_folder(), 'met_objects')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'MetObjects.csv')

        if name == 'dating_profiles':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'dating_profiles')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'profiles.csv')

        if name == 'colleges':
            '''Source: https://beachpartyserver.azurewebsites.net/VueBigData/
            DataFiles/Colleges.txt
            '''
            data_path = os.path.join(get_data_folder(), 'colleges')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'Colleges.txt')

        if name == 'cacao_flavors':
            '''Source: https://www.kaggle.com/rtatman/chocolate-bar-ratings/'''
            data_path = os.path.join(get_data_folder(), 'cacao_flavors')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'flavors_of_cacao.csv')

        if name == 'wine_reviews':
            '''Source: https://www.kaggle.com/zynicide/wine-reviews/home'''
            data_path = os.path.join(get_data_folder(), 'wine_reviews')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'winemag-data_first150k.csv')

        if name == 'intrusion_detection':
            '''Source:
            https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data'''
            data_path = os.path.join(get_data_folder(), 'kddcup99')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'kddcup.data_10_percent_corrected')

        if name == 'house_prices':
            '''Source: https://www.kaggle.com/c/
            house-prices-advanced-regression-techniques/data'''
            data_path = os.path.join(get_data_folder(), 'house_prices')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'train.csv')

        if name == 'kickstarter_projects':
            '''Source: https://www.kaggle.com/kemical/kickstarter-projects'''
            data_path = os.path.join(get_data_folder(), 'kickstarter_projects')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'ks-projects-201612.csv')

        if name == 'building_permits':
            '''Source:
            https://www.kaggle.com/chicago/chicago-building-permits'''
            data_path = os.path.join(get_data_folder(), 'building_permits')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'building-permits.csv')

        if name == 'california_housing':
            '''Source:
            https://github.com/ageron/handson-ml/tree/master/datasets/housing
            '''
            data_path = os.path.join(get_data_folder(), 'california_housing')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'housing.csv')

        if name == 'house_sales':
            '''Source: https://www.kaggle.com/harlfoxem/housesalesprediction'''
            data_path = os.path.join(get_data_folder(), 'house_sales')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'kc_house_data.csv')

        if name == 'vancouver_employee':
            '''Source: https://data.vancouver.ca/datacatalogue/
            employeeRemunerationExpensesOver75k.htm

            Remuneration and Expenses for Employees Earning over $75,000
            '''
            data_path = os.path.join(get_data_folder(), 'vancouver_employee')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw',
                '2017StaffRemunerationOver75KWithExpenses.csv')

        if name == 'firefighter_interventions':
            '''Source:
            https://www.data.gouv.fr/fr/datasets/interventions-des-pompiers/
            '''
            data_path = os.path.join(
                get_data_folder(), 'firefighter_interventions')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'interventions-hebdo-2010-2017.csv')

        # add here the path to a new dataset ##################################
        if name == 'new_dataset':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'new_dataset')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'data_file.csv')
        #######################################################################

        self.file = data_file
        self.path = data_path

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

    def make_configs(self, **kw):
        if self.df is None:
            raise ValueError('need data to make column config')
        self.configs = [Config(name=name, kind=self.col_action.get(name), **kw)
                        for name in self.df.columns
                        if name in self.col_action.keys()]
        self.configs = [c for c in self.configs
                        if not (c.kind in ('Delete', 'y'))]

    def get_df(self, preprocess_df=True):
        if self.name == 'adult':
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
            self.col_action = {
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
                'income': 'y'}
            self.clf_type = 'binary-clf'

        if self.name == 'beer_reviews':
            df = pd.read_csv(self.file)
            df.shape
            self.df = df.dropna(axis=0, how='any')

            # print_unique_values(df)
            self.col_action = {
                'brewery_id': 'Delete',
                'brewery_name': 'Delete',
                'review_time': 'Delete',
                'review_overall': 'Delete',
                'review_aroma': 'Numerical',
                'review_appearance': 'Numerical',
                'review_profilename': 'Delete',
                'beer_style': 'y',
                'review_palate': 'Numerical',
                'review_taste': 'Numerical',
                'beer_name': 'Special',
                'beer_abv': 'Delete',
                'beer_beerid': 'Delete'}
            self.clf_type = 'multiclass-clf'

        if self.name == 'midwest_survey':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'RespondentID': 'Delete',
                'In your own words, what would you call the part ' +
                'of the country you live in now?': 'Special',
                'Personally identification as a Midwesterner?':
                    'OneHotEncoderDense',
                'Illinois in MW?': 'OneHotEncoderDense-1',
                'Indiana in MW?': 'OneHotEncoderDense-1',
                'Iowa in MW?': 'OneHotEncoderDense-1',
                'Kansas in MW?': 'OneHotEncoderDense-1',
                'Michigan in MW?': 'OneHotEncoderDense-1',
                'Minnesota in MW?': 'OneHotEncoderDense-1',
                'Missouri in MW?': 'OneHotEncoderDense-1',
                'Nebraska in MW?': 'OneHotEncoderDense-1',
                'North Dakota in MW?': 'OneHotEncoderDense-1',
                'Ohio in MW?': 'OneHotEncoderDense-1',
                'South Dakota in MW?': 'OneHotEncoderDense-1',
                'Wisconsin in MW?': 'OneHotEncoderDense-1',
                'Arkansas in MW?': 'OneHotEncoderDense-1',
                'Colorado in MW?': 'OneHotEncoderDense-1',
                'Kentucky in MW?': 'OneHotEncoderDense-1',
                'Oklahoma in MW?': 'OneHotEncoderDense-1',
                'Pennsylvania in MW?': 'OneHotEncoderDense-1',
                'West Virginia in MW?': 'OneHotEncoderDense-1',
                'Montana in MW?': 'OneHotEncoderDense-1',
                'Wyoming in MW?': 'OneHotEncoderDense-1',
                'ZIP Code': 'Delete',
                'Gender': 'OneHotEncoderDense',
                'Age': 'OneHotEncoderDense',
                'Household Income': 'OneHotEncoderDense',
                'Education': 'OneHotEncoderDense',
                'Location (Census Region)': 'y'}
            le = LabelEncoder()
            ycol = [col for col in self.col_action
                    if self.col_action[col] == 'y']
            self.df[ycol] = le.fit_transform(self.df[ycol[0]].astype(str))
            self.clf_type = 'multiclass-clf'

        if self.name == 'indultos_espana':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'Fecha BOE': 'Delete',
                'Ministerio': 'OneHotEncoderDense-1',
                'Ministro': 'OneHotEncoderDense',
                'Partido en el Gobierno': 'OneHotEncoderDense-1',
                'Género': 'OneHotEncoderDense-1',
                'Tribunal': 'OneHotEncoderDense',
                'Región': 'OneHotEncoderDense',
                'Fecha Condena': 'Delete',
                'Rol en el delito': 'OneHotEncoderDense',
                'Delito': 'Delete',  # 'Special',
                'Año Inicio Delito': 'Numerical',
                'Año Fin Delito': 'Numerical',
                'Tipo de Indulto': 'y',
                'Fecha Indulto': 'Delete',
                'Categoría Cod.Penal': 'Special',
                'Subcategoría Cod.Penal': 'Delete',  # 'Special',
                'Fecha BOE.año': 'Numerical',
                'Fecha BOE.mes': 'Numerical',
                'Fecha BOE.día del mes': 'Numerical',
                'Fecha BOE.día de la semana': 'Numerical',
                'Fecha Condena.año': 'Numerical',
                'Fecha Condena.mes': 'Numerical',
                'Fecha Condena.día del mes': 'Numerical',
                'Fecha Condena.día de la semana': 'Numerical',
                'Fecha Indulto.año': 'Numerical',
                'Fecha Indulto.mes': 'Numerical',
                'Fecha Indulto.día del mes': 'Numerical',
                'Fecha Indulto.día de la semana': 'Numerical'}
            self.df['Tipo de Indulto'] = (
                self.df['Tipo de Indulto'] == 'indultar')
            self.clf_type = 'binary-clf'

        if self.name == 'docs_payments':
            # Variable names in Dollars for Docs dataset ######################
            pi_specialty = ['Physician_Specialty']
            drug_nm = ['Name_of_Associated_Covered_Drug_or_Biological1']
            #    'Name_of_Associated_Covered_Drug_or_Biological2',
            #    'Name_of_Associated_Covered_Drug_or_Biological3',
            #    'Name_of_Associated_Covered_Drug_or_Biological4',
            #    'Name_of_Associated_Covered_Drug_or_Biological5']
            dev_nm = ['Name_of_Associated_Covered_Device_or_Medical_Supply1']
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply2',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply3',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply4',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply5']
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
                                # remove all value thats are not in RSRCH
                                # for col in pi_specialty+drug_nm+dev_nm+corp:
                                #     print(col)
                                #     s1 = set(list(df[col].unique()))
                                #     s2 = unique_vals[col]
                                #     df = df.set_index(col).drop(labels=s1-s2)
                                #            .reset_index()
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
            self.clf_type = 'binary-clf'

        if self.name == 'medical_charge':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'State': 'OneHotEncoderDense',
                'Total population': 'Delete',
                'Median age': 'Delete',
                '% BachelorsDeg or higher': 'Delete',
                'Unemployment rate': 'Delete',
                'Per capita income': 'Delete',
                'Total households': 'Delete',
                'Average household size': 'Delete',
                '% Owner occupied housing': 'Delete',
                '% Renter occupied housing': 'Delete',
                '% Vacant housing': 'Delete',
                'Median home value': 'Delete',
                'Population growth 2010 to 2015 annual': 'Delete',
                'House hold growth 2010 to 2015 annual': 'Delete',
                'Per capita income growth 2010 to 2015 annual': 'Delete',
                '2012 state winner': 'Delete',
                'Medical procedure': 'Special',
                'Total Discharges': 'Delete',
                'Average Covered Charges': 'Numerical',
                'Average Total Payments': 'y'}
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'road_safety':
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
            # print_unique_values(df)
            # col_action = {'Casualty_Severity': 'y',
            #               'Casualty_Class': 'Numerical',
            #               'make': 'OneHotEncoderDense',
            #               'model': 'Special'}
            self.col_action = {
                'Sex_of_Driver': 'y',
                'model': 'Special',
                'make': 'OneHotEncoderDense'}
            self.clf_type = 'binary-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'
            self.file = self.file[0]

        if self.name == 'consumer_complaints':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'Date received': 'Delete',
                'Product': 'OneHotEncoderDense',
                'Sub-product': 'OneHotEncoderDense',
                'Issue': 'Special',
                'Sub-issue': 'Delete',
                'Consumer complaint narrative': 'Delete',  # too long
                'Company public response': 'OneHotEncoderDense',
                'Company': 'Delete',
                'State': 'Delete',
                'ZIP code': 'Delete',
                'Tags': 'Delete',
                'Consumer consent provided?': 'Delete',
                'Submitted via': 'OneHotEncoderDense',
                'Date sent to company': 'Delete',
                'Company response to consumer': 'OneHotEncoderDense',
                'Timely response?': 'OneHotEncoderDense-1',
                'Consumer disputed?': 'y',
                'Complaint ID': 'Delete'
                          }
            self.df = self.df.dropna(
                axis=0, how='any', subset=['Consumer disputed?'])
            self.df.loc[:, 'Consumer disputed?'] = (
                self.df['Consumer disputed?'] == 'Yes')
            self.clf_type = 'binary-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'traffic_violations':
            self.df = pd.read_csv(self.file, low_memory=False)
            # print_unique_values(df)
            self.col_action = {
                'Date Of Stop': 'Delete',
                'Time Of Stop': 'Delete',
                'Agency': 'Delete',
                'SubAgency': 'Delete',  # 'OneHotEncoderDense'
                'Description': 'Special',
                'Location': 'Delete',
                'Latitude': 'Delete',
                'Longitude': 'Delete',
                'Accident': 'Delete',
                'Belts': 'OneHotEncoderDense-1',
                'Personal Injury': 'Delete',
                'Property Damage': 'OneHotEncoderDense-1',
                'Fatal': 'OneHotEncoderDense-1',
                'Commercial License': 'OneHotEncoderDense-1',
                'HAZMAT': 'OneHotEncoderDense',
                'Commercial Vehicle': 'OneHotEncoderDense-1',
                'Alcohol': 'OneHotEncoderDense-1',
                'Work Zone': 'OneHotEncoderDense-1',
                'State': 'Delete',  #
                'VehicleType': 'Delete',  # 'OneHotEncoderDense'
                'Year': 'Numerical',
                'Make': 'Delete',
                'Model': 'Delete',
                'Color': 'Delete',
                'Violation Type': 'y',
                'Charge': 'Delete',  # 'y'
                'Article': 'Delete',  # 'y'
                'Contributed To Accident': 'Delete',  # 'y'
                'Race': 'OneHotEncoderDense',
                'Gender': 'OneHotEncoderDense',
                'Driver City': 'Delete',
                'Driver State': 'Delete',
                'DL State': 'Delete',
                'Arrest Type': 'OneHotEncoderDense',
                'Geolocation': 'Delete'}
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'crime_data':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'DR Number': 'Delete',
                'Date Reported': 'Delete',
                'Date Occurred': 'Delete',
                'Time Occurred': 'Delete',
                'Area ID': 'Delete',
                'Area Name': 'OneHotEncoderDense',
                'Reporting District': 'Delete',
                'Crime Code': 'Delete',
                'Crime Code Description': 'Special',
                'MO Codes': 'Delete',  # 'Special'
                'Victim Age': 'y',  # 'Numerical'
                'Victim Sex': 'OneHotEncoderDense',
                'Victim Descent': 'Delete',
                'Premise Code': 'Delete',
                'Premise Description': 'OneHotEncoderDense',
                'Weapon Used Code': 'Delete',
                'Weapon Description': 'OneHotEncoderDense',
                'Status Code': 'Delete',
                'Status Description': 'Delete',
                'Crime Code 1': 'Delete',
                'Crime Code 2': 'Delete',
                'Crime Code 3': 'Delete',
                'Crime Code 4': 'Delete',
                'Address': 'Delete',
                'Cross Street': 'Delete',  # 'Special'
                'Location ': 'Delete',
                          }
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'employee_salaries':
            df = pd.read_csv(self.file)
            self.col_action = {
                'Full Name': 'Delete',
                'Gender': 'OneHotEncoderDense',
                'Current Annual Salary': 'y',
                '2016 Gross Pay Received': 'Delete',
                '2016 Overtime Pay': 'Delete',
                'Department': 'Delete',
                'Department Name': 'OneHotEncoderDense',
                'Division': 'OneHotEncoderDense',  # 'Special'
                'Assignment Category': 'OneHotEncoderDense-1',
                'Employee Position Title': 'Special',
                'Underfilled Job Title': 'Delete',
                'Year First Hired': 'Numerical'
                          }
            df['Current Annual Salary'] = [float(s[1:]) for s
                                           in df['Current Annual Salary']]
            df['Year First Hired'] = [datetime.datetime.strptime(
                d, '%m/%d/%Y').year for d
                                      in df['Date First Hired']]
            self.df = df
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'product_relevance':
            self.df = pd.read_csv(self.file, encoding='latin1')
            self.col_action = {
                'id': 'Delete',
                'product_uid': 'Delete',
                'product_title': 'Special',
                'search_term': 'Special',
                'relevance': 'y'}
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'federal_election':
            df_dict = pd.read_csv(self.data_dict_file)
            self.df = pd.read_csv(self.file, sep='|', encoding='latin1',
                                  header=None, names=df_dict['Column Name'])
            self.col_action = {
                'CMTE_ID': 'Delete',
                'AMNDT_IND': 'Delete',
                'RPT_TP': 'Delete',
                'TRANSACTION_PGI': 'OneHotEncoderDense',
                'IMAGE_NUM': 'Delete',
                'TRANSACTION_TP': 'OneHotEncoderDense',
                'ENTITY_TP': 'OneHotEncoderDense',
                'NAME': 'Delete',
                'CITY': 'Delete',
                'STATE': 'OneHotEncoderDense',
                'ZIP_CODE': 'Delete',
                'EMPLOYER': 'Delete',
                'OCCUPATION': 'Special',  # 'Special'
                'TRANSACTION_DT': 'Delete',
                'TRANSACTION_AMT': 'y',
                'OTHER_ID': 'Delete',
                'TRAN_ID': 'Delete',
                'FILE_NUM': 'Delete',
                'MEMO_CD': 'Delete',
                'MEMO_TEXT': 'Special',  # 'Special',
                'SUB_ID': 'Delete'}
            # Some donations are negative
            self.df['TRANSACTION_AMT'] = self.df['TRANSACTION_AMT'].abs()
            # Predicting the log of the donation
            self.df['TRANSACTION_AMT'] = self.df[
                'TRANSACTION_AMT'].apply(np.log)
            self.df = self.df[self.df['TRANSACTION_AMT'] > 0]

            self.clf_type = 'regression'

        if self.name == 'public_procurement':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'ID_NOTICE_CAN': 'Delete',
                'YEAR': 'Delete',
                'ID_TYPE': 'OneHotEncoderDense',  # (3/100000)
                'DT_DISPATCH': 'Delete',
                'XSD_VERSION': 'Delete',
                'CANCELLED': 'Delete',  # very unbalanced
                'CORRECTIONS': 'Delete',  # very unbalanced
                'B_MULTIPLE_CAE': 'Delete',
                'CAE_NAME': 'Special',  # se (17388/100000)
                'CAE_NATIONALID': 'Delete',
                'CAE_ADDRESS': 'Delete',
                'CAE_TOWN': 'Delete',  # 'OneHotEncoderDense', #se(6184/100000)
                'CAE_POSTAL_CODE': 'Delete',
                'ISO_COUNTRY_CODE': 'OneHotEncoderDense',
                'B_MULTIPLE_COUNTRY': 'Delete',  # (32/100000)
                'ISO_COUNTRY_CODE_ALL': 'Delete',
                'CAE_TYPE': 'OneHotEncoderDense',  # (10/100000)
                'EU_INST_CODE': 'OneHotEncoderDense',  # (11/100000)
                'MAIN_ACTIVITY': 'OneHotEncoderDense',  # (200/100000)
                'B_ON_BEHALF': 'OneHotEncoderDense',  # (3/100000)
                'B_INVOLVES_JOINT_PROCUREMENT': 'Delete',
                'B_AWARDED_BY_CENTRAL_BODY': 'Delete',
                'TYPE_OF_CONTRACT': 'OneHotEncoderDense',  # (3/100000)
                'TAL_LOCATION_NUTS': 'Delete',  # (4238/542597)
                'B_FRA_AGREEMENT': 'OneHotEncoderDense',  # (3/100000)
                'FRA_ESTIMATED': 'Delete',
                'B_FRA_CONTRACT': 'Delete',
                'B_DYN_PURCH_SYST': 'Delete',
                'CPV': 'Delete',  # 'OneHotEncoderDense',  # (4325/542597)
                'ID_LOT': 'Delete',
                'ADDITIONAL_CPVS': 'Delete',
                'B_GPA': 'OneHotEncoderDense',  # 'OneHotEncoderDense'
                'LOTS_NUMBER': 'Numerical',  # 'Numerical',
                'VALUE_EURO': 'Delete',  # maybe this should be 'y'
                'VALUE_EURO_FIN_1': 'Delete',
                'VALUE_EURO_FIN_2': 'Delete',  # 'y'
                'B_EU_FUNDS': 'OneHotEncoderDense',  # (3/100000)
                'TOP_TYPE': 'OneHotEncoderDense',  # (9/100000)
                'B_ACCELERATED': 'Delete',
                'OUT_OF_DIRECTIVES': 'Delete',
                'CRIT_CODE': 'OneHotEncoderDense',  # (3/100000)
                'CRIT_PRICE_WEIGHT': 'Delete',
                'CRIT_CRITERIA': 'Delete',  # 'not enough data'
                'CRIT_WEIGHTS': 'Delete',  # needs to be treated
                'B_ELECTRONIC_AUCTION': 'Delete',  # 'OneHotEncoderDense',
                'NUMBER_AWARDS': 'Numerical',
                'ID_AWARD': 'Delete',
                'ID_LOT_AWARDED': 'Delete',
                'INFO_ON_NON_AWARD': 'Delete',
                'INFO_UNPUBLISHED': 'Delete',
                'B_AWARDED_TO_A_GROUP': 'Delete',
                'WIN_NAME': 'Delete',  # (216535/542597)
                'WIN_NATIONALID': 'Delete',
                'WIN_ADDRESS': 'Delete',
                'WIN_TOWN': 'Delete',  # (38550/542597)
                'WIN_POSTAL_CODE': 'Delete',
                'WIN_COUNTRY_CODE': 'OneHotEncoderDense',  # (81/100000)
                'B_CONTRACTOR_SME': 'Delete',
                'CONTRACT_NUMBER': 'Delete',
                'TITLE': 'Delete',  # 'Special' (230882/542597)
                'NUMBER_OFFERS': 'Delete',  # num'; noisy
                'NUMBER_TENDERS_SME': 'Delete',
                'NUMBER_TENDERS_OTHER_EU': 'Delete',
                'NUMBER_TENDERS_NON_EU': 'Delete',
                'NUMBER_OFFERS_ELECTR': 'Delete',
                'AWARD_EST_VALUE_EURO': 'Delete',  # 'y'
                'AWARD_VALUE_EURO': 'y',
                'AWARD_VALUE_EURO_FIN_1': 'Delete',  # 'y'
                'B_SUBCONTRACTED': 'Delete',  # 'OneHotEncoderDense'
                'DT_AWARD': 'Delete'}  # 'OneHotEncoderDense'
            ycol = [col for col in self.col_action
                    if self.col_action[col] == 'y'][0]
            self.df[ycol] = self.df[ycol].abs()
            # Predicting the log of the donation
            self.df[ycol] = self.df[ycol].apply(np.log)
            self.df = self.df[self.df[ycol] > 0]
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'drug_directory':
            self.df = pd.read_csv(self.file, sep='\t', encoding='latin1')
            self.col_action = {
                'PRODUCTID': 'Delete',
                'PRODUCTNDC': 'Delete',
                'PRODUCTTYPENAME': 'y',
                'PROPRIETARYNAME': 'Delete',  # 'Special'
                'PROPRIETARYNAMESUFFIX': 'Delete',
                'NONPROPRIETARYNAME': 'Special',  # 'Special'
                'DOSAGEFORMNAME': 'OneHotEncoderDense',  # 'OneHotEncoderDense'
                'ROUTENAME': 'OneHotEncoderDense',  # 'OneHotEncoderDense'
                'STARTMARKETINGDATE': 'Delete',
                'ENDMARKETINGDATE': 'Delete',
                'MARKETINGCATEGORYNAME': 'Delete',  # 'OneHotEncoderDense'
                'APPLICATIONNUMBER': 'Delete',
                'LABELERNAME': 'Delete',
                'SUBSTANCENAME': 'Delete',  # 'OneHotEncoderDense'
                'ACTIVE_NUMERATOR_STRENGTH': 'Delete',
                'ACTIVE_INGRED_UNIT': 'Delete',
                'PHARM_CLASSES': 'Delete',
                'DEASCHEDULE': 'Delete',
                'NDC_EXCLUDE_FLAG': 'Delete',
                'LISTING_RECORD_CERTIFIED_THROUGH': 'Delete'
                }
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'french_companies':
            df1 = pd.read_csv(self.file[0], sep=';')
            df2 = pd.read_csv(self.file[1])
            0/0
            self.df = pd.read_csv(self.file)
            self.col_action = {}
            self.clf_type = 'multiclass-clf'

        if self.name == 'journal_influence':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'avg_cites_per_paper': 'y',
                'journal_name': 'Special',
            }
            self.clf_type = 'regression'

        if self.name == 'met_objects':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'Object Number': 'Delete',
                'Is Highlight': 'Delete',
                'Is Public Domain': 'Delete',
                'Object ID': 'Delete',
                'Department': 'y',
                'Object Name': 'Special',
                'Title': 'Delete',
                'Culture': 'Delete',
                'Period': 'Delete',
                'Dynasty': 'Delete',
                'Reign': 'Delete',
                'Portfolio': 'Delete',
                'Artist Role': 'Delete',
                'Artist Prefix': 'Delete',
                'Artist Display Name': 'Delete',
                'Artist Display Bio': 'Delete',
                'Artist Suffix': 'Delete',
                'Artist Alpha Sort': 'Delete',
                'Artist Nationality': 'Delete',
                'Artist Begin Date': 'Delete',
                'Artist End Date': 'Delete',
                'Object Date': 'Delete',
                'Object Begin Date': 'Delete',
                'Object End Date': 'Delete',
                'Medium': 'Delete',
                'Dimensions': 'Delete',
                'Credit Line': 'Delete',
                'Geography Type': 'Delete',
                'City': 'Delete',
                'State': 'Delete',
                'County': 'Delete',
                'Country': 'Delete',
                'Region': 'Delete',
                'Subregion': 'Delete',
                'Locale': 'Delete',
                'Locus': 'Delete',
                'Excavation': 'Delete',
                'River': 'Delete',
                'Classification': 'Delete',
                'Rights and Reproduction': 'Delete',
                'Link Resource': 'Delete',
                'Metadata Date': 'Delete',
                'Repository': 'Delete',
            }
            self.clf_type = 'multiclass-clf'

        if self.name == 'dating_profiles':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'age': 'y',
                'body_type': 'Delete',
                'diet': 'Special',
                'drinks': 'OneHotEncoderDense',
                'drugs': 'OneHotEncoderDense',
                'education': 'OneHotEncoderDense',
                'essay0': 'Delete',
                'essay1': 'Delete',
                'essay2': 'Delete',
                'essay3': 'Delete',
                'essay4': 'Delete',
                'essay5': 'Delete',
                'essay6': 'Delete',
                'essay7': 'Delete',
                'essay8': 'Delete',
                'essay9': 'Delete',
                'ethnicity': 'Delete',
                'height': 'Numerical',  # 'Numerical',
                'income': 'Delete',
                'job': 'Special',
                'last_online': 'Delete',
                'location': 'Delete',
                'offspring': 'OneHotEncoderDense',
                'orientation': 'OneHotEncoderDense',
                'pets': 'OneHotEncoderDense',
                'religion': 'Delete',
                'sex': 'OneHotEncoderDense',
                'sign': 'OneHotEncoderDense',
                'smokes': 'OneHotEncoderDense',
                'speaks': 'Delete',  # 'Special',
                'status': 'OneHotEncoderDense',
            }
            self.clf_type = 'regression'

        if self.name == 'colleges':
            self.df = pd.read_csv(self.file, sep='\t', encoding='latin1')
            self.col_action = {
                'School Name': 'Special',
                'City': 'Delete',
                'State': 'OneHotEncoderDense',
                'ZIP': 'Delete',
                'Undergrad Size': 'Numerical',
                'Percent White': 'Numerical',
                'Percent Black': 'Numerical',
                'Percent Hispanic': 'Numerical',
                'Percent Asian': 'Numerical',
                'Percent Part Time': 'Numerical',
                'Spend per student': 'Numerical',
                'Percent Pell Grant': 'y',
                'Predominant Degree': 'OneHotEncoderDense',
                'Highest Degree': 'OneHotEncoderDense',
                'Ownership': 'OneHotEncoderDense',
                'Region': 'OneHotEncoderDense',
                'Gender': 'OneHotEncoderDense',
            }
            self.clf_type = 'regression'

        if self.name == 'cacao_flavors':
            self.df = pd.read_csv(self.file)
            self.df['Cocoa\nPercent'] = self.df[
                'Cocoa\nPercent'].astype(str).str[:-1].astype(float)
            self.col_action = {
                'Company \n(Maker-if known)': 'Delete',
                'Specific Bean Origin\nor Bar Name': 'Delete',
                'REF': 'Delete',
                'Review\nDate': 'Delete',
                'Cocoa\nPercent': 'Numerical',
                'Company\nLocation': 'Delete',
                'Rating': 'Numerical',
                'Bean\nType': 'y',
                'Broad Bean\nOrigin': 'Special',
            }
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'wine_reviews':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'country': 'OneHotEncoderDense',
                'description': 'Special',
                'designation': 'Delete',
                'points': 'y',
                'price': 'Numerical',
                'province': 'Delete',
                'region_1': 'Delete',
                'region_2': 'Delete',
                'variety': 'Delete',  # 'OneHotEncoderDense',
                'winery': 'Delete',
            }
            self.df['price'] = self.df['price'].apply(np.log)
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'intrusion_detection':
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
            self.df = pd.read_csv(self.file, header=None, names=col_names)
            # self.df['attack_type'] = (self.df['attack_type'] != 'normal.')

            self.col_action = {
                'duration': 'Numerical',
                'protocol_type': 'OneHotEncoderDense',
                'service': 'Special',
                'flag': 'OneHotEncoderDense',
                'src_bytes': 'Numerical',
                'dst_bytes': 'Numerical',
                'land': 'OneHotEncoderDense',
                'wrong_fragment': 'Numerical',
                'urgent': 'Numerical',
                'hot': 'Numerical',
                'num_failed_logins': 'Numerical',
                'logged_in': 'OneHotEncoderDense',
                'num_compromised': 'Numerical',
                'root_shell': 'Numerical',
                'su_attempted': 'Numerical',
                'num_root': 'Numerical',
                'num_file_creations': 'Numerical',
                'num_shells': 'Numerical',
                'num_access_files': 'Numerical',
                'num_outbound_cmds': 'Numerical',
                'is_host_login': 'OneHotEncoderDense',
                'is_guest_login': 'OneHotEncoderDense',
                'count': 'Numerical',
                'srv_count': 'Numerical',
                'serror_rate': 'Numerical',
                'srv_serror_rate': 'Numerical',
                'rerror_rate': 'Numerical',
                'srv_rerror_rate': 'Numerical',
                'same_srv_rate': 'Numerical',
                'diff_srv_rate': 'Numerical',
                'srv_diff_host_rate': 'Numerical',
                'dst_host_count': 'Numerical',
                'dst_host_srv_count': 'Numerical',
                'dst_host_same_srv_rate': 'Numerical',
                'dst_host_diff_srv_rate': 'Numerical',
                'dst_host_same_src_port_rate': 'Numerical',
                'dst_host_srv_diff_host_rate': 'Numerical',
                'dst_host_serror_rate': 'Numerical',
                'dst_host_srv_serror_rate': 'Numerical',
                'dst_host_rerror_rate': 'Numerical',
                'dst_host_srv_rerror_rate': 'Numerical',
                'attack_type': 'y',
            }
            self.clf_type = 'multiclass-clf'

        if self.name == 'house_prices':
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
            self.col_action = {
                'MSSubClass': 'Special',
                'MSZoning': 'OneHotEncoderDense',
                'LotFrontage': 'Numerical',
                'LotArea': 'Numerical',
                'Street': 'OneHotEncoderDense',
                'Alley': 'OneHotEncoderDense',
                'LotShape': 'OneHotEncoderDense',
                'LandContour': 'OneHotEncoderDense',
                'Utilities': 'Delete',
                'LotConfig': 'OneHotEncoderDense',
                'LandSlope': 'OneHotEncoderDense',
                'Neighborhood': 'OneHotEncoderDense',
                'Condition1': 'OneHotEncoderDense',
                'Condition2': 'Delete',
                'BldgType': 'OneHotEncoderDense',
                'HouseStyle': 'OneHotEncoderDense',
                'OverallQual': 'Numerical',
                'OverallCond': 'Numerical',
                'YearBuilt': 'Numerical',
                'YearRemodAdd': 'Numerical',
                'RoofStyle': 'OneHotEncoderDense',
                'RoofMatl': 'Delete',
                'Exterior1st': 'OneHotEncoderDense',
                'Exterior2nd': 'Delete',
                'MasVnrType': 'OneHotEncoderDense',
                'MasVnrArea': 'Numerical',
                'ExterQual': 'OneHotEncoderDense',
                'ExterCond': 'OneHotEncoderDense',
                'Foundation': 'OneHotEncoderDense',
                'BsmtQual': 'OneHotEncoderDense',
                'BsmtCond': 'OneHotEncoderDense',
                'BsmtExposure': 'OneHotEncoderDense',
                'BsmtFinType1': 'OneHotEncoderDense',
                'BsmtFinSF1': 'Numerical',
                'BsmtFinType2': 'Delete',
                'BsmtFinSF2': 'Delete',
                'BsmtUnfSF': 'Numerical',
                'TotalBsmtSF': 'Numerical',
                'Heating': 'OneHotEncoderDense',
                'HeatingQC': 'OneHotEncoderDense',
                'CentralAir': 'OneHotEncoderDense',
                'Electrical': 'OneHotEncoderDense',
                '1stFlrSF': 'Numerical',
                '2ndFlrSF': 'Numerical',
                'LowQualFinSF': 'Numerical',
                'GrLivArea': 'Numerical',
                'BsmtFullBath': 'Numerical',
                'BsmtHalfBath': 'Numerical',
                'FullBath': 'Numerical',
                'HalfBath': 'Numerical',
                'BedroomAbvGr': 'Numerical',
                'KitchenAbvGr': 'Numerical',
                'KitchenQual': 'OneHotEncoderDense',
                'TotRmsAbvGrd': 'Numerical',
                'Functional': 'OneHotEncoderDense',
                'Fireplaces': 'Numerical',
                'FireplaceQu': 'OneHotEncoderDense',
                'GarageType': 'OneHotEncoderDense',
                'GarageYrBlt': 'Numerical',
                'GarageFinish': 'OneHotEncoderDense',
                'GarageCars': 'Numerical',
                'GarageArea': 'Numerical',
                'GarageQual': 'OneHotEncoderDense',
                'GarageCond': 'OneHotEncoderDense',
                'PavedDrive': 'OneHotEncoderDense',
                'WoodDeckSF': 'Numerical',
                'OpenPorchSF': 'Numerical',
                'EnclosedPorch': 'Numerical',
                '3SsnPorch': 'Numerical',
                'ScreenPorch': 'Numerical',
                'PoolArea': 'Numerical',
                'PoolQC': 'OneHotEncoderDense',
                'Fence': 'OneHotEncoderDense',
                'MiscFeature': 'OneHotEncoderDense',
                'MiscVal': 'Delete',
                'MoSold': 'OneHotEncoderDense',
                'YrSold': 'Numerical',
                'SaleType': 'OneHotEncoderDense',
                'SaleCondition': 'OneHotEncoderDense',
                'SalePrice': 'y',
                }
            self.clf_type = 'regression'

        if self.name == 'kickstarter_projects':
            self.df = pd.read_csv(self.file, encoding='latin1', index_col=0)
            self.df = self.df[self.df['state '].isin(['failed', 'successful'])]
            self.df['state '] = (self.df['state '] == 'successful')
            self.df['usd pledged '] = (
                self.df['usd pledged '].astype(float) + 1E-10).apply(np.log)
            self.col_action = {
                'name ': 'Delete',
                'category ': 'Special',
                'main_category ': 'Delete',
                'currency ': 'Delete',
                'deadline ': 'Delete',
                'goal ': 'Delete',
                'launched ': 'Delete',
                'pledged ': 'Delete',
                'state ': 'y',
                'backers ': 'Delete',
                'country ': 'Delete',
                'usd pledged ': 'Numerical',
                'Unnamed: 13': 'Delete',
                'Unnamed: 14': 'Delete',
                'Unnamed: 15': 'Delete',
                'Unnamed: 16': 'Delete',
                }
            self.clf_type = 'binary-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'building_permits':
            self.df = pd.read_csv(self.file)
            self.df.columns = self.df.columns.str.strip()
            self.df['ESTIMATED_COST'] = (
                self.df['ESTIMATED_COST'].astype(float) + 1E-10).apply(np.log)
            self.col_action = {
                'ESTIMATED_COST': 'y',
                'PERMIT_TYPE': 'OneHotEncoderDense',
                'WORK_DESCRIPTION': 'Special',
                }
            self.clf_type = 'regression'

        if self.name == 'california_housing':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'longitude': 'Delete',
                'latitude': 'Delete',
                'housing_median_age': 'Delete',
                'total_rooms': 'Numerical',
                'total_bedrooms': 'Numerical',
                'population': 'Numerical',
                'households': 'Numerical',
                'median_income': 'Numerical',
                'median_house_value': 'y',
                'ocean_proximity': 'Special',
            }
            self.cat_type = dict()
            self.cat_type['ocean_proximity'] = ['Abbreviation']
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'house_sales':
            self.df = pd.read_csv(self.file, index_col=0)
            self.col_action = {
                'date': 'Delete',
                'price': 'y',
                'bedrooms': 'Numerical',
                'bathrooms': 'Numerical',
                'sqft_living': 'Numerical',
                'sqft_lot': 'Numerical',
                'floors': 'Numerical',
                'waterfront': 'Numerical',
                'view': 'Numerical',
                'condition': 'Numerical',
                'grade': 'Numerical',
                'sqft_above': 'Numerical',
                'sqft_basement': 'Numerical',
                'yr_built': 'Numerical',
                'yr_renovated': 'Delete',
                'zipcode': 'Special',
                'lat': 'Delete',
                'long': 'Delete',
                'sqft_living15': 'Delete',
                'sqft_lot15': 'Delete',
            }
            self.clf_type = 'regression'  # opts: 'regression',
            self.cat_type = dict()
            self.cat_type['zipcode'] = 'ZIP code'
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'vancouver_employee':
            self.df = pd.read_csv(self.file, header=3)
            self.df['Remuneration'] = self.df[
                'Remuneration'].apply(
                    lambda x: np.log(float(''.join(str(x).split(',')))))
            self.col_action = {
                'Name': 'Delete',
                'Department': 'OneHotEncoderDense',
                'Title': 'Special',
                'Remuneration': 'y',
                'Expenses': 'Delete',
            }
            self.cat_type = dict()
            self.clf_type = 'regression'

        if self.name == 'firefighter_interventions':
            self.df = pd.read_csv(self.file, sep=';')
            self.col_action = {
                'ope_code_insee': 'Delete',
                'nb_ope': 'Numerical',
                'ope_annee': 'Numerical',
                'ope_semaine': 'Numerical',
                'ope_categorie': 'y',
                'ope_code_postal': 'Special',
                'ope_nom_commune': 'Delete',
            }
            self.cat_type = dict()
            self.clf_type = 'multiclass-clf'

        # add here info about the dataset #####################################
        if self.name == 'new_dataset':
            self.df = pd.read_csv(self.file)
            self.col_action = {}
            self.cat_type = dict()
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'
        #######################################################################
        # if preprocess_df:
        #     self.preprocess()
        self.df = self.df[list(self.col_action)]
        # why not but not coherent with the rest --> self.preprocess
        return self
