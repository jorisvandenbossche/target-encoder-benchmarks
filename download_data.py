"""
Download the data.

Using the https://github.com/dirty-cat/datasets scripts, assuming this repo
was cloned in an upper directory (should later be on OpenML, now not yet
available):
- medical_charge
- employee_salaries

Downloaded directly (TODO: update by taking it from OpenML):
- adult

TODO:
- criteo 
- generated

"""
import os
import sys

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


## From dirty_cat/datasets

sys.path.insert(0, '../datasets/src/')

output_file = os.path.join(
        '.', 'data', 'medical_charge', 'raw',
        'Medicare_Provider_Charge_Inpatient_DRG100_FY2011.csv')
if os.path.exists(output_file):
    print("{} already exists".format(output_file))
else:
    import medical_charge
    medical_charge.get_medical_charge_df()

output_file = os.path.join('.', 'data', 'employee_salaries', 'raw', 'rows.csv')
if os.path.exists(output_file):
    print("{} already exists".format(output_file))
else:
    import employee_salaries
    employee_salaries.get_employee_salaries_df()

output_file = os.path.join('.', 'data', 'traffic_violations', 'raw', 'rows.csv')
if os.path.exists(output_file):
    print("{} already exists".format(output_file))
else:
    import traffic_violations
    traffic_violations.get_traffic_violations_df()


## Adult

output_file = os.path.join('.', 'data', 'adult_dataset', 'raw', 'adult.data')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

if os.path.exists(output_file):
    print("{} already exists".format(output_file))
else:
    urlretrieve(url, filename=output_file)
