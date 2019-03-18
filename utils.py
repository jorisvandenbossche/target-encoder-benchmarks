import glob
import json
import os

import numpy as np
import pandas as pd


SCORE_TYPES = {
    'regression': 'r2 score',
    'binary-clf': 'average precision score',
    'multiclass-clf': 'accuracy score'}


def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4)


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_results_df(datadir):
    """
    Return a long format dataframe of all result
    files included in the directory.

    """
    files = glob.glob(os.path.join(datadir, '*.json'))
    df_all = []

    for f in files:
        f_dict = read_json(f)
        df = pd.DataFrame(f_dict['results'])
        # df = df.drop_duplicates(subset=df.columns[1:])
        dataset = f_dict['dataset']
        df['dataset'] = dataset
        df['encoder'] = f_dict['encoder']
        df['clf'] = f_dict['clf'][0]
        df['n_splits'] = f_dict['n_splits']
        df['results_file'] = os.path.split(f)[1]

        # clf type
        df['score_type'] = SCORE_TYPES[f_dict['clf_type']]
        df_all.append(df)

    df_all = pd.concat(df_all, axis=0, ignore_index=True)
    return df_all
