# -*- coding: utf-8 -*-
""" Created on Wed Aug  5 22:45:08 2015

@author: Moran """ import os from os.path import expanduser

import pandas.io.sql as psql from sqlalchemy import create_engine import sys
import database.PostGISDatabase as PG from sklearn.cross_validation import
train_test_split import pandas as pd

try:
    from os import scandir
except ImportError:
    from scandir import scandir


def subdirs(path):
    """ Yield directory names not starting with '.' under given path. """
    for entry in scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name


def testing(engine, csv_folder, parameter, table_name="test"):
    """ dumps testing dataset_ to csv folder """
    sql_str = """ SELECT *
                  FROM\"{0}\"
                  WHERE \"{1}\" IS NOT NULL or
                  \"{1}\" != '' ORDER BY \"{1}\" """.format(table_name,
                                                            parameter)
    mydf = psql.read_sql(sql_str, engine)
    mydf = mydf[mydf[parameter].isin(paramdict[parameter])]
    a_df.insert(0, parameter, mydf[parameter])
    # if "/" in a_df[parameter]:
    #    a_df[parameter] = a_df[parameter].replace(to_replace='/', value='-')
    # if " " in a_df[parameter]:
    #    a_df[parameter] = re.sub(r"[\\s]", "_", a_df[parameter])
    cols = list(a_df)
    cols.insert(0, cols.pop(cols.index(parameter))) a_df = a_df.ix[:, cols]
    out_table = '_'.join(["train", parameter])
    a_df.to_sql(out_table, engine, if_exists='replace')
    file_path = os.path.join(csv_folder, "{0}.csv".format(out_table))
    a_df.astype(object) a_df.astype(object).convert_objects()
    print(a_df.dtypes)
    print("writing file: {0}".format(file_path))
    a_df.to_csv(file_path, index=False, encoding='utf-8')


def training(engine, csv_folder, paramdict, parameter='all',
             table_name='train'):
    # pd.set_option('display.float_format', float_format='{:f}')
    """ dumps training dataset_ to csv folder """
    sql_str = """ SELECT *
                  FROM \"{0}\"
                  WHERE \"{1}\" IS NOT NULL or
                  \"{1}\" != '' ORDER BY \"{1}\" """.format(table_name,
                                                            parameter)
    print(sql_str) mydf =
    psql.read_sql(sql_str, engine)
    if not parameter:
        raise ValueError print
        ("You must include a parameter! ")
    else:
        mydf = mydf[mydf[parameter].isin(paramdict[parameter])]
    a_df = mydf.select_dtypes(['float'])
    a_df = a_df.fillna(0, axis=1)
    a_df.insert(0, parameter, mydf[parameter])
    # if "/" in a_df[parameter]:
    #    a_df[parameter] = a_df[parameter].replace(to_replace='/', value='-')
    # if " " in a_df[parameter]:
    #    a_df[parameter] = re.sub(r"[\\s]", "_", a_df[parameter])
    cols = list(a_df) cols.insert(0, cols.pop(cols.index(parameter)))
    a_df = a_df.ix[:, cols]
    out_table = '_'.join([table_name, parameter])
    a_df.to_sql(out_table, engine, if_exists='replace')
    file_path = os.path.join(csv_folder, "{0}_seath.csv".format(out_table))
    a_df.astype(object)
    a_df.astype(object).convert_objects()
    print(a_df.dtypes)
    print("writing file: {0}".format(file_path))
    a_df = a_df.sample(n=20)
    a_df.to_csv(file_path, index=False, encoding='utf-8')

def get_table_names(myfolder): """ returns list of table names from the csv
files in given folder. """ return [fname.name.split('.csv')[0] for fname in
                                   scandir(myfolder) if not
                                   fname.name.startswith('.') and
                                   fname.is_file()]


def get_csv_folder(csv_folder="csvDump"):
    """" returns the csv folder where csv files should be dumped from database. """
    if os.name == 'nt':
        home = os.environ.get("USERPROFILE")
    else:
        home = expanduser("~")
    test_root = os.path.join(home, "test-rlp")
    csv_path = os.path.join(test_root, "csvDump")
    if csv_folder not in subdirs(test_root):
        os.mkdir(csv_path)
    return csv_path

if __name__ == '__main__':
    INPUT_TABLE = "test"
    paramdict = { "natflo_wetness": ["dry", "mesic", "very wet"],
                   "natflo_depression": ["0", "1"],
                    "natflo_hydromorphic": ["0", "1"],
                    "natflo_immature_soil": ["0", "1"],
                    "natflo_species_richness": ["species_poor", "species_rich"],
                    "natflo_usage": ["grazing", "mowing", "orchards",
                                     "vineyards"],
                    "natflo_usage_intensity": ["high", "medium", "low"],
                    "eagle_vegetationtype": ["graminaceous_herbaceous",
                                             "herbaceous", "shrub", "tree"]
                }
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    PARAM = "natflo_wetness"
    engine = create_engine(DSN)
    for param in paramdict:
        training(engine, os.path.join(os.getcwd(), 'data'), paramdict,
                 parameter=param, table_name='test') 
    #testing(engine, os.path.join(os.getcwd(), 'data'), PARAM, "test")
