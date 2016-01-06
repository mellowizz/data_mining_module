#!/usr/bin/python2
'''
Created on 11.12.2015

@author: Moran
'''
from __future__ import division
from sqlalchemy import create_engine
import pandas.io.sql as psql
import os


def training(engine, csv_folder, paramdict, parameter='all',
             table_name='train'):
    # pd.set_option('display.float_format', float_format='{:f}')
    """ dumps training dataset_ to csv folder """
    sql_str = """ SELECT *
                  FROM \"{0}\"
                  WHERE \"{1}\" IS NOT NULL or
                  \"{1}\" != '' ORDER BY \"{1}\" """.format(table_name,
                                                            parameter)
    print(sql_str)
    mydf = psql.read_sql(sql_str, engine)
    if not parameter:
        raise ValueError
        print ("You must include a parameter! ")
    else:
        mydf = mydf[mydf[parameter].isin(paramdict[parameter])]
    a_df = mydf.select_dtypes(['float'])
    a_df = a_df.fillna(0, axis=1)
    a_df.insert(0, parameter, mydf[parameter])
    cols = list(a_df)
    cols.insert(0, cols.pop(cols.index(parameter)))
    a_df = a_df.ix[:, cols]
    out_table = '_'.join([table_name, parameter])
    a_df.to_sql(out_table, engine, if_exists='replace')
    # conn = engine.connect()
    # conn.execute("vacuum analyze {}".format(out_table))
    '''
    file_path = os.path.join(csv_folder, "{0}_seath.csv".format(out_table))
    a_df.astype(object)
    a_df.astype(object).convert_objects()
    print(a_df.dtypes)
    print("writing file: {0}".format(file_path))
    '''


paramdict = {
    "natflo_wetness": ["dry", "mesic", "very_wet"],
    "natflo_hydromorphic": ["0", "1"],
    "natflo_immature_soil": ["0", "1"],
    "natflo_species_richness": ["species_poor", "species_rich"],
    "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
    "natflo_usage_intensity": ["high", "medium", "low"]
}
try:
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    # PARAM = "natfddlo_wetness"
    engine = create_engine(DSN)
    conn = engine.connect()
    #table_name = 'saarburg_tr'
    labels = paramdict.keys()
    print(labels)
    for param in paramdict:
        training(engine, os.path.join(os.getcwd(), 'data'), paramdict,
                 parameter=param, table_name='grasslands_train')
    '''
    sql_query = "SELECT * FROM {}".format(table_name)
    mydf = psql.read_sql(sql_query, engine)
    X, y = mydf.drop(labels, axis=1), mydf[labels]
    X = X.select_dtypes(['float'])
    # print("X: {}".format(X))
    # print("Y: {}".format(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4)
    for label in labels:
        # print(y)
        print("train: {} ".format(y_train.groupby(label).size()))
        print("test: {} ".format(y_test.groupby(label).size()))
        print(X_train)
        print(out_train)
    out_train = pd.concat(X_train, X_test)
    out_test = pd.concat(y_train, y_test)
    out_train.to_sql('_'.join(["grassland", "train", label]), engine,
                     if_exists='replace')
    out_test.to_sql('_'.join(["grassland", "test", label]), engine,
                    if_exists='replace')
    '''
except BaseException as error:
    print(error)
