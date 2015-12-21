#!/usr/bin/python2

# import spectral
# from spectral.algorithms import bdist
# from spectral.algorithms import TrainingClass
from sqlalchemy import create_engine
from pandas.io import sql as psql
# import numpy as np
import os
# from sklearn.preprocessing import LabelEncoder

paramdict = {
    "natflo_wetness": ["dry", "mesic", "very wet"],
    "natflo_hydromorphic": ["0", "1"],
    "natflo_immature_soil": ["0", "1"],
    "natflo_species_richness": ["species_poor", "species_rich"],
    "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
    "natflo_usage_intensity": ["high", "medium", "low"],
    "eagle_vegetationtype": ["graminaceous_herbaceous",
                             "herbaceous", "shrub", "tree"]
    }

DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
engine = create_engine(DSN)
conn = engine.connect()
# get data
parameter = "natflo_hydromorphic"
table_train = "_".join(["grasslands", "train", parameter]).lower()
table_test = "grasslands_test"
test_sql = "SELECT * FROM {}".format(table_test)
train_sql = "SELECT * FROM {}".format(table_train)
train = psql.read_sql(train_sql, engine)
test = psql.read_sql(test_sql, engine)

if __name__ == '__main__':
    homedir = os.path.expanduser('~')
    report_folder = os.path.join(homedir, "test-rlp", "training_cm")
    scikit_folder = os.path.join(homedir, "test-rlp", "sci-kit_rules")
    scores = []
    file_name = '_'.join([parameter, "report.txt"])
    file_name_kpca = '_'.join([parameter, "report_kpca.txt"])
    file_name_et = '_'.join([parameter, "report_et.txt"])
    file_name_reduced = '_'.join([parameter, "report_reduced.txt"])
    out_file = '/'.join([report_folder, file_name])
    out_file_kpca = '/'.join([report_folder, file_name_kpca])
    out_file_et = '/'.join([report_folder, file_name_et])
    out_file_reduced = '/'.join([report_folder, file_name_reduced])
    # prepare data
    train = train.drop(['index'], axis=1)
    train = train.fillna(0, axis=1)
    class_labels = paramdict[parameter]
    print("class labels: {}".format(class_labels))
    # test = test.fillna(0, axis=1)
    # class_combinations = [(x, y) for x in class_labels
    #                      for y in class_labels
    #                      if x != y]
    # print(class_combinations)
    # get 10 of each class:
    sample = train.groupby(parameter).head(10)
    for currcls in paramdict[parameter]:
        print("curr cls: {}".format(currcls))
        currdata = sample[sample[parameter] == currcls]
        # currdata = sample.get_group(currcls)
        currdata = currdata.select_dtypes(['float64'])
        mean = currdata.mean(0)
        stdev = currdata.std()
        print("mean: {}".format(mean))
        print("stdev: {}".format(stdev))
        # calculate bdist and jm
    '''
    X_train = train[parameter]
    X_test = test.drop([parameter], axis=1)
    y_train = train[parameter]
    y_test = test[parameter]
    X_train = X_train.select_dtypes(['float64'])
    X_test = X_test.select_dtypes(['float64'])
    '''
    # for feature in features:
    # calculate Bdist and JM for each class
    #    pass
    # class_labels = y_train
    # for value
    # for row in X_train.iterrows():
    #    print(row)

    '''
    le = LabelEncoder()
    print(le.fit(y_train))
    class_true = spectral.algorithms.TrainingClass(X_train, y_train, index=1)
    class_false = spectral.algorithms.TrainingClass(X_train, y_train, index=2)
    print(spectral.algorithms.bdist(class_true, class_false))
    '''
