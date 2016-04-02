#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:56:26 2015

@author: Moran
"""

from __future__ import division
import os
import subprocess
import sys
import io
from sklearn import cross_validation
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.base import TransformerMixin


class ColumnSelector(TransformerMixin):
    """
    Class for building sklearn Pipeline step. This class should be used to
    select a column from a pandas data frame.
    """

    def __init__(self, column=[]):
        self.columns = column

        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, y, **fit_params)
            return self.transform(X)

        def transform(self, X, **transform_params):
            return X[self.columns]

        def fit(self, X, y=None, **fit_params):
            return self


# logging name: data_mining_module.py
# parameter_log = ''.join([parameter, '.log'])
# logging_file = '/'.join([report_folder, parameter_log])
logging.basicConfig(filename='data_mining_module.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)

DSN = 'postgresql://postgres@localhost:5432/rlp_saarburg'

# load this from dict cursor?
paramdict = {
    # "natflo_hydromorphic": ["0", "1"],
    "natflo_immature_soil": ["0", "1"],
    "natflo_species_richness": ["species_poor", "species_rich"],
    # "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
    # "natflo_usage_intensity": ["high", "medium", "low"],
    "natflo_wetness": ["dry", "mesic", "wet"],
    "natflo_acidity": ["alkaline", "acid"]
    }

trainingparams = {'criterion': 'gini', 'max_depth': 8,
                  'max_features': 'auto', 'min_samples_leaf': 6,
                  'min_samples_split': 8, 'class_weight': 'balanced'}
et_params = {'n_estimators': 250, 'random_state': 0, 'n_jobs': -1,
             'class_weight': 'balanced'}

parameters = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': range(2, 12, 2),
              'criterion': ['gini', 'entropy'],
              'min_samples_split': range(2, 20, 2),
              'min_samples_leaf': range(2, 20, 2,),
              'class_weight': ['balanced']
              }


def decision_tree_neural(X, y):
    parameters = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': range(2, 12, 2),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(2, 20, 2),
        'class_weight': ['balanced']
    }
    gs_dtree = EvolutionaryAlgorithmSearchCV(
        estimator=DecisionTreeClassifier(),
        params=parameters,
        scoring="accuracy",
        cv=KFold(len(X), 5),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    gs_dtree.fit(X, y)
    return gs_dtree.best_params_


def extra_tree(X, y, trainingparams, num_feat=10, filename=None):
    print("y passed in to ET: {}".format(y))
    e_tree = ExtraTreesClassifier(**trainingparams).fit(X, y)
    assert(e_tree is not None)
    importances = e_tree.feature_importances_
    std = np.std([tree.feature_importances_ for tree in e_tree.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    logging.info("Feature ranking:")
    for f in range(num_feat*2):
        logging.info(("{} feature {} ({})".format(f + 1, X.columns[indices[f]],
                                                  importances[indices[f]])))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices][:10],
            color="r", yerr=std[indices][:10], align="center")
    plt.xticks(range(10), X.columns[indices][:10])
    plt.xlim([-1, 10])
    plt.xlabel("Feature")
    plt.show()
    if filename is not None:
        plt.savefig(filename)
    # classification_report(e_tree, X, y)  # , out_file)
    return e_tree, X.columns[indices][:num_feat]


def extra_tree_neural(X, y, num_feat=10):
    parameters = {
        'n_estimators': [600],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None],
        'min_samples_split': range(2, 22, 2),
        'min_samples_leaf': range(2, 22, 2)
    }
    es_etree = EvolutionaryAlgorithmSearchCV(
        estimator=ExtraTreesClassifier(),
        params=parameters,
        scoring="accuracy",
        cv=KFold(len(X), 5),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    es_etree.fit(X, y)
    logging.info("Best ET: {}".format(es_etree.best_score_,
                                      es_etree.best_params_))
    return es_etree.best_params_


def plot_feature_importance(forest, all_df, num_feat=10):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_feat), importances[indices][:num_feat],
            color="r", yerr=std[indices][:num_feat], align="center")
    plt.xticks(range(num_feat), all_df.columns[indices][:num_feat],
               rotation='vertical')
    plt.xlim([-1, num_feat])
    plt.xlabel("Feature")
    plt.show()


def classification_report(clf, x_test, y_test):  # , out_file=None):
    """Prints out a confusion matrix from the classifier object."""
    expected = y_test
    predicted = clf.predict(x_test)

    scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=5,
                                              n_jobs=-1)
    accuracy = "Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(),
                                                        scores.std() * 2)
    return metrics.confusion_matrix(expected, predicted), accuracy


def decision_tree(X, y, trainingparam):  # out_file):
    """Returns a decision tree classifier trained on parameters.

    Keyword arguments:
    trainingparam -- dict full of training parameters
    out_file -- file where the classification report is save to.

    """
    return DecisionTreeClassifier(**trainingparam).fit(X, y)


def get_lineage(tree, feature_names, wet_classes,
                output_file="default.csv"):
    """Iterates over tree and saves all nodes to file."""
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    try:
        if sys.version < '3':
            infile = io.open(output_file, 'wb')
        else:
            infile = io.open(output_file, 'wb')
        with infile as tree_csv:
            idx = np.argwhere(left == -1)[:, 0]

            def recurse(left, right, child, lineage=None):
                if lineage is None:
                    try:
                        lineage = [wet_classes[np.argmax(value[child])]]
                    except KeyError as f:
                        print(f)
                    except IndexError as f:
                        print("{}: {}".format(f, wet_classes))
                if child in left:
                    parent = np.where(left == child)[0].item()
                    split = '<='
                else:
                    parent = np.where(right == child)[0].item()
                    split = '>'

                lineage.append((features[parent], split,
                                threshold[parent], parent))

                if parent == 0:
                    lineage.reverse()
                    return lineage
                else:
                    return recurse(left, right, parent, lineage)

            for child in idx:
                for node in recurse(left, right, child):
                    if not node:
                        continue
                    if type(node) == tuple:
                        if not (isinstance(node[2], float) and
                                isinstance(node[3], int)):
                            print("skipping tuple..")
                            continue
                        a_feature, a_split, a_threshold, a_parent_node = node
                        tree_csv.write("{},{},{:5f},{}\n".format(a_feature,
                                                                 a_split,
                                                                 a_threshold,
                                                                 a_parent_node,
                                                                 ))
                    else:
                        print(node)
                        tree_csv.write(''.join([node, "\n"]))
    except ValueError as e:
        print(e)
    except KeyError as f:
        print(f)
    except UnboundLocalError as e:
        print(e.message)


def read_db_table(table, parameter):
    engine = create_engine(DSN)
    with engine.connect():
        ''' read data from table '''
        datatable = psql.read_sql('''SELECT * FROM {}
                                  where natflo_wetness is not null
                                  and natflo_immature_soil is not null
                                  and natflo_species_richness is not
                                  null'''.format(table),
                                  engine)
        if parameter == 'all':
            # "natflo_acidity",
            parameter = paramdict.keys()
            # all_ys = [datatable[x]  for x in parameter]

            # y = pd.concat(all_ys)
            #datatable = datatable[pd.notnull(datatable[parameter])]
            # datatable = datatable[datatable.isin(paramdict)]
        else:
            parameter = [parameter]
        try:
            for i in ["kul", "kn1", "kn2", "wert_kn2",
                      "we1min", "bodenart_kn1", "we2min"]:
                datatable = datatable.drop(i, axis=1)
        except ValueError as e:
            print(e.message)

        datatable = datatable.fillna(0, axis=1)
        y = datatable[parameter]
        # y = y.apply(str)
        X = datatable.drop(parameter, axis=1)
        X = X.select_dtypes(['float64'])
    return X, y

if __name__ == '__main__':
    # usage: data_mining_module.py -infolder -outfolder -train/test table
    # file stuff, get_data_from_db
    # train, test, write classification report
    # defaults to ./training_cm and ./sci-kit_rules
    if len(sys.argv) < 2:
        print("Please at least give name of table")
        sys.exit(1)
    try:
        table = sys.argv[1]
        parameter = sys.argv[2]
        num_feat = sys.argv[3]
    except IndexError:
        print("Not enough parameters given")
        print("DEFAULT: natflo_wetness, 25")
        num_feat = 25
    report_folder = os.path.join(os.getcwd(), "training_cm")
    rules_folder = os.path.join(os.getcwd(), "rules")
    rules_folder_reduced = os.path.join(os.getcwd(),
                                        "rules_reduced")
    ''' create folders if they do not exist '''
    if not os.path.exists(report_folder):
        os.mkdir(report_folder)
    elif not os.path.exists(rules_folder):
        os.mkdir(rules_folder)
    elif not os.path.exists(rules_folder_reduced):
        os.mkdir(rules_folder_reduced)

    feature_importances_png = os.path.join(report_folder,
                                           'feature_importances.png')
    X, y = read_db_table(table, parameter)
    # print("y: {}".format(y))
    # print("x: {} y: {}".format(len(X), len(y)))
    # kf = KFold(data.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    #for train, test in kf:
    for curr in paramdict:
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        y_train_curr = y_train[curr]
        y_test_curr = y_test[curr]
        dot_file = ''.join([curr, '.dot'])
        dot_file_reduced = ''.join([curr, '_', num_feat, '.dot'])
        my_out_file_reduced = ''.join([rules_folder_reduced, '/',
                                    curr, '.csv'])
        my_out_file = ''.join([rules_folder, '/', curr, '.csv'])
        print("my_out_file: {}".format(my_out_file))
        # dot files go in report_folder
        my_file_dot = os.path.join(report_folder, dot_file)
        my_file_dot_reduced = os.path.join(report_folder, dot_file_reduced)
        print("dot file: {}".format(my_file_dot))
        forest, important_features = extra_tree(X_train, y_train_curr,
                                                et_params,
                                                num_feat=int(num_feat),
                                                filename=feature_importances_png)
        logging.info("ET TEST: {}".format(classification_report(forest, X_test,
                                                        y_test_curr)))
        reduced = X_train[important_features]
        ''' fit classifiers!'''
        logging.info(" ".join(["*** Finding best parameters for DT",
                    "using EvolutionarySearchCV ***"]))
        neural_parameters_reduced = decision_tree_neural(reduced,
                                                        y_train_curr)
        neural_parameters_all = decision_tree_neural(X_train, y_train_curr)
        logging.info("Done fitting DT with EvolutionarySearchCV")
        logging.info(" ".join(["*** Fitting DT with parameters from",
                            "EvolutionarySearchCV and",
                            "{} selected features ***".format(num_feat)]))
        dt = decision_tree(reduced, y_train_curr, neural_parameters_reduced)
        logging.info("DT, num features: {}".format(num_feat))
        logging.info("TRAIN report: {}".format(classification_report(dt,
                                                                    reduced,
                                                                    y_train_curr)))

        logging.info("TEST report: {}".format(classification_report(dt,
                                                                    reduced,
                                                                    y_train_curr)))
        finish_dt_final = """Done fitting DT with EvolutionarySearchCV and {}
                        features""".format(num_feat)
        dt_all = decision_tree(X_train, y_train_curr, neural_parameters_all)
        logging.info("DT ALL features:")
        logging.info("TRAIN:{}".format(classification_report(dt_all, X_train,
                                                            y_train_curr)))
        logging.info("TEST:{}".format(classification_report(dt_all, X_test,
                                                            y_test_curr)))
        #  files
        get_lineage(dt, X_train.columns, paramdict[curr],
                    output_file=my_out_file_reduced)
        # write testing data to grasslands_test
        get_lineage(dt_all, X_train.columns, paramdict[curr],
                    output_file=my_out_file)
        with open(my_file_dot_reduced, 'w+') as f:
            f = tree.export_graphviz(dt_all, out_file=f,
                                    feature_names=X_train.columns,
                                    class_names=paramdict[curr],
                                    filled=True,
                                    rounded=True,
                                    special_characters=False)

        with open(my_file_dot, 'w+') as f:
            f = tree.export_graphviz(dt, out_file=f,
                                    feature_names=X_train.columns,
                                    class_names=paramdict[curr],
                                    filled=True,
                                    rounded=True,
                                    special_characters=False)
        subprocess.call(["dot", "-Tpng", my_file_dot_reduced, "-o",
                        ''.join([report_folder, '/', curr, '_',
                                num_feat, '.png'])])
        subprocess.call(["dot", "-Tpng", my_file_dot, "-o",
                        ''.join([report_folder, '/', curr, '.png'])])
    ''' write test data '''
    engine = create_engine(DSN)
    with engine.connect():
        test = pd.concat([X_test, y_test], axis=1)
        test.index.name = 'id'
        test_table = '_'.join(["test", parameter])
        test.to_sql(test_table, engine, if_exists='replace', index=True)
        test_table = '_'.join([test_table, "reduced"])
        test = pd.concat([reduced, y_test], axis=1)
        test.index.name = 'id'
        test.to_sql(test_table, engine, if_exists='replace',
                    index=True)
