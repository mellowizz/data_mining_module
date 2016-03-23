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
from sklearn.decomposition import KernelPCA
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
import pandas.io.sql as psql
# import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import logging
# from functools import wraps

# logging name: data_mining_module.py
# parameter_log = ''.join([parameter, '.log'])
# logging_file = '/'.join([report_folder, parameter_log])
logging.basicConfig(filename='data_mining_module.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)

DSN = 'postgresql://postgres@localhost:5432/rlp_saarburg'

# load this from dict cursor?
paramdict = {
    "natflo_hydromorphic": ["0", "1"],
    "natflo_immature_soil": ["0", "1"],
    "natflo_species_richness": ["species_poor", "species_rich"],
    "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
    "natflo_usage_intensity": ["high", "medium", "low"],
    "natflo_wetness": ["dry", "mesic", "very_wet"],
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
              'class_weight': [None, 'balanced']
              }


def decision_tree_neural(X, y):
    parameters = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': range(2, 12, 2),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(2, 20, 2),
        'class_weight': [None, 'balanced']
    }
    gs_dtree = EvolutionaryAlgorithmSearchCV(
        estimator=DecisionTreeClassifier(),
        params=parameters,
        scoring="accuracy",
        cv=StratifiedKFold(y, 5),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    gs_dtree.fit(X, y)
    return gs_dtree.best_params_


def extra_tree(X, y, trainingparams, num_feat=10, filename=None):
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
    # generate_classification_report(e_tree, X, y)  # , out_file)
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
        cv=StratifiedKFold(y, 5),
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


def generate_classification_report(clf, x_test, y_test):  # , out_file=None):
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
                    if type(node) == tuple:
                        a_feature, a_split, a_threshold, a_parent_node = node
                        tree_csv.write("{},{},{:5f},{}\n".format(a_feature,
                                                                 a_split,
                                                                 a_threshold,
                                                                 a_parent_node,
                                                                 ))
                    else:
                        tree_csv.write(''.join([node, "\n"]))
    except ValueError as e:
        print(e)
        print(node)
    except KeyError as f:
        print(f)


def evolutionary_pipeline(X, y, pipe_grid, out_file):
    pipeline = Pipeline(steps=[('kpca', KernelPCA()),
                               ('dt', DecisionTreeClassifier())])
    ev_search = EvolutionaryAlgorithmSearchCV(pipeline,
                                              params=pipe_grid,
                                              scoring="accuracy",
                                              cv=StratifiedKFold(y, 5),
                                              verbose=1,
                                              population_size=50,
                                              gene_mutation_prob=0.10,
                                              gene_crossover_prob=0.5,
                                              tournament_size=3,
                                              generations_number=10,
                                              n_jobs=-1).fit(X, y)
    # generate_classification_report(ev_search, X, y, out_file)
    return ev_search


def read_db_table():
    engine = create_engine(DSN)
    with engine.connect():
        ''' read data from table '''
        datatable = psql.read_sql("SELECT * FROM saarburg_grasslands", engine)
        for i in ["kul", "kn1", "kn2", "wert_kn2",
                  "we1min", "bodenart_kn1", "we2min"]:
            datatable = datatable.drop(i, axis=1)
        datatable = datatable.fillna(0, axis=1)
        y = datatable[parameter].apply(str)
        X = datatable.drop([parameter], axis=1)
        X = X.select_dtypes(['float64'])
    return X, y


if __name__ == '__main__':
    # usage: data_mining_module.py -infolder -outfolder -train/test table
    # file stuff, get_data_from_db
    # train, test, write classification report
    # defaults to ./training_cm and ./sci-kit_rules
    try:
        parameter = sys.argv[1]
        num_feat = sys.argv[2]
    except IndexError:
        print("*** Number of features not given!!")
        print("DEFAULT: 25")
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
    dot_file = ''.join([parameter, '.dot'])
    dot_file_reduced = ''.join([parameter, '_', num_feat, '.dot'])
    my_out_file_reduced = ''.join([rules_folder_reduced, '/',
                                   parameter, '.csv'])
    my_out_file = ''.join([rules_folder, '/', parameter, '.csv'])
    print("my_out_file: {}".format(my_out_file))
    # dot files go in report_folder
    my_file_dot = os.path.join(report_folder, dot_file)
    my_file_dot_reduced = os.path.join(report_folder, dot_file_reduced)
    print("dot file: {}".format(my_file_dot))
    X, y = read_db_table()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    forest, important_features = extra_tree(X_train, y_train,
                                            et_params, num_feat=25,
                                            filename=feature_importances_png)
    logging.info("ET: {}".format(generate_classification_report(forest, X_test,
                                                                y_test)))
    reduced = X_train[important_features]
    ''' fit classifiers!'''
    logging.info(" ".join(["*** Finding best parameters for DT",
                 "using EvolutionarySearchCV ***"]))
    neural_parameters_reduced = decision_tree_neural(reduced,
                                                     y_train)
    neural_parameters_all = decision_tree_neural(X_train, y_train)
    logging.info("Done fitting DT with EvolutionarySearchCV")
    logging.info(" ".join(["*** Fitting DT with parameters from",
                          "EvolutionarySearchCV and",
                           "25 selected features ***"]))
    dt = decision_tree(reduced, y_train, neural_parameters_reduced)
    logging.info("DT, num features: {}".format(num_feat))
    logging.info("report: {}".format(generate_classification_report(dt,
                                                                    reduced,
                                                                    y_train)))
    finish_dt_final = """Done fitting DT with EvolutionarySearchCV and {}
                    features""".format(num_feat)
    dt_all = decision_tree(X_train, y_train, neural_parameters_all)
    logging.info("DT ALL features:")
    logging.info("{}".format(generate_classification_report(dt_all, X_train,
                                                            y_train)))
    #  files
    get_lineage(dt, X_train.columns, paramdict[parameter],
                output_file=my_out_file_reduced)
    # write testing data to grasslands_test
    get_lineage(dt_all, X_train.columns, paramdict[parameter],
                output_file=my_out_file)
    with open(my_file_dot_reduced, 'w+') as f:
        f = tree.export_graphviz(dt_all, out_file=f,
                                 feature_names=X_train.columns,
                                 class_names=paramdict[parameter],
                                 filled=True,
                                 rounded=True,
                                 special_characters=False)

    with open(my_file_dot, 'w+') as f:
        f = tree.export_graphviz(dt, out_file=f,
                                 feature_names=X_train.columns,
                                 class_names=paramdict[parameter],
                                 filled=True,
                                 rounded=True,
                                 special_characters=False)
    subprocess.call(["dot", "-Tpng", my_file_dot_reduced, "-o",
                     ''.join([report_folder, '/', parameter, '_',
                              num_feat, '.png'])])
    subprocess.call(["dot", "-Tpng", my_file_dot, "-o",
                     ''.join([report_folder, '/', parameter, '.png'])])
