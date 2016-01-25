#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:56:26 2015

@author: Moran
"""

from __future__ import division
import os
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
import pandas.io.sql as psql
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import logging

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
              'class_weight': ['balanced']
              }


def decision_tree_neural(X_train, y_train):
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
        cv=StratifiedKFold(y_train, n_folds=10),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    gs_dtree.fit(X_train, y_train)
    return gs_dtree.best_params_


def extra_tree(X, y, trainingparams, num_feat=10, filename=None):  # out_file
    """ Extra Tree Classifier """
    e_tree = ExtraTreesClassifier(**trainingparams).fit(X, y)
    importances = e_tree.feature_importances_
    std = np.std([tree.feature_importances_ for tree in e_tree.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_feat*2):
        print("{} feature {} ({})".format(f + 1, X.columns[indices[f]],
                                          importances[indices[f]]))

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
    generate_classification_report(e_tree, X, y)  # , out_file)
    return e_tree, X.columns[indices][:num_feat]


def extra_tree_neural(X_train, y_train, num_feat=10):
    parameters = {
        'n_estimators': [600],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None],  # range(2,18,2),
        'min_samples_split': range(2, 22, 2),
        'min_samples_leaf': range(2, 22, 2)
    }
    es_etree = EvolutionaryAlgorithmSearchCV(
        estimator=ExtraTreesClassifier(),
        params=parameters,
        scoring="accuracy",
        cv=StratifiedKFold(y_train, n_folds=10),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    es_etree.fit(X_train, y_train)
    print(es_etree.best_score_, es_etree.best_params_)
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
    report = """Classification report {}:
    {}\n""".format(clf, metrics.classification_report(expected, predicted))
    confusion = """Confusion matrix:
    {}\n""".format(metrics.confusion_matrix(expected, predicted))
    scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=5,
                                              n_jobs=-1)
    accuracy = "Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(),
                                                        scores.std() * 2)
    logging.info(report)

    logging.info(report)
    logging.info(confusion)
    logging.info(accuracy)
    logging.info(scores)
    '''
    if out_file is not None:
        try:
            if sys.version < '3':
                infile = io.open(out_file, 'wb')
            else:
                infile = io.open(out_file, 'wb')
            with infile as classification:
                classification.write(report)
                classification.write(confusion)
                classification.write(accuracy)
                classification.write("{}".format(scores))
        except IOError:
            print("Sorry can't read: {}".format(out_file))
    '''
    return metrics.confusion_matrix(expected, predicted)


def decision_tree(X, y, trainingparam):  # out_file):
    """Returns a decision tree classifier trained on parameters.

    Keyword arguments:
    trainingparam -- dict full of training parameters
    out_file -- file where the classification report is save to.

    """
    d_tree = DecisionTreeClassifier(**trainingparam).fit(X, y)
    generate_classification_report(d_tree, X, y)  # , out_file)
    # dot_data = StringIO()
    return d_tree


def get_lineage(tree, feature_names, wet_classes,
                output_file="default.csv"):
    """Iterates over tree and saves all nodes to file."""
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    # print("value: I{0}".format(value))

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
                                              cv=StratifiedKFold(y_train,
                                                                 n_folds=10),
                                              verbose=1,
                                              population_size=50,
                                              gene_mutation_prob=0.10,
                                              gene_crossover_prob=0.5,
                                              tournament_size=3,
                                              generations_number=10,
                                              n_jobs=-1).fit(X, y)
    # generate_classification_report(ev_search, X, y, out_file)
    return ev_search


if __name__ == '__main__':
    parameter = sys.argv[1]
    homedir = os.path.expanduser('~')
    report_folder = os.path.join(homedir, "test-rlp", "training_cm")
    scikit_folder = os.path.join(homedir, "test-rlp", "sci-kit_rules")
    parameter_log = ''.join([parameter, '.log'])
    feature_importances_png = ''.join([parameter, '.png'])
    feature_full = '/'.join([report_folder, feature_importances_png])
    rule_filename = ''.join([parameter, '.csv'])
    rule_dot = ''.join([parameter, '.dot'])
    my_out_file = '/'.join([scikit_folder, rule_filename])
    my_out_file_dot = '/'.join([scikit_folder, rule_dot])
    logging_file = '/'.join([report_folder, parameter_log])
    logging.basicConfig(filename=logging_file, level=logging.INFO)
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    engine = create_engine(DSN)
    conn = engine.connect()
    # get data
    table_train = "_".join(["grasslands", "train", parameter]).lower()
    table_test = "grasslands_test"
    test_sql = "SELECT * FROM {}".format(table_test)
    train_sql = "SELECT * FROM {}".format(table_train)
    train = psql.read_sql(train_sql, engine)
    test = psql.read_sql(test_sql, engine)
    train = train.fillna(0, axis=1)
    test = test.fillna(0, axis=1)
    X_train = train.drop([parameter], axis=1)
    X_test = test.drop([parameter], axis=1)
    y_train = train[parameter].apply(str)
    y_test = test[parameter].apply(str)
    X_train = X_train.select_dtypes(['float64'])
    X_test = X_test.select_dtypes(['float64'])
    print("train: {}".format(y_train))
    # get 10 most important features
    forest, important_features = extra_tree(X_train, y_train,
                                            et_params, num_feat=25,
                                            filename=feature_full)
    generate_classification_report(forest, X_test, y_test)
    X_train_reduced = train[important_features]
    ''' fit classifiers!'''
    start_ev_search = """*** Fitting DT with 25 features
                        and EvolutionarySearchCV ***"""
    logging.info(start_ev_search)
    neural_parameters = decision_tree_neural(X_train_reduced, y_train)
    finish_ev_search = "Done fitting DT with EvolutionarySearchCV"
    logging.info(finish_ev_search)
    start_dt_final = """*** Fitting DT with parameters from EvolutionarySearchCV and 25
                        selected features ***"""
    logging.info(start_dt_final)
    dt = decision_tree(X_train_reduced, y_train, neural_parameters)
    finish_dt_final = """Done fitting DT with EvolutionarySearchCV and 25
                      features"""
    logging.info(finish_dt_final)
    #  files
    get_lineage(dt, X_train.columns, paramdict[parameter],
                output_file=my_out_file)
    with open(my_out_file_dot, 'w+') as f:
        f = tree.export_graphviz(dt, out_file=f,
                                 feature_names=X_train.columns,
                                 class_names=["dry", "mesic", "very_wet"],
                                 filled=True,
                                 rounded=True,
                                 special_characters=False)
