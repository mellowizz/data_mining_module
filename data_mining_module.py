#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:56:26 2015

@author: Moran
"""

from __future__ import division
from sknn.platform import cpu32, threading
# from IPython.display import Image
# from sklearn.externals.six import StringIO
# import pydot
import os
import sys
import io
# from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
# from sklearn.decomposition import RandomizedPCA
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
import pandas.io.sql as psql
from sqlalchemy import create_engine
import numpy as np
# import pydot
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
                  'max_features': 'auto', 'min_samples_leaf': 2,
                  'min_samples_split': 8, 'class_weight': 'balanced'}
et_params = {'n_estimators': 250, 'random_state': 0, 'n_jobs': -1,
             'class_weight': 'balanced'}


parameters = {'dt__max_features': ['auto', 'sqrt', 'log2'],
              'dt__max_depth': range(2, 10, 2),
              'dt__criterion': ['gini', 'entropy'],
              'dt__min_samples_split': range(2, 20, 2),
              'dt__min_samples_leaf': range(4, 10, 2,),
              'dt__class_weight': ['balanced'],
              'spca__n_components': [20, 25, 30],
              #'spca__kernel': ["poly", "rbf"],
              'spca__n_jobs': [-1],
              'et__n_estimators': range(600, 800, 100),
              'et__max_features': ['auto', 'sqrt', 'log2'],
              'et__min_samples_split': range(2, 26, 2),
              'et__min_samples_leaf': range(2, 26, 2)
              }

'''
parameters = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': range(2, 12, 2),
              'criterion': ['gini', 'entropy'],
              'min_samples_split': range(2, 20, 2),
              'min_samples_leaf': range(2, 20, 2,),
              'class_weight': ['balanced']
              }

'''

def decision_tree_neural(X_train, y_train):
    parameters = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': range(2, 14, 2),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': range(2, 22, 2),
        'min_samples_leaf': range(2, 22, 2),
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
        gene_crossover_prob=0.5,
        tournament_size=3,
        generations_number=10)
    gs_dtree.fit(X_train, y_train)
    return gs_dtree.best_params_


def extra_tree_neural(X_train, y_train):
    parameters = {
        'n_estimators': range(600, 800, 200),
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None],  # range(2,18,2),
        'min_samples_split': range(2, 22, 2),
        'min_samples_leaf': range(2, 22, 2)
    }
    es_etree = EvolutionaryAlgorithmSearchCV(
        estimator=ExtraTreesClassifier(),
        param_grid=parameters,
        scoring="accuracy",
        cv=StratifiedKFold(y_train, n_folds=10),
        verbose=False,
        population_size=50,
        mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    es_etree.fit(X_train, y_train)
    print(es_etree.best_score_, es_etree.best_params_)


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


def generate_classification_report(clf, x_test, y_test, out_file=None):
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
    print(report)
    print(confusion)
    print(accuracy)
    print(scores)
    if out_file is not None:
        try:
            if sys.version < '3':
                infile = io.open(out_file, '+a')
            else:
                infile = io.open(out_file, '+a')
            with infile as classification:
                classification.write(report)
                classification.write(confusion)
                classification.write(accuracy)
                classification.write("{}".format(scores))
        except IOError:
            print("Sorry can't read: {}".format(out_file))
    return metrics.confusion_matrix(expected, predicted)


def extra_tree(X, y, trainingparams, out_file, num_feat=10):
    """ Extra Tree Classifier """
    e_tree = ExtraTreesClassifier(**trainingparams).fit(X, y)
    importances = e_tree.feature_importances_
    np.std([tree.feature_importances_ for tree in e_tree.estimators_],
           axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_feat*2):
        print("{} feature {} ({})".format(f + 1, X.columns[indices[f]],
                                          importances[indices[f]]))

    # Plot the feature importances of the forest
    '''
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_feat), importances[indices][:num_feat],
            color="r", yerr=std[indices][:num_feat], align="center")
    plt.xticks(range(num_feat), X.columns[indices][:num_feat])
    plt.xlim([-1, num_feat])
    plt.xlabel("Feature")
    plt.show()
    '''
    generate_classification_report(e_tree, X, y, out_file)
    return e_tree, X.columns[indices][:num_feat]


def decision_tree(X, y, trainingparam, out_file):
    """Returns a decision tree classifier trained on parameters.

    Keyword arguments:
    trainingparam -- dict full of training parameters
    out_file -- file where the classification report is save to.

    """
    d_tree = DecisionTreeClassifier(**trainingparam).fit(X, y)
    generate_classification_report(d_tree, X, y, out_file)
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


if __name__ == '__main__':

    parameter = sys.argv[1]
    homedir = os.path.expanduser('~')
    report_folder = os.path.join(homedir, "test-rlp", "training_cm")
    scikit_folder = os.path.join(homedir, "test-rlp", "sci-kit_rules")
    log_filename = '_'.join([parameter, "report_pipeline.log"])
    neural_filename = '_'.join([parameter, "report_neural.txt"])
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
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y%I:%M:%S')
    # prepare data
    train = train.fillna(0, axis=1)
    test = test.fillna(0, axis=1)
    X_train = train.drop([parameter], axis=1)
    X_test = test.drop([parameter], axis=1)
    y_train = train[parameter].apply(str)
    y_test = test[parameter].apply(str)
    X_train = X_train.select_dtypes(['float64'])
    X_test = X_test.select_dtypes(['float64'])
    neural_parameters = decision_tree_neural(X_train, y_train)
    print("*** Using neural parameters to train DT:")
    dt_neural = decision_tree(X_train, y_train, neural_parameters,
                              neural_filename)
    # rules
    neural_rules = ''.join([parameter, "_neural.csv"])
    neural_dot = u''.join([parameter, ".dot"])
    get_lineage(dt_neural, X_train.columns, paramdict[parameter],
                output_file="/".join([scikit_folder, neural_rules]))
    with open(neural_dot, 'w+') as f:
        f = tree.export_graphviz(dt_neural, out_file=f,
                                 feature_names=X_train.columns,
                                 class_names=paramdict[parameter],
                                 filled=True,
                                 rounded=True,
                                 special_characters=False)
