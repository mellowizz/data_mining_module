#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:56:26 2015

@author: Moran
"""

from __future__ import division
from IPython.display import Image
from time import time
from sklearn.externals.six import StringIO
import pydot
import os
import sys
import io
from sklearn import cross_validation
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import KernelPCA
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
import pandas.io.sql as psql
from sqlalchemy import create_engine
import numpy as np
# from sklearn.externals.six import StringIO
# import pydot
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA


def decision_tree_neural(X_train, y_train):
    parameters = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': range(2, 12, 2),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(2, 20, 2)
    }
    gs_dtree = EvolutionaryAlgorithmSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=parameters,
        scoring="accuracy",
        cv=StratifiedKFold(y_train, n_folds=10),
        verbose=False,
        population_size=50,
        mutation_prob=0.10,
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
    return metrics.confusion_matrix(expected, predicted)


def extra_tree(X, y, trainingparams, out_file):
    """ Extra Tree Classifier """
    e_tree = ExtraTreesClassifier(**trainingparams).fit(X, y)
    importances = e_tree.feature_importances_
    std = np.std([tree.feature_importances_ for tree in e_tree.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(20):
        print("{} feature {} ({})".format(f + 1, X.columns[indices[f]],
                                          importances[indices[f]]))

    num_feat = 10
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_feat), importances[indices][:num_feat],
            color="r", yerr=std[indices][:num_feat], align="center")
    plt.xticks(range(num_feat), X.columns[indices][:num_feat])
    plt.xlim([-1, num_feat])
    plt.xlabel("Feature")
    plt.show()
    generate_classification_report(e_tree, X, y, out_file)
    return e_tree


def decision_tree(X, y, trainingparam, out_file):
    """ Decision Tree Classifier """
    d_tree = DecisionTreeClassifier(**trainingparam).fit(X, y)
    generate_classification_report(d_tree, X, y, out_file)
    # dot_data = StringIO()
    return d_tree


def plotPCALDA(X_r, X_r2, pca, lda, target_names, y):
    print("explained variance ratio: {0!s}".format(
        pca.explained_variance_ratio_))
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('PCA')

    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('LDA')

    plt.show()


def get_lineage(tree, feature_names, wet_classes,
                output_file="default.csv"):
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
    pipeline = Pipeline(steps=[('pca', RandomizedPCA()),
                               ('kpca', KernelPCA()),
                               ('dt', DecisionTreeClassifier())])
    ev_search = EvolutionaryAlgorithmSearchCV(pipeline, pipe_grid,
                                              scoring=None,
                                              verbose=True,
                                              n_jobs=-1,
                                              population_size=5).fit(X,
                                                                     y)
    # generate_classification_report(ev_search, X, y, out_file)
    return ev_search

if __name__ == '__main__':
    t0 = time()
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
    homedir = os.path.expanduser('~')
    parameter = "natflo_usage_intensity"
    table_train = "_".join(["grasslands", "train", parameter]).lower()
    table_test = "grasslands_test"  # ).lower()
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    engine = create_engine(DSN)
    conn = engine.connect()
    all_sql = "SELECT * FROM {} UNION ALL SELECT * FROM {}".format(table_train,
                                                                   table_test)
    test_sql = "SELECT * FROM {}".format(table_test)
    train_sql = "SELECT * FROM {}".format(table_train)
    train = psql.read_sql(train_sql, engine)
    test = psql.read_sql(test_sql, engine)
    report_folder = os.path.join(homedir, "test-rlp", "training_cm")
    tree_file = '_'.join([parameter, "graph.png"])
    file_name = '_'.join([parameter, "report.txt"])
    file_name_pca = '_'.join([parameter, "report_pca.txt"])
    file_name_kpca = '_'.join([parameter, "report_kpca.txt"])
    file_name_neural_pca = '_'.join([parameter, "report_neural_pca.txt"])
    file_name_et = '_'.join([parameter, "report_et.txt"])
    file_name_pipe = '_'.join([parameter, "report_pipeline.txt"])
    out_file = '/'.join([report_folder, file_name])
    out_file_pca = '/'.join([report_folder, file_name_pca])
    out_file_kpca = '/'.join([report_folder, file_name_kpca])
    out_file_neural_pca = '/'.join([report_folder, file_name_neural_pca])
    out_file_et = '/'.join([report_folder, file_name_et])
    out_file_pipe = '/'.join([report_folder, file_name_pipe])
    train = train.fillna(0, axis=1)
    test = test.fillna(0, axis=1)
    X_train = train.drop([parameter], axis=1)
    X_test = test.drop([parameter], axis=1)
    y_train = train[parameter]
    y_test = test[parameter]
    X_train = X_train.select_dtypes(['float64'])
    X_test = X_test.select_dtypes(['float64'])
    ''' fit classifiers! '''
    trainingparams = {'criterion': 'gini', 'max_depth': 10,
                      'max_features': 'auto', 'min_samples_leaf': 16,
                      'min_samples_split': 14}
    et_params = {'n_estimators': 800, 'max_features': 'sqrt', 'n_jobs': -1,
                 'max_depth': None, 'criterion': 'entropy'}
    et_params_simple = {'n_estimators': 250, 'random_state': 0, 'n_jobs': -1}
    forest = extra_tree(X_train, y_train, et_params_simple, out_file_et)
    print("Performing PCA")
    n_components = 5
    parameters = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': range(2, 12, 2),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(2, 20, 2)
    }
    pipe_grid = {
        'dt__criterion': ['gini', 'entropy'],
        'dt__max_features': ['auto', 'sqrt', 'log2'],
        'dt__min_samples_split': range(2, 18, 2),
        'dt__min_samples_leaf': range(2, 18, 2),
        'kpca__n_components': [10, 20],
        'pca__n_components': [10, 20],
        'pca__whiten': [True, False]
        }
    # print(X_train.dtypes)
    # ev_pipe = evolutionary_pipeline(X_train, y_train, pipe_grid, out_file_pipe)
    pca = RandomizedPCA(n_components=n_components, whiten=False).fit(X_train)
    kpca = KernelPCA(n_components=n_components).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("Done fitting Randomized PCA in {:0.3f}".format(time() - t0))
    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    print("Done fitting Kernel PCA in {:0.3f}".format(time() - t0))
    print("*** DT with Randomized PCA ***")
    dt_pca = decision_tree(X_train_pca, y_train, trainingparams, out_file_pca)
    print("*** DT with Kernel PCA transform: ***")
    dt_kpca = decision_tree(X_train_kpca, y_train, trainingparams,
                            out_file_kpca)
    neural_parameters = decision_tree_neural(X_train, y_train)
    print("*** Using neural parameters to train DT:")
    dt_neural = decision_tree(X_train, y_train, neural_parameters, out_file)
    print("*** Using neural parameters to train DT WITH PCA:")
    dt_neural_pca = decision_tree(X_train, y_train, neural_parameters,
                                  out_file_neural_pca)
    # print("*** Randomized PCA ***")
    # parameters_pca = decision_tree_neural(X_train_pca, y_train)
    # print("*** Kernel PCA ***")
    # parameters_kpca = decision_tree_neural(X_train_kpca, y_train)
    # decision_tree(X_train, y_train, new_parameters, out_file)
    my_out_file = ''.join(["C:\\Users\\Moran\\test-rlp\\sci-kit_rules\\",
                           parameter, "_neural.csv"])

    my_out_file_kpca = ''.join(["C:\\Users\\Moran\\test-rlp\\sci-kit_rules\\",
                               parameter, "_kpca.csv"])
    get_lineage(dt_kpca, X_train.columns, paramdict[parameter],
                output_file=my_out_file_kpca)
    get_lineage(dt_neural, X_train.columns, paramdict[parameter],
                output_file=my_out_file)
    '''
    with open(tree_file, 'w') as f:
        #dot_data = StringIO()
        f = tree.export_graphviz(dt_pca, out_file=f,
                             feature_names=X_train.columns,
                             class_names=y_train,
                             filled=True, rounded=True,
                             special_characters=False)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())
    graph.write_png(tree_file)
    '''
