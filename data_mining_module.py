#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:56:26 2015

@author: Moran
"""

from __future__ import division

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from evolutionary_search import EvolutionaryAlgorithmSearchCV
# from sklearn.pipeline import Pipeline
import commonutils
# from sklearn import tree
from sklearn.cross_validation import StratifiedKFold
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
    print(gs_dtree.best_score_, gs_dtree.best_params_)
    # best_parameters, score, _ = max(gs_dtree.best_score_, key=lambda x: x[1])
    # return param_dict


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


def extra_tree(X, y):
    """ Decision Tree Classifier """
    e_tree = ExtraTreesClassifier(n_estimators=800,
                                  max_features='sqrt',
                                  n_jobs=-1, max_depth=None,
                                  criterion='entropy').fit(X_train, y_train)
    cu.generate_classification_report(e_tree, X, y)
    cu.plot_feature_importance(e_tree, X)
    return e_tree


def decision_tree(X, y):
    """ Decision Tree Classifier """
    d_tree = DecisionTreeClassifier(
        criterion='gini', max_depth=10,
        max_features='auto', min_samples_leaf=2,
        min_samples_split=4).fit(X_train, y_train)
    cu.generate_classification_report(d_tree, X, y)
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

if __name__ == '__main__':
    paramdict = {
        "natflo_wetness": ["dry", "mesic", "very wet"],
        "natflo_depression": ["0", "1"],
        "natflo_hydromorphic": ["0", "1"],
        "natflo_immature_soil": ["0", "1"],
        "natflo_species_richness": ["species_poor", "species_rich"],
        "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
        "natflo_usage_intensity": ["high", "medium", "low"],
        "eagle_vegetationtype": ["graminaceous_herbaceous",
                                 "herbaceous", "shrub", "tree"]
        }
    parameter = "eagle_vegetationtype"

    table_train = "_".join(["train", parameter]).lower()
    table_test = "test"  # ).lower()

    cu = commonutils.Commonutils({'user': 'postgres',
                                  'database': 'rlp_spatial'})
    train = cu.get_munged_data(table_train, paramdict, parameter)
    test = cu.get_munged_data(table_test, paramdict, parameter)
    # features, labels = df.drop([parameter], axis=1), df[parameter]
    X_train = train.drop([parameter], axis=1)
    y_train = train[parameter]
    X_test = test.drop([parameter], axis=1)
    y_test = test[parameter]

    '''
    pca = PCA(n_components=5)
    X_r = pca.fit(X_train, y_train).transform(X_train)
    lda = LinearDiscriminantAnalysis(n_components=5)
    X_r2 = lda.fit(X_train, y_train).transform(X_train)

    plotPCALDA(X_r, X_r2, pca, lda, usage)
    '''
    # params = decision_tree_neural(X_train, y_train)
    my_dt = decision_tree(X_train, y_train)
    # , params) # wet_classes) #, params)
    BOOLEAN = ["0", "1"]

    my_out_file = ''.join(["C:\\Users\\Moran\\test-rlp\\sci-kit_rules\\",
                           parameter, ".csv"])

    cu.get_lineage(my_dt, X_train.columns, paramdict["eagle_vegetationtype"],
                   out_file=my_out_file)
    '''
    tree.export_graphviz(d_tree, out_file=dot_data,
                         feature_names=X.columns,
                         class_names=wet_classes,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("C:\\Users\\Moran\\test-rlp\\sci-kit_rules\\natlfo_wetness.png")
    # print(type(X_train.columns), type(wet_classes))
    '''
