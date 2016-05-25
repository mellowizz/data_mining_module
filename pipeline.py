#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division
from builtins import input
import subprocess
import sys
import io
import os
# from sklearn import cross_validation
# from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn import pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

logging.basicConfig(filename='pipeline.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)

DSN = 'postgresql://postgres@localhost:5432/rlp_saarburg'

report_folder = os.path.join(os.getcwd(), "training_cm")
rules_folder = os.path.join(os.getcwd(), "rules")

# load this from dict cursor?
paramdict = {
    "natflo_wetness": ["dry", "mesic", "wet"],
    "natflo_alkaline": ["0", "1"]
}


def plot_top_features(atree, X, parameter, num_feat=10):
    importances = atree.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(num_feat):
        logging.info(("{} feature {} ({})".format(f + 1, X.columns[indices[f]],
                                                  importances[indices[f]])))
    plt.figure()
    plt.title("Feature importances")
    plt.xticks(range(10), X.columns[indices][:10], rotation=45)
    plt.margins(0.8)
    plt.xlim([-1, 10])
    plt.xlabel("Feature")
    if 'ExtraTrees' in type(atree).__name__:
        std = np.std([tree.feature_importances_ for tree in atree.estimators_],
                     axis=0)
        # Plot the feature importances of the forest
        plt.bar(range(10), importances[indices][:10],
                color="r", yerr=std[indices][:10], align="center")
    else:
        # Plot the feature importances of the forest
        plt.bar(range(10), importances[indices][:10],
                color="r", align="center")

    save = os.path.join(os.getcwd(), 'training_cm')
    save += '/' + type(atree).__name__
    save += parameter + '.png'
    plt.savefig(save, dpi=300, format='png')
    plt.show()
    print("saving to: {}".format(save))


def read_db_table(table, parameter='all'):
    engine = create_engine(DSN)
    with engine.connect():
        ''' read data from table'''
        datatable = psql.read_sql('''SELECT * FROM {}
                                  '''.format(table),
                                  engine, index_col='id')

        if parameter == 'all':
            parameter = paramdict.keys()
        else:
            parameter = [parameter]
        try:
            for i in ["kul", "kn1", "kn2", "wert_kn2",
                      "we1min", "bodenart_kn1", "we2min"]:
                datatable = datatable.drop(i, axis=1)
        except ValueError as e:
            logging.info(e.message)

        datatable = datatable.fillna(0, axis=1)
        # datatable = datatable.dropna()
        # datatable.drop_duplicates()
        y = pd.DataFrame(datatable[parameter]) #, dtype=str)
        X = datatable.drop(parameter, axis=1)
        X = X.select_dtypes(['float64'])
        return X, y


def get_lineage(tree, feature_names, wet_classes,
                output_file="default.csv"):
    """Iterates over tree and saves all nodes to file."""
    try:

        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value
        if sys.version < '3':
            infile = io.open(output_file, 'wb')
        else:
            infile = io.open(output_file, 'wb')
        with infile as tree_csv:
            idx = np.argwhere(left == -1)[:, 0]

            def recurse(left, right, child, lineage=None):
                if lineage is None:
                    try:
                        # logging.info(str(value[child]))
                        lineage = [wet_classes[np.argmax(value[child])]]
                    except KeyError as f:
                        logging.info(f)
                    except IndexError as f:
                        logging.info("{}: {}".format(f, wet_classes))
                    except AttributeError as r:
                        logging.info("{}".format(r))
                if child in left:
                    parent = np.where(left == child)[0].item()
                    split = '<='
                else:
                    parent = np.where(right == child)[0].item()
                    split = '>'

                if lineage is None:
                    return
                lineage.append((features[parent], split,
                                threshold[parent], parent))

                if parent == 0:
                    lineage.reverse()
                    return lineage
                else:
                    return recurse(left, right, parent, lineage)

            for child in idx:
                try:
                    for node in recurse(left, right, child):
                        if not node:
                            continue
                        if type(node) == tuple:
                            if not (isinstance(node[2], float) and
                                    isinstance(node[3], int)):
                                logging.info("skipping tuple..")
                                continue
                            a_feature, a_split, a_threshold, a_parent_node = node
                            tree_csv.write("{},{},{:5f},{}\n".format(a_feature,
                                                                     a_split,
                                                                     a_threshold,
                                                                     a_parent_node,
                                                                    ))
                        else:
                            tree_csv.write(''.join([node, "\n"]))
                except TypeError as e:
                    logging.info(e)
    except ValueError as e:
        logging.info(e)
    except KeyError as f:
        logging.info(f)
    except UnboundLocalError as e:
        logging.info(e.message)

if __name__ == '__main__':
    table = sys.argv[1]
    if not os.path.exists(report_folder):
        choice = input("Create folder: {} y/n?".format(report_folder))
        if choice.lower() == 'y':
            try:
                os.mkdir(report_folder)
                print("folder: {} created sucessfully".format(report_folder))
            except OSError as e:
                print("Error: {}".format(e.message))
                sys.exit(1)
        else:
            sys.exit("goodbye!")

    if not os.path.exists(rules_folder):
        choice = input("Create folder: {} y/n?".format(rules_folder))
        if choice.lower() == 'y':
            try:
                os.mkdir(rules_folder)
                print("folder: {} created sucessfully".format(rules_folder))
            except OSError as e:
                print("Error: {}".format(e.message))
                sys.exit(1)
        else:
            sys.exit("goodbye!")

    X, y = read_db_table(table)
    engine = create_engine(DSN)
    msk = np.random.rand(len(X)) < 0.6
    X_train, X_test = X[msk], X[~msk]
    y_train, y_test = y[msk], y[~msk]
    with engine.connect():
        ''' write test data to DB'''
        pandas_sql = pd.io.sql.pandasSQL_builder(engine, schema=None,
                                                    flavor=None)
        test = pd.concat([X_test, y_test], axis=1,
                        join_axes=[X_test.index]).drop_duplicates()
        test.index.name = 'id'
        test_table = '_'.join(["test", table])
        test.index.name = 'id'
        test.to_sql(test_table, engine, if_exists='replace',
                    index_label='id')
    for curr in paramdict:
        y_test_curr = y_test[curr]
        y_train_curr = y_train[curr]
        print("X: {}, y: {}".format(len(X), len(y)))
        print("X_train: {}, y_train: {}".format(len(X_train), len(y_train_curr)))
        print("X_test: {}, y_test: {}".format(len(X_test), len(y_test)))
        print("curr parameter: {}".format(curr))
        dt_params = {
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': range(2, 20, 2),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': range(2, 20, 2),
            'min_samples_leaf': range(2, 20, 2),
            'class_weight': ['balanced']
        }

        et_params = {'n_estimators': 600,
                    'class_weight': 'balanced'
                }
        grid = {
            'dt__max_features': ['auto', 'sqrt', 'log2'],
            'dt__max_depth': range(2, 12, 2),
            'dt__criterion': ['gini', 'entropy'],
            'dt__min_samples_split': range(2, 12, 2),
            'dt__min_samples_leaf': range(2, 12, 2),
            'dt__class_weight': ['balanced']
            #'feature_selection': range(10, 100, 10)
            }

        steps = [('feature_selection',
                #SelectKBest()),
                SelectFromModel(estimator=ExtraTreesClassifier(**et_params))),
                ('dt', DecisionTreeClassifier())]
        pipe = pipeline.Pipeline(steps)

        logging.info("*** ET *** ")
        logging.info("{}".format(curr))
        etree = ExtraTreesClassifier(**et_params)
        etree.fit(X_train, y_train_curr)
        # score = ross_val_score(etree, X_train, y_train_curr)
        # logging.info("Training cross validation score: {}".format(score))
        plot_top_features(etree, X_train, curr)
        y_pred = etree.predict(X_test)
        report = metrics.classification_report(y_test_curr, y_pred)
        # save report
        logging.info(report)
        dt = EvolutionaryAlgorithmSearchCV(
            estimator=DecisionTreeClassifier(),
            params=dt_params,
            scoring="accuracy",
            cv=StratifiedKFold(y_train_curr),
            verbose=1,
            population_size=50,
            gene_mutation_prob=0.10,
            tournament_size=3,
            generations_number=10,
            n_jobs=-1) ## !! 1 wenn es nicht funktioniert
        dt.fit(X_train, y_train_curr)
        y_pred = dt.predict(X_test)
        report = metrics.classification_report(y_test_curr, y_pred)
        logging.info("*** DT *** ")
        # save report
        logging.info(report)
        dt = DecisionTreeClassifier(**dt.best_params_).fit(X_train, y_train_curr)
        y_pred = dt.predict(X_test)
        logging.info("DT fitted with best parameters")
        report = metrics.classification_report(y_test_curr, y_pred)
        plot_top_features(dt, X_train, curr)
        # save report
        logging.info(report)
        my_out_file = ''.join([rules_folder, '/', curr, '.csv'])
        get_lineage(dt, X_train.columns, paramdict[curr],
                    output_file=my_out_file)
        with open(curr + '.dot', 'w+') as f:
            f = tree.export_graphviz(dt, out_file=f,
                                    feature_names=X_train.columns,
                                    class_names=paramdict[curr],
                                    filled=True,
                                    rounded=True,
                                    special_characters=False)

        subprocess.call(["dot", "-Tpng", curr + '.dot', "-o",
                            ''.join([report_folder, '/', curr, '.png'])])
