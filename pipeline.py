#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division
import subprocess
import sys
import io
import os
from sklearn import cross_validation
from sklearn import metrics
from sklearn import pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import logging


class ColumnSelector(object):

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self


logging.basicConfig(filename='data_mining_module.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)

DSN = 'postgresql://postgres@localhost:5432/rlp_saarburg'

# load this from dict cursor?
paramdict = {
    "natflo_wetness": ["dry", "mesic", "wet"],
    "natflo_acidity": ["alkaline", "acid"]
}

et_params = {'n_estimators': 400, 'random_state': 0, 'n_jobs': 4,
             'class_weight': 'balanced'}


def read_db_table(table, parameter='all'):
    engine = create_engine(DSN)
    with engine.connect():
        ''' read data from table'''
        datatable = psql.read_sql('''SELECT * FROM {}
                                  --where natflo_wetness is not null
                                  --and natflo_immature_soil is not null
                                  --and natflo_species_richness is not null
                                  --and natflo_acidity is not null
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
            print(e.message)

        datatable = datatable.fillna("unclass", axis=1)
        y = pd.DataFrame(datatable[parameter])
        X = datatable.drop(parameter, axis=1)
        X = X.select_dtypes(['float64'])
        return X, y


def print_tree(t, root=0, depth=1):
    if depth == 1:
        print('def predict(X_i):')
    indent = '    '*depth
    print(indent + '# node %s: impurity = %.2f' %
          (str(root), t.impurity[root]))
    left_child = t.children_left[root]
    right_child = t.children_right[root]

    if left_child == tree._tree.TREE_LEAF:
        print(indent + 'return %s # (node %d)' % (str(t.value[root]), root))
    else:
        print(indent + 'if X_i[%d] < %.2f: # (node %d)' % (t.feature[root],
                                                           t.threshold[root],
                                                           root))
        print_tree(t, root=left_child, depth=depth+1)

        print(indent + 'else:')
        print_tree(t, root=right_child, depth=depth+1)


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
                        print(str(value[child]))
                        lineage = [wet_classes[np.argmax(value[child])]]
                    except KeyError as f:
                        print(f)
                    except IndexError as f:
                        print("{}: {}".format(f, wet_classes))
                    except AttributeError as r:
                        print("{}".format(r))
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
                except TypeError as e:
                    print(e)
    except ValueError as e:
        print(e)
    except KeyError as f:
        print(f)
    except UnboundLocalError as e:
        print(e.message)

if __name__ == '__main__':
    table = sys.argv[1]
    X, y = read_db_table(table)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        stratify=y)

    report_folder = os.path.join(os.getcwd(), "training_cm")
    rules_folder = os.path.join(os.getcwd(), "rules")
    for curr in paramdict:
        y_train_curr = y_train[curr]
        y_test_curr = y_test[curr]

        parameters = {
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': range(2, 20, 2),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': range(2, 20, 2),
            'min_samples_leaf': range(2, 20, 2),
            'class_weight': ['balanced']
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

        cv = EvolutionaryAlgorithmSearchCV(
            estimator=DecisionTreeClassifier(),  # pipe,
            params=parameters,  # grid,
            scoring="accuracy",
            cv=StratifiedKFold(y_train_curr),
            verbose=1,
            population_size=50,
            gene_mutation_prob=0.10,
            tournament_size=3,
            generations_number=10,
            n_jobs=4)
        cv.fit(X_train, y_train_curr)
        y_pred = cv.predict(X_test)
        report = metrics.classification_report(y_test_curr, y_pred)
        print(report)
        mydt = cv.best_params_
        dt = DecisionTreeClassifier(**mydt).fit(X_train, y_train_curr)
        y_pred = dt.predict(X_test)
        report = metrics.classification_report(y_test_curr, y_pred)
        #print("best estimator: {}".format(cv.best_estimator_))
        #print("named step: {}".format(pipe.named_steps['dt']))
        #print("best params: {}".format(cv.best_params_))
        my_out_file = ''.join([rules_folder, '/', curr, '.csv'])
        get_lineage(dt, X_train.columns, paramdict[curr],
                    output_file=my_out_file)

        report = metrics.classification_report(y_test_curr, y_pred)
        # X_new =SelectKBest(k=pipe.named_steps['feature_selection']).fit_transform(X_train,
        #                                                                    y_train_curr)
        # mydt = DecisionTreeClassifier(cv.best_params_).fit(X_new, y_train_curr)
        with open(curr + '.dot', 'w+') as f:
            f = tree.export_graphviz(dt, out_file=f,
                                     feature_names=X_train.columns,
                                     class_names=paramdict[curr],
                                     filled=True,
                                     rounded=True,
                                     special_characters=False)

        subprocess.call(["dot", "-Tpng", curr + '.dot', "-o",
                            ''.join([report_folder, '/', curr, '.png'])])
    ''' write test data '''
    engine = create_engine(DSN)
    with engine.connect():
        test = pd.concat([X_test, y_test], axis=1)
        test.index.name = 'id'
        test_table = '_'.join([table, "test"])
        test.to_sql(test_table, engine, if_exists='replace', index=True)
        '''
        features = cv.steps[-1]
        print(X.columns[features.transform(np.arange(len(X.columns)))])
        # feature_importances_
        importances = pipe.named_steps['feature_selection'].feature_importances_
        std = np.std([tree.feature_importances_ for tree in dt.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        logging.info("Feature ranking:")
        for f in range(num_feat):
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
        print(X.columns[indices][:10])
        '''
