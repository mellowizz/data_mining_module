#!/usr/bin/python2

from __future__ import division
import logging
from time import time
# from IPython.display import Image
# from sklearn.externals.six import StringIO
# import pydot
import io
import os
import sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.pipeline import Pipeline
import pandas.io.sql as psql
from sqlalchemy import create_engine
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold


def get_lineage(tree, feature_names, wet_classes,
                output_file="default.csv"):
    """Iterates over tree and saves all nodes to file."""
    print("dict: {}".format(tree.named_steps))
    print("keys: {}".format(tree.named_steps.keys()))
    print("values: {}".format(tree.named_steps.values()))
    left = tree.named_steps['dt'].tree_.children_left
    right = tree.named_steps['dt'].tree_.children_right
    threshold = tree.named_steps['dt'].tree_.threshold
    features = [feature_names[i]
                for i in tree.named_steps['dt'].tree_.feature]
    value = tree.named_steps['dt'].tree_.value

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


def generate_classification_report(clf, x_test, y_test):  # , out_file=None):
    """Prints out a confusion matrix from the classifier object."""
    expected = y_test
    predicted = clf.predict(x_test)
    report = """dt report {}:
    {}\n""".format(clf, metrics.classification_report(expected, predicted))
    confusion = """Confusion matrix:
    {}\n""".format(metrics.confusion_matrix(expected, predicted))
    scores = cross_val_score(clf, x_test, y_test, cv=5,
                             n_jobs=-1)
    accuracy = "Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(),
                                                        scores.std() * 2)

    logging.info(report)

    logging.info(report)
    logging.info(confusion)
    logging.info(accuracy)
    logging.info(scores)

if __name__ == '__main__':
    paramdict = {
        "natflo_hydromorphic": ["0", "1"],
        "natflo_immature_soil": ["0", "1"],
        "natflo_species_richness": ["species_poor", "species_rich"],
        "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
        "natflo_usage_intensity": ["high", "medium", "low"],
        "natflo_wetness": ["dry", "mesic", "very_wet"],
        }
    parameter = sys.argv[1]
    homedir = os.path.expanduser('~')
    report_folder = os.path.join(homedir, "test-rlp", "training_cm")
    scikit_folder = os.path.join(homedir, "test-rlp", "sci-kit_rules")
    log_filename = '_'.join([parameter, "report_pipeline.log"])
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S%p')
    logging.warning('is when this event was logged.')
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
    homedir = os.path.expanduser('~')
    scores = []
    report_filename = '_'.join([parameter, "report_pipeline.txt"])
    rules_filename = ''.join([parameter, ".csv"])
    t0 = time()
    # prepare data
    train = train.fillna(0, axis=1)
    test = test.fillna(0, axis=1)
    X_train = train.drop([parameter], axis=1)
    X_test = test.drop([parameter], axis=1)
    y_train = train[parameter].apply(str)
    y_test = test[parameter].apply(str)
    X_train = X_train.select_dtypes(['float64'])
    X_test = X_test.select_dtypes(['float64'])
    # parameters = {'criterion': 'entropy', 'max_depth': 10,
    #              'max_features': 'auto', 'min_samples_leaf': 2,
    #              'min_samples_split': 6, 'class_weight': 'balanced'}
    parameters = {
        'dt__max_features': ['auto', 'sqrt', 'log2'],
        'dt__max_depth': range(2, 12, 2),
        'dt__criterion': ['gini', 'entropy'],
        'dt__min_samples_split': range(2, 12, 2),
        'dt__min_samples_leaf': range(2, 12, 2),
        'dt__class_weight': ['balanced']
    }
    steps = [
            ('et',
             SelectFromModel(ExtraTreesClassifier(n_estimators=400,
                                                  class_weight='balanced',
                                                  n_jobs=-1))),
            ('dt', DecisionTreeClassifier())  # **parameters))
    ]
    pipeline = Pipeline(steps)
    ev_search = EvolutionaryAlgorithmSearchCV(
        estimator=pipeline,
        params=parameters,
        scoring="accuracy",
        cv=StratifiedKFold(y_train, n_folds=10),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        tournament_size=3,
        generations_number=10)
    ev_search.fit(X_train, y_train)
    # ev_search = pipeline.fit(X_train, y_train)
    generate_classification_report(ev_search, X_train, y_train)
    generate_classification_report(ev_search, X_test, y_test)
    #  files
    get_lineage(ev_search, X_train.columns, paramdict[parameter],
                output_file=''.join([parameter, '_pipeline', '.csv']))
    with open(''.join([parameter, '_pipeline', '.dot']), 'w+') as f:
        f = export_graphviz(ev_search.named_steps['dt'],
                            out_file=f,
                            feature_names=X_train.columns,
                            class_names=paramdict[parameter],
                            filled=True,
                            rounded=True,
                            special_characters=False)
