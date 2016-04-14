#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division
import os
import subprocess
import sys
import io
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
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.base import TransformerMixin


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

trainingparams = {'criterion': 'gini', 'max_depth': 8,
                  'max_features': 'auto', 'min_samples_leaf': 6,
                  'min_samples_split': 8, 'class_weight': 'balanced'}
et_params = {'n_estimators': 600, 'random_state': 0, 'n_jobs': -1,
             'class_weight': 'balanced', 'bootstrap': False}

parameters = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': range(2, 12, 2),
              'criterion': ['gini', 'entropy'],
              'min_samples_split': range(2, 20, 2),
              'min_samples_leaf': range(2, 20, 2,),
              'class_weight': ['balanced']}


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

if __name__ == '__main__':
    table = sys.argv[1]
    X, y = read_db_table(table)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        stratify=y)
    for curr in paramdict:
        y_train_curr = y_train[curr]
        y_test_curr = y_test[curr]
        grid = {
            'dt__max_features': ['auto', 'sqrt', 'log2'],
            'dt__max_depth': range(2, 12, 2),
            'dt__criterion': ['gini', 'entropy'],
            'dt__min_samples_split': range(2, 12, 2),
            'dt__min_samples_leaf': range(2, 12, 2),
            'dt__class_weight': ['balanced'],
            'feature_selection': range(10, 100, 10)
            }

        steps = [('feature_selection', SelectKBest()),
                ('dt', DecisionTreeClassifier())
                ]
        pipeline = pipeline.Pipeline(steps)

        cv = EvolutionaryAlgorithmSearchCV(
            estimator=pipeline,
            params=grid,
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
