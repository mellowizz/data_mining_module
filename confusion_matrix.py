#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on 04.06.2015

@author: Moran
"""
from __future__ import division
from sqlalchemy import create_engine
from os import path
from sklearn.metrics.classification import confusion_matrix
# from sklearn.metrics.classification import precision_score, recall_score
import pandas.io.sql as psql
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


def create_save_file(var_fname):
    return path.join(r'c:\Users\Moran\test-rlp\accuracy_assessments',
                     var_fname)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(paramdict[parameter]))
    plt.xticks(tick_marks, paramdict[parameter], rotation=45)
    plt.yticks(tick_marks, paramdict[parameter])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    paramdict = {
        "natflo_hydromorphic": ["0", "1"],
        "natflo_immature_soil": ["0", "1"],
        "natflo_species_richness": ["species_poor", "species_rich"],
        "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
        "natflo_usage_intensity": ["high", "medium", "low"],
        "natflo_wetness": ["dry", "mesic", "very_wet"]
    }
    parameter = "natflo_wetness"
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    engine = create_engine(DSN)
    conn = engine.connect()
    try:
        sql_query = """SELECT {}, classified
                        FROM {}""".format(parameter,
                                          '_'.join(["results", parameter]))
        mydf = psql.read_sql(sql_query, engine)
        y_true = mydf[parameter].apply(str)
        y_pred = mydf["classified"].apply(str)
        print(y_true)
        print(y_pred)
        print(y_true.dtypes)
        print(y_pred.dtypes)
        print(confusion_matrix(y_true, y_pred, labels=paramdict[parameter]))
        print(classification_report(y_true, y_pred))
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        plot_confusion_matrix(cm)
        plt.show()
    except BaseException as e:
        print("Sorry: {}".format(e))
