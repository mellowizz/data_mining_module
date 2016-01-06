#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from sqlalchemy import create_engine
from os import path
from sklearn.metrics.classification import confusion_matrix
# from sklearn.metrics.classification import precision_score, recall_score
import pandas.io.sql as psql
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_similarity_score
# import io


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


dry = '''
select substr(ut_saarburg_mad_all.eunis,1,2) as eunis, 'E1' as classified
from ut_saarburg_mad_all where id in (
select id from results_natflo_wetness where classified = 'dry')
and id in (select id from results_natflo_usage)
and id in (select ndsm_nm_id from ndsm_saarburg where ndsm_nm_max < 1)
'''

mesic = '''
SELECT substr(ut_saarburg_mad_all.eunis,1,2) as eunis,
'E2' as classified
FROM ut_saarburg_mad_all
WHERE id IN (
SELECT id FROM results_natflo_wetness
WHERE classified = 'mesic')
AND
id IN (SELECT id
from results_natflo_usage)
AND
id IN (SELECT ndsm_nm_id
FROM ndsm_saarburg
WHERE ndsm_nm_max < 1)
'''
very_wet = '''
select substr(ut_saarburg_mad_all.eunis,1,2) as eunis, 'E3' as classified
from ut_saarburg_mad_all where id in (
select id from results_natflo_wetness where classified = 'very_wet')
and id in (select id from results_natflo_usage)
and id in (select ndsm_nm_id from ndsm_saarburg where ndsm_nm_max < 1)
'''

if __name__ == '__main__':
    paramdict = {
        "natflo_hydromorphic": ["0", "1"],
        "natflo_immature_soil": ["0", "1"],
        "natflo_species_richness": ["species_poor", "species_rich"],
        "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
        "natflo_usage_intensity": ["high", "medium", "low"],
        "natflo_wetness": ["dry", "mesic", "very_wet"],
        "eunis": ['D5', 'E1', 'E2', 'E3', 'F3', 'FB', 'G1', 'H2', 'I1', 'J1']
    }
    parameter = "eunis"
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    engine = create_engine(DSN)
    conn = engine.connect()
    try:

        sql_query = ' UNION ALL '.join([dry, mesic, very_wet])
        mydf = psql.read_sql(sql_query, engine)
        y_true = mydf[parameter].apply(str)
        y_pred = mydf["classified"].apply(str)
        my_matrix = ''.join(["classification", '_level0_all', '.txt'])
        cm = confusion_matrix(y_true, y_pred,
                              labels=paramdict[parameter])
        print(cm)
        with open(my_matrix, 'wt') as f:
            print(classification_report(y_true, y_pred,
                                        labels=paramdict[parameter]),
                  file=f)
        print("jaccard similarty score:")
        print("{}".format(jaccard_similarity_score(y_true,
                                                   y_pred)))
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        '''
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        plot_confusion_matrix(cm)
        plt.show()
        '''
    except BaseException as e:
        print("Sorry: {}".format(e))
