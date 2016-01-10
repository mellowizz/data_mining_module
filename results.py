#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
try:
    from os import scandir
except:
    from scandir import scandir
from sqlalchemy import create_engine
from os import path
from sklearn.metrics.classification import confusion_matrix
from os.path import join as osjoin
from collections import defaultdict
# from sklearn.metrics.classification import precision_score, recall_score
import pandas.io.sql as psql
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from os.path import expanduser
import sys


def subdirs(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name, entry.path


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
    parameter = "eunis"
    DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
    engine = create_engine(DSN)
    conn = engine.connect()
    homedir = expanduser('~')
    homesubdirs = defaultdict()
    for i, j in subdirs(homedir):
        homesubdirs[i] = j
    output_folder = ''
    # print(homesubdirs.keys())
    try:
        if 'tubCloud' in homesubdirs:
            output_folder = homesubdirs['tubCloud']
        elif 'ownCloud' in homesubdirs:
            output_folder = homesubdirs['ownCloud']

        currtable = sys.argv[1]
        wetness = currtable.split('_')[1]
        desc_eunis = {'dry': 'E1', 'mesic': 'E2',
                      'wet': 'E3', 'mytable': currtable}
        sql_query = '''select result.id,
                        substr(eunis,1,2) as eunis,
                        '{mywet}' as classified
                       from {mytable} as result,
                        ut_saarburg_mad_all as ut
                       where result.id = ut.id
                    '''.format(mywet=desc_eunis[wetness], **desc_eunis)
        print("sql: {}".format(sql_query))

        mydf = psql.read_sql(sql_query, engine)
        y_true = mydf[parameter]   # .apply(str)
        y_pred = mydf["classified"]  # .apply(str)
        output_folder = osjoin(homedir, output_folder)
        my_matrix = ''.join([currtable, '.txt'])
        my_matrix = osjoin(output_folder, my_matrix)
        print("Saving to: {}".format(my_matrix))
        cm = confusion_matrix(y_true, y_pred)
        # labels=paramdict[parameter])
        print(cm)
        with open(my_matrix, 'wt') as f:
            print(classification_report(y_true, y_pred), file=f)
        # labels=paramdict[parameter]),
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
