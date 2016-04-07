#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Created on 04.06.2015

@author: Moran
"""
from __future__ import division
from __future__ import print_function
try:
    from os import scandir
except:
    from scandir import scandir
import sys
from sqlalchemy import create_engine
from os import path
from sklearn.metrics.classification import confusion_matrix
from collections import defaultdict
# from sklearn.metrics.classification import precision_score, recall_score
import pandas.io.sql as psql
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as osjoin
from os.path import expanduser


def subdirs(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name, entry.path


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
        # "natflo_hydromorphic": ["0", "1"],
         "natflo_immature_soil": ["0", "1"],
         "natflo_species_richness": ["species_poor", "species_rich"],
        # "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
        # "natflo_usage_intensity": ["high", "medium", "low"],
         "natflo_wetness": ["dry", "mesic", "very_wet"]
    }
    dt_rule_folder = ""
    DSN = 'postgresql://postgres@localhost:5432/rlp_saarburg'
    engine = create_engine(DSN)
    conn = engine.connect()
    algorithmn = "" # "" for dt or seath
    homedir = expanduser('~')
    homesubdirs = defaultdict()
    for i, j in subdirs(homedir):
       homesubdirs[i] = j

    try:
        if 'tubCloud' in homesubdirs:
            output_folder = homesubdirs['tubCloud']
        elif 'ownCloud' in homesubdirs:
            output_folder = homesubdirs['ownCloud']
        dt_rule_folder = osjoin(homedir, "git", "data_mining_module", "rules")
        output_folder = osjoin(output_folder, "accuracy_assessments")
        test_area = sys.argv[1]
        for parameter in paramdict:
            print("current parameter: {}".format(parameter))
            if algorithmn == '':
                base_rulename = parameter
            else:
                base_rulename = "_".join([parameter, algorithmn])
            results_tbl = '_'.join(["results", base_rulename, test_area])
            sql_query = """SELECT {}, classified
                            FROM {}""".format(parameter, results_tbl)
            mydf = psql.read_sql(sql_query, engine)
            if " " in mydf[parameter]:
                mydf[parameter] = re.sub(r"[\\s]", "_", my_df[parameter])
            y_true = mydf[parameter].apply(str)
            y_pred = mydf["classified"].apply(str)
            # print(confusion_matrix(y_true, y_pred))
            # , labels=paramdict[parameter]))
            confusion_out = ''.join([base_rulename, '_', test_area, ".txt"])
            confusion_out = osjoin(output_folder, confusion_out)
            print("confusion matrix located: {}".format(confusion_out))
            with open(confusion_out, 'w+') as f:
                f.write(classification_report(y_true, y_pred))
    except IOError as myerror:
        print(myerror)
    except BaseException as e:
        print("Sorry: {}".format(e))
