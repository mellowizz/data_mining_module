# -*- coding: utf-8 -*-
"""
Created on 04.06.2015

@author: Moran
"""

from __future__ import division
import csv
from collections import Counter
from collections import defaultdict
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from os import path
import sys


def fetch_data_from_table(validation_tbl, results_tbl, parameter="NATFLO_wetness"):
    engine = create_engine('postgresql://postgres:BobtheBuilder@localhost/RLP')
    conn = engine.connect()
    s = text("SELECT classified, \"{0}\" from \"{1}\"".format(parameter, results_tbl))
    v_wetness = text("SELECT \"{0}\"  from \"{1}\"".format(parameter, validation_tbl))
    results = conn.execute(s).fetchall()
    v_results = conn.execute(v_wetness).fetchall()
    c_cntr = defaultdict(Counter)
    w_cntr = Counter()
    v_cntr = Counter()
    for wetness in v_results:
        wet = wetness[0]
        if '/' in wet:
            wet = wet.split('/')[1]
        if ' ' in wet:
            wet = wet.replace(' ', '_')
        v_cntr[wet] += 1
    for row in results:
        classified = row[0]
        currwetness = row[1]
        if '/' in currwetness:
            currwetness = currwetness.split('/')[1]
        if ' ' in currwetness:
            currwetness = currwetness.replace(' ', '_')
        w_cntr[currwetness] += 1
        c_cntr[classified][currwetness] += 1
    sorted(w_cntr.keys())
    sorted(c_cntr.keys())
    sorted(v_cntr.keys())
    return c_cntr, w_cntr, v_cntr


def create_save_file(var_fname):
    return path.join(r'c:\Users\Moran\test-rlp\accuracy_assessments', var_fname)

if __name__ == '__main__':
    p = 1  # print to stdout? saarburg_200_testing_300_zstats_1seath_rules_results
    #validation_tbl = "testing_600"
    num_objects = "600"
    #testing_eagle_vegetationtype_1_natflo_wetness_600_1seath_rules_results
    parameter_2 = "EAGLE_vegetationType_1"
    parameter_1 = "NATFLO_wetness"
    validation_tbl = "_".join(["testing", parameter_1, parameter_2, num_objects]).lower()

    #training_eagle_vegetationtype_1_seath_1seath_rules_results
    algorithm = 'cart'
    num_rules = str(10)+algorithm
    results_tbl = '_'.join([validation_tbl, num_rules, 'rules', 'results'])
    print("resultstbl: {}".format(results_tbl))
    classified_cntr, wetness_cntr, validation_cntr = fetch_data_from_table(validation_tbl, results_tbl, parameter_1)
    print(classified_cntr)
    ''' DANGER using classified_cntr because of unclassified classes!'''
    header = [k for k in wetness_cntr]
    fname = results_tbl + '.csv'
    file_name = create_save_file(fname)
    total_wetness = [validation_cntr[j] for j in header]
    total_wetness.append(sum(total_wetness))
    unclassed = [validation_cntr[wet] - wetness_cntr[wet] for wet in header]
    if sys.version < '3':
        infile = open(file_name, 'wb')
    else:
        infile = open(file_name, 'w', newline='')
    with infile as csvfile:
        resultwriter = csv.writer(csvfile, delimiter=',')
        resultwriter.writerow(['class'] + header + ['total', 'Producer', 'User'])
        resultwriter.writerow(['unclass'] + unclassed)  # [validation_cntr[wetness] for wetness in classified_cntr])
        correctly_classified = 0.0
        #for i in classified_cntr:
        for index, i in enumerate(header):  # wetness_cntr:
            correctly_classified += classified_cntr[i][i]
            curr_classes = [classified_cntr[i][j] for j in header]
            row_total = sum(classified_cntr[i].values())
            if row_total == 0:
                producer = 0
            else:
                producer = classified_cntr[i][i] / row_total * 100.0
            if total_wetness[index] == 0:
                user = 0
            else:
                user = classified_cntr[i][i] / total_wetness[index] * 100.0
                #user = classified_cntr[i][i] / wetness_cntr[i] * 100.0
            print("i:{0}, classified_cntr: {1}, wetness_cntr:{2}".format(i, classified_cntr[i][i], wetness_cntr[i]))
            resultwriter.writerow([i] + curr_classes + [sum(classified_cntr[i].values())] +
                                  ["%.2f" % producer, "%.2f" % user])
        if p:
            print("{0}\t {1}".format('total', ','.join([str(wetness_cntr[j]) for j in classified_cntr])))
        resultwriter.writerow(['total'] + total_wetness)
        overall_accuracy = correctly_classified / total_wetness.pop() * 100.0
        resultwriter.writerow(['overall accuracy', '{0:.2f}'.format(overall_accuracy)])
