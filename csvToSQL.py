#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Niklas Moran
"""
# from __future__ import division
try:
    from os import scandir
except:
    from scandir import scandir

from os import path
# import sys
import io
from sqlalchemy import create_engine
import csv
from collections import defaultdict


def listcsvs(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in scandir(path):
        if entry.is_file() and entry.name.endswith(".csv"):
            yield entry.name, entry.path

paramdict = {
    "natflo_hydromorphic": ["0", "1"],
    "natflo_immature_soil": ["0", "1"],
    "natflo_species_richness": ["species_poor", "species_rich"],
    "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
    "natflo_usage_intensity": ["high", "medium", "low"],
    "natflo_wetness": ["dry", "mesic", "very_wet"],
    }

DSN = 'postgresql://postgres@localhost:5432/rlp_spatial'
engine = create_engine(DSN)
# conn = engine.connect()


def filter_dict(dict1, dict2):
    '''
    Takes two dictionaries and deletes matching records;
    Dict1 is the main dictionary;
    Dict2 is the secondary dictionary.
    RETURNS: dictionary Dict1 of
    unique values.
    '''

    for key in dict2.keys():
            if key in dict1.keys():
                del dict1[key]
                del dict2[key]
    return dict1  # [dict1, dict2]

if __name__ == '__main__':
    homedir = path.expanduser('~')
    scikit_folder = path.join(homedir, "test-rlp", "sci-kit_rules")
    sql_folder = path.join(homedir, "test-rlp", "sql_rules")
    for csv_name, csv_path in listcsvs(scikit_folder):
        sql_rules_out = "/".join([sql_folder, csv_name.replace(".csv",
                                                               ".sql")])
        rules = defaultdict(list)
        with io.open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            ruleset = []
            for row in reader:
                if len(row) == 1:
                    # if row[0] in paramdict[csv_name.split('.')[0]]:
                    # this is a class
                    # print("class: {}".format(row[0]))
                    rules[row[0]].append(list(ruleset))
                    ruleset.clear()
                else:
                    # print ("appending: {} ".format(row))
                    ruleset.append("{} {} {}".format(*row))
                # print("row: {}".format(row))
        test_table = "grasslands_test"
        class_name = csv_name.split(".csv")[0]
        with io.open(sql_rules_out, 'w') as sql_rule:
            for key, values in rules.items():
                sql_rule.write("--{}\n".format(key))
                sql_rule.write("SELECT id, {0}".format(class_name))
                sql_rule.write("FROM {} WHERE (\n".format(test_table))
                mylist = [' and '.join(i) for i in values]
                sql_rule.write("{}\n".format(') or \n('.join(mylist)))
                sql_rule.write(")")
                sql_rule.write("\n")
        with engine.begin() as conn:
            sql_query = []
            for key, values in rules.items():
                # make new table
                mylist = [' and '.join(i) for i in values]
                union_rule = "{}\n".format(') or \n('.join(mylist))
                # print("-- **** {0} values: {1} ****".format(class_name, key))
                conn.execute("""DROP TABLE
                                IF EXISTS results_{}""".format(class_name))
                conn.execute("""CREATE TABLE results_{0} (
                            id bigint, {0} VARCHAR(25),
                            classified VARCHAR(25),
                            PRIMARY KEY(id))""".format(class_name))
                sql_query.append("""SELECT id, {0}, \'{1}\' as classified
                         FROM {2}
                         WHERE ({3})\n""".format(class_name,
                                                 key,
                                                 test_table,
                                                 union_rule))
            sql = ' UNION '.join(sql_query)
            values_list = conn.execute(sql).fetchall()
            for row in values_list:
                insert = """INSERT INTO results_{0}
                            (id, {0}, classified)
                            VALUES (%s, %s,
                            %s)""".format(class_name)
                # print(insert)
                conn.execute(insert, (row[0], row[1], row[2]))
                # row[0],
                #                        value,
                #                        key))
            '''
            for key in rules.items():
                #for myid, value in values_dict.items():
                insert = """INSERT INTO results_{0}
                            (id, {0}, classified)
                            VALUES (%{id}s, %{class_name}s,
                            %{classified}s)""".format(class_name)
                conn.execute(insert, (**values_dictmyid,
                                        value,
                                        key))
            '''
