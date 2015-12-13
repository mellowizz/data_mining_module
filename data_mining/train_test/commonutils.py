#!/usr/bin/python2
# coding: utf-8

import io
import sys
from sys import platform
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from sqlalchemy import create_engine
import pandas.io.sql as psql
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import psycopg2
import pandas as pd

class Commonutils(object):
    def __init__(self, dsn_dict):
        self.dsn = 'postgresql://{user}@localhost/{database}'.format(**dsn_dict)

    def get_data(self, table_name):
        with create_engine(self.dsn) as engine:
            sql_str = "SELECT * FROM \"{}\"".format(table_name)
            df = psql.read_sql(sql_str, engine)
        if engine is not None:
            del (engine)
        return df.drop("ogc_fid", axis=1)


    def get_munged_data(self, table_name, paramdict, parameter):
        # TODO: add database parameter and settle on pycopg2 or sqlalchemy!
        """ Returns pandas dataframe with only relevant data."""
        parameter_values = None
        try:
            engine = create_engine(self.dsn)
        except Exception as e:
            print(e.msg)

        sql_str = "SELECT * FROM {}".format(table_name)
        df = psql.read_sql(sql_str, engine) #, params={'table_name': table_name})
        df = df[df[parameter].isin(paramdict[parameter])]
        a_df = df.select_dtypes(['float64'])  # 'int64'])
        a_df = a_df.fillna(0, axis=1)  
        a_df.insert(0, parameter, df[parameter])
        #if " " in a_df[parameter]:
        #    a_df[parameter] = re.sub(r"[\\s]", "_", a_df[parameter])
        cols = list(a_df)
        cols.insert(0, cols.pop(cols.index(parameter)))
        a_df = a_df.ix[:, cols]
        return a_df

    class GradientBoostingClassifierWithCoef(GradientBoostingClassifier):

        def fit(self, *args, **kwargs):
            super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
            self.coef_ = self.feature_importances_


    class RandomForestClassifierWithCoef(RandomForestClassifier):

        def fit(self, *args, **kwargs):
            super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
            self.coef_ = self.feature_importances_


    def get_labels_features(debug=0):
        """ DEPRECATED? """
        engine = create_engine(self.dsn)
        # 21068 rows
        sql_str = """SELECT *
                   FROM {0}
                   WHERE {1} is not NULL 
                   ORDER BY """.format(table_name, column)

        wet_classes = ["aquatic", "dry", "mesic", "wet/very_wet"]
        # all_file = path.join(csv_dir, "rlp_eunis_all_parameters.csv")
        df = psql.read_sql(sql_str, engine)

        df = df[df["wetness"].isin(wet_classes)]
        a_df = df.select_dtypes(include=[np.float])  # + df["wetness"]

        if (debug):
            print("total: {}".format(len(a_df)))

        drop_columns = [i for i in a_df.columns if i.endswith("fid") and
                        "ogc_fid" not in i]
        if (debug):
            print("Dropping columns: {}".format(drop_columns))
        for j in drop_columns:
            a_df.drop(j, axis=1, inplace=True)

        # a_df.drop("Shape_area", axis=1, inplace=True)
        # a_df.drop("Shape_length", axis=1, inplace=True)
        a_df = a_df.dropna()
        # a_df.drop()
        a_df.insert(0, a_df[parameter], df[parameter])
        # a_df["wetness"].replace(
        # to_replace="wet/very_wet", value="very_wet")  # .values

        return a_df[parameter], a_df.drop([parameter], axis=1)


    def print_feature_importance(clf, all_df, nb_to_display=10):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(nb_to_display):
            print("{} feature {} ({})".format(f + 1,
                                              all_df.columns[indices[f]],
                                              importances[indices[f]]))


    def get_distinct_values(table_name, column_name):
        """Return the distinct values in the given column and table name."""
        rows = []
        try:
            engine = create_engine(self.dsn)
            query = """SELECT DISTINCT %(column_name)s
                        FROM %(table_name)s
                        WHERE %(column_name)s IS NOT NULL""" % column_name, table_name
            result = engine.execute(query,)
            rows = [row[column_name] for row in result]
        except psycopg2.Error as myexception:
            print(myexception)
        finally:
            if result is not None:
                result.close()
        return rows  # [x[0] for x in rows if x[0] != "None"]


    def plot_feature_importance(self, forest, all_df, num_feat=10):
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(num_feat), importances[indices][:num_feat],
                color="r", yerr=std[indices][:num_feat], align="center")
        plt.xticks(range(num_feat), all_df.columns[indices][:num_feat],
                   rotation='vertical')
        plt.xlim([-1, num_feat])
        plt.xlabel("Feature")
        plt.show()


    def generate_classification_report(self, clf, x_test, y_test):
        """Prints out a confusion matrix from the classifier object."""
        expected = y_test
        predicted = clf.predict(x_test)

        print("Classification report for classifier %s:\n%s\n"
              % (clf, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" %
              metrics.confusion_matrix(expected, predicted))
        return metrics.confusion_matrix(expected, predicted)


    def get_lineage(self, tree, feature_names, wet_classes,
                    out_file="C:\\Users\\Moran\\test-rlp\\sci-kit_rules\\default.csv"):
        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value
        # print("value: I{0}".format(value))
        
        try:
            if sys.version < '3':
                infile = io.open(out_file, 'wb')
            else:
                infile = io.open(out_file, 'wb')
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
                            a_feature, a_split, a_threshold, a_parent_node=node
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


    def get_code(self, tree, feature_names, wet_classes):
        """ Traverses decision tree and emits rules."""
        ''' TODO: write to file '''
        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
            if (threshold[node] != -2):
                print("{},<=,{}".format(features[node], str(threshold[node])))
                if left[node] != -1:
                    recurse(left, right, threshold, features, left[node])
                print("} else {")
                if right[node] != -1:
                    recurse(left, right, threshold, features, right[node])
                print("}")
            else:
                print("{0}".format(wet_classes[np.argmax(value[node])]))

        recurse(left, right, threshold, features, 0)


    def print_best_parameters(self, parameters, best_parameters):
        my_dict = dict()
        for param_name in sorted(parameters.keys()):
            print("{}: {}".format(param_name, best_parameters[param_name]))
            my_dict[param_name] = best_parameters[param_name]
        return my_dict
