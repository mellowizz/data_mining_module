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
# import io

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


dry = '''
select ut.id, substr(eunis,1,2) as eunis, 'E1' as classified 
from ut_saarburg_mad_all as ut, results_natflo_wetness as wet,
 results_natflo_usage_intensity as intensity
where ut.id = wet.id and ut.id = intensity.id and wet.classified = 'dry'
 and intensity.classified = 'low'
'''
some = '''
and (
b_min <= 48.500000 and re_ndvi_mean <= 0.517389 and tpi2_sum > 1154.854858 and 
difi_std <= 0.114556 and mrrt_std > 0.785374) or 
(b_min <= 48.500000 and re_ndvi_mean <= 0.517389 and tpi2_sum > 1154.854858 and
difi_std > 0.114556 and pan4_glcm_dis_0 <= 0.019477 and pan4_glcm_cor_45 <=
1883.409180 and b_min > 22.500000 and difi_max > 186.755646 and mpi_max <=
0.493550 and r_min <= 34.500000) or (b_min <= 48.500000 and re_ndvi_mean <=
0.517389 and tpi2_sum > 1154.854858 and difi_std > 0.114556 and pan4_glcm_dis_0
<= 0.019477 and pan4_glcm_cor_45 <= 1883.409180 and b_min > 22.500000 and
difi_max > 186.755646 and mpi_max <= 0.493550 and r_min > 34.500000) or (b_min
<= 48.500000 and re_ndvi_mean <= 0.517389 and tpi2_sum > 1154.854858 and
difi_std > 0.114556 and pan4_glcm_dis_0 > 0.019477 and slop_median > 13.846483 )
-- and id in (select ndsm_nm_id from ndsm_saarburg where ndsm_nm_max < 1)
'''

mesic_hydro = '''
SELECT id, substr(ut_saarburg_mad_all.eunis,1,2) as eunis,
'E2' as classified
FROM ut_saarburg_mad_all
WHERE id IN (
SELECT id FROM results_natflo_wetness
WHERE classified = 'mesic')
AND
id IN (SELECT id
from results_natflo_usage)
AND id IN (SELECT id from results_natflo_usage_intensity
WHERE classified = 'low' or classified = 'medium')
AND
(mrvb_median <= 1.543150 and ndvi_std <= 0.058207) or  (mrvb_median <= 1.543150
and ndvi_std > 0.058207 and ndwi_min <= -0.435071 and g_sum <= 1114749.500000
and caar_min <= 413.378204 and tpi5_min <= -0.570950 and ndvi_max <= 0.579725)
or (mrvb_median <= 1.543150 and ndvi_std > 0.058207 and ndwi_min <= -0.435071
and g_sum <= 1114749.500000 and caar_min <= 413.378204 and tpi5_min <= -0.570950
and ndvi_max > 0.579725 and g_min <= 44.500000 and re_ndvi_re_max <= 0.377492
and swi_sum > 2878.907715) or (mrvb_median <= 1.543150 and ndvi_std > 0.058207
and ndwi_min <= -0.435071 and g_sum <= 1114749.500000 and caar_min <= 413.378204
and tpi5_min <= -0.570950 and ndvi_max > 0.579725 and g_min <= 44.500000 and
re_ndvi_re_max > 0.377492) or (mrvb_median <= 1.543150 and ndvi_std > 0.058207
and ndwi_min <= -0.435071 and g_sum <= 1114749.500000 and caar_min <= 413.378204
and tpi5_min <= -0.570950 and ndvi_max > 0.579725 and g_min > 44.500000) or
(mrvb_median <= 1.543150 and ndvi_std > 0.058207 and ndwi_min <= -0.435071 and
g_sum <= 1114749.500000 and caar_min <= 413.378204 and tpi5_min > -0.570950) or
(mrvb_median <= 1.543150 and ndvi_std > 0.058207 and ndwi_min <= -0.435071 and
g_sum <= 1114749.500000 and caar_min > 413.378204) or (mrvb_median <= 1.543150
and ndvi_std > 0.058207 and ndwi_min <= -0.435071 and g_sum > 1114749.500000) or
(mrvb_median <= 1.543150 and ndvi_std > 0.058207 and ndwi_min > -0.435071) or
(mrvb_median > 1.543150 and pan4_glcm_var_135 <= 0.035906 and re_ndre_max <=
0.300928) or (mrvb_median > 1.543150 and pan4_glcm_var_135 <= 0.035906 and
re_ndre_max > 0.300928 and tpi2_median <= -8.432250 and nir_std <= 10.652500) or
(mrvb_median > 1.543150 and pan4_glcm_var_135 <= 0.035906 and re_ndre_max >
0.300928 and tpi2_median <= -8.432250 and nir_std > 10.652500 and prcu_min <=
-1.795377) or (mrvb_median > 1.543150 and pan4_glcm_var_135 <= 0.035906 and
re_ndre_max > 0.300928 and tpi2_median <= -8.432250 and nir_std > 10.652500 and
prcu_min > -1.795377 and ndvi_mean <= 0.203127) or (mrvb_median > 1.543150 and
pan4_glcm_var_135 <= 0.035906 and re_ndre_max > 0.300928 and tpi2_median <=
-8.432250 and nir_std > 10.652500 and prcu_min > -1.795377 and ndvi_mean >
0.203127 and ddvi_min <= 96.500000) or (mrvb_median > 1.543150 and
pan4_glcm_var_135 <= 0.035906 and re_ndre_max > 0.300928 and tpi2_median <=
-8.432250 and nir_std > 10.652500 and prcu_min > -1.795377 and ndvi_mean >
0.203127 and ddvi_min > 96.500000 and re_ndre_median <= 0.360556 and swi_sum <=
2862.500977) or (mrvb_median > 1.543150 and pan4_glcm_var_135 <= 0.035906 and
re_ndre_max > 0.300928 and tpi2_median <= -8.432250 and nir_std > 10.652500 and
prcu_min > -1.795377 and ndvi_mean > 0.203127 and ddvi_min > 96.500000 and
re_ndre_median > 0.360556) or (mrvb_median > 1.543150 and pan4_glcm_var_135 <=
0.035906 and re_ndre_max > 0.300928 and tpi2_median > -8.432250) or (mrvb_median
> 1.543150 and pan4_glcm_var_135 > 0.035906 and swi_min <= 12.902250) or
(mrvb_median > 1.543150 and pan4_glcm_var_135 > 0.035906 and swi_min > 12.902250
and pan4_glcm_asm_135 <= 0.031156
)
'''

mesic_meadows = '''
SELECT id, substr(ut_saarburg_mad_all.eunis,1,4) as eunis,
'E2.1' as classified
FROM ut_saarburg_mad_all
WHERE id IN (
SELECT id FROM results_natflo_wetness
WHERE classified = 'mesic')
AND
id IN (SELECT id from results_natflo_usage where classified = 'grazing')
AND id IN (SELECT id from results_natflo_usage_intensity
WHERE classified = 'high' or classified = 'medium')
--AND id IN (SELECT id from results_natflo_species_richness
--WHERE classified = 'species_rich')
-- AND id IN (SELECT ndsm_nm_id FROM ndsm_saarburg WHERE ndsm_nm_max < 1)
'''
very_wet = '''
select id, substr(ut_saarburg_mad_all.eunis,1,2) as eunis, 'E3' as classified
from ut_saarburg_mad_all where id in (
select id from results_natflo_wetness where classified = 'very_wet')
and id in (select id from results_natflo_usage)
and id in (select id from results_natflo_usage_intensity 
where classified = 'medium') 
--and id in (select ndsm_nm_id from ndsm_saarburg where ndsm_nm_max < 1)
'''

hay_meadows = '''
-- e2.22
SELECT id, eunis,
'E2.22' as classified
FROM ut_saarburg_mad_all
WHERE id IN (
SELECT id FROM results_natflo_wetness
WHERE classified = 'mesic')
AND
id IN (SELECT id from results_natflo_usage where classified = 'mowing')
AND
id in (SELECT id from results_natflo_usage_intensity
where classified = 'medium')
'''


if __name__ == '__main__':
    paramdict = {
        "natflo_hydromorphic": ["0", "1"],
        "natflo_immature_soil": ["0", "1"],
        "natflo_species_richness": ["species_poor", "species_rich"],
        "natflo_usage": ["grazing", "mowing", "orchards", "vineyards"],
        "natflo_usage_intensity": ["high", "medium", "low"],
        "natflo_wetness": ["dry", "mesic", "very_wet"]
        #"eunis": ['D5', 'E1', 'E2', 'E3', 'E5', 'FA', 'F3', 'F4', 'FB', 'G1', 
        #          'G5', 'H2', 'H3', 'I1', 'J1', 'Y.']
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
        sql_query = mesic_meadows #dry  ' UNION '.join([dry, mesic, very_wet])
        mydf = psql.read_sql(sql_query, engine)
        y_true = mydf[parameter].apply(str)
        y_pred = mydf["classified"].apply(str)
        output_folder = osjoin(homedir, output_folder)
        my_matrix = ''.join(['classification', '_e2.1',
                             '.txt'])
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
