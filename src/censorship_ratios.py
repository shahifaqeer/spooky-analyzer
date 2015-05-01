from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict

## GET CENSORSHIP INFO
def get_ratios(dfin, GROUP=0):
    ''' assume df_count is indexed'''
    if GROUP:
        df_count = dfin.groupby(['sIP', 'domain', 'subcat',
        'country', 'case'])['port'].count().unstack().fillna(0)
    else:
        df_count = dfin.copy()
    df_count['tot'] = df_count.sum(axis=1)

    df_count['err'] = 0
    if (0 in df_count.columns):
        df_count['err']+= df_count[0]
    if (4 in df_count.columns):
        df_count['err']+= df_count[4]

    df_count['tot'] = df_count['tot'] - df_count['err']

    if 1 in df_count.columns:
        df_count['case1'] = df_count[1]/df_count['tot']
    if 2 in df_count.columns:
        df_count['case2'] = df_count[2]/df_count['tot']
    if 3 in df_count.columns:
        df_count['case3'] = df_count[3]/df_count['tot']
    return df_count

def get_censorship_by_country_sIP_subcat(df_val, dimension='censorship'):
    """
    groupby sIP subcat + country. Must use disjoint to duplicate measurements
    conclusions drawn by sIP not by subcat
    """
    censorship = df_val.groupby(['sIP', 'domain', 'subcat',
                                 'country', 'case'])['port'].count().unstack().fillna(0)
    censorship = get_ratios(censorship)
    global_censorship = df_val.groupby(['sIP', 'domain',
                                        'subcat', 'case'])['port'].count().unstack().fillna(0)
    global_censorship = get_ratios(global_censorship)
    if dimension == 'censorship':
        censor_country = (1 - censorship['case2']).unstack()
        censor_global = (1 - global_censorship['case2'])
    else:
        # dimension can be err, tot, case1, case2, case3, 1, 2, 3, 4, 0 apart from censorship
        censor_country = censorship[dimension].unstack()
        censor_global = global_censorship[dimension]

    censor_country['global'] = censor_global
    #censor_country= censor_country.reset_index()
    return censor_country

def get_censorship_by_country_sIP(df_val, dimension='censorship'):
    """
    only groupby sIP + country, no need to use disjoint repeated measurements
    conclusions drawn by sIP not by subcat
    """
    censorship = df_val.groupby(['sIP', 'country', 'case'])['port'].count().unstack().fillna(0)
    censorship = get_ratios(censorship)
    global_censorship = df_val.groupby(['sIP','case'])['port'].count().unstack().fillna(0)
    global_censorship = get_ratios(global_censorship)
    if dimension == 'censorship':
        censor_country = (1 - censorship['case2']).unstack()
        censor_global = (1 - global_censorship['case2'])
    else:
        # dimension can be err, tot, case1, case2, case3, 1, 2, 3, 4, 0 apart from censorship
        censor_country = censorship[dimension].unstack()
        censor_global = global_censorship[dimension]

    censor_country['global'] = censor_global
    #censor_country= censor_country.reset_index()
    return censor_country

def get_censorship_technology_combined(df_val, sliced_sIP_list):
    """ df_val can be filtered to some sIPs only """
    df_sliced = df_val[df_val['sIP'].isin(sliced_sIP_list)]

    data_sliced_sIP = defaultdict(int)
    data_sliced_sIP['1'] = get_censorship_by_country_sIP(df_sliced, 1).sum()
    data_sliced_sIP['3'] = get_censorship_by_country_sIP(df_sliced, 3).sum()
    data_sliced_sIP['2'] = get_censorship_by_country_sIP(df_sliced, 3).sum()
    data_sliced_sIP['tot'] = get_censorship_by_country_sIP(df_sliced, 'tot').sum()
    data_sliced_sIP['err'] = get_censorship_by_country_sIP(df_sliced, 'err').sum()

    df_sliced_sIP = pd.DataFrame(data_sliced_sIP)
    df_sliced_sIP['unknown']=data_sliced_sIP['err']/(data_sliced_sIP['tot']+data_sliced_sIP['err'])
    df_sliced_sIP['case1']=data_sliced_sIP['1']/data_sliced_sIP['tot']
    df_sliced_sIP['case3']=data_sliced_sIP['3']/data_sliced_sIP['tot']
    df_sliced_sIP['case13']=df_sliced_sIP['case1']+ df_sliced_sIP['case3']
    df_sliced_sIP['case1/case13']=df_sliced_sIP['case1']/df_sliced_sIP['case13']

    return df_sliced_sIP

def get_censorship_technology_sIP(df_val):
    """ df_val can be filtered to some sIPs only """
    # by sIP
    case1_tot = get_censorship_by_country_sIP(df_val, 1)
    case3_tot = get_censorship_by_country_sIP(df_val, 3)
    tot = get_censorship_by_country_sIP(df_val, 'tot')
    unknown = get_censorship_by_country_sIP(df_val, 'err')

    technology = (case1_tot)/(case1_tot+case3_tot)
    censorship = (case1_tot+case3_tot)/tot
    unknown = unknown/tot
    return censorship, technology, unknown
