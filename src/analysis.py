
# coding: utf-8

# In[20]:

import pandas as pd
import os, sys
from caseDetection import detect_case


# In[2]:

df = pd.read_pickle("data/detected_case_all_Apr12.pkl")


# In[10]:

df.head(1)


# In[ ]:




# ## Sanitize by case
# - remove case = 0, -1, 4
# - case = -1 default: why??
# - treat case -1 as 0 (did not enter else)

# In[24]:

get_ipython().magic(u'load caseDetection.py')


# In[27]:

df_case_default = df[df['case'] == -1]
x = df_case_default.ix[138003,:]
print x


# In[43]:

#!/usr/bin/env/ python
from intervention import *
import numpy as np
import random,glob
import string
import datetime as dt

INTERVENTION = 20 #number of forged packets
ALPHA = 0.01
ao_alpha = 0.05
diffList=[6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,10,10,11,9,9,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,13,9,9,9,9,None,None,10,9,9,9,9,9,9,None,None,9,9,9,None,None,None,9,9,None,None,9,9,10,9,9,9]
gIP="71.44.3.191"
sIP="198.211.105.99"
ts=[60,64,70]

def detect_case2(gIP,sIP,diffList,ts,error_case,INTERVENTION,DEBUG=False):

 #   if len(ts)<3: return(4,None,None)
 #   if error_case!=None: return(4,None,None)
    try:
        ir=intervention_at_time(diffList,ts, [0.5 * INTERVENTION,            (min(2,((len(ts)+1)/2))) * INTERVENTION], 0.05, True) #FIXME
        if DEBUG:print "ir.pvalues",ir.pvalues[0],ALPHA
        case =- 1
        #print ir.pvalues, ALPHA
        if ir.pvalues[0] <= ALPHA:
            case = 1
        elif ir.pvalues[0] > (1.0 - ALPHA):
            print ir.pvalues[1], (1.0 - ALPHA), ir.pvalues[1] < ALPHA, ir.pvalues[1] > (1.0 - ALPHA)
            if ir.pvalues[1] < ALPHA:
                case = 2
            elif ir.pvalues[1] > 1.0 - ALPHA:
                case = 3
        else:
            case = 0
        if DEBUG: print gIP,sIP,case,ir
        return(case,ir.pvalues[1],ir.intervention)
    except Exception:
        if DEBUG:print gIP,sIP, ts, "ERROR in detect case ARMA",diffList
        case = 4
        return(case,None,None)

#if __name__=="__main__":
#    print detect_case(gIP,sIP,diffList,ts,None,INTERVENTION,True)


# In[44]:

detect_case2(x['gIP'],  x['sIP'], x['diff_list'], x['ts'], None, 20, True)


# In[101]:

df_bad = df [  df['case'].isin([0, -1, 4])  ]
df_good = df [  df['case'].isin([1,2,3])  ]


# In[102]:

len(df_good)


# In[103]:

len(df_bad)


# In[104]:

# Save df_bad "bad_cases_detected" [sip, gip, country] 
df_bad[['sIP', 'gIP', 'country']].to_csv("data/case_er_redo")
df_bad.to_pickle("data/case_er_redo.pkl")


# ## Look at SIP and country
# - due to measurement snafu same sIP and country pairs might have ran two or more measurements
# - each of these measurements should usually end up in the same "case"
# - but if two of these measurement have different "case" numbers => further analysis is needed
# - only look at case 1,2,3

# In[122]:

grouped_sIP_CO = df_good.groupby(["sIP", "country"])
df_sIP_CO = grouped_sIP_CO['case'].unique().reset_index()


# In[124]:

df_sIP_CO["num_unique_cases"] = df_sIP_CO['case'].apply(lambda x: len(x))
df_redo = df_sIP_CO[df_sIP_CO['num_unique_cases'] > 1]
df_redo = df_redo.rename(columns={'case':'unique_cases'})
df_redo.head()


# In[125]:

# number of sIP challenged throughout
len(df_redo['sIP'].unique())


# In[126]:

# number of countrie challenged throughout
len(df_redo['country'].unique())


# In[109]:

# per country, how many unique sIPs
#df_temp = df_redo.groupby(['country'])['sIP'].count()
#df_temp.sort(ascending=False)
#df_temp


# In[110]:

# per sIP, how many unique countries
#df_temp = df_redo.groupby(['sIP'])['country'].count()
#df_temp.sort(ascending=False)
#df_temp


# ### Flag these cases
# - with sIP, country in df_redo
# - further inversigation in df_good
# - merge df_redo and df_good on ['sIP', 'country'] to get num_unique_cases
# - expectedly from 63 x 532, all should have multiple cases

# In[128]:

df_merged_unique_cases = df_good.merge(df_redo, on=['sIP', 'country'])


# In[132]:

# which ones had just one result?

df_unique = df_merged_unique_cases[df_merged_unique_cases['num_unique_cases'] < 2]
print len(df_unique)
df_unique


# # DATA ANALYSIS
# - per country num obs
# - per country num unique sIPs tested

# In[190]:

# meaningful num of countries
THRESHOLD = 350


# In[191]:

# per country, how many unique sIPs
df_unique_sIP_per_country = df_good.groupby(['country'])['sIP'].unique().apply(lambda x:len(x))
df_current = df_unique_sIP_per_country[df_unique_sIP_per_country > THRESHOLD]


# In[196]:

valid_countries = df_current.index
print valid_countries, len(valid_countries)

# filter df_good to only valid countries
# remove extra columns
df = df_good[  df_good['country'].isin(valid_countries)   ] [['sIP', 'gIP', 'country', 'diff_list', 'domain', 'subcat', 'case']]


# In[197]:

df_obs_per_country = df.groupby('country').count().reset_index().sort('sIP', ascending=False)
# number of measurements per country
df_obs_per_country


# In[198]:

# num of unique sIPs per country
df_unique_sIP_per_country = df.groupby('country')['sIP'].unique().apply(lambda x: len(x)).sort(inplace=False, ascending=False)
pd.DataFrame(df_unique_sIP_per_country).to_html('unique_sIP_per_country-filtered350.htm')


# ### case 1, 2, 3 per (country, subcat)

# In[199]:

grouped_by_country_subcat_case = df.groupby(['country', 'subcat', 'case'])


# In[200]:

count_per_country_subcat_case = grouped_by_country_subcat_case.count()


# In[206]:

# replace nan with 0
df_case_count = count_per_country_subcat_case.unstack(level=-1)['sIP']
df_case_count['total'] = df_case_count.sum(axis=1)
df_case_count.to_html('cases_per_country_subcat-filtered350.htm')


# #### Censorship by country for each subcat
# - create num(subcat) bar plots comparing case2 with (case1 + case3)

# In[215]:

# clear all except df_case_count
del df, df_bad, df_case_default, df_current, df_good, df_merged_unique_cases, df_obs_per_country, df_redo
del df_sIP_CO, df_temp, df_unique, df_unique_sIP_per_country


# In[216]:

group_by_subcat = df_case_count.reset_index().groupby('subcat')


# In[227]:

subcat = 'adult'
temp_df = df_case_count.ix[ group_by_subcat.groups[subcat] ].reset_index()
temp_df['censored'] = temp_df[1]+temp_df[3]
temp_df = temp_df.rename(columns={2:'uncensored'})[['country', 'censored', 'uncensored']].set_index('country')


# In[235]:

temp_df


# In[243]:

fig1, ax1 = subplots(1,1)
ax1.hist((temp_df['censored'], temp_df['uncensored']))
ax1.set_xticks(range(len(temp_df)))

#labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = temp_df.index

ax1.set_xticklabels(labels)


# In[ ]:



