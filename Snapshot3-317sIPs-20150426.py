from __future__ import division
get_ipython().magic(u'matplotlib nbagg')
import pandas as pd
import os, sys
import numpy as np
from collections import defaultdict
from caseDetection import detect_case
from world_map_maker import create_world_map
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
#import matplotlib as mpl
#mpl.use("GTK3cairo")
#import matplotlib.pyplot as plt
#plt.plot([1,2,3,4],'*-')


SUBCAT_TYPE = 'disjoint'
#'unshared'
#'general'

# In[63]:

SAMPLENAME = 'Snapshot3_317sIPs_Apr27'
RESULTS = "results/" + SAMPLENAME + "/"
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


# In[64]:

df_sIP_subcat = pd.read_csv("data/Servers_IMC.txt")
valid_sIP = list(df_sIP_subcat['sIP'])
df_sIP_subcat['subcat'] = df_sIP_subcat['subcat'].apply(lambda x: x.replace('kids_and_teens', 'teens'))
print len( df_sIP_subcat ), len(valid_sIP)
df_sIP_subcat.head(1)


# In[65]:

subcat_duplicated = defaultdict(list)
for ix, row in df_sIP_subcat.iterrows():
    if len(row[2].split("|")) > 1:
        shared = 1
    else:
        shared = 0
    for subcat in row[2].split("|"):
        if subcat == 'kids_and_teens':
            subcat = 'teens'
        subcat_duplicated['sIP'].append(row[0])
        subcat_duplicated['domain'].append(row[1])
        subcat_duplicated['subcat'].append(subcat)
        subcat_duplicated['shared'].append(shared)
df_sIP_subcat_disjoint = pd.DataFrame(subcat_duplicated)
df_sIP_subcat_disjoint.to_csv("data/Servers_IMC_disjoint_sIP.txt")


# ## Dealing with Duplicate Subcats
# - general: anything with multiple subcats is just made general
# - unshared: anything with multiple subcats is simply removed
# - disjoint: multiple subcats are repeated (=> repeated measurements for the same site in diff subcats)

# In[66]:

df_sIP_subcat_general = df_sIP_subcat.copy()
df_sIP_subcat_general['subcat'] = df_sIP_subcat_general['subcat'].apply(lambda x: 'general' if '|' in x else x)
df_sIP_subcat_unshared = df_sIP_subcat_disjoint[df_sIP_subcat_disjoint['shared']==0]
df_sIP_subcat_shared = df_sIP_subcat_disjoint[df_sIP_subcat_disjoint['shared']==1]


# In[67]:

subcat_hist = pd.DataFrame({'general': df_sIP_subcat_general.groupby('subcat')['domain'].count(),
                           'unshared': df_sIP_subcat_unshared.groupby('subcat')['domain'].count(),
                           'disjoint': df_sIP_subcat_disjoint.groupby('subcat')['domain'].count()})
print subcat_hist


# ## Load DATA

# In[68]:

df_all1 = pd.read_pickle("data/case_detected_all_20150422.pkl")
df_all2 = pd.read_pickle("data/case_detected_all_20150427.pkl")


# In[69]:

df_all = pd.concat([df_all1,df_all2])
print len(df_all1), len(df_all2), len(df_all)


# In[70]:

df_all.head(1)


# # SANITIZE based on good 252 sIPs

# In[71]:

df_final = df_all[df_all['sIP'].isin( valid_sIP )]


# In[72]:

print len(df_all), len(df_final)


# # Sets of data: shared, disjoint, general

# In[73]:

data = {}
# disjoint duplicate entries
data['disjoint'] = df_final[['sIP', 'country', 'case']].merge(df_sIP_subcat_disjoint, on=['sIP'])
# unshared entries
data['unshared'] = df_final[['sIP', 'country', 'case']].merge(df_sIP_subcat_unshared, on=['sIP'])
# general entries with categ replacement
data['general'] = df_final[['sIP', 'country', 'case']].merge(df_sIP_subcat_general, on=['sIP'])

print "Number of measurements: ", {k:len(v) for k,v in data.items()}


# In[74]:

measurements_per_subcat = pd.DataFrame( {k:v.groupby('subcat')['domain'].count() for k,v in data.items()} )

print measurements_per_subcat


# # Helper functions

# In[97]:

## GET CENSORSHIP INFO
def get_ratios(df_count):
    ''' assume df_count is indexed'''
    #df_count = dfin.groupby(['sIP', 'domain', 'subcat', 'slash24',
    #'country', 'case'])['port'].count().unstack().fillna(0)
    df_count['tot'] = df_count.sum(axis=1)

    df_count['err'] = 0
    if (0 in df_count.columns):
        df_count['err']+= df_count[0]
    if (4 in df_count.columns):
        df_count['err']+= df_count[4]

    #df_count['tot'] = df_count['tot'] - df_count['err']

    if 1 in df_count.columns:
        df_count['case1'] = df_count[1]/df_count['tot']
    if 2 in df_count.columns:
        df_count['case2'] = df_count[2]/df_count['tot']
    if 3 in df_count.columns:
        df_count['case3'] = df_count[3]/df_count['tot']
    return df_count

def get_censorship_by_country_sIP(df_val, dimension='censorship'):
    censorship = df_val.groupby(['sIP', 'domain', 'subcat',
                                 'country', 'case'])['port'].count().unstack().fillna(0)
    get_ratios(censorship)
    global_censorship = df_val.groupby(['sIP', 'domain',
                                        'subcat', 'case'])['port'].count().unstack().fillna(0)
    get_ratios(global_censorship)
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


# In[98]:

# case1 + case3 ratio
# censorship = get_censorship_by_country_sIP(df_all)


# # SUBCATS INCLUDED (df_general) VS EXCLUDED (df_unshared) VS DOUBLED (df_disjoint)
# - the following analysis is from Snapshot3-plot
# - now includes both Apr22 and Apr27 data

# In[99]:

#########################################################################################################################
subcat_type = SUBCAT_TYPE
df_analysis = data[subcat_type]


# In[100]:

grouped_by_country_subcat_case = df_analysis.groupby(['country', 'subcat', 'case'])
df_count_per_country_subcat_case = grouped_by_country_subcat_case.count().rename(columns={'sIP':'count'})[['count']]
#df_count_per_country_subcat_case.head(10)


# In[101]:

# replace nan with 0
df_case_count = df_count_per_country_subcat_case.unstack(level=-1)['count'].fillna(0)
df_case_count['total'] = df_case_count.sum(axis=1)
df_case_count.to_html(RESULTS+'cases_per_country_subcat-'+subcat_type+'.htm')
#df_case_count.head()


# ## GLOBAL SUBCAT BARPLOT

# In[102]:

df_censorship = df_case_count.reset_index().groupby('subcat').sum()
df_censorship = df_censorship.rename( columns= {k:str(k) for k in df_censorship.columns} )
df_censorship['tot_err'] = df_censorship[['0','4']].sum(axis=1)
#df_censorship['total'] = df_censorship['total'] - df_censorship['total_err']
df_censorship = df_censorship[['1','3','2','tot_err','total']]

df_censorship['case1'] = df_censorship['1']/df_censorship['total']
df_censorship['case3'] = df_censorship['3']/df_censorship['total']
df_censorship['case2'] = df_censorship['2']/df_censorship['total']
df_censorship['error'] = df_censorship['tot_err']/df_censorship['total']


censorship_ratio = defaultdict(int)
censorship_ratio['case1'] = df_censorship['case1'].to_dict()
censorship_ratio['case2'] = df_censorship['case2'].to_dict()
censorship_ratio['case3'] = df_censorship['case3'].to_dict()
censorship_ratio['error'] = df_censorship['error'].to_dict()

df_censorship.sort('case2', inplace=True)
df_censorship.to_html(RESULTS+ 'subcat_censorship_total-'+subcat_type+'.htm')
#print df_censorship


# In[103]:

fig1, ax1 = plt.subplots(1,1, figsize=(10,6))
df = df_censorship[['case2', 'case1', 'case3']].rename(columns = {'case2':'no-packets-dropped',
                                                                  'case1':'server-to-client-dropped',
                                                                  'case3':'client-to-server-dropped'})
df.plot(kind='bar', stacked=True, ax=ax1)
ax1.legend(loc='best', prop={'size':14})
fig1.tight_layout()
fig1.savefig(RESULTS + "cases_ratio_by_subcat_stacked_bar-"+subcat_type)

df.to_html(RESULTS + "cases_ratio_by_subcat_stacked_bar-"+subcat_type+".html")


# ## PER SUBCAT ANALYSIS (INC. ERRORS)

# In[107]:

group_by_subcat = df_case_count.reset_index().groupby('subcat')


# In[108]:

def get_ratios(subcat, indices, df_case_count):
    #if subcat == 'all':

    temp_df = df_case_count.ix[ group_by_subcat.groups[subcat] ].reset_index()
    #for CASE_NUM in [0,1,2,3,4]:
    #    if not (CASE_NUM in temp_df.columns):
    #        temp_df[CASE_NUM] = 0
    temp_df['censored'] = temp_df[1]+temp_df[3]
    temp_df['unknown'] = temp_df[0]+temp_df[4]
    temp_df = temp_df.rename(columns={2:'uncensored'})[['country', 'censored', 'uncensored', 'unknown' ,1 ,3]].set_index('country')
    #temp_df = temp_df.rename(columns={2:'uncensored'})[['country', 'censored', 'uncensored', 1, 3]].set_index('country')

    # calculate ratios
    temp_df['total'] = temp_df.sum(axis=1)

    temp_df['ratio-case1'] = temp_df[1]/temp_df['total']
    temp_df['ratio-case3'] = temp_df[3]/temp_df['total']
    temp_df['ratio-censored'] = temp_df['censored']/(temp_df['total'])
    temp_df['ratio-uncensored'] = temp_df['uncensored']/(temp_df['total'])
    temp_df['ratio-unknown'] = temp_df['unknown']/(temp_df['total'])

    # replace no entries with 0
    df_country_case = temp_df.fillna(0)
    return df_country_case


# In[109]:

country_case_per_subcat = defaultdict(int)

# iterate over groupby object, indices
for subcat, indices in group_by_subcat.groups.iteritems():
    country_case_per_subcat[subcat] = get_ratios(subcat, indices, df_case_count)
    #print subcat, len(country_case_per_subcat[subcat])

df_country_case_subcat = pd.concat(country_case_per_subcat)


# In[110]:

# save list of countries
pd.Series(df_country_case_subcat.reset_index()['country'].unique()).to_csv(RESULTS+"list_of_countries.csv")


# In[111]:

df_experiment_stats = df_country_case_subcat['total'].unstack()
per_country = df_experiment_stats.describe().T[['count','mean','std','min','max']].rename(
    columns={'count':'num_countries', 'mean':'avg_num_measurements_per_subcat'})
per_country.to_html(RESULTS + "experiment_total_stats_per_country-"+subcat_type+".html")
per_subcat = df_experiment_stats.T.describe().T[['count','mean','std','min','max']].rename(
    columns={'count':'num_subcats', 'mean':'avg_num_measurements_per_country'})
per_subcat.to_html(RESULTS + "experiment_total_stats_per_subcat-"+subcat_type+".html")


# In[112]:

all_results = list(df_country_case_subcat.columns)
print all_results


# In[113]:

censorship_ratio = defaultdict(int)
for key in ['ratio-case1', 'ratio-case3', 'ratio-censored', 'ratio-uncensored', 'total']:
    censorship_ratio[key] = df_country_case_subcat[key].unstack(0)
    censorship_ratio[key].to_html(RESULTS + 'censorship_'+key+'.html')


# In[114]:

#censorship_ratio['ratio-censored'].head()


# In[116]:

df_subcat_country = df_case_count.reset_index().groupby(['subcat', 'country']).sum()
df_subcat_country = df_subcat_country.rename( columns= {k:str(k) for k in df_subcat_country.columns} )
df_subcat_country['tot_err'] = df_subcat_country[['0','4']].sum(axis=1)
#df_censorship['total'] = df_censorship['total'] - df_censorship['total_err']
df_subcat_country = df_subcat_country[['1','3','2', 'tot_err', 'total']]
df_subcat_country['case1'] = df_subcat_country['1']/df_censorship['total']
df_subcat_country['case3'] = df_subcat_country['3']/df_censorship['total']
df_subcat_country['case2'] = df_subcat_country['2']/df_censorship['total']
df_subcat_country['error'] = df_subcat_country['tot_err']/df_censorship['total']


# In[117]:

df_censorship2 = df_subcat_country.reset_index()


# In[118]:

currFolder = RESULTS + "subcat-"+subcat_type+"/"
if not os.path.exists(currFolder):
    os.makedirs(currFolder)

for subcat in df_censorship2.subcat.unique():

    df_per_subcat = df_censorship2[ df_censorship2['subcat']==subcat ].sort('case2')

    #fig1, ax1 = plt.subplits(1, 1, figsize=(10,7))
    fig1, ax1 = plt.subplots(1,1, figsize=(24,6))
    df = df_per_subcat.set_index('country')[['case2', 'case1','case3']].rename({'case2':'no-packets-dropped',
                                                                  'case1':'server-to-client-dropped',
                                                                  'case3':'client-to-server-dropped'})
    df.plot(kind='bar', stacked=True, ax=ax1)
    ax1.legend(loc='best', prop={'size':14})
    fig1.tight_layout()
    fig1.savefig(currFolder + "cases_ratio_by_country_stacked_bar-"+subcat)
    plt.close()

    df_per_subcat.to_html(currFolder + "cases_ratio_by_country_stacked_bar-"+subcat+".html")


# # PCA/TSNE

# In[119]:

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


# In[120]:

tsne = TSNE(n_components=2, random_state=0)
pca = PCA(n_components=2)


# In[121]:

df_censorship2.subcat.unique()


# In[122]:

df_filtered = df_censorship2[['subcat','country','case1','case2','case3']].copy()
df_filtered['case1/case13'] = df_filtered['case1']/( df_filtered['case1'] + df_filtered['case3'] )
df_filtered['case1/case13'] = df_filtered['case1/case13'].fillna(0)
df_filtered['case13'] = df_filtered['case1'] + df_filtered['case3']


# In[123]:

df_multidim = df_filtered.pivot(index='country', columns='subcat')


# In[124]:

# TSNE
model='tsne'
mat = df_multidim.as_matrix()
df4 = pd.DataFrame(tsne.fit_transform(mat)).set_index(df_multidim.index)
fig1, ax1 = plt.subplots(1,1, figsize=(10,10))
ax1.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax1.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax1.grid(1)
ax1.set_title("TSNE for server-to-client, client-to-server, and no-blocking for all subcats")
fig1.savefig(RESULTS + model+"-countries_all_cols-"+subcat_type)
#fig1.show()
plt.close()
df4.to_html(RESULTS + model+"-countries_all_cols-"+subcat_type+".html")


# PCA
model = 'pca'
mat = df_multidim.as_matrix()
df4 = pd.DataFrame(pca.fit_transform(mat)).set_index(df_multidim.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title("PCA for server-to-client, client-to-server, and no-blocking for all subcats")
fig2.savefig(RESULTS + model + "-countries_all_cols-"+subcat_type)
#fig2.show()
plt.close()
df4.to_html(RESULTS + model+"-countries_all_cols-"+subcat_type+".html")


# # GLOBAL SCATTER

# In[125]:

from itertools import cycle
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

col=cycle(tableau20)
mark=cycle(['o','*','v','d','s','^'])

fig2, ax2 = plt.subplots(1,1, figsize=(7,7))

#subcat = 'adult'
for subcat in df_censorship2.subcat.unique():
    df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case2', 'case3','case1/case13']]
    if subcat == 'kids_and_teen':
        subcat = 'kids'
    model = 'scatter'

    ax2.scatter((df4['case1']+df4['case3']), df4['case1/case13'], c=col.next(), marker=mark.next(),
                s=df4['case2']*50, label=subcat, alpha=0.5, edgecolor=None)
    #for label, x, y in zip(df4.index, df4['case1'], df4['case3']):
    #    ax2.annotate(
    #        label,
    #        xy = (x, y), xytext = (-20, 20),
    #        textcoords = 'offset points', ha = 'right', va = 'bottom',
    #        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.set_title(model + ' ' + subcat + ' case: 1+3 vs 1/1+3')
ax2.set_ylabel('server-to-client/total-tcp-ip-censorship')
ax2.set_xlabel('total-censorship-ratio')
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.grid(1)
ax2.legend(loc='best')
fig2.savefig(RESULTS + model + '-global-density-by-subcat-ratios')
plt.close()
df4.to_html(RESULTS + model + '-global-density-by-subcat-ratios.html')


# # SCATTER/PCA/TSNE PER CATEGORY

# In[126]:

#subcat = 'adult'
#df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case3', 'case2', 'case1/case13']]


# In[ ]:

#subcat = 'adult'
for subcat in df_censorship2.subcat.unique():
    df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case3', 'case2', 'case1/case13']]
    if subcat == 'kids_and_teen':
        subcat = 'kids'
    model = 'scatter'
    fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
    ax2.scatter((df4['case1']+df4['case3']),  df4['case1/case13'])
    for label, x, y in zip(df4.index, (df4['case1']+df4['case3']), df4['case1/case13']):
        ax2.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax2.set_ylabel('server-to-client/total-tcp-ip-censorship')
    ax2.set_xlabel('total-censorship-ratio')
    ax2.set_title(model + " for server-to-client/total-censorship ratio and no-blocking for subcat "+subcat)
    ax2.grid(1)
    fig2.savefig(RESULTS + "subcat-"+subcat_type+"/" + model + '-' + subcat)
    plt.close()
    df4.to_html(RESULTS + "subcat-"+subcat_type+"/" + model + '-' + subcat + ".html")

    # try PCA and TSNE - should be same result
    df3 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case2', 'case1/case13']]
    mat = df3.as_matrix()

    # TSNE
    model = 'tsne'
    df4 = pd.DataFrame(tsne.fit_transform(mat)).set_index(df3.index)
    fig1, ax1 = plt.subplots(1,1, figsize=(10,10))
    ax1.scatter(df4[0], df4[1])
    for label, x, y in zip(df4.index, df4[0], df4[1]):
        ax1.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax1.grid(1)
    ax1.set_title(model + " for server-to-client/total-censorship ratio and no-blocking for subcat "+subcat)
    fig1.savefig(RESULTS + "subcat-"+subcat_type+"/" + model + '-' + subcat)
    #fig1.show()
    plt.close()
    df4.to_html(RESULTS + "subcat-"+subcat_type+"/" + model + '-' + subcat + ".html")

    # PCA
    model = 'pca'
    df4 = pd.DataFrame(pca.fit_transform(mat)).set_index(df3.index)
    fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
    ax2.scatter(df4[0], df4[1])
    for label, x, y in zip(df4.index, df4[0], df4[1]):
        ax2.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax2.grid(1)
    ax2.set_title(model + " for server-to-client/total-censorship ratio and no-blocking  for subcat "+subcat)
    fig2.savefig(RESULTS + "subcat-"+subcat_type+"/" + model + '-' + subcat)
    #fig2.show()
    plt.close()
    df4.to_html(RESULTS + "subcat-"+subcat_type+"/" + model + '-' + subcat + ".html")


# # SCATTER GLOBAL

# In[ ]:

df_censorship = df_case_count.reset_index().groupby('subcat').sum()
df_censorship = df_censorship.rename( columns= {k:str(k) for k in df_censorship.columns} )
df_censorship['tot_err'] = df_censorship[['0','4']].sum(axis=1)
#df_censorship['total'] = df_censorship['total'] - df_censorship['total_err']
df_censorship = df_censorship[['1','3','2','tot_err','total']]

df_censorship['case1'] = df_censorship['1']/df_censorship['total']
df_censorship['case3'] = df_censorship['3']/df_censorship['total']
df_censorship['case2'] = df_censorship['2']/df_censorship['total']
df_censorship['error'] = df_censorship['tot_err']/df_censorship['total']


censorship_ratio = defaultdict(int)
censorship_ratio['case1'] = df_censorship['case1'].to_dict()
censorship_ratio['case2'] = df_censorship['case2'].to_dict()
censorship_ratio['case3'] = df_censorship['case3'].to_dict()
censorship_ratio['error'] = df_censorship['error'].to_dict()

df_censorship.sort('case2', inplace=True)
#df_censorship


# In[ ]:

subcat = 'global'
df4 = df_censorship.copy()
df4['case1/case13'] = df4['case1']/(df4['case1']+df4['case3'])
model = 'scatter'
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter((df4['case1']+df4['case3']),  df4['case1/case13'])
for label, x, y in zip(df4.index, (df4['case1']+df4['case3']), df4['case1/case13']):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

ax2.set_ylabel('server-to-client/total-tcp-ip-censorship')
ax2.set_xlabel('total-censorship-ratio')
ax2.set_title(model + ' ' + subcat + ' case 1+3 vs case1/1+3')
ax2.grid(1)
fig2.savefig(RESULTS + model + '-' + subcat)
plt.close()
df4.to_html(RESULTS + model + '-' + subcat + '.html')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Sanitize and Unbiasing
# - see how many are case 0 and 4
# - select best case for those with multiple case entries per sIP

# In[10]:

print "sIP, country with discrepancy: ", len( df_per_sIP_country[df_per_sIP_country['unique_count'] > 1] ) / len (df_per_sIP_country)


# #### choose first valid index of each sip,country pair

# In[11]:

def choose_valid_case(case_list):
    """ return first case that was either 1, 2, 3"""
    for x in case_list:
        if (x == 1) or (x == 2) or (x == 3):
            return x
    return case_list[0]


# In[12]:

df_unique = gp_good_sIP_country['case'].apply(choose_valid_case).reset_index()
# drop duplicate rows with same sIP, country, case
df_unique = df_unique.drop_duplicates()
print len( df_unique.sIP.unique() )
print len( df_unique.country.unique() )
print len( df_unique )


# In[13]:

# Quick check
df_temp = df_unique.groupby(['sIP', 'country'])['case'].unique().reset_index()
df_temp['count'] = df_unique.groupby(['sIP', 'country'])['case'].count().reset_index()['case']
df_temp['unique_count'] = df_per_sIP_country['case'].apply(lambda x: len(x))
df_temp[df_temp['count']>1]


# ## Merge unique selected entries with subcats

# In[23]:

df_unbiased = df_unique.merge(df_sIP_subcat_disjoint, on=['sIP'])
df_unbiased.subcat.unique()


# In[24]:

# Quick check
df_unbiased[df_unbiased['domain']=='history.com'].head()


# ## Data description
# - total experiments: means, var per country
# - total experiments: means, var per subcat

# In[36]:

# save list of countries
pd.Series(df_unbiased.reset_index()['country'].unique()).to_csv(RESULTS+"list_of_countries.csv")


# In[37]:

df_experiment_stats = df_unbiased.groupby(['country', 'subcat']).count()['sIP'].unstack()


# In[38]:

per_country = df_experiment_stats.describe().T[['count','mean','std','min','max']].rename(
    columns={'count':'num_countries', 'mean':'avg_num_measurements_per_subcat'})
per_country.to_html(RESULTS + "experiment_total_stats_per_country.html")
per_subcat = df_experiment_stats.T.describe().T[['count','mean','std','min','max']].rename(
    columns={'count':'num_subcats', 'mean':'avg_num_measurements_per_country'})
per_subcat.to_html(RESULTS + "experiment_total_stats_per_subcat.html")


# ### Save domain/country/case

# In[135]:

df_unbiased.to_pickle(RESULTS + 'domain_censorship_results_repeated-categ.pkl')
df_unbiased.to_csv(RESULTS + 'domain_censorship_results_repeated-categ.csv')
df_unbiased.head(10)


# <hr/>
# # BELOW THIS IS ANALYSIS
# <hr/>

# ## CENSORSHIP RATIOS
# - count country, case, subcat

# In[54]:

df_country_case_subcat = df_unbiased.groupby(['country', 'case', 'subcat'])['sIP'].count().unstack(1).fillna(0)
df_country_case_subcat.head(10)


# ## ADD relevant ratio columns

# In[78]:

df_country_case_subcat = df_unbiased.groupby(['country', 'case', 'subcat'])['sIP'].count().unstack(1).fillna(0)
df_country_case_subcat = df_country_case_subcat.rename(columns={1:'1', 2:'2', 3:'3'})
df_country_case_subcat['total'] = df_country_case_subcat[['1','2','3']].sum(axis=1)
df_country_case_subcat['case1'] = df_country_case_subcat['1']/df_country_case_subcat['total']
df_country_case_subcat['case2'] = df_country_case_subcat['2']/df_country_case_subcat['total']
df_country_case_subcat['case3'] = df_country_case_subcat['3']/df_country_case_subcat['total']
df_country_case_subcat['case13'] = (df_country_case_subcat['1'] + df_country_case_subcat['3'])/df_country_case_subcat['total']
df_country_case_subcat['case1/case13'] = df_country_case_subcat['1']/(df_country_case_subcat['1'] + df_country_case_subcat['3'])
df_country_case_subcat.head()


# In[64]:

all_results = list(df_country_case_subcat.columns)
print all_results


# In[79]:

censorship_ratio = defaultdict(int)
for key in ['case1', 'case3', 'case13', 'case2', 'total']:
    censorship_ratio[key] = df_country_case_subcat[key].unstack(0)
    censorship_ratio[key].to_html(RESULTS + 'ALL_Apr20_censorship_'+key+'_updated.html')


# In[82]:

# check
df_country_case_subcat[['case1','case2','case3']].sum(axis=1)


# # BY SUBCAT per COUNTRY and vice versa
# - draw above plots for each country
# - each subplot

# In[132]:

df_censorship = df_country_case_subcat.reset_index()
df_per_subcat = df_censorship[ df_censorship['subcat']==subcat ].sort('case2')
df_censorship.to_html(RESULTS + 'all_case_ratios.html')
df_censorship.to_pickle(RESULTS + 'all_case_ratios.pkl')
# check
len(df_per_subcat.set_index('country'))


# In[94]:

currFolder = RESULTS + "subcat/"
if not os.path.exists(currFolder):
    os.makedirs(currFolder)

for subcat in df_censorship.subcat.unique():

    df_per_subcat = df_censorship[ df_censorship['subcat']==subcat ].sort('case2')

    #fig1, ax1 = plt.subplits(1, 1, figsize=(10,7))
    fig1, ax1 = plt.subplots(1,1, figsize=(20,6))
    df = df_per_subcat.set_index('country')[['case2', 'case1','case3']].rename({'case2':'no-packets-dropped',
                                                                  'case1':'server-to-client-dropped',
                                                                  'case3':'client-to-server-dropped'})
    df.plot(kind='bar', stacked=True, ax=ax1)
    ax1.legend(loc='best', prop={'size':14})
    fig1.tight_layout()
    fig1.savefig(currFolder + "cases_ratio_by_country_stacked_bar_"+subcat)
    plt.close()


# # PCA

# In[95]:

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


# In[96]:

tsne = TSNE(n_components=2, random_state=0)
pca = PCA(n_components=2)


# In[97]:

df_censorship.subcat.unique()


# In[99]:

df_censorship.head()


# In[100]:

df_filtered = df_censorship[['subcat','country','case1','case2','case3', 'case13', 'case1/case13']]
#df_filtered['case1/case13'] = df_filtered['case1']/( df_filtered['case1'] + df_filtered['case3'] )
#df_filtered2 = df_filtered[['subcat','country','case1','case2','case3']]
#df_filtered3 = df_filtered[['subcat','country','case1/case13','case2']]


# In[105]:

df_multidim = df_filtered.pivot(index='country', columns='subcat')
df_multidim2 = df_filtered[['subcat','country','case1','case2','case3']].pivot(index='country', columns='subcat')
df_multidim3 = df_filtered[['subcat','country','case13','case1/case13']].pivot(index='country', columns='subcat')


# In[108]:

mat = df_multidim2.as_matrix()
df4 = pd.DataFrame(tsne.fit_transform(mat)).set_index(df_multidim.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title("TSNE for server-to-client, client-to-server, and no-blocking for all subcats")
fig2.savefig(currFolder + "tsne_countries_all_cols")
fig2.show()

mat = df_multidim2.as_matrix()
df4 = pd.DataFrame(pca.fit_transform(mat)).set_index(df_multidim.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title("PCA for server-to-client, client-to-server, and no-blocking for all subcats")
fig2.savefig(currFolder + "pca_countries_all_cols")
#fig2.show()
#plt.close()


# ### This doesn't work due to nans

# In[109]:

mat = df_multidim3.as_matrix()
df4 = pd.DataFrame(tsne.fit_transform(mat)).set_index(df_multidim.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title("TSNE for ratio(server-to-client/total), and no-blocking for all subcats")
fig2.savefig(currFolder + "tsne_countries_ratios")
fig2.show()

mat = df_multidim3.as_matrix()
df4 = pd.DataFrame(pca.fit_transform(mat)).set_index(df_multidim.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title("PCA for ratio(server-to-client/total), and no-blocking  for all subcats")
fig2.savefig(currFolder + "pca_countries_ratios")
fig2.show()


# In[110]:

#subcat = 'adult'

for subcat in df_censorship.subcat.unique():
    df = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case2','case3']]
    mat = df.as_matrix()
    df4 = pd.DataFrame(pca.fit_transform(mat)).set_index(df.index)

    model = 'pca'
    fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
    ax2.scatter(df4[0], df4[1])
    for label, x, y in zip(df4.index, df4[0], df4[1]):
        ax2.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax2.set_title(model + ' ' + subcat + ' case: 1,2,3')
    ax2.grid(1)
    fig2.savefig(RESULTS + "subcat/" + model + '-' + subcat)

    model = 'tsne'
    fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
    ax2.scatter(df4[0], df4[1])
    for label, x, y in zip(df4.index, df4[0], df4[1]):
        ax2.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax2.set_title(model + ' ' + subcat + ' case: 1,2,3')
    ax2.grid(1)
    fig2.savefig(RESULTS + "subcat/" + model + '-' + subcat)

    plt.close()
#df_multidim = df.pivot(index='country', columns='subcat')


# In[111]:

df_censorship.head()


# In[112]:

df_global = df_censorship.groupby('subcat')[['1','2','3', 'total']].sum()
df_global['1'] = df_global['1']/df_global['total']
df_global['2'] = df_global['2']/df_global['total']
df_global['3'] = df_global['3']/df_global['total']
df_global = df_global[['1','2','3']]


# ## GLOBAL - ignore country

# In[113]:

mat = df_global.as_matrix()

model = 'pca'
df4 = pd.DataFrame(pca.fit_transform(mat)).set_index(df_global.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title(model + '-global-subcat')
fig2.savefig(RESULTS + model + '-global-subcat')

model = 'tsne'
df4 = pd.DataFrame(tsne.fit_transform(mat)).set_index(df_global.index)
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
ax2.scatter(df4[0], df4[1])
for label, x, y in zip(df4.index, df4[0], df4[1]):
    ax2.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.grid(1)
ax2.set_title(model + '-global-subcat')
fig2.savefig(RESULTS + model + '-global-subcat')


# # SCATTER PLOTS PER SUBCAT

# In[115]:

#subcat = 'adult'
for subcat in df_censorship.subcat.unique():
    df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case3']]
    if subcat == 'kids_and_teen':
        subcat = 'kids'
    model = 'scatter'
    fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
    ax2.scatter(df4['case1'], df4['case3'])
    for label, x, y in zip(df4.index, df4['case1'], df4['case3']):
        ax2.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax2.set_title(model + ' ' + subcat + ' TCP/IP blocking case 1 vs case 3')
    ax2.grid(1)
    ax2.set_xlabel('server-to-client-drop') #case1
    ax2.set_ylabel('client-to-server-drop') #case3
    fig2.savefig(RESULTS + "subcat/" + model + '-' + subcat)
    plt.close()


# In[116]:

from itertools import cycle
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

col=cycle(tableau20)
mark=cycle(['o','*','v','d','s','^'])

fig2, ax2 = plt.subplots(1,1, figsize=(7,7))

#subcat = 'adult'
for subcat in df_censorship.subcat.unique():
    df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case3', 'case2']]
    if subcat == 'kids_and_teen':
        subcat = 'kids'
    model = 'scatter'

    ax2.scatter(df4['case1'], df4['case3'], c=col.next(), marker=mark.next(),
                s=df4['case2']*50, label=subcat, alpha=0.5, edgecolor=None)
    #for label, x, y in zip(df4.index, df4['case1'], df4['case3']):
    #    ax2.annotate(
    #        label,
    #        xy = (x, y), xytext = (-20, 20),
    #        textcoords = 'offset points', ha = 'right', va = 'bottom',
    #        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.set_title(model + ' ' + subcat + ' TCP/IP blocking case 1 vs case 3')
ax2.set_xlabel('server-to-client-drop') #case1
ax2.set_ylabel('client-to-server-drop') #case3
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.grid(1)
ax2.legend(loc='best')
fig2.savefig(RESULTS + model + '-global-density-by-subcat')


# In[119]:

#df_filtered['case1/case13'] = df_filtered['case1']/(df_filtered['case3']+df_filtered['case1'])
df_filtered.head()


# In[124]:

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

col=cycle(tableau20)
mark=cycle(['o','*','v','d','s','^'])

fig2, ax2 = plt.subplots(1,1, figsize=(10,10))

#subcat = 'adult'
for subcat in df_censorship.subcat.unique():
    df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case13','case2','case1/case13']]
    if subcat == 'kids_and_teen':
        subcat = 'kids'
    model = 'scatter'

    ax2.scatter(df4['case13'], df4['case1/case13'], c=col.next(), marker=mark.next(),
                s=df4['case2']*50, label=subcat, alpha=0.5, edgecolor=None)
    #for label, x, y in zip(df4.index, df4['case1'], df4['case3']):
    #    ax2.annotate(
    #        label,
    #        xy = (x, y), xytext = (-20, 20),
    #        textcoords = 'offset points', ha = 'right', va = 'bottom',
    #        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.set_title(model + ' ' + subcat )
ax2.set_ylabel('server-to-client/total-tcp-ip-censorship')+ ' case(1+3) vs case1/case(1+3)'
ax2.set_xlabel('total-tcp-ip-censorship')
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.grid(1)
ax2.legend(loc='best', prop={'size':12})
fig2.savefig(RESULTS + model + '-global-density-by-subcat-ratios')


# In[127]:

from itertools import cycle
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

col=cycle(tableau20)
mark=cycle(['o','*','v','d','s','^'])

fig2, ax2 = plt.subplots(1,1, figsize=(7,7))

#subcat = 'adult'
for subcat in df_censorship.subcat.unique():
    df4 = df_filtered[df_filtered['subcat']==subcat].set_index('country')[['case1','case3', 'case2']]
    if subcat == 'kids_and_teen':
        subcat = 'kids'
    model = 'scatter'

    ax2.scatter(df4['case1'], df4['case3'], c=col.next(), marker=mark.next(),
                s=df4['case2']*50, label=subcat, alpha=0.5, edgecolor=None)
    #for label, x, y in zip(df4.index, df4['case1'], df4['case3']):
    #    ax2.annotate(
    #        label,
    #        xy = (x, y), xytext = (-20, 20),
    #        textcoords = 'offset points', ha = 'right', va = 'bottom',
    #        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax2.set_title(model + ' ' + subcat + ' case1 vs case3')

ax2.set_xlabel('server-to-client-dropped')
ax2.set_ylabel('client-to-server-dropped')
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.grid(1)
ax2.legend(loc='best', prop={'size':12})
fig2.savefig(RESULTS + model + '-global-density-by-subcat')


# # COSINE SIMILARTY - TODO

# In[131]:

from scipy.spatial.distance import cosine
df_censorship.head()
#df_multidim = df_filtered.pivot(index='case', columns='subcat')
#df_filtered.pivot()
#df_multidim.corr()

