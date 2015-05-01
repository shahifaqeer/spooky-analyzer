from __future__ import division
import pandas as pd
import subprocess, os
import httplib
import numpy as np
import json
from netaddr import IPNetwork, IPAddress
from collections import defaultdict

SAMPLENAME = "sIP_sanitize_20150426/"
#SAMPLENAME = "validate_20150422"
DATAPATH = "data/sIP_sanitize/"
DATAPATH2 = "results/validated_20150422/"

RESULTS = "results/" + SAMPLENAME
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


# # SIP and DOMAINS to REJECT

# Load sIP:domain:subcat list
df_sIP_info = pd.read_csv(DATAPATH + "Servers_IMC.txt")
df_sIP_info['subcat'] = df_sIP_subcat['subcat'].apply(lambda x:x.split("|"))
print len(df_sIP_info)
df_sIP_info.head()

# # Checks: IP-reachability, domain-IP-relevance, high-censorship
# - maintain an sIP_info table using df_sIP_subcat
# - add IP-reachability, domain-IP-relevance, and censorship info to it
#
# ## IP-reachability: Spooky-scan fails on anycast
# - ANYCAST_DESC: does IP belong to Anycast list: Cloudflare, amazon, etc
# - REVERSE_DNS: sIP reverse DNS using dig @8.8.8.8 contains aws etc.
# - SLASH24, SHARED_SLASH24: IPs that share a slash24 with others on the list: suspected CDNs
#
# ## domain-IP-relevance: sIP to Domain mapping
# - DNS_LIST, DNS_CONTAINS_IPADDR: DNS query for domain: contains sIP?
# - HTTP_DOMAIN, HTTP_IP, HTTP_IP_CONTAINS_DOMAIN, HTTP_DIFF: http download for domain works (check1) and is the same as http download for sIP (check2)
#
# ## high-censorship: Censorship ratios for US and SE
# - type1
# - type2
# - type3
# - type4

#################################################################################################################
# #1. IP-reachability
################################################################################################################

# - Spooky scan fails to detect censorship at gIP if the sIP itself is protecting its IP layer
#
# - sIPs can use firewalls that don't allow measurements without knowing the host
#     - for example popular websites like reddit are hosted on cloudflare
#     - although we can connect (telnet port 80) to them, they do not allow direct access
#
# - sIPs can be anycasted (for example popular CDNs) and refer to different machine
#     - similar to firewall issue, a proxy machine may decide where to route the connection based on host info
#
# - these websites are usually not reacheable directly via IP, but are valid websites
# - if we do not weed them out, they will appear as censorship. In reality they are limitations of spooky-scan that are server-side unreacheable (unmeasureable?) due to firewalls or anycast. (they are usually CDNs).
#
# #### Weeding out sIPs spooky scan cannot measure:
# - anycast sIPs: unfortunately detecting anycast is a huge research problem not tackled by us
#     - we download list of popular anycasting CDNs
#     - we find their IP prefixes using BGP updates
#     - we find sIPs that may belong to anycast
# - reverse DNS: may show akamai or aws, which are also popular anycast
# - slash24: many google sIPs seem to belong to the same /24 prefix. In case something is not in the anycast list, we find sIPs with common /24

# #### Create a copy frame to add IP-reachability info to

sIP_info = df_sIP_info.copy()

# #### Load Anycast

# Load ANYCASTERS
df_anycast_temp = pd.read_csv(DATAPATH + "anycasters.txt", names=["ip_prefix", "description"], delimiter='\t')

with open(DATAPATH + 'anycast-ip-range_amazon.json') as data_file:
    data = json.load(data_file)
amazon = pd.DataFrame(data['prefixes'])
amazon['description'] = amazon['description'] = amazon['region']+"|"+amazon['service']

df_anycast = pd.concat([df_anycast_temp, amazon[['ip_prefix', 'description']]])
del df_anycast_temp, amazon

df_anycast.to_csv(DATAPATH + "anycast_complete.csv")

# #### Store anycast if any sIP belongs to the prefixes

def check_ip_prefix(sIP):
    for prefix in list( df_anycast['ip_prefix'] ):
        if IPAddress(sIP) in IPNetwork(prefix):
            return df_anycast[df_anycast['ip_prefix'] == prefix]['description'].values[0]
    return -1

sIP_info['anycast'] = sIP_info['sIP'].apply(check_ip_prefix)

# #### Get reverse DNS and store in dig_x column

def get_revDNS(ipaddr):
    try:
        return subprocess.check_output("dig -x "+ipaddr+" @8.8.8.8 +short", shell=True).split()
    except:
        return -1

sIP_info['dig_x'] = sIP_info['sIP'].apply(get_revDNS)

# #### sIPs with shared /24 within the sIP list

sIP_info['slash24'] = sIP_info['sIP'].apply(lambda x:x[:x.rfind(".")])
df_IP_reachability = sIP_info.merge( sIP_info.groupby('slash24')['sIP'].count().reset_index().rename(
        columns={'sIP':'slash24_shared'}), on='slash24')


# #### Store this data frame regarding IP reachability

df_IP_reachability.to_pickle(RESULTS + "df_IP_reachability.pkl")
df_IP_reachability.head()

##############################################################################################################
# #2. Domain-IP relevance
##############################################################################################################

# #### Use original df_sIP_info without reachability data

sIP_info = df_sIP_info.copy()

# #### DNS Query for domain: does it contain the IPAddr?


def get_DNS(domain):
    try:
        dig = subprocess.check_output("dig @8.8.8.8 "+domain+" +short", shell=True).split()
    except:
        dig = -1
    return dig


sIP_info['dig'] = sIP_info['domain'].apply(get_DNS)

sIP_info['DNS_contains'] = sIP_info.apply(lambda row: row['sIP'] in row['dig'], axis=1)

# #### HTTP Check domain
# - http connect to domain
#     - store the code (should be 200)
#     - if error code try www
#     - if error code try https
#     - whoever gives no error, store in results/curl/QUERY.html using a curl subprocess

def get_code(domain):
    try:
        req =  urllib2.urlopen("http://"+domain+"/", timeout=2)
        return req.getcode()
    except urllib2.HTTPError, err:
        return err.code
    except Exception,err:
        if 'code' in err:
            return err.code
        else:
            # try 10 times
            for i in range(10):
                try:
                    req =  urllib2.urlopen("http://"+domain+"/", timeout=16)
                    code = req.getcode()
                    print domain, code, "loop: ", i
                    return code
                except:
                    pass
            print domain, err, "final"
            return -1

# httplib is actually unable to get many 200 codes
http_errors = defaultdict(list)

def get_http_code(domain):
    try:
        # this catches relocation
        conn = httplib.HTTPConnection(domain)
        conn.request("HEAD", "")
        res = conn.getresponse()
        if res.status !=200:
            print domain, res.status, res.reason

            http_errors['domain'].append(domain)
            http_errors['status'].append(res.status)
            http_errors['reason'].append(res.reason)
            http_errors['header'].append(res.getheaders())

        return res.status
    except Exception, e:
        print domain, e
        return -1

sIP_info['http_domain'] = sIP_info['domain'].apply(get_code)

# Do not use this it really sucks, even though it catches redirections its a shit library
# sIP_info['http_domain2'] = sIP_info['domain'].apply(get_http_code)


# #### Apart from torDir, there shouldn't be any codes other than 200 here
# - if code other than 200 exists, try the domain with httplib: this gives relocations (www or https or a new relocated home page)
# - manual check shows that www works for most missing domains, and https for vpn domains.

retry_domain = sIP_info[ (sIP_info['http_domain']!=200) & (sIP_info['domain']!='torDir')]
print len(retry_domain)
retry_domain.head()


sIP_info['http_domain2'] = retry_domain['domain'].apply(get_http_code)

# store the http headers showing redirects
df_http_error_ipaddr = pd.DataFrame(http_errors)
df_http_error_domain.to_pickle(RESULTS + "df_http_error_domain.pkl")

# #### Domains that have not relocated (with code 301/302) don't exist?
# - 301 or 302 means the real domain is www or https (it exists)
# - on checking 503 (server unavailable) also exist (maybe at www) and can be downloaded with curl
# - 403 are forbidden to access with curl or urllib, these probably need host info for access and only work with a browser
# - 416 is also a protocol range error
# - 405 method not allowed

sIP_info['http_domain2'] = retry_domain['domain'].apply(lambda x: get_http_code('www.'+x))


#sIP_info[ (sIP_info['http_domain']!=200) & (~sIP_info['http_domain2'].isin([200, 301, 302, 503, 403, 416, 405]))
#         & (sIP_info['domain']!='torDir')]


# #### Bad Domains to be culled (also rechecked manually by browser)
# - fortawesome.github.io: doesn't exist anymore
#
# #### good domains:
# - 302, 301 in http_domain2
# - 503, 416, 403, 500 exist but are weird and firewalled
# - tomshardware.com: can't be reached (gets a suspicious activity message from distilnetworks) but reacheable by www. on a browser
# - justice.gov.za: 403 forbidden, but reacheable by www. and /home
# - sun.ac.za: 403 forbidden, but reacheable by www. and /home
# - planetminecraft.com: redirects to www.planetminecraftcom. Very popular but unreacheable by urllib, httplib, curl,
# - greatfire, ironsocket seem to be reacheable over https but not using curl http etc.

# #### HTTP Check ipaddr
# - http connect to ipaddr
#     - store the error code (should be 200)
#     - if 'domain' occurs in valid (200) web page, return 200
#     - if webpage is valid (200) but domain doesn't occur, return 0
#         - these are usua;lly ngix services, apache etc, with open ports
#     - if return code is not 200, then return the error code or -1 |for timeouts
#     - also store all error info in df_http_error_ipaddr

http_error_ipaddr = defaultdict(list)
def get_http_ipaddr(ipaddr, domain):
    try:
        req =  urllib2.urlopen("http://"+ipaddr+"/", timeout=2)
        code = req.getcode()
        readreq = req.read()
        if (code==200):
            http_error_ipaddr['html'].append(readreq)
        else:
            http_error_ipaddr['html'].append('')

        if (code==200) and not (domain in readreq):
            #print "domain not in page 200:", ipaddr, domain, code
            code = 0
    except urllib2.HTTPError, err:
        code = err.code
        http_error_ipaddr['html'].append(str(err))
    except Exception, e:
        print ipaddr, domain, e
        http_error_ipaddr['html'].append(str(e))
        code = -1

    http_error_ipaddr['domain'].append(domain)
    http_error_ipaddr['sIP'].append(ipaddr)
    http_error_ipaddr['code'].append(code)

    return code

# check
ipaddr = '171.67.215.200'
domain = 'stanford.edu'
print get_http_ipaddr(ipaddr, domain)
row = sIP_info.iloc[10]
print get_http_ipaddr(row['sIP'], row['domain'])


sIP_info['http_ipaddr'] = sIP_info.apply(lambda row: get_http_ipaddr(row['sIP'], row['domain']), axis=1)


# #### Save errors and all current domain-ip mapping info

df_http_error_ipaddr = pd.DataFrame(http_error_ipaddr)
df_http_error_ipaddr.to_pickle(RESULTS + "df_http_error_ipaddr.pkl")


df_domain_IP_mapping = sIP_info.copy()
df_domain_IP_mapping['sIP_http_reach'] = (df_domain_IP_mapping['http_ipaddr']==200)
df_domain_IP_mapping.to_pickle(RESULTS + "df_domain_IP_mapping.pkl")


# #### Static IPs directly addressible: Technique should work for these
# - even if DNS response does not contain the IP, it is still directly reachable

static_ips = sIP_info[(sIP_info['http_ipaddr']==200)]
print len(static_ips)

# #### IPs not reacheable on HTTP, but their domain gives the same error code
#  - some of these were found later using www. or redirection (their http_domain2 is 200, 301, 302) but the IP couldn't find them

sIP_info[(sIP_info['http_ipaddr']!=200) & (sIP_info['http_ipaddr']==sIP_info['http_domain'])]

#########################################################################################################################
# #3. Ground Truth: high censorship ratio check
########################################################################################################################
# - load original data
# - get censorship by country for each site
# - add total, censorship ratio, 1, 3 to sIP_info table

# #### Load original data (contains only 540 sIPs not 580)


df_all = pd.read_pickle("data/case_detected_all_20150412.pkl").reset_index()
df_all2 = pd.read_pickle("data/case_detected_all_20150422.pkl").reset_index()
df_full = pd.concat([df_all, df_all2])


# #### Get ASNs info for gIPs for queries later

## for extra details about gIP, get the ASNs
name_row = ['ASN', 'gIP', 'IPnet', 'country', 'agency', 'date', 'name']
df_ASN_gIP = pd.read_csv(DATAPATH2 + "all_asn_list.csv", skiprows=1, delimiter='|', skipinitialspace=True,
                      names=name_row,
                      converters={y : lambda x: x.strip() for y in name_row})
df_val = df_full.merge(df_ASN_gIP[['ASN', 'gIP']], on='gIP')
df_val = df_val[['sIP', 'domain', 'subcat', 'gIP', 'ASN', 'country', 'case', 'port']]

del df_ASN_gIP
df_val.head()

# #### Helper functions to get censorship by country, and global censorship

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

    df_count['tot'] = df_count['tot'] - df_count['err']
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

# #### Get censorship by country for full dataset (including april 12: global will be wrong due to bias)

censorship = get_censorship_by_country_sIP(df_val)[['US','SE']]
totals = get_censorship_by_country_sIP(df_val, 'tot')[['US','SE']]
case1 = get_censorship_by_country_sIP(df_val, 1)[['US','SE']]
case3 = get_censorship_by_country_sIP(df_val, 3)[['US','SE']]

censorship['US-tot'] = totals['US']
censorship['SE-tot'] = totals['SE']
censorship['US-1'] = case1['US']
censorship['US-3'] = case3['US']
censorship['SE-1'] = case1['SE']
censorship['SE-3'] = case3['SE']

del totals, case1, case3, df_val

censorship = censorship.reset_index()
censorship.head()

# #### Merge these ratios back with sIP_info table

sIP_info = df_sIP_info.copy()

df_censorship_US_SE = sIP_info.merge(censorship[['sIP', 'US', 'US-tot', 'US-1', 'US-3',
                                                 'SE', 'SE-tot', 'SE-1', 'SE-3']], on='sIP')

df_censorship_US_SE.to_pickle(RESULTS + "df_censorship_US_SE.pkl")
df_censorship_US_SE.head(1)

###########################################################################################################################
# # MERGE ALL TOGETHER
###########################################################################################################################
# - ip_reachability
# - domain-ip mapping
# - censorship ratios
#
# ### CAVEATS: sIP, domain are primary keys to merge on
# - don't use subcat: lists are unhashable so drop it in two frames and only keep it in the third
# - the actual data (df_censorship) contains only 540 entries, so use outer joins

print len(df_IP_reachability)
COLUMNS = {name: 'IP.'+name for name in df_IP_reachability.columns if not name in ['sIP', 'domain', 'subcat']}
df1 = df_IP_reachability.rename(columns = COLUMNS)
df1.head()

print len(df_domain_IP_mapping)
#del df_domain_IP_mapping['subcat']
COLUMNS = {name: 'DOMAIN.'+name for name in df_domain_IP_mapping.columns if not name in ['sIP', 'domain', 'subcat']}
df2 = df_domain_IP_mapping.rename(columns = COLUMNS)
#del df2['subcat']
df2.head()

print len(df_censorship_US_SE)
#del df_censorship_US_SE['subcat']
COLUMNS = {name: 'CENSOR.'+name for name in df_censorship_US_SE.columns if not name in ['sIP', 'domain', 'subcat']}
df3 = df_censorship_US_SE.rename(columns = COLUMNS)
#del df3['subcat']
df3.head()

# merge in order to preserve COLUMNS order
df_temp = df1.merge( df2, on=['sIP', 'domain'], how='outer' )
df_temp2 = df_temp.merge( df3, on=['sIP', 'domain'], how='outer' )
print df_temp2.columns

# #### Sort by US censorship (increasing) / US totals (increasing)

df_final = df_temp2.sort(columns=["CENSOR.US", "CENSOR.US-tot"], ascending=[False, False]).reset_index()
del df_final['index']
df_final.head()

# ## SAVE THIS TABLE TO PICKLE, CSV, HTML, AND EXCEL (GOOGLE DOC) FOR ANALYSIS

df_final.to_pickle(RESULTS + "df_sanitize_sIP.pkl")
df_final.to_csv(RESULTS + "df_sanitize_sIP.csv")
df_final.to_html(RESULTS + "df_sanitize_sIP.html")

# ### selected columns only

COLUMNS = [u'sIP', u'domain', u'subcat', u'IP.dig_x', u'IP.anycast',
           u'DOMAIN.DNS_contains', u'DOMAIN.http_domain', u'DOMAIN.http_domain2',
           u'DOMAIN.http_ipaddr',
           u'CENSOR.US', u'CENSOR.US-tot', u'CENSOR.US-1', u'CENSOR.US-3',
           u'CENSOR.SE', u'CENSOR.SE-tot', u'CENSOR.SE-1', u'CENSOR.SE-3']
df_short = df_final[COLUMNS].copy()

def display_list(list_of_anything):
    if len(list_of_anything) > 1:
        return "|".join(list_of_anything)
    elif len(list_of_anything) == 1:
        return list_of_anything[0]
    else:
        return ""
# convert lists to strings
df_short['IP.dig_x'] = df_short['IP.dig_x'].apply(display_list)
df_short['subcat'] = df_short['subcat'].apply(display_list)

df_short.to_html(RESULTS + "df_sanitize_sIP_short.html")
df_short.to_excel(RESULTS + "df_sanitize_sIP_short.xlsx")



#TODO
# # DATA ANALYSIS: Types US/SE censorship expected
#- previous plot shows a CDF knee at censorship = 0.2, basically those 21% or so of the sIPs must be cloudflare.
#- type1: highly censored in US/SE
#- type2: US blocked, SE unblocked
#- type3: US unblocked, SE blocked
#- type4: uncensored (static IPs?)
# ### Analysis of slices is in exploring....ipynb

print RESULTS
print df_short

###########################################################################################################################
# # OLD SIP LIST
##########################################################################################################################
# - 1. sIP not found in dig lookup error = 1: 119 such sIPs (including 4 tor directories that we ignore)
# - 2. censorship in US > 0.8
# ## Current logic:
# - 1. sIP not found in dig is a DOMAIN level check: DOMAIN.DNS_contains but not the only check
# - 2. censorship ratios need to be studied not discarded
# ## Real Checks
# - 1. ANYCASTED domains need to be seen, 2. do digs also not correspond, 3. is dig -x reverse DNS a cdn, 4. is censorship too much? IGNORE
# - 5. Does direct IP download on http not work? 6. Does direct http domain download work then? 7. Is the direct IP download = 0, i.e., it connects but is not the original website?

df_sIP_subcat= pd.read_csv(DATAPATH + "Servers_IMC.txt")

df_digs = pd.read_html('results/validated_20150422/' + 'updated_sIP_domain_to_check.html')[0]
del df_digs['Unnamed: 0']
reject1 = df_digs[df_digs['error']==1]['sIP'].unique()
reject2 = censorship[censorship['us'] > 0.80]['sIP'].unique()
print len(reject1)
print len(reject2)
print len(set(reject1 + reject2))
sIP_trash = reject1+reject2

# Reshuffle and chunk
valid_sIP = df_sIP_subcat[~df_sIP_subcat['sIP'].isin(sIP_trash)]
indices = list(valid_sIP.index)
random.shuffle(indices)
valid_sIP = valid_sIP.reindex(indices).set_index('sIP')
N = len(valid_sIP)

# save chunks to disk
chunk1 = valid_sIP.iloc[0:N//3]
chunk2 = valid_sIP.iloc[N//3:2*N//3]
chunk3 = valid_sIP.iloc[2*N//3:]
chunk1.to_csv(RESULTS + "alexa-ips-2015-04-24-T.txt", header=False)
chunk2.to_csv(RESULTS + "alexa-ips-2015-04-24-F.txt", header=False)
chunk3.to_csv(RESULTS + "alexa-ips-2015-04-24-S.txt", header=False)

# useful little function
# http check
def curl_http(anything, www=0):
    if www:
        query = "www."+anything
    else:
        query = anything
    # do curl on ip or domain
    try:
        curl = subprocess.check_output("curl http://"+query, shell=True)
        open("results/curl/"+query+".html", "w").write(curl)
        curl_head = subprocess.check_output("curl http://"+query+" --head", shell=True)
        open("results/curl/"+query+"_HEAD.html", "w").write(curl_head)
    except:
        curl = '-1'
    return curl
