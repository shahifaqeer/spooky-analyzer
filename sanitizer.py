import pandas as pd
import os, sys
from intervention import *
import GeoIP
from caseDetection import *
import logging
from collections import defaultdict


# Global filename and date
fname = "ALL_APR7"
fdate = "Apr7"

# Global Path for DATA (Unix convention only)
DATAPATH = "data/"
print "NOTE: MAKE SURE THAT ALL DATA FILES ARE IN " + DATAPATH
if not os.path.exists(DATAPATH):
    os.makedirs(DATAPATH)

##########################################################

# UTILITY FUNCTIONS

gi1 = GeoIP.open(DATAPATH + "GeoIPCity.dat",GeoIP.GEOIP_STANDARD)
logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
logger.setLevel(level)


def get_country(ip):
    ipInfo=gi1.record_by_addr(ip)
    if ipInfo is None:
        return "roya"
    return ipInfo["country_code"]

def split_int(y):
    x = y.split(",")
    if len(x)<=1:
        return None
    z = []
    for e in x:
        e = e.strip()
        if "None" in e:
            z.append(None)
        else:
            try:
                z.append(int(e))
            except:
                z.append(float(e))
    return z

def count_not_none(some_list):
    counter1 = 0
    for x in some_list:
        if not( x == None ):
            counter1+=1
    return counter1

############################################################


# MAIN FUNCTIONS

def load_and_remove_errors():
    """
    Load raw df;
    Add countries;
    remove null entries in ipids;
    Save
    """
    fname=DATAPATH + filename
    print "Find file in "+fname
    temp = "gIP,sIP,port,ipids,diff_list,ts,k1,k2,retransmit_times,NotGlobal,NotEnough,NotEnoughRetrans, date"
    header_row = temp.split(",")
    df_all = pd.read_csv(fname,delimiter="|", names=header_row)

    # Add countries
    df_all["country"] = df_all["gIP"].apply(lambda x: get_country(x))

    if not os.path.exists(DATAPATH + "sanitize"):
        os.makedirs(DATAPATH + "sanitize")

    # ipid nulls and nonnulls sets
    df_nulls = df_all[ df_all["ipids"].isnull() ]
    df = df_all[ df_all["ipids"].notnull() ]

    # save both sets in DATAPATH
    df_nulls[["sIP", "gIP", "country"]].to_csv("cond_er_redo")
    df_nulls.to_pickle(DATAPATH + "sanitize/null_ipid_cond_er_redo.pkl")
    df.to_csv(DATAPATH + "cond_pass_basic_"+fdate, sep="|")
    df.to_pickle( DATAPATH + "sanitize/phase1_ipid_nonnull_"+fdate+".pkl")

    return df

def sanitize_converge_data(filename="cond_pass_basic"+fdate):
    """
    Load cond_pass_basic_$(fdate);
    Convert types by splitting;
    Add subcategories;
    Save
    """

    try:
        sIP_server = pd.read_csv(DATAPATH + "Servers_IMC.txt")
    except:
        print "Please put Servers_IMC.txt in " + DATAPATH

    fname = DATAPATH + filename
    convert_dic2 = {"k1":float, "k2": float}

    # read csv and convert columns
    df_san = pd.read_csv(fname,delimiter="|", converters=convert_dic2)
    df_conv = df_san[["ipids", "ts", "diff_list", "retransmit_times"]].applymap(split_int)

    # copy rest of the columns
    for colname in ["gIP", "sIP", "port", "k1", "k2", "country"]:
        df_conv[colname] = df_san[colname]

    del df_san

    # get country and first value in "ts"
    df_conv["first_ts"] = df_conv["ts"].apply(lambda x: x[0])

    # if country was not added earlier...
    if "country" not in df_conv.columns:
        df_conv["country"] = df_conv["gIP"].apply(lambda x: get_country(x))

    # get non-None values in
    for LIM, gx in df_conv.groupby("first_ts"):
        df_conv["diff_p1"] = gx["diff_list"].apply(lambda x: count_not_none( x[:LIM] ))
        df_conv["diff_p2"] = gx["diff_list"].apply(lambda x: count_not_none( x[LIM:] ))

    # merge with server, subcat list
    df2 = df_conv.merge(sIP_server, on='sIP', how='outer')

    # SAVE TO DATAPATH
    df2.to_pickle(DATAPATH + "sanitize/full_merged_ipid_sanitized"+fdate+".pkl")
    return df2

def sanitize_semi_finalize(df2):

    df_redo = df2[ (df2["diff_p1"]<10) | (df2["diff_p2"]<10) ]

    # SAVE sets to redo due to low ipid in DATAPATH
    df_redo[["sIP", "gIP", "country", "domain", "subcat", "diff_p1", "diff_p2"]].to_csv(DATAPATH + "low_ipid_er_redo_"+fdate)
    df_redo.to_pickle(DATAPATH + "sanitize/low_ipid_er_redo"+fdate+".pkl")

    # semi-finalized
    df = df2[ ~ ( (df2["diff_p1"]<10) | (df2["diff_p2"]<10) ) ]

    # SAVE TO DATAPATH
    df.to_pickle(DATAPATH + "sanitize/final_ts60_ipid_sanitized_"+fdate+".pkl")
    df[["gIP", "sIP", "port", "ipids", "ts", "k1", "k2", "country",
        "domain", "subcat", "diff_list", "retransmit_times"]].to_csv( DATAPATH + "ready_for_R_"+fdate)
    return df

###################################################################################################

def sanitize(fname):
    df1 = load_and_remove_errors(fname)
    df2 = sanitize_converge_data('cond_pass_basic_'+fdate)
    df3 = sanitize_semi_finalize(df2)
    logger.debug("DONE SANITIZING: Save csv to "+DATAPATH+"ready_for_R_"+fdate)
    return df3


if __name__ == "__main__":
    #df3 = sanitize(fname)
    pass

#temp = df2[["sIP", "gIP", "country", "domain", "subcat", "diff_p1", "diff_p2"]].groupby('country').count()
#temp['country']


#view_header = ["sIP", "gIP", "country", "domain", "subcat", "case", "pvalue", "intervention", "diff_p1", "diff_p2"]
#df_joined[view_header].head(10)
