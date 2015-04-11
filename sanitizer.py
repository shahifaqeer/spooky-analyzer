import pandas as pd
import os, sys
from intervention import *
import GeoIP
from caseDetection import *
import logging
from collections import defaultdict


##########################################################

# UTILITY FUNCTIONS
f = open(os.devnull, 'w')

gi1 = GeoIP.open("GeoIPCity.dat",GeoIP.GEOIP_STANDARD)
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

def load_and_remove_errors(filename="ALL_APR7"):
    """
    Load raw df;
    Add countries;
    remove null entries in ipids;
    Save
    """
    fname=filename
    temp = "gIP,sIP,port,ipids,diff_list,ts,k1,k2,retransmit_times,NotGlobal,NotEnough,NotEnoughRetrans, date"
    header_row = temp.split(",")
    df_all = pd.read_csv(fname,delimiter="|", names=header_row)

    # Add countries
    df_all["country"] = df_all["gIP"].apply(lambda x: get_country(x))

    if not os.path.exists("sanitize"):
        os.makedirs("sanitize")

    # save those for which ipids was not recorded
    df_nulls = df_all[ df_all["ipids"].isnull() ]
    df_nulls[["sIP", "gIP", "country"]].to_csv("cond_er_redo")
    df_nulls.to_pickle("sanitize/null_ipid_cond_er_redo.pkl")

    df = df_all[ df_all["ipids"].notnull() ]
    df.to_csv("cond_pass_basic_Apr7", sep="|")
    df.to_pickle("sanitize/phase1_ipid_nonnull.pkl")

    return df

def sanitize_converge_data(filename="cond_pass_basic_Apr7"):
    """
    Load cond_pass_basic_Apr7;
    Convert types by splitting;
    Add subcategories;
    Save
    """

    sIP_server = pd.read_csv("Servers_IMC.txt")

    fname= filename
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
    if "country" not in df_conv.columns:
        df_conv["country"] = df_conv["gIP"].apply(lambda x: get_country(x))

    # get non-None values in
    for LIM, gx in df_conv.groupby("first_ts"):
        df_conv["diff_p1"] = gx["diff_list"].apply(lambda x: count_not_none( x[:LIM] ))
        df_conv["diff_p2"] = gx["diff_list"].apply(lambda x: count_not_none( x[LIM:] ))

    # merge with server, subcat list
    df2 = df_conv.merge(sIP_server, on='sIP', how='outer')

    df2.to_pickle("sanitize/full_merged_ipid_sanitized.pkl")
    return df2

def sanitize_semi_finalize(df2):

    df_redo = df2[ (df2["diff_p1"]<10) | (df2["diff_p2"]<10) ]
    df_redo[["sIP", "gIP", "country", "domain", "subcat", "diff_p1", "diff_p2"]].to_csv("low_ipid_er_redo")
    df_redo.to_pickle("sanitize/low_ipid_er_redo.pkl")

    # semi-finalized
    df = df2[ ~ ( (df2["diff_p1"]<10) | (df2["diff_p2"]<10) ) ]
    df.to_pickle("sanitize/final_ts60_ipid_sanitized.pkl")
    df[["gIP", "sIP", "port", "ipids", "ts", "k1", "k2", "country",
        "domain", "subcat", "diff_list", "retransmit_times"]].to_csv("sanitized_Apr7")
    return df

###################################################################################################
# READY FOR R: SPLIT INTO 1000 steps
##################################################################################################

def dataframe_splitter(df3, STEP=1000):
    if not os.path.exists("splits"):
        os.makedirs("splits")
    part = 0
    for split_start in xrange(0, len(df3), STEP):
        print part, split_start, split_start + STEP

        # To save ALL columns
        df_new = df3.ix[split_start: split_start + STEP, :]

        # To save only 4 columns not all?
        #df_new = df3.ix[split_start: split_start + STEP, ["gIP", "sIP", "diff_list", "ts"]]

        df_new.to_pickle("splits/ready_for_R_"+str(part)+".pkl")
        part += 1
    logger.debug("DONE SPLITTING")
    return


def sanitize(fname):
    if fname=="ALL_APR7":
        fdate = "Apr7"
        df1 = load_and_remove_errors(fname)
        df2 = sanitize_converge_data('cond_pass_basic_'+fdate)
        df3 = sanitize_semi_finalize(df2)
    logger.debug("DONE SANITIZING")
    return df3


if __name__ == "__main__":
    df3 = sanitize("ALL_APR7")
    dataframe_splitter(df3)
    pass

#temp = df2[["sIP", "gIP", "country", "domain", "subcat", "diff_p1", "diff_p2"]].groupby('country').count()
#temp['country']


#view_header = ["sIP", "gIP", "country", "domain", "subcat", "case", "pvalue", "intervention", "diff_p1", "diff_p2"]
#df_joined[view_header].head(10)
