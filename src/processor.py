import pandas as pd
import multiprocessing as mp
import os, sys, glob
from intervention import *
import GeoIP
from caseDetection import *
import logging
from collections import defaultdict

import const
import sanitizer

##########################################################


# UTILITY FUNCTIONS
f = open(os.devnull, 'w')
#f2 = open("case_detector.log", 'a+')

gi1 = GeoIP.open(const.CONSTDATAPATH + "GeoIPCity.dat", GeoIP.GEOIP_STANDARD)

mp.log_to_stderr()
logger = mp.get_logger()
logger = logging.getLogger('myapp')
#hdlr = logging.FileHandler('case_detection.log')
#formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#hdlr.setFormatter(formatter)
#logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


def get_country(ip):
    ipInfo=gi1.record_by_addr(ip)
    if ipInfo is None:
        return "roya"
    return ipInfo["country_code"]



# PROCESSING: SPLIT INTO 1000 steps
##################################################################################################

# Loader
def load_dataframe(filename):
    """ input: filename of pickle dataframe ready_for_R csv"""

    if os.path.isfile(const.DATAPATH + filename + ".pkl"):
        return pd.read_pickle( const.DATAPATH + filename + ".pkl")
    else:
        logger.error("NO SANITIZED PICKLE: "+ const.DATAPATH + filename)
        logger.error("SANITIZE DATA FIRST")
        df = sanitizer.sanitize()
    return df


# MAIN FUNCTIONS
############################################################
# Splitter
def dataframe_splitter(df3, STEP=1000):
    """
    input: dataframe with sanitized ipids
    output: splits of STEP=1000 by default
    """

    if not os.path.exists(const.SPLITFOLDER):
        os.makedirs(const.SPLITFOLDER)

    part = 0
    for split_start in xrange(0, len(df3), STEP):
        print part, split_start, split_start + STEP

        # To save ALL columns
        df_new = df3.ix[split_start: split_start + STEP, :]

        # To save only 4 columns not all?
        #df_new = df3.ix[split_start: split_start + STEP, ["gIP", "sIP", "diff_list", "ts"]]

        df_new.to_pickle(const.SPLITFOLDER + "ready_for_R_"+str(part)+".pkl")
        part += 1
    logger.debug("DONE SPLITTING")
    return

# Detector
def get_each_case(df, OUTNULL=False):
    """ uses the R code to detect case for each 1000 enty batch. stdoutput to dev/null so nothing will print once this global setting is called """

    print "ENTER get_each_case"

    if OUTNULL:
        sys.stdout = f

    mydata = defaultdict(list)

    for ix, row in df.iterrows():
        case, pvalue0, pvalue1, intervention, aos, order, interventions, null_interventions = detect_case(row['gIP'], row['sIP'], row['diff_list'], row['ts'], None,
                const.CENSORPLANET_INTERVENTION, True)
>>>>>>> master

        mydata['index'].append(ix)
        mydata['case'].append(case)
        mydata['pvalue0'].append(pvalue0)
        mydata['pvalue1'].append(pvalue1)
        mydata['intervention'].append(intervention)
        mydata['aos'].append(aos)
        mydata['order'].append(order)
        mydata['interventions'].append(interventions)
        mydata['null_interventions'].append(null_interventions)

    return pd.DataFrame(mydata).set_index('index')

# Wrapper
def mp_case_detection_per_df(files):
    """ wrapper function for get_each_case() to load and save individual case detected"""

    part = files.split("_")[-1].strip(".pkl")
    print "LOAD " + files
    df = pd.read_pickle(files)
    df_out = get_each_case(df, True)
    #print "DONE + SAVE " + part
    df_out.to_pickle(const.DETECTFOLDER+"case_detected_"+part+".pkl")

    # delete df otherwise becomes super slow
    del df, df_out
    return

def parallel_case_detection():
    """ CREATE CPU-2 parallel processes to handle
    1000 entry databases """

    # DATAPATH to save detected output
    if not os.path.exists(const.DETECTFOLDER):
        os.makedirs(const.DETECTFOLDER)
        print "make "+const.DETECTFOLDER

    # use Pool instead of Process to limit num of cores in async jobs
    CORES = mp.cpu_count()-2    #avoid lockdown
    pool = mp.Pool(processes=CORES)
    print "start pool of "+ str(CORES) +" cores"

    # TESTING - 25 files took about 5 mins with 6 cores on my machine
    print "detecting files in " + const.SPLITFOLDER
    for files in glob.glob(const.SPLITFOLDER + "*.pkl"):
    # Takes 1 - 1.5 hours on 6 cores for 230 files, 230,000 entries
        print files
        pool.apply_async(mp_case_detection_per_df, args=(files, ))

    pool.close()
    pool.join()

    logger.debug("Done ALL")
    return

def single_threaded():

    print "THIS WILL FUCK UP THE CPU"

    for files in glob.glob(const.SPLITFOLDER + "*"):
    #for files in glob.glob("splits/*")[3:6]:
        part = files.split("_")[-1]
        logger.debug(files)
        df = pd.read_pickle(files)
        df_out = get_each_case(df, True)
        logger.debug("DONE " + part)
        df_out.to_pickle(const.DETECTFOLDER+"case_detected_"+part+".pkl")
        del df, df_out
    return

# PROCESSING: COMBINE BACK TO DATAFRAME
#############################################################

def join_case_with_data():
    full_df = []

    for files in glob.glob(const.SPLITFOLDER + "*.pkl"):
        part = files.split("_")[-1].strip(".pkl")

        # original: no case detection
        df1 = pd.read_pickle(files)

        # load case detected parts, join to original, concat into a complete dataframe
        df2 = pd.read_pickle(const.DETECTFOLDER +"case_detected_"+part+".pkl")
        full_df.append( df1.join(df2) )

        del df2, df1

    df_full = pd.concat(full_df)

    # SAVE TO DATAPATH
    df_full.to_pickle(const.DATAPATH+"case_detected_all_"+const.fdate+".pkl")
    df_full.to_csv(const.DATAPATH+"case_detected_all_"+const.fdate)

    return df_full

#############################################################

if __name__ == "__main__":
    #pass

    fname = "ready_for_R_" + const.fdate
    df3 = load_dataframe(fname)

    dataframe_splitter(df3, STEP=1000)
    del df3

    parallel_case_detection()
    join_case_with_data()
