import pandas as pd

from sanitizer import *
from processor import *
import const
import sys

# constants
fdate = const.final_concat_fdate
datapath = const.CONSTDATAPATH[:-1]



def massive_concat():
    machines = ['Apr7', 'Apr11_T', 'Apr11_F', 'Apr11_S']
    full_detected = []

    for mc in machines:
        df = pd.read_pickle(datapath + "_"+mc+"/case_detected_all_"+mc+".pkl")
        print " LEN " + mc + " = " + str( len(df) )
        full_detected.append( df )
        del df

    df_concat = pd.concat(full_detected)
    print "TOTAL ", len(df_concat)
    df_concat.to_pickle(datapath + "detected_case_all_"+fdate+".pkl")
    return df_concat


if __name__ == "__main__":

    # Global Path for DATA (Unix convention only)
    if not os.path.exists(const.DATAPATH):
        os.makedirs(const.DATAPATH)

    # sanitize
    df1 = load_and_remove_errors()
    df2 = sanitize_converge_data('cond_pass_basic_'+ const.fdate)
    df3 = sanitize_semi_finalize(df2)
    logger.debug("DONE SANITIZING: Save pkl to "+ const.DATAPATH+"ready_for_R_"+ const.fdate+".pkl")
    del df1, df2

    # process
    dataframe_splitter(df3, STEP=1000)
    parallel_case_detection()
    join_case_with_data()

    print "DONE"
