import pandas as pd

from sanitizer import *
from processor import *
import const


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
    dataframe_splitter(df3, STEP=1000):
    parallel_case_detection()
    join_case_with_data()

    print "DONE"
