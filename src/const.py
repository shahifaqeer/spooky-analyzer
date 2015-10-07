# filename (use dates) in data/ folder
fname = "20150921_prelim_exp2.csv"
#fname = "20150412_biased.csv"
fdate = "20150921"

# massive concat data (if in use)
final_concat_fdate = "20150921"

# parameters
# intervention for case detection
CENSORPLANET_INTERVENTION = 20

# PATHS
CONSTDATAPATH = "../data/"
DATAPATH = "../data_"+fdate+"/"
SPLITFOLDER = DATAPATH + "splits_"+fdate+"/"
DETECTFOLDER = DATAPATH + "detects_"+fdate+"/"
