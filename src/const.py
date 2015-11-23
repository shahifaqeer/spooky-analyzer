# filename (use dates) in data/ folder
fname = "20151120_unbiased.csv"
#fname = "20150412_biased.csv"
fdate = "20151120"

# massive concat data (if in use)
final_concat_fdate = "20151120"

# parameters
# intervention for case detection - read from 3rd last param
#CENSORPLANET_INTERVENTION = 20

# Threshold for min number of phase measurements - case 5 (can't be passed to R)
IPID_MIN_RESPONSES = 10

# PATHS
CONSTDATAPATH = "../data/"
DATAPATH = "../data_"+fdate+"/"
SPLITFOLDER = DATAPATH + "splits_"+fdate+"/"
DETECTFOLDER = DATAPATH + "detects_"+fdate+"/"
