# filename (use dates) in data/ folder
fname = "20150422_unbiased.csv"
#fname = "20150412_biased.csv"
fdate = "20150422"

# massive concat data (if in use)
final_concat_fdate = "20150422"

# parameters
# intervention for case detection
CENSORPLANET_INTERVENTION = 20

# PATHS
CONSTDATAPATH = "data/"
DATAPATH = "data_"+fdate+"/"
SPLITFOLDER = DATAPATH + "splits_"+fdate+"/"
DETECTFOLDER = DATAPATH + "detects_"+fdate+"/"
