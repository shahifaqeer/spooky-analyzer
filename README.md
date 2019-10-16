# Censor Planet - Spooky Scan

Needs a data/ folder containing all initial files (server IPs, GeoCity database, initial CSVs)

Saves output to data/ folder

### sanitizer.py
used to sanitize ipids data

input is initial csv (eg: ALL_APR7) in data/ folder

output is ipids filtered (eg: ready_for_R_Apr7) in data/ folder


### processor.py
multiprocessing code used to split, detect_cases, combine filtered csv

input is ipids filtered csv (eg: ready_for_R_Apr7)

stage1: splits it in steps of 1000 in data/splits/

stage2: calculates cases and saves it in data/detected/ in multiprocessing fashion using CPU_CORES - 2 parallel processes

stage3: combines detected cases and merges with original data, saves it to data/case_detected_all


