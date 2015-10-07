#!/usr/bin/env/ python
from intervention import *
import numpy as np
import random,glob
import string
import datetime as dt

INTERVENTION = 20 #number of forged packets
ALPHA = 0.01
ao_alpha = 0.05
diffList=[6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,10,10,11,9,9,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,13,9,9,9,9,None,None,10,9,9,9,9,9,9,None,None,9,9,9,None,None,None,9,9,None,None,9,9,10,9,9,9]
gIP="71.44.3.191"
sIP="198.211.105.99"
ts=[60,64,70]

def detect_case(gIP,sIP,diffList,ts,error_case,INTERVENTION,DEBUG=False):

 #   if len(ts)<3: return(4,None,None)
 #   if error_case!=None: return(4,None,None)
    try:
        ir=intervention_at_time(diffList,ts, [0.5 * INTERVENTION,\
            (min(2,((len(ts)+1)/2))-0.1) * INTERVENTION], 0.05, True) #FIXME
        if DEBUG:print "ir.pvalues",ir.pvalues[0],ALPHA
        case = -1
        if ir.pvalues[0] <= ALPHA:
            case = 1
        elif ir.pvalues[0] > 1.0 - ALPHA:
            if ir.pvalues[1] < ALPHA:
                case = 2
            elif ir.pvalues[1] > 1.0 - ALPHA:
                case = 3
            else:
                case = 0
        else:
            case = 0
        if DEBUG: print gIP,sIP,case,ir
        return(case,ir.pvalues[0],ir.pvalues[1],ir.intervention)
    except Exception:
        if DEBUG:print gIP,sIP, ts, "ERROR in detect case ARMA",diffList
        case = 4
        return(case,None,None)

if __name__=="__main__":
    print detect_case(gIP,sIP,diffList,ts,None,INTERVENTION,True)
