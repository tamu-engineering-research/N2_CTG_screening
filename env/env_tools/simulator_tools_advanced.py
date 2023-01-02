"""
Use SimuAtuo scripts to do simulation via ESA
"""

import os
import sys
import time

from esa import SAW
import numpy as np
import pandas as pd
from tqdm import tqdm
import yacs

def advanced_CTG_loader(saw_case, CTG_file):
    """
    Description:
    Parameters:
        saw_case: SAW class
        CTG_file: string, name of PTI CON file
    Returns:
        saw_case: SAW class
    """
    saw_case.RunScriptCommand("CTGReadFilePTI(\""+CTG_file+"\");") # CTGReadFilePTI("filename");
    return saw_case

def advanced_CTG_solve_single(saw_case, CTG_name):
    saw_case.RunScriptCommand("CTGSolve(\""+CTG_name+"\");")
    return saw_case

def advanced_CTG_solve_all(saw_case,):
    saw_case.RunScriptCommand("CTGSolveALL(NO,YES)")
    return

def advanced_CTG_restore(saw_case):
    # saw_case.RunScriptCommand("CTGClearAllResults;") #CTGClearAllResults;
    saw_case.RunScriptCommand("CTGRestoreReference;") # CTGRestoreReference;
    return saw_case

def advanced_CTG_set_ref(saw_case):
    saw_case.RunScriptCommand("CTGSetAsReference;") # CTGSetAsReference;
    return saw_case

def advanced_simulator(saw_case, 
    CTG_name_list:list = None,):
    """
    Description:
    Parameters:
        saw_case: SAW class object
        CTG_name_list: list, 
    Returns:
    """
    if CTG_name_list is None:
        print("No contingencies are simulated")
    else:
        saw_case.RunScriptCommand("EnterMode(RUN);")
        for CTG_name in CTG_name_list:
            advanced_CTG_solve_single(
                saw_case = saw_case,
                CTG_name = CTG_name,)
        # saw_case.RunScriptCommand("")
    return

if __name__=="__main__":

    a=0