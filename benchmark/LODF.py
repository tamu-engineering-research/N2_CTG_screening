import os
import sys
import time
import pickle

import json
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.abspath(os.getcwd()))
from env import env_tools
from env.builder_advanced import env_advanced
from env.env_tools import CTG_compiler


def LODF_bruteforce(CTG_dict, matrix_lodf, islanding, df_branch, df_bus):
    """
    Description: 
        Fast brute-force screening based on LODF
    Parameters: 
        CTG_dict: dict, contains names, components and scripts of existing contingecies
        matrix_lodf: array, M*M
        flow: array, M
        limit: array, M
        func: function
    Returns: 
        CTG_dict_critical: dict, contains names, components and scripts of all contingecies contianed in PTI .con files
    """
    # filter out CTGs that have islanding lines
    CTG_noisl_name = []
    CTG_noisl_branch_index = {}
    CTG_isl_name = []
    CTG_isl_branch_index = {}
    df_branch['BusNum'] = df_branch['BusNum'].astype(int)
    df_branch['BusNum:1'] = df_branch['BusNum:1'].astype(int)
    for i in tqdm(range(len(CTG_dict['name']))): 
        flg = True
        branch_index_list = []
        for name, action in CTG_dict['components'][i].items():
            # if not (action[0] =="BRANCH"):
            #     flg = False
            #     break
            bus_num1 = action[1]
            bus_num2 = action[2]
            ckt_num = action[3]
            df_tmp = df_branch.loc[(df_branch['BusNum']==bus_num1)&(df_branch['BusNum:1']==bus_num2)&(df_branch['LineCircuit']==str(ckt_num))].copy()
            if df_tmp.empty:
                df_tmp = df_branch.loc[(df_branch['BusNum']==bus_num2)&(df_branch['BusNum:1']==bus_num1)&(df_branch['LineCircuit']==str(ckt_num))].copy()
            if not df_tmp.empty:
                index_tmp = df_tmp.index[0]
            else:
                flg = False
                break
            if islanding[index_tmp]: # if having any line resulting in islanding
                flg = False
                break
            else:
                branch_index_list.append(index_tmp)
        if flg:
            CTG_noisl_name.append(CTG_dict['name'][i])
            CTG_noisl_branch_index[len(CTG_noisl_branch_index)] = branch_index_list
        else:
            CTG_isl_name.append(CTG_dict['name'][i])
            CTG_isl_branch_index[len(CTG_noisl_branch_index)] = branch_index_list
    print("Find"+str(len(CTG_noisl_name))+" non-islanding CTGs out of "+str(len(CTG_dict['name']))+" CTGs")
    
    ## find critical combos
    CTG_combo_critical = []
    CTG_combo_critical_value = []
    CTG_combo_c2islanding = []
    limit = df_branch['LineLimMVA'].astype(float).to_numpy().flatten()*110/100
    flow = df_branch['LineMW'].astype(float).to_numpy().flatten()
    branch_index_full = np.arange(len(matrix_lodf[:,0])).tolist()
    matrix_lodf[np.isnan(matrix_lodf)] = 0.0 #### nan means the branch is OPEN ####
    count = len(CTG_noisl_name)
    for i in tqdm(range(count-1)):
        for j in range(i+1, count):
            if True:
                combo_name = [i,j]#[CTG_noisl_name[i], CTG_noisl_name[j]]
                ## get branch indices
                branch_index_tmp = list(set(CTG_noisl_branch_index[i]+CTG_noisl_branch_index[j]))
                branch_index_tmp.sort()
                branch_index_remaining_tmp = list(set(branch_index_full)-set(branch_index_tmp))
                branch_index_remaining_tmp.sort()
                ## construct M
                num_branches = len(branch_index_tmp)
                M = np.zeros([num_branches, num_branches])
                for k1 in range(num_branches):
                    for k2 in range(num_branches):
                        if k1==k2:
                            M[k1,k2] = 1
                        else:
                            M[k1,k2] = - matrix_lodf[branch_index_tmp[k2],branch_index_tmp[k1]]
                if (abs(np.linalg.det(M))<=1e-8) or (np.isnan(abs(np.linalg.det(M)))): # if det~0. meaning non-invertible, means islanding
                    CTG_combo_c2islanding.append(combo_name)
                    continue
                ## construct L
                L = matrix_lodf[np.ix_(branch_index_tmp, branch_index_remaining_tmp)]
                L = L.T
                ## construct f
                F = flow[branch_index_tmp]
                ## calculate delta_F = L*M^(-1)*F
                delta_F_remaining = L@np.linalg.inv(M)@F
                F_remaining_new = flow[branch_index_remaining_tmp]+delta_F_remaining
                det =  np.abs(F_remaining_new)-limit[branch_index_remaining_tmp]
                violation_sum_MVA = np.sum(np.maximum(det, np.zeros(len(F_remaining_new))))
                if violation_sum_MVA>=1.0:
                    CTG_combo_critical.append(combo_name)
                    CTG_combo_critical_value.append(violation_sum_MVA)
    print("Finished screening"+str(int(count*(count-1)/2))+" CTG combos")
    print("There are "+str(len(CTG_combo_c2islanding))+" CTG combos resulting in islanding")
    print("There are "+str(len(CTG_combo_critical))+" CTG combos causing thermal violation")
    # save results
    results = {}
    results['CTG_dict'] = CTG_dict
    results['CTG_noisl_name'] = CTG_noisl_name 
    results['CTG_noisl_branch_index'] = CTG_noisl_branch_index
    results['CTG_isl_name'] = CTG_isl_name
    results['CTG_isl_branch_index'] = CTG_isl_branch_index
    results['CTG_combo_critical'] = CTG_combo_critical # 2 indices in CTG_noisl_name
    results['CTG_combo_critical_value'] = CTG_combo_critical_value
    results['CTG_combo_c2islanding'] = CTG_combo_c2islanding
    return results


def _get_LODF(saw_case, save_root_path = None, save_file = None):
    """
    credited to ESA saw.get_lodf_matrix
    Note that it cannot accomandate large df
    """
    saw_case.pw_order = True
    statement = "CalculateLODFMatrix(OUTAGES,ALL,ALL,YES,DC,ALL,YES)"
    saw_case.RunScriptCommand(statement)
    count = saw_case.ListOfDevices('branch').shape[0]
    array = [f"LODFMult:{x}" for x in range(count)]
    # separate array into pieces and then merge them
    # it is because GetParametersMultipleElement cannot hold a ~6000X6000 matrix
    df = pd.DataFrame()
    step = 500 # to split the whole LODF into multiple pieces
    index_list = np.arange(0, count, 500).tolist()
    for i in tqdm(range(len(index_list))):
        if i<len(index_list)-1:
            df_tmp = saw_case.GetParametersMultipleElement('branch', array[index_list[i]:index_list[i]+step])
            if len(df)==0:
                df = df_tmp.copy()
            else:
                df = pd.concat([df, df_tmp.copy()], axis=1)
        elif i==len(index_list)-1:
            if index_list[i]<=count-1:
                df_tmp = saw_case.GetParametersMultipleElement('branch', array[index_list[i]:])
                if len(df)==0:
                    df = df_tmp.copy()
                else:
                    df = pd.concat([df, df_tmp.copy()], axis=1)
    # modify matrix_lodf and create islaning lines index
    matrix_lodf = df.to_numpy(dtype=float)/100 # covert df percentage into numpy
    islanding = np.any(matrix_lodf>=10, axis=1) # if one contingency cause islanding, then the matrix element -> infinite
    matrix_lodf[islanding, :] = 0.0
    matrix_lodf[islanding, islanding] = -1.0
    if save_root_path is not None:
        _save_LODF(matrix_lodf, islanding, save_root_path, save_file)
    return matrix_lodf, islanding


def _save_LODF(matrix_lodf, islanding, save_root_path, save_file):
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    with open(save_root_path+save_file, 'wb') as f:
        pickle.dump([matrix_lodf, islanding], f)
    return

def _restore_LODF(save_root_path, save_file):
    with open(save_root_path+save_file, 'rb') as f:
        matrix_lodf, islanding = pickle.load(f)
    return matrix_lodf, islanding

def _save_LODF_bruteforce_results(save_root_path, save_file, results):
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    with open(save_root_path+save_file, 'wb') as f:
        pickle.dump([results], f)

def _restore_LODF_bruteforce_results(save_root_path, save_file):
    with open(save_root_path+save_file, 'rb') as f:
        results = pickle.load(f)
    return results

if __name__=="__main__":
    # initialize some parameters
    case_name = 'LOWLD_20220320'
    sim_model_path = 'E:/RL_N2_Contingency/dataset/Scenarios/'+case_name+'/LOWLD_03_20_2022_0328v30.pwb'
    CTG_pool_path = 'E:/RL_N2_contingency/dataset/Scenarios/'+case_name+'/contingency_pool_v0_s138kV_noMaLo.con'
    CTG_dict = CTG_compiler(CTG_file_list = [CTG_pool_path])
    env = env_advanced(
        env_file = sim_model_path,
        CTG_file = CTG_pool_path,
        func_metric = None,)
    env.env_interact.order = True
    env.env_interact.SolvePowerFlow()
    ## test _get_LODF: ~8 min
    matrix_lodf, islanding = _get_LODF(saw_case = env.env_interact,
                                        save_root_path="E:/RL_N2_Contingency/dataset/Scenarios/"+case_name+"/LODF/",
                                        save_file="matrix_LODF.pkl") 
    
    ## test LODF_bruteforce: HWLL takes 1h55m, HW 3h08m, 
    matrix_lodf, islanding = _restore_LODF(save_root_path="E:/RL_N2_Contingency/dataset/Scenarios/"+case_name+"/LODF/",
                                           save_file="matrix_LODF.pkl")
    df_branch = env.env_interact.GetParametersMultipleElement("branch", ['BusNum', 'BusNum:1', 'LineCircuit', 'LineMW', 'LineLimMVA'])
    df_bus = env.env_interact.GetParametersMultipleElement('bus', ['BusNum', 'BusNomVolt'])
    results = LODF_bruteforce(
        CTG_dict = CTG_dict, 
        matrix_lodf = matrix_lodf, 
        islanding = islanding,
        df_branch = df_branch,
        df_bus = df_bus)
    _save_LODF_bruteforce_results(save_root_path="E:/RL_N2_Contingency/dataset/Scenarios/"+case_name+"/LODF/",
                                save_file="results_LODF_bruteforce.pkl",
                                results = results)
    results_rstr = _restore_LODF_bruteforce_results(save_root_path="E:/RL_N2_Contingency/dataset/Scenarios/"+case_name+"/LODF/",
                                save_file="results_LODF_bruteforce.pkl")
    pass