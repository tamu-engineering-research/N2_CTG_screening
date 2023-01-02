import os
import sys

import json
import numpy as np
from tqdm import tqdm
import pandas as pd


def CTG_pool_creater(
    CTG_file_list,
    CTG_pool_root_path, 
    CTG_pool_name,):
    """
    Description: 
        To create a common set of all .con files and save it as a new .con file
    Parameters: 
        CTG_file_list: A list of paths of .con files
        CTG_pool_root_path: A root directory of contingency pool .con file
        CTG_pool_name: Name of contingency pool .con file
    Returns: 
        CTG_dict: A dict variable that contains names, components and scripts of contingency pool .con file
    """
    CTG_dict = CTG_compiler(CTG_file_list = CTG_file_list)
    _CTG_writer(CTG_pool_root_path, CTG_pool_name, CTG_dict) # packed module of the follows
    return CTG_dict

def CTG_pool_comparator(
    CTG_pool_dict = None,
    CTG_target_dict = None,
    CTG_pool_dict_file = None,
    CTG_target_dict_file = None):
    """
    Description: 
        To check if target contingency dict are in the contingency pool
    Parameters: 
        CTG_pool_dict: dict, n CTGs, a dict of contingency pool,
        CTG_target_dict: dict, m CTGs, a dict of target contingency,
        CTG_pool_dict_file: string, name of contingency pool .con file ,
        CTG_target_dict_file: string, Name of target contingency .con file 
    Returns: 
        array_overlapped_contingency: A list of indices of overlapped contingencies in the pool dict
    """
    # read contingency pool
    if (CTG_pool_dict is not None):
        print('\'CTG_pool_dict\' is being used.')
    else:
        print('\'CTG_pool_dict_file\' is being used.')
        CTG_pool_dict = CTG_compiler(CTG_file_list = [CTG_pool_dict_file])
    # read target contingency
    if (CTG_target_dict is not None):
        print('\'CTG_target_dict\' is being used.')
    else:
        print('\'CTG_target_dict_file\' is being used.')
        CTG_target_dict = CTG_compiler(CTG_file_list= [CTG_target_dict_file])
    # compare contingency pool and target
    array_overlapped_contingency, array_missed_contingency, array_new_contingency = _CTG_dict_duplicate_checker(
        CTG_dict_ref = CTG_pool_dict,
        CTG_dict_target = CTG_target_dict,)
    # notification
    print('Detect '+str(len(array_overlapped_contingency))+" CTGs in the pool, while "+str(len(array_missed_contingency))+" CTGs are missed.")
    print(str(len(array_new_contingency))+' new CTGs in the target file will not be considered in the following analysis.')
    return array_overlapped_contingency

def CTG_compiler(
    CTG_file_list:list, 
    CTG_dict_exist:dict={'components':[],'name':[], 'scripts':[]}):
    """
    Description: 
        To compile a list of PTI .con files
    Parameters: 
        CTG_file_list: A list of paths of .con files
        CTG_dict_exist: (Optional) A dict variable that contains names, components and scripts of existing contingecies
    Returns: 
        CTG_dict: A dict variable that contains names, components and scripts of all contingecies contianed in PTI .con files
    """
    CTG_dict = {'components':[],'name':[], 'scripts':[]}
    for file_tmp in CTG_file_list:
        event_total_index, event_new_index = 0, 0
        event_flg = 0
        CTG_info_tmp = {}
        with open(file_tmp) as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                if line.startswith('CONTINGENCY'):
                    event_flg = 1
                    component_index = 0
                    CTG_info_tmp['name'] = line.split()[1]
                    CTG_info_tmp['components'] = {}
                    CTG_info_tmp['scripts'] = {}
                elif line.startswith('END'):
                    event_flg = 0
                    event_total_index+= 1
                    if not _CTG_duplicate_checker(CTG_dict['name'], CTG_info_tmp['name']): # if new CTG
                        CTG_dict['components'].append(CTG_info_tmp['components'])
                        CTG_dict['name'].append(CTG_info_tmp['name'])
                        CTG_dict['scripts'].append(CTG_info_tmp['scripts'])
                        event_new_index +=1
                else:
                    if event_flg==0:# skip unknown lines when flg is negative
                        continue
                    elif event_flg==1:
                        components = line.split()
                        if (components[0]=='OPEN')& ((components[1]=='BRANCH')|(components[1]=='LINE')):
                            CTG_info_tmp['components'][component_index] = ['BRANCH', int(components[4]), int(components[7]), int(components[9])] # type, bus1, bus2, circuit
                            CTG_info_tmp['scripts'][component_index] = line
                            component_index += 1
                        elif (components[0]=='REMOVE')& (components[1]=='LOAD'):
                            CTG_info_tmp['components'][component_index] = ['LOAD', int(components[5]), int(components[2])] # type, bus, load
                            CTG_info_tmp['scripts'][component_index] = line
                            component_index += 1
                        elif (components[0]=='REMOVE')& (components[1]=='MACHINE'):
                            CTG_info_tmp['components'][component_index] = ['MACHINE', int(components[5]), int(components[2])] # type, bus, machine
                            CTG_info_tmp['scripts'][component_index] = line
                            component_index += 1
                        else:
                            print('Unexpected type of contingency')
        print('Finish reading contingency in '+file_tmp)
        print('Find total '+str(event_total_index)+'contingencies, among which '+str(event_new_index)+' ones are new.')
    return CTG_dict

def CTG_selector(CTG_dict, saw_model, voltage_threshold = 138, CTG_pool_root_path = "", CTG_pool_name = ""):
    """
    Description: 
        To select CTGs that satisfy the requirement
    Parameters: 
        CTG_dict: A dict variable that contains names, components and scripts of existing contingecies
        saw_model: SAW class object
        voltage_threshold: int

    Returns: 
        CTG_dict_selected
    """
    CTG_dict_selected = {'components':[],'name':[], 'scripts':[]}
    df_bus = saw_model.GetParametersMultipleElement('bus', ['BusNum', 'BusNomVolt'])
    num_infeasible = 0
    num_selected = 0
    for i in tqdm(range(len(CTG_dict['name']))):
        flg_tmp = True
        for _, action_tmp in CTG_dict['components'][i].items():
            if (action_tmp[0] == "MACHINE") or (action_tmp[0] == "LOAD"):
                flg_tmp = False
                break
            elif action_tmp[0] == "BRANCH":
                num_bus0 = action_tmp[1]
                num_bus1 = action_tmp[2]
                try:
                    nominal_vol_bus0 = df_bus.loc[df_bus['BusNum']==num_bus0]['BusNomVolt'].values[0]
                    nominal_vol_bus1 = df_bus.loc[df_bus['BusNum']==num_bus1]['BusNomVolt'].values[0]
                except:
                    num_infeasible += 1
                    flg_tmp = False
                    break
                if not ((nominal_vol_bus0>=voltage_threshold) or (nominal_vol_bus1>=voltage_threshold)):
                    flg_tmp = False
            else:
                print("Unexpected contingency type")
                flg_tmp = False
            if not flg_tmp:
                break
        if flg_tmp:
            num_selected +=1
            CTG_dict_selected['components'].append(CTG_dict['components'][i])
            CTG_dict_selected['name'].append(CTG_dict['name'][i])
            CTG_dict_selected['scripts'].append(CTG_dict['scripts'][i])
    _CTG_writer(CTG_pool_root_path, CTG_pool_name, CTG_dict_selected)
    print("Totally " +str(num_selected)+" selected contingencies and "+str(num_infeasible)+" infeasible contingencies for this model.")
    return CTG_dict_selected

def CTG_attribute_creator(CTG_dict, saw_model,):
    """
    Description: 
        To create attibutes of CTGs in terms of voltage, types, islandings
    Parameters: 
        CTG_dict: A dict variable that contains names, components and scripts of existing contingecies
        saw_model: SAW class object
    Returns: 
        CTG_dict_comment: df with attributes in terms of type, voltage and islandings
    """
    CTG_dict_comment = pd.DataFrame(columns=["voltage_kV", "num_branch", "num_machine", "num_load", "feasible", 
                                            "list_branch_id", "list_machine_id" ,"list_load_id",
                                            "list_machine_MW" ,"list_load_MW",])
    saw_model.pw_order = True
    df_bus = saw_model.GetParametersMultipleElement('bus', ['BusNum', 'BusNomVolt'])
    df_bus['BusNum'] = df_bus['BusNum'].astype(int)
    df_bus['BusNomVolt'] = df_bus['BusNomVolt'].astype(float)
    df_branch = saw_model.GetParametersMultipleElement("branch", ['BusNum', 'BusNum:1', 'LineCircuit', ])
    df_branch['BusNum'] = df_branch['BusNum'].astype(int)
    df_branch['BusNum:1'] = df_branch['BusNum:1'].astype(int)
    df_branch['LineCircuit'] = df_branch['LineCircuit'].astype(int)
    df_machine = saw_model.GetParametersMultipleElement("gen", ['BusNum', 'GenID', 'GenMW', 'GenMWSetPoint'])
    df_machine['BusNum'] = df_machine['BusNum'].astype(int)
    df_machine['GenID'] = df_machine['GenID'].astype(int)
    df_machine['GenMW'] = df_machine['GenMW'].astype(float)
    df_machine['GenMWSetPoint'] = df_machine['GenMWSetPoint'].astype(float)
    df_load = saw_model.GetParametersMultipleElement("load", ['BusNum', 'LoadID', 'LoadSMW', 'LoadMW'])
    df_load['BusNum'] = df_load['BusNum'].astype(int)
    df_load['LoadID'] = df_load['LoadID'].astype(int)
    df_load['LoadSMW'] = df_load['LoadSMW'].astype(float)
    df_load['LoadMW'] = df_load['LoadMW'].astype(float)
    for i in tqdm(range(len(CTG_dict['name']))):
        voltage_highest = 0
        flg_feasible = True
        num_branch = 0
        num_machine = 0
        num_load = 0
        list_branch_id = []
        list_machine_id = []
        list_load_id = []
        list_machine_MW = []
        list_load_MW = []
        for _, action_tmp in CTG_dict['components'][i].items():
            if (action_tmp[0] == "MACHINE"): 
                num_machine += 1
                num_bus0 = action_tmp[1]
                num_bus1 = action_tmp[1]
                machine_id = action_tmp[2]
                list_machine_id.append(num_bus0)
                list_machine_MW.append(df_machine.loc[(df_machine['BusNum']==num_bus0)&(df_machine['GenID']==machine_id)]['GenMW'].values[0])
            elif (action_tmp[0] == "LOAD"):
                num_load += 1
                num_bus0 = action_tmp[1]
                num_bus1 = action_tmp[1]
                load_id = action_tmp[2]
                list_load_id.append(num_bus0)
                list_load_MW.append(df_load.loc[(df_load['BusNum']==num_bus0)&(df_load['LoadID']==load_id)]['LoadMW'].values[0])
            elif action_tmp[0] == "BRANCH":
                num_branch += 1
                num_bus0 = action_tmp[1]
                num_bus1 = action_tmp[2]
                num_circuit = action_tmp[3]
                # swap the order of two buses
                df_tmp = df_branch.loc[(df_branch['BusNum']==num_bus0)&(df_branch['BusNum:1']==num_bus1)&(df_branch['LineCircuit']==num_circuit)]
                if df_tmp.empty:
                    df_tmp = df_branch.loc[(df_branch['BusNum']==num_bus0)&(df_branch['BusNum:1']==num_bus1)&(df_branch['LineCircuit']==num_circuit)]
                # check if the branch exists in the power flow model
                if df_tmp.empty:
                    flg_feasible = False
                    branch_id = -1
                else:
                    branch_id = df_tmp.index[0]
                list_branch_id.append(branch_id)
            else:
                print("Unexpected contingency type")
                flg_feasible = False
            try:
                nominal_vol_bus0 = df_bus.loc[df_bus['BusNum']==num_bus0]['BusNomVolt'].values[0]
                nominal_vol_bus1 = df_bus.loc[df_bus['BusNum']==num_bus1]['BusNomVolt'].values[0]
            except:
                nominal_vol_bus0 = 0
                nominal_vol_bus1 = 0
                flg_feasible = False
            voltage_highest =  np.max([voltage_highest, nominal_vol_bus0, nominal_vol_bus1])
        CTG_dict_comment.loc[len(CTG_dict_comment.index)] = [
            voltage_highest, num_branch, num_machine, num_load, flg_feasible,
            list_branch_id, list_machine_id, list_load_id,
            list_machine_MW, list_load_MW]
    ## print attributes
    num_branch = CTG_dict_comment[CTG_dict_comment["num_branch"]>0].count()["num_branch"]
    num_machine = CTG_dict_comment[CTG_dict_comment["num_machine"]>0].count()["num_machine"]
    num_load = CTG_dict_comment[CTG_dict_comment["num_load"]>0].count()["num_load"]
    num_feasible = CTG_dict_comment[CTG_dict_comment["feasible"]==True].count()["feasible"]
    num_high_volt = CTG_dict_comment[CTG_dict_comment["voltage_kV"]>=138].count()["voltage_kV"]
    print("Find "+str(num_branch)+" out of "+str(len(CTG_dict['name']))+"  CTGs that contains branches.")
    print("Find "+str(num_machine)+" out of "+str(len(CTG_dict['name']))+"  CTGs that contains machines.")
    print("Find "+str(num_load)+" out of "+str(len(CTG_dict['name']))+"  CTGs that contains loads.")
    print("Find "+str(num_feasible)+" out of "+str(len(CTG_dict['name']))+"  CTGs that are feasible for the power flow model.")
    print("Find "+str(num_high_volt)+" out of "+str(len(CTG_dict['name']))+"  CTGs that are >= 138 kV.")
    return CTG_dict_comment


def _CTG_duplicate_checker(CTG_list, CTG_tmp):
    """
    Description:
        Check if one contingency is in the list.
    Parameters:
        CTG_list: A list of string, name of contingencies; 
        CTG_tmp: A string;
    Returns
        Boolean: if CTG_tmp is in the CTG_list
    """
    if CTG_tmp in CTG_list:
        return True
    else:
        return False

def _CTG_dict_duplicate_checker(CTG_dict_ref, CTG_dict_target):
    """
    Description:
        Compare two dicts
    Parameters:
        CTG_dict_ref: dict, contingency pool
        CTG_dict_target: dict, target contingency dict
    Returns:
        array_missed_contingency: indices of (CTG_dict_ref / CTG_dict_target) in CTG_dict_ref
        array_new_contingency: indices of (CTG_dict_target / CTG_dict_ref) in CTG_dict_target
        array_overlapped_contingency: indices of  (CTG_dict_target & CTG_dict_ref) in CTG_dict_ref
    """
    array_missed_contingency = 0
    array_new_contingency = 0
    array_overlapped_contingency = 0
    list_ref_name = CTG_dict_ref['name']
    list_target_name = CTG_dict_target['name']
    list_ref_minus_target = list(set(list_ref_name)-set(list_target_name))
    list_target_minus_ref = list(set(list_target_name)-set(list_ref_name))
    list_overlapped_contingency, array_overlapped_contingency, _ = np.intersect1d(list_ref_name, list_target_name, return_indices=True)
    list_missed_contingency, array_missed_contingency, _ = np.intersect1d(list_ref_name, list_ref_minus_target, return_indices=True)
    list_new_contingency, array_new_contingency, _ = np.intersect1d(list_target_name, list_target_minus_ref, return_indices=True)
    return array_overlapped_contingency, array_missed_contingency, array_new_contingency

def _CTG_writer(CTG_pool_root_path, CTG_pool_name, CTG_dict):
    with open(CTG_pool_root_path+CTG_pool_name, 'w') as f:
        for i in range(len(CTG_dict['name'])):
            f.write('CONTINGENCY '+CTG_dict['name'][i]+'\n')
            for key, script_tmp in CTG_dict['scripts'][i].items():
                f.write(script_tmp)
            f.write('END\n')


if __name__=="__main__":
    CTG_file_list = []
    CTG_file_list.append('E:/RL_N2_Contingency/dataset/Scenarios/HWLL_20211224/HWLL_20211224.con') # 5665 contingencies
    CTG_file_list.append('E:/RL_N2_Contingency/dataset/Scenarios//HIGHWD_20220320/HIGHWD_03_20_2022_2010.con') # 6978 contingencies
    CTG_file_list.append('E:/RL_N2_Contingency/dataset/Scenarios//HIGHLD_20210824/HIGHLD_08_24_2021_1655.con') # 7075 contingencies
    CTG_file_list.append('E:/RL_N2_Contingency/dataset/Scenarios//LOWLD_20220320/LOWLD_03_20_2022_0328.con') # 6965 contingencies
    CTG_file_list.append('E:/RL_N2_Contingency/dataset/Scenarios//LOWWD_20220318/LOWWD_03_18_2022_2303.con') # 7077 contingencies
    
    # ## test CTG_compiler
    # CTG_list = CTG_compiler(CTG_file_list = CTG_file_list)

    # ## test CTG_pool_creater
    # CTG_dict = CTG_pool_creater(
    #             CTG_file_list,
    #             CTG_pool_root_path = r'E:/RL_N2_contingency/dataset/Scenarios/Contingency_Pool/', 
    #             CTG_pool_name = 'contingency_pool_v0.con',)

    # ## test CTG_pool_comparator
    # array_overlapped_contingency = CTG_pool_comparator(
    #     CTG_pool_dict_file = 'E:/RL_N2_contingency/dataset/Scenarios/Contingency_Pool/contingency_pool_v0.con',
    #     CTG_target_dict_file = 'E:/RL_N2_Contingency/dataset/Scenarios/HWLL_20211224/HWLL_20211224.con')
    # a=0

    ## test CTG_selector: successful
    from esa import SAW
    env_file = 'E:/RL_N2_Contingency/dataset/Scenarios/LOWLD_20220320/LOWLD_03_20_2022_0328v30.pwb'
    # env_file = 'E:/RL_N2_Contingency/dataset/Scenarios/LOWWD_20220318/LOWWD_03_18_2022_2303v30.pwb'
    # env_file = 'E:/RL_N2_Contingency/dataset/Scenarios/HIGHWD_20220320/HIGHWD_03_20_2022_2010v30.pwb'
    # env_file = 'E:/RL_N2_Contingency/dataset/Scenarios/HWLL_20211224/HWLL_20211224_v30.pwb'

    CTG_pool_path = 'E:/RL_N2_contingency/dataset/Scenarios/LOWLD_20220320/LOWLD_03_20_2022_0328.con'
    # CTG_pool_path = 'E:/RL_N2_contingency/dataset/Scenarios/LOWWD_20220318/LOWWD_03_18_2022_2303.con'
    # CTG_pool_path = 'E:/RL_N2_contingency/dataset/Scenarios/HIGHWD_20220320/HIGHWD_03_20_2022_2010.con'
    # CTG_pool_path = 'E:/RL_N2_contingency/dataset/Scenarios/Contingency_Pool/contingency_pool_v0.con'

    CTG_dict = CTG_compiler(CTG_file_list = [CTG_pool_path])

    CTG_pool_root_path = "E:/RL_N2_contingency/dataset/Scenarios/LOWLD_20220320/"
    # CTG_pool_root_path = "E:/RL_N2_contingency/dataset/Scenarios/LOWWD_20220318/"
    # CTG_pool_root_path = "E:/RL_N2_contingency/dataset/Scenarios/HIGHWD_20220320/"
    # CTG_pool_root_path = "E:/RL_N2_contingency/dataset/Scenarios/Contingency_Pool/"

    saw_case = SAW(FileName = env_file, 
                        CreateIfNotFound=True,
                        early_bind=True,)
    CTG_dict_selected = CTG_selector(
        CTG_dict =  CTG_dict, 
        saw_model = saw_case, 
        voltage_threshold = 138,
        CTG_pool_root_path = CTG_pool_root_path,
        CTG_pool_name = 'contingency_pool_v0_s138kV_noMaLo.con')
    a=0
