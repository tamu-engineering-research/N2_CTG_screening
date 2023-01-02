import os
import sys
import time

import numpy as np
import pandas as pd
from esa import SAW
from tqdm import tqdm
import yacs, yaml

def simple_env_modifier(saw_case, action_dict, dict_key_fields):
    #### get key fields ####
    bus_key_fields_wr = dict_key_fields['bus_writable']
    branch_key_fields_wr = dict_key_fields['branch_writable']
    gen_key_fields_wr = dict_key_fields['gen_writable']
    load_key_fields_wr = dict_key_fields['load_writable']
    #### read df of key fields ####
    df_gen = saw_case.GetParametersMultipleElement('gen', gen_key_fields_wr)
    df_branch = saw_case.GetParametersMultipleElement('branch', branch_key_fields_wr)
    df_load = saw_case.GetParametersMultipleElement('load', load_key_fields_wr)
    #### change components according to actions ####
    for name, action in action_dict.items():
        if action[0] == "BRANCH":
            bus_num1 = action[1]
            bus_num2 = action[2]
            ckt_num = action[3]
            df_tmp = df_branch.loc[(df_branch['BusNum']==bus_num1)&(df_branch['BusNum:1']==bus_num2)&(df_branch['LineCircuit']==str(ckt_num))].copy()
            if df_tmp.empty:
                df_tmp = df_branch.loc[(df_branch['BusNum']==bus_num2)&(df_branch['BusNum:1']==bus_num1)&(df_branch['LineCircuit']==str(ckt_num))].copy()
            if not df_tmp.empty:
                df_tmp['LineStatus'] = 'Open'
                saw_case.change_and_confirm_params_multiple_element('branch', df_tmp)
        elif action[0] == "MACHINE":
            bus_num = action[1]
            machine_num = action[2]
            df_tmp = df_gen.loc[(df_gen['BusNum']==bus_num)&(df_gen['GenID']==machine_num)].copy()
            if not df_tmp.empty:
                df_tmp['GenStatus'] = 'Open'
                saw_case.change_and_confirm_params_multiple_element('gen', df_tmp)
        elif action[0] == "LOAD":
            bus_num = action[1]
            load_num = action[2]
            df_tmp = df_load.loc[(df_load['BusNum']==bus_num)&(df_load['LoadID']==load_num)].copy()
            if not df_tmp.empty:
                df_tmp['LoadStatus'] = 'Open'
                saw_case.change_and_confirm_params_multiple_element('load', df_tmp)
        else:
            print('Unexpected type of contingency')
    return saw_case

def simple_env_modifier_fast(saw_case, action_dict, dict_key_fields):
    #### change components according to actions ####
    for name, action in action_dict.items():
        if action[0] == "BRANCH":
            try:
                d = {'BusNum':[action[1]], 'BusNum:1':[action[2]], 'LineCircuit':[str(action[3])], 'LineStatus':['Open']}
                saw_case.change_and_confirm_params_multiple_element('branch', pd.DataFrame(data=d))
            except:
                d = {'BusNum':[action[2]], 'BusNum:1':[action[1]], 'LineCircuit':[str(action[3])], 'LineStatus':['Open']}
                saw_case.change_and_confirm_params_multiple_element('branch', pd.DataFrame(data=d))
        elif action[0] == "MACHINE":
            d = {'BusNum':[action[1]], 'GenID':[action[2]], 'GenStatus':['Open']}
            saw_case.change_and_confirm_params_multiple_element('gen', pd.DataFrame(data=d))
        elif action[0] == "LOAD":
            d = {'BusNum':[action[1]], 'LoadID':[action[2]], 'LoadStatus':['Open']}
            saw_case.change_and_confirm_params_multiple_element('load', pd.DataFrame(data=d))
        else:
            print('Unexpected type of contingency')
    return saw_case


def simple_state_reader(saw_case, dict_key_fields):
    #### get variables according to key fields ####
    df_bus = saw_case.GetParametersMultipleElement('bus', dict_key_fields['bus_readable'])
    df_branch = saw_case.GetParametersMultipleElement('branch', dict_key_fields['branch_readable'])
    df_gen = saw_case.GetParametersMultipleElement('gen', dict_key_fields['gen_readable'])
    dict_state = {'bus':df_bus, 'branch':df_branch, 'gen':df_gen}
    return dict_state

def simple_simulator(
    saw_case,
    dict_key_fields, 
    action_present,
    action_cumulative, 
    state_full = None,
    state_scenario = None
    ):
    #### modify the saw case with removal of components based on actions ####
    start = time.time()
    if not (len(action_present)==0):
        action_dict = action_present #action_cumulative + action_present
        # saw_case_mdf = simple_env_modifier(saw_case, action_dict, dict_key_fields)
        saw_case_mdf = simple_env_modifier_fast(saw_case, action_dict, dict_key_fields)
    else:
        action_dict = {}
        saw_case_mdf = saw_case
    end = time.time()
    # print(end-start)
    #### simulate via PowerWorld ####
    start = time.time()
    try:
        saw_case_mdf.SolvePowerFlow()
        state_tmp = saw_case_mdf.get_power_flow_results('bus')
        flg_solved = True
        state_updated = None #simple_state_reader(saw_case = saw_case_mdf) 
    except:
        flg_solved = False
        state_updated = None
    end = time.time()
    # print(end-start)
    return action_dict, state_updated, flg_solved


