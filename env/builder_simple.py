import os
import sys

import numpy as np
import pandas as pd
from esa import SAW
from tqdm import tqdm
import yacs

sys.path.append('.')
from env_tools import simple_state_reader, simple_env_modifier, simple_simulator
from env_tools import CTG_compiler
from env_tools import sim_PWD_compiler
from utils import BaseEnvModel



class env_simple():
    """
    Create an interactive model, including intialization, interaction, evaluation
    """
    def __init__(
        self, cfg = None
        ):
        self.cfg = cfg
        self.TASK_FUNCTION_MAP = TASK_FUNCTION_MAP
        self.dict_key_fields = DICT_KEY_FIELDS

    def initialize_env(
        self, env_path=None
        ):
        """
        Description: Initialize an environment of one simulation case. 
        Parameters: env_path: string.
        Returns: action_list: list. 
        """
        #### create interactive environment for power flow ####
        self.env_original = SAW(FileName=env_path, CreateIfNotFound=True, early_bind=True)
        self.env_original.RunScriptCommand("EnterMode(EDIT);")
        self.env_interactive = SAW(FileName=env_path, CreateIfNotFound=True, early_bind=True)
        #### calculate initial state of system model ####
        action_list, state_initialized, flg_solved = simple_simulator(
            saw_case = self.env_interactive, dict_key_fields = self.dict_key_fields,
            action_present = [], action_cumulative = [], 
            state_full = None, state_scenario = None)
        return action_list, state_initialized, flg_solved
        
    def interact_env(
        self, action_present = None, state_full = None,
        action_cumulative = None, state_scenario = None, 
        flg_restore=False,
        ):
        """
        Description: Interact with the environment
        Input:
            action_present:
            state_full:
            action_cumulative:
            state_scenario:
            flg_restore:
        Output:
        """
        # store the 
        env_original = self.env_interactive
        # initialize output variables
        flg_solved = False
        # modify the saw case according to contingencies actions
        action_dict, state_updated, flg_solved = simple_simulator(
            self.env_interactive, dict_key_fields = self.dict_key_fields,
            action_present = action_present, action_cumulative = [], 
            state_full = None, state_scenario = None)
        if flg_restore:
            self.env_interactive = env_original
        return action_dict, state_updated, flg_solved
    
    def quit_env(self):
        self.env_original.exit()

    def eval_violation(self):
        pass
        

#############  Only for internal test  #############

def _test_env_simple():
    # obtain the contingency list
    CTG_file_list = []
    CTG_file_list.append('E:/RL_N2_Contingency/dataset/Scenarios/HWLL_20211224/HWLL_20211224.con') # 5665 contingencies
    CTG_list = CTG_compiler(CTG_file_list)
    # start
    sim_model_path = 'E:/RL_N2_Contingency/dataset/Scenarios/HWLL_20211224/HWLL_20211224_v30.pwb'
    
    env = env_simple()
    env.initialize_env(env_path = sim_model_path)
    for CTG_single in tqdm(CTG_list['components']):
        env.interact_env(action_present = CTG_single, flg_restore=True)
    return


if __name__=="__main__":
    # _test_env_creater()
    _test_env_simple()