import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
from esa import SAW
from tqdm import tqdm
import yacs

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from env_tools import advanced_simulator, advanced_CTG_loader, advanced_CTG_restore
    from env_tools import CTG_compiler
    from env_tools import sim_PWD_compiler
except NameError:
    from .env_tools import advanced_simulator, advanced_CTG_loader, advanced_CTG_restore
    from .env_tools import CTG_compiler
    from .env_tools import sim_PWD_compiler
from utils import BaseEnvModel

class env_advanced(BaseEnvModel):
    """
    Create an interactive model, including intialization, interaction, evaluation
    """
    def __init__(self, 
        env_file = None,
        CTG_file = None,
        func_metric = None):
        self.initialize(env_file = env_file,
                        CTG_file = CTG_file,
                        func_metric = func_metric)

    def initialize(self, 
                env_file = None,
                CTG_file = None,
                func_metric = None):
        """
        Description: 
            Initialize an environment of one simulation case. 
        Parameters: 
            env_file: string, location of PWD file.
            CTG_file: string, location of contingency pool .con file.
        Returns:
            None 
        """
        if func_metric is not None:
            self.func_metric = func_metric
        self.env_interact = SAW(FileName = env_file, 
                                CreateIfNotFound=True,
                                early_bind=True,)
        advanced_CTG_loader(saw_case = self.env_interact,
                            CTG_file = CTG_file,)
        return
        
    def simulate(self, 
                CTG_name_list = None,
                flg_restore = False,):
        """
        Description: Interact with the environment
        Parameters:
            CTG_name_list: list, a list of (string) contingency names that should be in the imported .con file 
            flg_restore: boolean, whether to restore to the reference 
        Returns:
        """
        # initialize output variables
        flg_solved = False
        # modify the saw case according to contingencies actions
        flg_solved = advanced_simulator(saw_case = self.env_interact,
                                        CTG_name_list = CTG_name_list,)
        if flg_restore:
            advanced_CTG_restore(self.env_interact)
        return flg_solved
    
    def restore(self):
        advanced_CTG_restore(saw_case = self.env_interact)
    
    def quit(self):
        self.env_interact.exit()
        return

    def run_power_flow(self):
        self.env_interact.SolvePowerFlow()
    
    def order(self, flg=True):
        self.env_interact.pw_order = flg

    def evaluate(self, 
                state = None,
                state_margin = None,):
        if state is None:
            state = self.state.copy()
        if state_margin is None:
            state_margin = self.state_margin.copy()
        score = self.func_metric(state = state,
                                state_margin = state_margin,)
        return score

#############  Only for internal test  #############

def _test_env_advanced():
    # initialize some parameters
    sim_model_path = 'E:/RL_N2_Contingency/dataset/Scenarios/HWLL_20211224/HWLL_20211224_v30.pwb'
    CTG_pool_path = 'E:/RL_N2_contingency/dataset/Scenarios/Contingency_Pool/contingency_pool_v0.con'
    CTG_dict = CTG_compiler(CTG_file_list = [CTG_pool_path]) #{'name': xx, 'components': xx}
    func_metric = None
    ## initialize advanced environment
    env = env_advanced(
        env_file = sim_model_path,
        CTG_file = CTG_pool_path,
        func_metric = func_metric,)

    # ## test speed of simulating single contingency analysis: ~15 min
    # for CTG_single in tqdm(CTG_dict['name']): # CTG_single example: "'DABNWS28'"
    #     env.simulate(CTG_name_list = [CTG_single[1:-1]])

    # ## test restore: it works successful!!
    # env.run_power_flow()
    # state0 = env.read_state()
    # env.simulate(CTG_name_list = ['DABNWS28'])
    # state1 = env.read_state()
    # ca_results = env.read_CA_results()
    # env.restore()
    # state2 = env.read_state()

    # ## test saw built-in contingency analysis module: temporarily error?
    # env.env_interact.run_contingency_analysis(option='N-2', validate=True)

    # ## test saw built-in GetParametersMultipleElement('Contingency', ['Name', 'Solved', 'Violations']): it works!
    # env.env_interact.pw_order = True
    # env.simulate(CTG_name_list = ['DABNWS28'])
    # ca_results = env.read_CA_results()

    ## test saw built-in get_LODF and its speed: creating LODF in pw is fast, the problem here is df too large 
    start = time.time()
    env.env_interact.pw_order = True
    statement = "CalculateLODFMatrix(OUTAGES,ALL,ALL,YES,DC,ALL,YES)"
    env.env_interact.RunScriptCommand(statement)
    count = env.env_interact.ListOfDevices('branch').shape[0]
    end1 = time.time()
    array = [f"LODFMult:{x}" for x in range(1000)]
    df = env.env_interact.GetParametersMultipleElement('branch', array)
    end2 = time.time()
    return


if __name__=="__main__":
    _test_env_advanced()