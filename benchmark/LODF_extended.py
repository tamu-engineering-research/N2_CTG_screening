import os
import sys
import time
import pickle
from datetime import datetime
import copy

import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx

sys.path.append(os.path.abspath(os.getcwd()))
from env import env_tools
from env.builder_advanced import env_advanced
from env.env_tools import CTG_compiler, CTG_attribute_creator


class LODF_extended_BF():
    """LODF brute-force search that accomandates islanding cases"""
    ######## common modules ##########
    def __init__(self, root_path, scenario, approach,
        sim_model_file, CTG_file, LODF_file, PTDF_file, CTG_attribute_file,
        df_branch_file, df_bus_file):
        self.root_path = root_path+"/"
        self.scenario_path = root_path+"/"+scenario+"/"
        self.approach_path = root_path+"/"+scenario +"/"+approach+ "/"
        self.sim_model_file = root_path+"/"+scenario+"/"+sim_model_file
        self.CTG_file = root_path+"/"+scenario+"/"+CTG_file
        self.LODF_file = root_path+"/"+scenario +"/"+approach+"/"+LODF_file
        self.PTDF_file = root_path+"/"+scenario +"/"+approach+"/"+PTDF_file
        self.CTG_attribute_file = root_path+"/"+scenario +"/"+approach+"/"+CTG_attribute_file
        self.df_branch_file = root_path+"/"+scenario +"/"+approach+"/"+df_branch_file
        self.df_bus_file = root_path+"/"+scenario +"/"+approach+"/"+df_bus_file
        if not os.path.exists(self.approach_path):
            os.makedirs(self.approach_path)
        pass

    def preprocessing(self, flg_read_LODF=True, flg_read_PTDF=True, flg_read_CTG_attribute= True, flg_read_df_branch=True, flg_read_df_bus=True, ):
        ## prepare env
        print("*****************************************************")
        print("Start preparing interactive simulation environment ...")
        self.env = env_advanced(
            env_file = self.sim_model_file,
            CTG_file = self.CTG_file,
            func_metric = None,)
        self.env.env_interact.pw_order=True
        self.env.run_power_flow()
        print("Finish preparing interactive simulation environment.")
        print("*****************************************************")

        ## create attributes of power world model
        print("*****************************************************")
        print("Start creating attibutes of power flow model...")
        if flg_read_df_branch and flg_read_df_bus and os.path.exists(self.df_branch_file):
            self.restore_df_branch(save_file=self.df_branch_file)
            self.restore_df_bus(save_file=self.df_bus_file)
        else:
            self.env.env_interact.pw_order=True
            self.df_branch = self.env.env_interact.GetParametersMultipleElement("branch", ['BusNum', 'BusNum:1', 'LineCircuit', 'LineMW', 'LineLimMVA','BusNomVolt','BusNomVolt:1','LineStatus'])
            self.df_bus = self.env.env_interact.GetParametersMultipleElement('bus', ['BusNum', 'BusNomVolt','BusSlack'])
            self.df_branch['BusNum'] = self.df_branch['BusNum'].astype(int)
            self.df_branch['BusNum:1'] = self.df_branch['BusNum:1'].astype(int)
            self.df_branch['LineCircuit'] = self.df_branch['LineCircuit'].astype(int)
            self.df_branch['BusNomVolt'] = self.df_branch['BusNomVolt'].astype(float)
            self.df_branch['BusNomVolt:1'] = self.df_branch['BusNomVolt:1'].astype(float)
            self.df_bus['BusNum'] = self.df_bus['BusNum'].astype(int)
            self.df_bus['BusNomVolt'] = self.df_bus['BusNomVolt'].astype(float)
            self.save_df_branch(save_file=self.df_branch_file)
            self.save_df_bus(save_file=self.df_bus_file)
        self.slack_bus = self.df_bus.loc[self.df_bus['BusSlack']=="YES"].index
        print("Finish creating attibutes of power flow model.")
        print("*****************************************************")
        
        ## get LODF
        print("*****************************************************")
        if flg_read_LODF and os.path.exists(self.LODF_file):
            print("Start reading LODF..")
            self.restore_LODF(save_file = self.LODF_file)
            print("Finish reading LODF.")
        else:
            print("Start calculating LODF.")
            self.get_LODF_v2()
            self.save_LODF(save_file = self.LODF_file)
            print("Finish calculating LODF.")
        print("*****************************************************")
        
        ## get PTDF
        print("*****************************************************")
        if flg_read_PTDF and os.path.exists(self.PTDF_file):
            print("Start reading PTDF...")
            self.restore_PTDF(save_file = self.PTDF_file)
            print("Finish reading PTDF.")
        else:
            print("Start calculating PTDF...")
            self.get_PTDF()
            self.save_PTDF(save_file = self.PTDF_file)
            print("Finish calculating PTDF.")
        print("*****************************************************")
        
        ## create a graph
        self.create_graph()
        
        ## get CTGs attributes
        print("*****************************************************")
        if flg_read_CTG_attribute and os.path.exists(self.CTG_attribute_file):
            print("Start reading CTG attributes...")
            self.CTG_dict = CTG_compiler(CTG_file_list = [self.CTG_file])
            self.restore_CTG_attribute(save_file = self.CTG_attribute_file)
            print("Finish reading CTG attributes.")
        else:
            print("Start calculating CTG attributes...")
            self.CTG_dict = CTG_compiler(CTG_file_list = [self.CTG_file])
            self.create_CTG_attribute()
            self.save_CTG_attribute(save_file = self.CTG_attribute_file)
            print("Finish calculating CTG attributes.")
        print("*****************************************************")

        ## update CTG attribute, df_branch
        self.update_CTG_attribute()
        self.update_df_branch()
        return

    def analysis(self):
        """LODF extended brute-force search that accomandate machine, load, and islanding issues."""
        ## set NaN (open branch) in self.matrix_lodf as 0.0
        self.matrix_lodf[np.isnan(self.matrix_lodf)] = 0.0
        
        ## get list of combos that contain selected CTGs, e.g. high-voltage
        df_selected = self.CTG_dict_attribute.copy()
        df_selected = df_selected.loc[self.CTG_dict_attribute['voltage_kV']>=138].copy() # select high voltage CTGs
        df_selected = df_selected.loc[df_selected['feasible']].copy() # select CTGs that have feasible components in the PF model
        self.list_CTG_index_selected = df_selected.index
        count = len(self.list_CTG_index_selected)
        # self.list_combo_CTG_index = [[self.list_CTG_index_selected[i], self.list_CTG_index_selected[j]] for i in range(count-1) for j in range(i+1, count)]
        print("Select "+str(len(self.list_CTG_index_selected))+" out of "+str(len(self.CTG_dict_attribute.index))+" contingencies")

        ## get list of branches whose power flow we care about, e.g. closed high-voltage branches
        self.list_branch_index_full = self.df_branch.index
        self.list_branch_index_closed = self.df_branch.loc[self.df_branch['LineStatus']=='Closed'].index
        self.list_branch_index_selected = self.df_branch.loc[(self.df_branch['LineStatus']=='Closed')&((self.df_branch['BusNomVolt']>=137)&(self.df_branch['BusNomVolt:1']>=137))].index
        print("Select "+str(len(self.list_branch_index_selected))+" out of "+str(len(self.list_branch_index_closed))+" closed branches, out of total "+str(len(self.df_branch.index))+" branches.")
        
        ## find critical combos
        self.list_result_head = ['num_isolated_devices', 'total_thermal_violation',]
        self.list_result_value = []
        self.list_branch_position = self.df_branch[['BusNum_position', 'BusNum:1_position']].values.tolist()
        self.list_bus_num = self.df_bus['BusNum'].tolist()
        branch_thermal_limit = self.df_branch['LineLimMVA'].astype(float).to_numpy().flatten()*110/100
        branch_flow = self.df_branch['LineMW'].astype(float).to_numpy().flatten()
        list_branch = self.df_branch[['BusNum', 'BusNum:1']].values.tolist()
        attribute_list_machine_id_position = self.CTG_dict_attribute['list_machine_id_position'].tolist()
        attribute_list_machine_MW = self.CTG_dict_attribute['list_machine_MW'].tolist()
        attribute_list_load_id_position = self.CTG_dict_attribute['list_load_id_position'].tolist()
        attribute_list_load_MW = self.CTG_dict_attribute['list_load_MW'].tolist()
        attribute_list_branch_id = self.CTG_dict_attribute['list_branch_id'].tolist()
        attribute_list_branch_attribute_position = self.CTG_dict_attribute['list_branch_attribute_position'].tolist()
        attribute_flg_islanding = self.CTG_dict_attribute['flg_islanding'].tolist()
        attribute_list_node_in_island = self.CTG_dict_attribute['list_node_in_island'].tolist()
        num_combo_island = 0 # Single CTg does not result in islanding but both together do.
        print("Start enumeration over "+str(int(count*(count-1)/2))+" combinations...")
        for i in tqdm(range(count-1)):
          for j in tqdm(range(i+1, count), leave=False):
            combo = [self.list_CTG_index_selected[i], self.list_CTG_index_selected[j]] #[i ,j]
            list_result_value_tmp = []
            flg_islanding_tmp = False
            flg_islanding_combo_tmp  = False
            combo_list_machine_id_position = attribute_list_machine_id_position[combo[0]]+attribute_list_machine_id_position[combo[1]]
            combo_list_machine_MW = attribute_list_machine_MW[combo[0]]+attribute_list_machine_MW[combo[1]]
            combo_list_load_id_position = attribute_list_load_id_position[combo[0]]+attribute_list_load_id_position[combo[1]]
            combo_list_load_MW = attribute_list_load_MW[combo[0]] + attribute_list_load_MW[combo[1]]
            combo_list_branch_id = attribute_list_branch_id[combo[0]] + attribute_list_branch_id[combo[1]]
            combo_list_branch_attribute_position = attribute_list_branch_attribute_position[combo[0]] + attribute_list_branch_attribute_position[combo[1]]
            combo_list_node_in_island = attribute_list_node_in_island[combo[0]] + attribute_list_node_in_island[combo[1]]

            if len(combo_list_branch_id)>0: # check if existing islands 
                if attribute_flg_islanding[combo[0]] or attribute_flg_islanding[combo[1]]: # single CTG can result in islanding
                    flg_islanding_tmp = True
                else:
                    ## calculate M to verify if existing islands
                    open_branch_index = combo_list_branch_id
                    open_branch_index = list(set(open_branch_index))
                    open_branch_index.sort()
                    M = np.array(self.matrix_lodf[np.ix_(open_branch_index, open_branch_index)]).copy()
                    M = -M.T
                    if (abs(np.linalg.det(M))<=1e-8): # non-invertible M means islanding
                        flg_islanding_tmp = True
                        flg_islanding_combo_tmp = True
                        num_combo_island +=1
                    elif (np.isnan(abs(np.linalg.det(M)))):
                        flg_islanding_tmp = True
                        flg_islanding_combo_tmp = True
                        num_combo_island +=1
            if not flg_islanding_tmp:# The simplest case: no splitted subgraph
                branch_flow_tmp = np.copy(branch_flow)
                ## calculate impact of machines
                if len(combo_list_machine_id_position)>0:
                    PTDF_machine = self.matrix_ptdf[combo_list_machine_id_position,:].T.copy()
                    delta_branch_flow_from_machine = PTDF_machine @ np.array(combo_list_machine_MW)
                    branch_flow_tmp += delta_branch_flow_from_machine
                ## calculate impact of loads
                if len(combo_list_load_id_position)>0:
                    PTDF_load = self.matrix_ptdf[combo_list_load_id_position,:].T.copy()
                    delta_branch_flow_from_load = - PTDF_load @ np.array(combo_list_load_MW)
                    branch_flow_tmp += delta_branch_flow_from_load
                ## calculate impact of opening branches
                if len(combo_list_branch_id)>0:
                    ## calculate imapct of branches, note that necessary to imcorporate the impact of 
                    L = self.matrix_lodf[np.ix_(open_branch_index, self.list_branch_index_full)]
                    L = L.T
                    F = branch_flow_tmp[open_branch_index]
                    delta_branch_flow_from_branch = L@np.linalg.inv(M)@F
                    branch_flow_tmp +=delta_branch_flow_from_branch
                ## evaluate number of isolated devices
                num_isolated_devices = 0 
            else: # results in splitted subgraph
                branch_flow_tmp = np.copy(branch_flow)
                ## calculate impact of machines
                if len(combo_list_machine_id_position)>0:
                    PTDF_machine = self.matrix_ptdf[combo_list_machine_id_position,:].T.copy()
                    delta_branch_flow_from_machine = PTDF_machine @ np.array(combo_list_machine_MW)
                    branch_flow_tmp += delta_branch_flow_from_machine
                ## calculate impact of loads
                if len(combo_list_load_id_position)>0:
                    PTDF_load = self.matrix_ptdf[combo_list_load_id_position,:].T.copy()
                    delta_branch_flow_from_load = - PTDF_load @ np.array(combo_list_load_MW)
                    branch_flow_tmp += delta_branch_flow_from_load
                ## calculate impact of branches
                if len(combo_list_branch_id)>0:
                    ###############calculate_delta_flow#############
                    def calculate_delta_flow(list_branch_attribute, open_branch_index, branch_flow_tmp,):
                        ## Only boundary and main should be considered, exclude open branches in islands
                        list_branch_attribute_nonisl = []
                        open_branch_index_nonisl = []
                        for branch_attribute, branch_index in zip(list_branch_attribute, open_branch_index):
                            if not (branch_attribute[0] == "Island"):
                                list_branch_attribute_nonisl.append(branch_attribute)
                                open_branch_index_nonisl.append(branch_index)
                        ## construct L, M and F, delta_F = L*M^-1*F
                        M = np.zeros([len(open_branch_index_nonisl), len(open_branch_index_nonisl)])
                        L = []
                        F = []
                        for ii in range(len(open_branch_index_nonisl)):
                            # construct one column in L
                            if list_branch_attribute_nonisl[ii][0]=="MainSystem":# get L from lodf
                                branch_index = open_branch_index_nonisl[ii]
                                L.append(list(self.matrix_lodf[branch_index, self.list_branch_index_full]))
                            elif list_branch_attribute_nonisl[ii][0]=="Boundary":# get L from ptdf
                                absorb_bus_index = list_branch_attribute_nonisl[ii][1]
                                node_position = absorb_bus_index#list_bus.index(absorb_bus_index) # BusNum not equal Node_index
                                L.append(list(self.matrix_ptdf[node_position, self.list_branch_index_full]))
                            else:
                                print("unexpected type of open branch")
                            # construct on entry in column vector F
                            F.append(branch_flow_tmp[open_branch_index_nonisl[ii]])
                            # construct one row in M
                            for jj in range(len(open_branch_index_nonisl)):
                                if ii==jj:
                                    if list_branch_attribute_nonisl[jj][0]=="MainSystem":
                                        branch_index = open_branch_index_nonisl[ii]
                                        bus_from = self.list_branch_position[branch_index][0]
                                        bus_to = self.list_branch_position[branch_index][1]
                                        node_from_position = bus_from#list_bus.index(bus_from) # BusNum not equal Node_index
                                        node_to_position = bus_to#list_bus.index(bus_to) # BusNum not equal Node_index
                                        M[ii,jj] = 1-(-self.matrix_ptdf[node_from_position, branch_index]+self.matrix_ptdf[node_to_position, branch_index])
                                    elif list_branch_attribute_nonisl[jj][0]=="Boundary":
                                        branch_index = open_branch_index_nonisl[ii]
                                        absorb_bus_index = list_branch_attribute_nonisl[jj][1]
                                        absorb_node_position = absorb_bus_index#list_bus.index(absorb_bus_index) # BusNum not equal Node_index
                                        if list_branch_attribute_nonisl[jj][2]=="FromBus":# Whether the boundary node is FromBus or ToBus matters
                                            M[ii,jj] = -1 - self.matrix_ptdf[absorb_node_position, branch_index]
                                        elif list_branch_attribute_nonisl[jj][2]=="ToBus":
                                            M[ii,jj] = 1 - self.matrix_ptdf[absorb_node_position, branch_index]
                                else:
                                    if list_branch_attribute_nonisl[jj][0]=="MainSystem":
                                        branch_index_0 = open_branch_index_nonisl[ii]
                                        branch_index = open_branch_index_nonisl[jj]
                                        bus_from = self.list_branch_position[branch_index][0]
                                        bus_to = self.list_branch_position[branch_index][1]
                                        node_from_position = bus_from#list_bus.index(bus_from) # BusNum not equal Node_index
                                        node_to_position = bus_to#list_bus.index(bus_to) # BusNum not equal Node_index
                                        M[ii,jj] = -(-self.matrix_ptdf[node_from_position, branch_index_0]+self.matrix_ptdf[node_to_position, branch_index_0])
                                    elif list_branch_attribute_nonisl[jj][0]=="Boundary":
                                        branch_index_0 = open_branch_index_nonisl[ii]
                                        absorb_bus_index = list_branch_attribute_nonisl[jj][1]
                                        absorb_node_position = absorb_bus_index#list_bus.index(absorb_bus_index) # BusNum not equal Node_index
                                        M[ii,jj] = -self.matrix_ptdf[absorb_node_position, branch_index_0]
                        # modify columns in M that correspond to MainSystem open branches
                        for ii in range(len(open_branch_index_nonisl)):
                            if list_branch_attribute_nonisl[ii][0]=="MainSystem":
                                branch_index = open_branch_index_nonisl[ii]
                                bus_from = self.list_branch_position[branch_index][0]
                                bus_to = self.list_branch_position[branch_index][1]
                                node_from_position = bus_from#list_bus.index(bus_from) # BusNum not equal Node_index
                                node_to_position = bus_to#list_bus.index(bus_to) # BusNum not equal Node_index
                                dominator = 1-(-self.matrix_ptdf[node_from_position, branch_index]+self.matrix_ptdf[node_to_position, branch_index])
                                M[:,ii] = M[:,ii]/dominator
                        L = np.array(L).T
                        F = np.array(F)
                        try:
                            delta_flow = L@np.linalg.inv(M)@F
                            flg_solved = True
                        except np.linalg.LinAlgError:
                            delta_flow = 0
                            flg_solved = False
                            
                        return flg_solved, delta_flow

                    ## get islands, boundary, open-branches on the boundary from self.CTG_dict_attribute
                    open_branch_index = combo_list_branch_id
                    list_branch_attribute_position = combo_list_branch_attribute_position
                    list_node_in_island = combo_list_node_in_island
                    num_isolated_devices = len(list_node_in_island)
                    flg_solved, delta_flow = calculate_delta_flow(list_branch_attribute_position, open_branch_index, branch_flow_tmp)
                    if flg_solved:
                        branch_flow_tmp+=delta_flow
                    else:
                        print("Cannot solve inverse matrix with calculated islands of single CTG. Try calculating islands of CTG combos")
                        ## identify islands, boundary, open-branches on the boundary or not
                        open_branch_index = combo_list_branch_id
                        open_branch_index = list(set(open_branch_index))
                        open_branch_index.sort()
                        list_branch_attribute, list_node_in_island = self.identify_islanding(open_branch_index = open_branch_index)
                        num_isolated_devices = len(list_node_in_island) # list_node_in_island is NOT list_node_in_island_position
                        list_branch_attribute_position = []
                        for ii in list_branch_attribute:
                            if ii[0]=='Boundary':
                                list_branch_attribute_position.append([ii[0], self.list_bus_num.index(ii[1]), ii[2]])
                            else:
                                list_branch_attribute_position.append(ii)
                        flg_solved, delta_flow = calculate_delta_flow(list_branch_attribute_position, open_branch_index, branch_flow_tmp)
                        if flg_solved:
                            branch_flow_tmp+=delta_flow
                        else:
                            print("Combo of CTG "+str(i)+" and "+str(j)+" cannot be solved by two island identification methods.")
                            num_isolated_devices = 100
                else:
                    num_isolated_devices = 0
                    print("Impossible! M matrix indicates islanding but no island found!")

            ## evaluate the severity of thermal violation
            list_thermal_violation = np.abs(branch_flow_tmp[self.list_branch_index_selected])-branch_thermal_limit[self.list_branch_index_selected] # only high-voltage closed branches considered
            total_thermal_violation = np.sum(np.maximum(list_thermal_violation, np.zeros(len(list_thermal_violation))))
            self.list_result_value.append([num_isolated_devices, total_thermal_violation])
        print("Finish enumeration over "+str(count*(count-1)/2)+" combinations.")

        ## save results
        date_time = datetime.now().strftime("%Y%m%d_%I_%M_%p")
        save_file = self.approach_path+"result_"+date_time+".pkl"
        self.save_results(save_file = save_file)
        print("Finish saving results to "+save_file)
    
    def save_results(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(
                {'CTG_dict':self.CTG_dict,
                'CTG_dict_attribute':self.CTG_dict_attribute,
                'df_branch':self.df_branch, # new var
                'df_bus':self.df_bus,# new var
                'list_branch_index_selected':self.list_branch_index_selected,# new var
                'list_CTG_index_selected': self.list_CTG_index_selected,# new var
                'list_result_head':self.list_result_head,
                'list_result_value':self.list_result_value},f)

    def restore_results(self, save_file):
        with open(save_file, 'rb') as f:
            self.result = pickle.load(f)

    def create_CTG_attribute(self):
        self.CTG_dict_attribute = CTG_attribute_creator(
            CTG_dict = self.CTG_dict, 
            saw_model = self.env.env_interact)
        self.CTG_dict_attribute["flg_islanding"] = self.check_CTG_contain_islanding()
        print("Start identifying islands derived from signle CTGs....")
        df_list_branch_attribute, df_list_node_in_island = self.identify_islanding_single_CTG()
        self.CTG_dict_attribute["list_branch_attribute"] = df_list_branch_attribute
        self.CTG_dict_attribute['list_node_in_island'] = df_list_node_in_island
        print("Finish identifying islands derived from signle CTGs.")

    def update_CTG_attribute(self):
        df_busnum = self.df_bus['BusNum'].tolist()
        # update list_machine_id_position, list_load_id_position
        list_machine_id_position = []
        list_load_id_position = []
        for i in range(len(self.CTG_dict_attribute.index)):
            machine_id = self.CTG_dict_attribute.loc[i]['list_machine_id']
            load_id = self.CTG_dict_attribute.loc[i]['list_load_id']
            machine_id_position = []
            load_id_position = []
            if len(machine_id)>0:
                for ii in machine_id:
                    machine_id_position.append(df_busnum.index(ii))
            if len(load_id)>0:
                for ii in load_id:
                    load_id_position.append(df_busnum.index(ii))
            list_machine_id_position.append(machine_id_position)
            list_load_id_position.append(load_id_position)
        self.CTG_dict_attribute['list_machine_id_position'] = list_machine_id_position
        self.CTG_dict_attribute['list_load_id_position'] = list_load_id_position
        # update list_branch_attribute_position, list_node_in_island_position
        list_branch_attribute_position = []
        list_node_in_island_position = []
        for i in range(len(self.CTG_dict_attribute.index)):
            branch_attribute = self.CTG_dict_attribute.loc[i]['list_branch_attribute']
            node_in_island = self.CTG_dict_attribute.loc[i]['list_node_in_island']
            branch_attribute_position = []
            node_in_island_position = []
            for ii in node_in_island:
                node_in_island_position.append(df_busnum.index(ii))
            for ii in branch_attribute:
                if ii[0]=='Boundary':
                    branch_attribute_position.append([ii[0], df_busnum.index(ii[1]), ii[2]])
                else:
                    branch_attribute_position.append(ii)
            list_branch_attribute_position.append(branch_attribute_position)
            list_node_in_island_position.append(node_in_island_position)
        self.CTG_dict_attribute['list_branch_attribute_position'] = list_branch_attribute_position
        self.CTG_dict_attribute['list_node_in_island_position'] = list_node_in_island_position

    def update_df_branch(self):
        # update df branch
        df_branch_bus0_position = []
        df_branch_bus1_position = []
        df_branch_bus0 = self.df_branch['BusNum'].tolist()
        df_branch_bus1 = self.df_branch['BusNum:1'].tolist()
        df_busnum = self.df_bus['BusNum'].tolist()
        for i in df_branch_bus0:
            df_branch_bus0_position.append(df_busnum.index(i))
        for i in df_branch_bus1:
            df_branch_bus1_position.append(df_busnum.index(i))
        self.df_branch['BusNum_position'] = df_branch_bus0_position
        self.df_branch['BusNum:1_position'] = df_branch_bus1_position


    ######### ad hoc modules ###########
    
    def get_LODF_v2(self):
        """get Original LODF matrix from PW"""
        self.matrix_lodf = []
        saw_case = self.env.env_interact
        saw_case.pw_order = True
        list_branch_num = self.df_branch[['BusNum','BusNum:1','LineCircuit','LineStatus']].values.tolist()
        count_branch = len(list_branch_num)
        for branch_num in tqdm(list_branch_num):
            statement = "CalculateLODF([BRANCH "+str(branch_num[0])+" "+str(branch_num[1])+" "+str(branch_num[2])+"],DC)"
            saw_case.RunScriptCommand(statement)
            df_tmp = saw_case.GetParametersMultipleElement('branch', ['LineLODF'])
            if branch_num[-1]=="Closed":
                self.matrix_lodf.append((df_tmp.to_numpy(dtype=float)/100).flatten().tolist())
            elif branch_num[-1]=="Open": # pw calculate LCDF if open, so we manually calculate it
                row = [0.0]*count_branch
                self.matrix_lodf.append(row)
        self.matrix_lodf = np.array(self.matrix_lodf)
        self.matrix_lodf[np.diag_indices_from(self.matrix_lodf)] = -1.0
        

    def get_PTDF(self): # TODO:read ptdf from  
        """get Original PTDF of slack bus and other buses from PW"""
        self.matrix_ptdf = []
        saw_case = self.env.env_interact
        saw_case.pw_order = True
        list_bus_num = self.df_bus['BusNum'].tolist()
        for bus_num in tqdm(list_bus_num):
            statement = "CalculatePTDF([SLACK],[BUS "+str(bus_num)+"],DC)"
            saw_case.RunScriptCommand(statement)
            df_tmp = saw_case.GetParametersMultipleElement('branch', ['LinePTDF'])
            self.matrix_ptdf.append((df_tmp.to_numpy(dtype=float)/100).flatten().tolist())
        self.matrix_ptdf = np.array(self.matrix_ptdf)

    def save_PTDF(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(self.matrix_ptdf, f)

    def restore_PTDF(self, save_file):
        with open(save_file, 'rb') as f:
            self.matrix_ptdf = pickle.load(f)

    def save_LODF(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(self.matrix_lodf, f)

    def restore_LODF(self, save_file):
        with open(save_file, 'rb') as f:
            self.matrix_lodf = pickle.load(f)
    
    def save_CTG_attribute(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(self.CTG_dict_attribute, f)

    def restore_CTG_attribute(self, save_file):
        with open(save_file, 'rb') as f:
            self.CTG_dict_attribute = pickle.load(f)
    
    def save_df_branch(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(self.df_branch, f)

    def restore_df_branch(self, save_file):
        with open(save_file, 'rb') as f:
            self.df_branch = pickle.load(f)

    def save_df_bus(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(self.df_bus, f)

    def restore_df_bus(self, save_file):
        with open(save_file, 'rb') as f:
            self.df_bus = pickle.load(f)

    def check_CTG_contain_islanding(self):
        list_flg_CTG_islanding = [False]*len(self.CTG_dict["name"])
        list_flg_branch_islanding = np.any(self.matrix_lodf>=10, axis=1)
        df_branch = self.df_branch.copy()
        for i in tqdm(range(len(self.CTG_dict["name"]))):
            flg_tmp = False
            for name, action in self.CTG_dict['components'][i].items():
                if (action[0] =="BRANCH"):
                    bus_num1 = action[1]
                    bus_num2 = action[2]
                    ckt_num = action[3]
                    df_tmp = df_branch.loc[(df_branch['BusNum']==bus_num1)&(df_branch['BusNum:1']==bus_num2)&(df_branch['LineCircuit']==ckt_num)].copy()
                    if df_tmp.empty:
                        df_tmp = df_branch.loc[(df_branch['BusNum']==bus_num2)&(df_branch['BusNum:1']==bus_num1)&(df_branch['LineCircuit']==ckt_num)].copy()
                    if not df_tmp.empty:
                        index_tmp = df_tmp.index[0]
                        if list_flg_branch_islanding[index_tmp]: # if having any line resulting in islanding
                            flg_tmp = True
                            break
                    else:
                        flg_tmp = True
                        break
            list_flg_CTG_islanding[i] = flg_tmp
        print("Find "+str(list_flg_CTG_islanding.count(True))+" out of "+str(len(self.CTG_dict["name"]))+" single contingencies that can result in islanding")
        return list_flg_CTG_islanding
    
    def identify_islanding_single_CTG(self):
        df_list_branch_attribute = []
        df_list_node_in_island = []
        for i in tqdm(range(len(self.CTG_dict_attribute.index))):
            if self.CTG_dict_attribute.loc[i]["num_branch"]==0:
                df_list_branch_attribute.append([])
                df_list_node_in_island.append([])
            else:
                open_branch_index = self.CTG_dict_attribute.loc[i]["list_branch_id"]
                list_branch_attribute, list_node_in_island = self.identify_islanding(open_branch_index = open_branch_index)
                df_list_branch_attribute.append(list_branch_attribute)
                df_list_node_in_island.append(list_node_in_island)
        return df_list_branch_attribute, df_list_node_in_island


    def create_graph(self):
        ## build a simple graph that reflects the connectivity
        df_branch_closed = self.df_branch.loc[self.df_branch['LineStatus']=="Closed"].copy()
        self.graph_simple = nx.from_pandas_edgelist(df_branch_closed, "BusNum", "BusNum:1", create_using=nx.MultiGraph)

    def identify_islanding(self, open_branch_index):
        # get set of nodes
        open_branches = self.df_branch.loc[open_branch_index].copy()
        list_node = list(set(open_branches['BusNum'].tolist() + open_branches['BusNum:1'].tolist()))
        list_branch_bidirection = open_branches[['BusNum', 'BusNum:1']].values.tolist()+open_branches[['BusNum:1', 'BusNum']].values.tolist()
        # traverse the graph and classify nodes into "MainSystem" or "Island"
        list_node_attribute = []
        list_node_in_island = []
        for node_tmp in list_node:
            depth_limit = 15 # eprical setting, assuming any island cannot have diameter more than 10
            flg_islanding = True
            # start deep-first-search
            list_node_index = [node_tmp]
            list_node_dist_to_source = [0]
            front, rear = 0, 0
            while front<=rear:
                try: #list_node_index[front] may not exist, because self.graph_simple excludes all pre-CTG isolated nodes 
                    next_branches = list(self.graph_simple.edges(list_node_index[front]))
                except:
                    next_branches = []
                for next_branch in next_branches:
                    if not(list(next_branch) in list_branch_bidirection): # not open
                        from_node_dist = list_node_dist_to_source[front]
                        to_node_dist = from_node_dist + 1
                        if not (next_branch[1] in list_node_index): # if new node
                            list_node_index.append(next_branch[1])
                            list_node_dist_to_source.append(to_node_dist)
                            rear += 1
                            if to_node_dist>=depth_limit:
                                flg_islanding = False
                                break
                if flg_islanding==False:
                    break
                front +=1
            # create node attribute
            if flg_islanding:
                list_node_attribute.append('Island')
                list_node_in_island += list_node_index
            else:
                list_node_attribute.append('MainSystem')
        list_node_in_island = list(set(list_node_in_island))
        # classify branch according to attributes of nodes into "boundary", 'main', 'island'
        list_branch_attribute = []
        for open_branch in open_branches[['BusNum', 'BusNum:1']].values.tolist():
            node_from = open_branch[0]
            node_to = open_branch[1]
            if list_node_attribute[list_node.index(node_from)]=='MainSystem':
                if list_node_attribute[list_node.index(node_to)]=='MainSystem':
                    list_branch_attribute.append(['MainSystem', 0 , 'None'])
                elif list_node_attribute[list_node.index(node_to)]=='Island':
                    list_branch_attribute.append(['Boundary', node_from, 'FromBus'])
            elif list_node_attribute[list_node.index(node_from)]=='Island':
                if list_node_attribute[list_node.index(node_to)]=='MainSystem':
                    list_branch_attribute.append(['Boundary', node_to, 'ToBus'])
                elif list_node_attribute[list_node.index(node_to)]=='Island':
                    list_branch_attribute.append(['Island', 0 , 'None'])
        return list_branch_attribute, list_node_in_island


if __name__=="__main__":
    # cases
    cases = {
        'HighWind':{
            'scenario': "HIGHWD_20220320",
            'sim_model_file':'HIGHWD_03_20_2022_2010v30.pwb',
            'CTG_file':'HIGHWD_03_20_2022_2010.con',
        },
        'LowLoad':{
            'scenario': "LOWLD_20220320",
            'sim_model_file':'LOWLD_03_20_2022_0328v30.pwb',
            'CTG_file':'LOWLD_03_20_2022_0328.con',
        },
        'HighLoad':{
            'scenario': "HIGHLD_20210824",
            'sim_model_file':'HIGHLD_08_24_2021_1655v30.pwb',
            'CTG_file':'HIGHLD_08_24_2021_1655.con',
        },
        'LowWind':{
            'scenario': "LOWWD_20220318",
            'sim_model_file':'LOWWD_03_18_2022_2303v30.pwb',
            'CTG_file':'LOWWD_03_18_2022_2303.con',
        },
    }
    # initialize some parameters
    root_path = 'E:/RL_N2_Contingency/dataset/Scenarios'
    case_name = 'HighLoad'
    scenario = cases[case_name]['scenario']
    approach = "LODF_extended_BF"
    sim_model_file = cases[case_name]['sim_model_file']
    CTG_file = cases[case_name]['CTG_file']
    LODF_file = "matrix_LODF.pkl"
    PTDF_file = "matrix_PTDF.pkl"
    CTG_attribute_file = "CTG_attribute.pkl"
    df_branch_file = "df_branch.pkl"
    df_bus_file = "df_bus.pkl"

    method = LODF_extended_BF(
        root_path = root_path, 
        scenario = scenario, 
        approach = approach, 
        sim_model_file = sim_model_file, 
        CTG_file = CTG_file, 
        LODF_file = LODF_file,
        PTDF_file = PTDF_file,
        CTG_attribute_file=CTG_attribute_file,
        df_branch_file=df_branch_file,
        df_bus_file=df_bus_file)

    method.preprocessing(
        flg_read_LODF= True,)

    method.analysis()
    pass