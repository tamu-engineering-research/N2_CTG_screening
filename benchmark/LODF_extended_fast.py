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
from numpy.linalg import multi_dot, det, solve, inv

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import env_tools
from env.builder_advanced import env_advanced
from env.env_tools import CTG_compiler, CTG_attribute_creator


class LODF_extended_fast():
    """LODF brute-force search that accomandates islanding cases"""
    ######## common modules ##########
    def __init__(self, root_path, scenario, approach, 
        sim_model_file, CTG_file, LODF_file, PTDF_file, CTG_attribute_file,
        df_branch_file, df_bus_file, fast_screening_file):
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
        self.fast_screening_file = root_path+"/"+scenario +"/"+approach+"/"+fast_screening_file
        if not os.path.exists(self.approach_path):
            os.makedirs(self.approach_path)
        pass

    def preprocessing(self, 
        flg_read_LODF=True, flg_read_PTDF=True, flg_read_CTG_attribute= True,
        flg_read_df_branch=True, flg_read_df_bus=True, ):

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
            # self.CTG_dict = CTG_compiler(CTG_file_list = [self.CTG_file])
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
    
    
    def fast_screening_preprocessing(self):
        """
        A generalized version of the meta-heuristic approach proposed in an MIT's paper "Fast and reliable screening of N-2 contingencies"
        """
        ## set NaN (open branch) in self.matrix_lodf as 0.0
        self.matrix_lodf[np.isnan(self.matrix_lodf)] = 0.0

        ## get list of combos that contain selected CTGs, e.g. high-voltage
        df_selected = self.CTG_dict_attribute.copy()
        df_selected = df_selected.loc[self.CTG_dict_attribute['voltage_kV']>=138].copy() # select high voltage CTGs
        df_selected = df_selected.loc[df_selected['feasible']].copy() # select CTGs that have feasible components in the PF model
        self.list_CTG_index_selected = df_selected.index
        print("Select "+str(len(self.list_CTG_index_selected))+" out of "+str(len(self.CTG_dict_attribute.index))+" contingencies")

        ## get list of branches whose power flow we care about, e.g. closed high-voltage branches
        self.list_branch_index_full = self.df_branch.index
        self.list_branch_index_closed = self.df_branch.loc[self.df_branch['LineStatus']=='Closed'].index
        self.list_branch_index_selected = self.df_branch.loc[(self.df_branch['LineStatus']=='Closed')&((self.df_branch['BusNomVolt']>=137)&(self.df_branch['BusNomVolt:1']>=137))].index
        print("Select "+str(len(self.list_branch_index_selected))+" out of "+str(len(self.list_branch_index_closed))+" closed branches, out of total "+str(len(self.df_branch.index))+" branches.")
        
        ## initialize variables
        count_CTG = len(self.CTG_dict_attribute.index)
        count_branch = len(self.df_branch.index)
        self.list_branch_position = self.df_branch[['BusNum_position', 'BusNum:1_position']].values.tolist()
        
        ## branch upper and lower limits
        thermal_rate_coef = 150/100
        branch_flow = self.df_branch['LineMW'].astype(float).to_numpy().flatten()
        branch_upper_limit = self.df_branch['LineLimMVA'].astype(float).to_numpy().flatten()*thermal_rate_coef
        branch_lower_limit = -self.df_branch['LineLimMVA'].astype(float).to_numpy().flatten()*thermal_rate_coef
        branch_upper_margin = branch_upper_limit - branch_flow
        branch_lower_margin = branch_flow - branch_lower_limit
        if np.any(branch_upper_margin<=0) or np.any(branch_lower_margin<=0):
            print("Find violation when no CTGs")
        branch_upper_margin_denominator = (1 / branch_upper_margin).reshape((-1,1))
        branch_lower_margin_denominator = (1 / branch_lower_margin).reshape((-1,1))

        # prepare xi_positive, xi_negative: count_branch * [count_branch * count_open_compenents]
        # prepare Gamma [count_CTG, count_CTG, count_open_components]
        print("############ Start preparing positive and negative xi ############")
        xi_positive = []
        xi_negative = []
        G = [[0]*count_CTG for _ in range(count_CTG)]
        solvable = np.ones((count_CTG, count_CTG)) - np.diag(np.ones(count_CTG))
        attribute_list_machine_id_position = self.CTG_dict_attribute['list_machine_id_position'].tolist()
        attribute_list_machine_MW = self.CTG_dict_attribute['list_machine_MW'].tolist()
        attribute_list_load_id_position = self.CTG_dict_attribute['list_load_id_position'].tolist()
        attribute_list_load_MW = self.CTG_dict_attribute['list_load_MW'].tolist()
        attribute_list_branch_id = self.CTG_dict_attribute['list_branch_id'].tolist()
        attribute_list_branch_attribute_position = self.CTG_dict_attribute['list_branch_attribute_position'].tolist()
        for i in tqdm(range(count_CTG)):
            xi_positive_tmp = []
            xi_negative_tmp = []
            gamma = []
            CTG_i_index = i
            CTG_dict_attribute_tmp = self.CTG_dict_attribute.loc[CTG_i_index].copy()
            branch_flow_tmp = np.copy(branch_flow)
            ## calculate xi derived from machines
            if CTG_dict_attribute_tmp['num_machine']>0:
                machine_bus_id = CTG_dict_attribute_tmp['list_machine_id_position']
                machine_MW = CTG_dict_attribute_tmp['list_machine_MW']
                machine_node_position = machine_bus_id #[list_bus.index(kk) for kk in machine_bus_id]
                PTDF_machine = self.matrix_ptdf[machine_node_position,:].T.copy()
                branch_flow_tmp += PTDF_machine @ np.array(machine_MW)
                delta_branch_flow_from_machine_each = PTDF_machine * np.array(machine_MW).reshape((1,-1))
                xi_positive_tmp +=delta_branch_flow_from_machine_each.T.tolist()
                xi_negative_tmp +=delta_branch_flow_from_machine_each.T.tolist()
                gamma +=[1]*len(machine_bus_id)
            ## calculate xi derived from loads
            if CTG_dict_attribute_tmp['num_load']>0:
                load_bus_id = CTG_dict_attribute_tmp['list_load_id_position']
                load_MW = CTG_dict_attribute_tmp['list_load_MW']
                load_node_position = load_bus_id #[list_bus.index(kk) for kk in load_bus_id]
                PTDF_load = self.matrix_ptdf[load_node_position,:].T.copy()
                branch_flow_tmp += - PTDF_load @ np.array(load_MW)
                delta_branch_flow_from_load_each = - PTDF_load * np.array(load_MW).reshape((1,-1))
                xi_positive_tmp += delta_branch_flow_from_load_each.T.tolist()
                xi_negative_tmp += delta_branch_flow_from_load_each.T.tolist()
                gamma += [1]*len(load_bus_id)
            ## calculate xi derived from branches
            if CTG_dict_attribute_tmp['num_branch']>0:
                ## calculate part of xi_positive and xi_negative
                open_branch_index = CTG_dict_attribute_tmp['list_branch_id']
                list_branch_attribute_position = CTG_dict_attribute_tmp['list_branch_attribute_position']
                flg_solved, delta_flow, delta_flow_each, injection_each = self.calculate_delta_flow(
                    list_branch_attribute_position, open_branch_index, branch_flow_tmp)
                xi_positive_tmp += delta_flow_each.T.tolist()
                xi_negative_tmp += delta_flow_each.T.tolist()
                if not flg_solved:
                    solvable[CTG_i_index, :] = 0
                
                gamma_branch_denominator = injection_each.copy() # M^-1*F with only CTG i

                ## prepare one row of Gamma, Gamma means interaction between CTG i and j
                for j in tqdm(range(count_CTG), leave=False):
                    CTG_j_index = j
                    if not (CTG_i_index == CTG_j_index):
                        list_branch_attribute_position = attribute_list_branch_attribute_position[CTG_i_index]+attribute_list_branch_attribute_position[CTG_j_index]#a+self.CTG_dict_attribute.loc[CTG_j_index]['list_branch_attribute_position']
                        open_branch_index = attribute_list_branch_id[CTG_i_index]+attribute_list_branch_id[CTG_j_index]#b+self.CTG_dict_attribute.loc[CTG_j_index]['list_branch_id']
                        # list_branch_attribute_position = self.CTG_dict_attribute.loc[CTG_i_index]['list_branch_attribute_position']+self.CTG_dict_attribute.loc[CTG_j_index]['list_branch_attribute_position']
                        # open_branch_index = self.CTG_dict_attribute.loc[CTG_i_index]['list_branch_id']+self.CTG_dict_attribute.loc[CTG_j_index]['list_branch_id']
                        
                        # get M
                        flg_solved, M, open_branch_index_nonisl = self.calculate_M(list_branch_attribute_position, open_branch_index)
                        if flg_solved:
                            # get branch concerned F
                            machine_id_position = attribute_list_machine_id_position[CTG_i_index]+attribute_list_machine_id_position[CTG_j_index]
                            load_id_position = attribute_list_load_id_position[CTG_i_index]+attribute_list_load_id_position[CTG_j_index]
                            machine_MW = attribute_list_machine_MW[CTG_i_index]+attribute_list_machine_MW[CTG_j_index]
                            load_MW = attribute_list_load_MW[CTG_i_index] + attribute_list_load_MW[CTG_j_index]
                            # machine_id_position = self.CTG_dict_attribute.loc[CTG_i_index]['list_machine_id_position']+self.CTG_dict_attribute.loc[CTG_j_index]['list_machine_id_position']
                            # load_id_position = self.CTG_dict_attribute.loc[CTG_i_index]['list_load_id_position']+self.CTG_dict_attribute.loc[CTG_j_index]['list_load_id_position']
                            # machine_MW = self.CTG_dict_attribute.loc[CTG_i_index]['list_machine_MW']+self.CTG_dict_attribute.loc[CTG_j_index]['list_machine_MW']
                            # load_MW = self.CTG_dict_attribute.loc[CTG_i_index]['list_load_MW']+self.CTG_dict_attribute.loc[CTG_j_index]['list_load_MW']

                            # F = branch_flow[open_branch_index_nonisl].copy()
                            F = self.calculate_F_for_gamma(# F with CTG i and j
                                branch_flow_tmp =  branch_flow[open_branch_index_nonisl].copy(), 
                                branch_index = open_branch_index_nonisl, 
                                machine_id_position = machine_id_position, 
                                load_id_position = load_id_position, 
                                machine_MW = machine_MW, 
                                load_MW = load_MW)

                            # get M-1* F with CTG i and j
                            gamma_numerator = np.linalg.inv(M)@ F
                            gamma_numerator = gamma_numerator[0:len(list(gamma_branch_denominator.ravel()))]
                            # get gamma
                            G[CTG_i_index][CTG_j_index] = gamma.copy()+list((gamma_numerator/gamma_branch_denominator.ravel()).ravel())
                        else:
                            solvable[CTG_i_index, CTG_j_index] = 0
                            G[CTG_i_index][CTG_j_index] = []
                        
                pass
            else: # if CTG i only has machine or load, no open branches
                branch_flow_ij_0 = np.copy(branch_flow)
                for j in tqdm(range(count_CTG)):
                    CTG_j_index = j
                    G[CTG_i_index][CTG_j_index] = gamma.copy()

            ## calculate xi_positive and xi_negative
            xi_positive.append(np.array(xi_positive_tmp).T * branch_upper_margin_denominator) 
            xi_negative.append(-np.array(xi_negative_tmp).T * branch_lower_margin_denominator) # NOTICE this negative sign
        with open(self.fast_screening_file, 'wb') as f:
            pickle.dump(
                {'xi_positive':xi_positive, 'xi_negative':xi_negative,
                'G':G, 'solvable':solvable},
                f)
    
    def fast_screening(self):
        ## set NaN (open branch) in self.matrix_lodf as 0.0
        self.matrix_lodf[np.isnan(self.matrix_lodf)] = 0.0

        ## get list of combos that contain selected CTGs, e.g. high-voltage
        df_selected = self.CTG_dict_attribute.copy()
        df_selected = df_selected.loc[self.CTG_dict_attribute['voltage_kV']>=137].copy() # select high voltage CTGs
        df_selected = df_selected.loc[df_selected['feasible']].copy() # select CTGs that have feasible components in the PF model
        self.list_CTG_index_selected = df_selected.index
        print("Select "+str(len(self.list_CTG_index_selected))+" out of "+str(len(self.CTG_dict_attribute.index))+" contingencies")

        ## get list of branches whose power flow we care about, e.g. closed high-voltage branches
        self.list_branch_index_full = self.df_branch.index
        self.list_branch_index_closed = self.df_branch.loc[self.df_branch['LineStatus']=='Closed'].index
        self.list_branch_index_selected = self.df_branch.loc[(self.df_branch['LineStatus']=='Closed')&((self.df_branch['BusNomVolt']>=137)|(self.df_branch['BusNomVolt:1']>=137))].index
        print("Select "+str(len(self.list_branch_index_selected))+" out of "+str(len(self.list_branch_index_closed))+" closed branches, out of total "+str(len(self.df_branch.index))+" branches.")
        
        # restore xi_postive, xi_negative, solvable, and G (Gamma)
        with open(self.fast_screening_file, 'rb') as f:
            fast_screening_var = pickle.load(f)
        xi_positive = fast_screening_var['xi_positive']
        xi_negative = fast_screening_var['xi_negative']
        solvable = fast_screening_var['solvable']
        G = fast_screening_var['G']
        count_CTG = len(self.CTG_dict_attribute.index)
        count_branch = len(self.df_branch.index)

        ## initialize S and C, (only keep high-voltage closed branches in S)
        ## REMEMBER to initialize S and C according to the selected CTG, commenet out 
        self.S = np.zeros([count_CTG, count_branch])
        self.S[np.ix_(self.list_CTG_index_selected, self.list_branch_index_selected)]=1
        self.C = np.zeros([count_CTG, count_CTG])
        self.C[np.ix_(self.list_CTG_index_selected, self.list_CTG_index_selected)]=1
        np.fill_diagonal(self.C, 0)
        self.C = np.tril(self.C, -1).T # upper triangle
        # for i in tqdm(range(solvable.shape[0])):
        #     for j in range(solvable.shape[1]):
        #         if not solvable[i,j]==1:
        #             self.C[i,j] = 0

        ## iterative
        iter_k = 10
        for i in tqdm(range(iter_k)):
            size_S_0 = np.sum(self.S.ravel())
            size_C_0 = np.sum(self.C.ravel())
            print("\n S size is "+str(size_S_0)+" and C size is "+str(size_C_0))

            # step 1  get xi_positive/negative_max/min
            xi_positive_max = []
            xi_positive_min = []
            xi_negative_max = []
            xi_negative_min = []
            for j in tqdm(range(count_CTG)):
                # get Z_alpha = {gamma |(alpha, gamma) in S}
                Z_index = list(np.nonzero(self.S[j].ravel())[0])
                if len(Z_index)>0:
                    xi_positive_max.append(np.amax(xi_positive[j][Z_index], axis=0))
                    xi_positive_min.append(np.amin(xi_positive[j][Z_index], axis=0))
                    xi_negative_max.append(np.amax(xi_negative[j][Z_index], axis=0)) 
                    xi_negative_min.append(np.amin(xi_negative[j][Z_index], axis=0))
                else:
                    xi_positive_max.append([])
                    xi_positive_min.append([])
                    xi_negative_max.append([]) 
                    xi_negative_min.append([])


            # step 1 get G_max/min
            G_max = []
            G_min = []
            for j in tqdm(range(count_CTG)):
                B_index = list(np.nonzero(self.C[j].ravel())[0])
                if len(B_index)>0:
                    G_max.append(np.amax(np.array(G[j])[B_index], axis=0))
                    G_min.append(np.amin(np.array(G[j])[B_index], axis=0))
                else:
                    G_max.append([])
                    G_min.append([])

            
            ## step 2 update C
            for j in tqdm(range(count_CTG)):
                for k in range(j, count_CTG):
                    if self.C[j][k]==1:
                        gamma_jk = list(G[j][k])
                        gamma_kj = list(G[k][j])
                        U_positive = 0
                        U_negative = 0
                        for index, l in enumerate(gamma_jk):
                            if l>0:
                                U_positive+=xi_positive_max[j][index]*l
                                U_negative+=xi_negative_max[j][index]*l
                            else:
                                U_positive+=xi_positive_min[j][index]*l
                                U_negative+=xi_negative_min[j][index]*l
                        for index, l in enumerate(gamma_kj):
                            if l>0:
                                U_positive+=xi_positive_max[k][index]*l
                                U_negative+=xi_negative_max[k][index]*l
                            else:
                                U_positive+=xi_positive_min[k][index]*l
                                U_negative+=xi_negative_min[k][index]*l
                        if not ((U_positive>1) or (U_negative>1)):
                            self.C[j][k]=0
            
            ## step 3 update S
            # for j in range(count_CTG):
            #     for k in range(count_branch):
            #         if self.S[j][k]==1:
            #             pass

            size_S = np.sum(self.S.ravel())
            size_C = np.sum(self.C.ravel())
            if (size_C==size_C_0) and (size_S==size_S_0):
                break
            pass 

        ## summarize fast screening results
        count_combo_to_check =  np.sum(self.C.ravel())

    def calculate_delta_flow(self, list_branch_attribute_position, open_branch_index, branch_flow_tmp):
        ## Only boundary and main should be considered, exclude open branches in islands
        list_branch_attribute_nonisl = []
        open_branch_index_nonisl = []
        for branch_attribute, branch_index in zip(list_branch_attribute_position, open_branch_index):
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
                node_position = absorb_bus_index #list_bus.index(absorb_bus_index) # BusNum not equal Node_index
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
                        node_from_position = bus_from #list_bus.index(bus_from) # BusNum not equal Node_index
                        node_to_position = bus_to #list_bus.index(bus_to) # BusNum not equal Node_index
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
                        node_from_position = bus_from #list_bus.index(bus_from) # BusNum not equal Node_index
                        node_to_position = bus_to #list_bus.index(bus_to) # BusNum not equal Node_index
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
                node_from_position = bus_from #list_bus.index(bus_from) # BusNum not equal Node_index
                node_to_position = bus_to #list_bus.index(bus_to) # BusNum not equal Node_index
                dominator = 1-(-self.matrix_ptdf[node_from_position, branch_index]+self.matrix_ptdf[node_to_position, branch_index])
                M[:,ii] = M[:,ii]/dominator
        L = np.array(L).T
        F = np.array(F)
        try:
            injection_each = (np.linalg.inv(M)@F).reshape((-1,1))
            delta_flow = L @ injection_each #L@np.linalg.inv(M)@F
            delta_flow_each = L* injection_each.reshape((1,-1))
            flg_solved = True
        except np.linalg.LinAlgError:
            delta_flow = 0
            delta_flow_each = 0
            injection_each = 0
            flg_solved = False
        return flg_solved, delta_flow, delta_flow_each, injection_each
    


    def calculate_M(self, list_branch_attribute_position, open_branch_index):
        ## Only boundary and main should be considered, exclude open branches in islands
        list_branch_attribute_nonisl = []
        open_branch_index_nonisl = []
        for branch_attribute, branch_index in zip(list_branch_attribute_position, open_branch_index):
            if not (branch_attribute[0] == "Island"):
                list_branch_attribute_nonisl.append(branch_attribute)
                open_branch_index_nonisl.append(branch_index)
        ## construct M
        M = np.zeros([len(open_branch_index_nonisl), len(open_branch_index_nonisl)])
        for ii in range(len(open_branch_index_nonisl)):
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
                        absorb_node_position = absorb_bus_index # BusNum not equal Node_index
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
                        absorb_node_position = absorb_bus_index #list_bus.index(absorb_bus_index) # BusNum not equal Node_index
                        M[ii,jj] = -self.matrix_ptdf[absorb_node_position, branch_index_0]
        # modify columns in M that correspond to MainSystem open branches
        for ii in range(len(open_branch_index_nonisl)):
            if list_branch_attribute_nonisl[ii][0]=="MainSystem":
                branch_index = open_branch_index_nonisl[ii]
                bus_from = self.list_branch_position[branch_index][0]
                bus_to = self.list_branch_position[branch_index][1]
                node_from_position = bus_from #list_bus.index(bus_from) # BusNum not equal Node_index
                node_to_position = bus_to #list_bus.index(bus_to) # BusNum not equal Node_index
                dominator = 1-(-self.matrix_ptdf[node_from_position, branch_index]+self.matrix_ptdf[node_to_position, branch_index])
                M[:,ii] = M[:,ii]/dominator
        try:
            M_inv = np.linalg.inv(M)
            flg_solved = True
        except np.linalg.LinAlgError:
            flg_solved = False
        return flg_solved, M, open_branch_index_nonisl

    def calculate_F_for_gamma(self, branch_flow_tmp, branch_index, machine_id_position, load_id_position, machine_MW, load_MW):
        ## calculate xi derived from machines
        if len(machine_id_position)>0:
            machine_node_position = machine_id_position #[list_bus.index(kk) for kk in machine_bus_id]
            PTDF_machine = self.matrix_ptdf[np.ix_(machine_node_position,branch_index)].T.copy()
            branch_flow_tmp += PTDF_machine @ np.array(machine_MW)
        ## calculate xi derived from loads
        if len(load_id_position)>0:
            load_node_position = load_id_position
            PTDF_load = self.matrix_ptdf[np.ix_(load_node_position,branch_index)].T.copy()
            branch_flow_tmp += - PTDF_load @ np.array(load_MW)
        return branch_flow_tmp

        
    def analysis(self):
        """LODF extended brute-force search that accomandate machine, load, and islanding issues."""
        
        ## find critical combos
        self.list_result_head = ['num_isolated_devices', 'total_thermal_violation',]
        self.list_result_value = []
        branch_thermal_limit = self.df_branch['LineLimMVA'].astype(float).to_numpy().flatten()*110/100
        branch_flow = self.df_branch['LineMW'].astype(float).to_numpy().flatten()
        list_bus = self.df_bus['BusNum'].tolist()
        list_branch = self.df_branch[['BusNum', 'BusNum:1']].values.tolist()
        num_combo_island = 0 # Single CTg does not result in islanding but both together do.
        
        
        count=1
        print("Start enumeration over "+str(count*(count-1)/2)+" combinations...")
        for i in tqdm(range(count-1)):#tqdm(range(count-1)):#####################
          for j in tqdm(range(i+1, count), leave=False):
            combo = [i ,j]
            list_result_value_tmp = []
            flg_islanding_tmp = False
            flg_islanding_combo_tmp  = False
            CTG_dict_attribute_tmp = self.CTG_dict_attribute.loc[combo].copy()
            if CTG_dict_attribute_tmp['num_branch'].sum()>0: # check if existing islands 
                if CTG_dict_attribute_tmp['flg_islanding'].any(): # single CTG can result in islanding
                    flg_islanding_tmp = True
                else:
                    ## calculate M to verify if existing islands
                    open_branch_index = CTG_dict_attribute_tmp.iloc[0]['list_branch_id']+CTG_dict_attribute_tmp.iloc[1]['list_branch_id']
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
                if CTG_dict_attribute_tmp['num_machine'].sum()>0:
                    machine_bus_id = CTG_dict_attribute_tmp.iloc[0]['list_machine_id']+CTG_dict_attribute_tmp.iloc[1]['list_machine_id']
                    machine_MW = CTG_dict_attribute_tmp.iloc[0]['list_machine_MW']+CTG_dict_attribute_tmp.iloc[1]['list_machine_MW']
                    machine_node_position = [list_bus.index(kk) for kk in machine_bus_id]
                    PTDF_machine = self.matrix_ptdf[machine_node_position,:].T.copy()
                    delta_branch_flow_from_machine = PTDF_machine @ np.array(machine_MW)
                    branch_flow_tmp += delta_branch_flow_from_machine
                ## calculate impact of loads
                if CTG_dict_attribute_tmp['num_load'].sum()>0:
                    load_bus_id = CTG_dict_attribute_tmp.iloc[0]['list_load_id']+CTG_dict_attribute_tmp.iloc[1]['list_load_id']
                    load_MW = CTG_dict_attribute_tmp.iloc[0]['list_load_MW']+CTG_dict_attribute_tmp.iloc[1]['list_load_MW']
                    load_node_position = [list_bus.index(kk) for kk in load_bus_id]
                    PTDF_load = self.matrix_ptdf[load_node_position,:].T.copy()
                    delta_branch_flow_from_load = - PTDF_load @ np.array(load_MW)
                    branch_flow_tmp += delta_branch_flow_from_load
                ## calculate impact of opening branches
                if CTG_dict_attribute_tmp['num_branch'].sum()>0:
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
                if CTG_dict_attribute_tmp['num_machine'].sum()>0:
                    machine_bus_id = CTG_dict_attribute_tmp.iloc[0]['list_machine_id']+CTG_dict_attribute_tmp.iloc[1]['list_machine_id']
                    machine_MW = CTG_dict_attribute_tmp.iloc[0]['list_machine_MW']+CTG_dict_attribute_tmp.iloc[1]['list_machine_MW']
                    machine_node_position = [list_bus.index(kk) for kk in machine_bus_id]
                    PTDF_machine = self.matrix_ptdf[machine_node_position,:].T.copy()
                    delta_branch_flow_from_machine = PTDF_machine @ np.array(machine_MW)
                    branch_flow_tmp += delta_branch_flow_from_machine
                ## calculate impact of loads
                if CTG_dict_attribute_tmp['num_load'].sum()>0:
                    load_bus_id = CTG_dict_attribute_tmp.iloc[0]['list_load_id']+CTG_dict_attribute_tmp.iloc[1]['list_load_id']
                    load_MW = CTG_dict_attribute_tmp.iloc[0]['list_load_MW']+CTG_dict_attribute_tmp.iloc[1]['list_load_MW']
                    load_node_position = [list_bus.index(kk) for kk in load_bus_id]
                    PTDF_load = self.matrix_ptdf[load_node_position,:].T.copy()
                    delta_branch_flow_from_load = - PTDF_load @ np.array(load_MW)
                    branch_flow_tmp += delta_branch_flow_from_load
                ## calculate impact of branches
                if CTG_dict_attribute_tmp['num_branch'].sum()>0:
                    ###############calculate_delta_flow#############
                    def calculate_delta_flow(list_branch_attribute, open_branch_index, branch_flow_tmp, list_bus, list_branch):
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
                                node_position = list_bus.index(absorb_bus_index) # BusNum not equal Node_index
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
                                        bus_from = list_branch[branch_index][0]
                                        bus_to = list_branch[branch_index][1]
                                        node_from_position = list_bus.index(bus_from) # BusNum not equal Node_index
                                        node_to_position = list_bus.index(bus_to) # BusNum not equal Node_index
                                        M[ii,jj] = 1-(-self.matrix_ptdf[node_from_position, branch_index]+self.matrix_ptdf[node_to_position, branch_index])
                                    elif list_branch_attribute_nonisl[jj][0]=="Boundary":
                                        branch_index = open_branch_index_nonisl[ii]
                                        absorb_bus_index = list_branch_attribute_nonisl[jj][1]
                                        absorb_node_position = list_bus.index(absorb_bus_index) # BusNum not equal Node_index
                                        if list_branch_attribute_nonisl[jj][2]=="FromBus":# Whether the boundary node is FromBus or ToBus matters
                                            M[ii,jj] = -1 - self.matrix_ptdf[absorb_node_position, branch_index]
                                        elif list_branch_attribute_nonisl[jj][2]=="ToBus":
                                            M[ii,jj] = 1 - self.matrix_ptdf[absorb_node_position, branch_index]
                                else:
                                    if list_branch_attribute_nonisl[jj][0]=="MainSystem":
                                        branch_index_0 = open_branch_index_nonisl[ii]
                                        branch_index = open_branch_index_nonisl[jj]
                                        bus_from = list_branch[branch_index][0]
                                        bus_to = list_branch[branch_index][1]
                                        node_from_position = list_bus.index(bus_from) # BusNum not equal Node_index
                                        node_to_position = list_bus.index(bus_to) # BusNum not equal Node_index
                                        M[ii,jj] = -(-self.matrix_ptdf[node_from_position, branch_index_0]+self.matrix_ptdf[node_to_position, branch_index_0])
                                    elif list_branch_attribute_nonisl[jj][0]=="Boundary":
                                        branch_index_0 = open_branch_index_nonisl[ii]
                                        absorb_bus_index = list_branch_attribute_nonisl[jj][1]
                                        absorb_node_position = list_bus.index(absorb_bus_index) # BusNum not equal Node_index
                                        M[ii,jj] = -self.matrix_ptdf[absorb_node_position, branch_index_0]
                        # modify columns in M that correspond to MainSystem open branches
                        for ii in range(len(open_branch_index_nonisl)):
                            if list_branch_attribute_nonisl[ii][0]=="MainSystem":
                                branch_index = open_branch_index_nonisl[ii]
                                bus_from = list_branch[branch_index][0]
                                bus_to = list_branch[branch_index][1]
                                node_from_position = list_bus.index(bus_from) # BusNum not equal Node_index
                                node_to_position = list_bus.index(bus_to) # BusNum not equal Node_index
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
                    open_branch_index = CTG_dict_attribute_tmp.iloc[0]['list_branch_id']+CTG_dict_attribute_tmp.iloc[1]['list_branch_id']
                    list_branch_attribute = CTG_dict_attribute_tmp.iloc[0]['list_branch_attribute']+CTG_dict_attribute_tmp.iloc[1]['list_branch_attribute']
                    list_node_in_island = CTG_dict_attribute_tmp.iloc[0]['list_node_in_island']+CTG_dict_attribute_tmp.iloc[1]['list_node_in_island']
                    num_isolated_devices = len(list_node_in_island)
                    flg_solved, delta_flow = calculate_delta_flow(list_branch_attribute, open_branch_index, branch_flow_tmp, list_bus, list_branch)
                    if flg_solved:
                        branch_flow_tmp+=delta_flow
                    else:
                        print("Cannot solve inverse matrix with calculated islands of single CTG. Try calculating islands of CTG combos")
                        ## identify islands, boundary, open-branches on the boundary or not
                        open_branch_index = CTG_dict_attribute_tmp.iloc[0]['list_branch_id']+CTG_dict_attribute_tmp.iloc[1]['list_branch_id']
                        open_branch_index = list(set(open_branch_index))
                        open_branch_index.sort()
                        list_branch_attribute, list_node_in_island = self.identify_islanding(open_branch_index = open_branch_index)
                        num_isolated_devices = len(list_node_in_island)
                        flg_solved, delta_flow = calculate_delta_flow(list_branch_attribute, open_branch_index, branch_flow_tmp, list_bus, list_branch)
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
    case_name = 'LowWind'
    root_path = 'E:/RL_N2_Contingency/dataset/Scenarios' #'/Users/xiangtianzheng/Project/RL_N2_contingency'#'E:/RL_N2_Contingency/dataset/Scenarios'
    scenario = cases[case_name]['scenario']
    approach = "LODF_extended_fast"
    sim_model_file = cases[case_name]['sim_model_file']
    CTG_file = cases[case_name]['CTG_file']
    LODF_file = "matrix_LODF.pkl"
    PTDF_file = "matrix_PTDF.pkl"
    CTG_attribute_file = "CTG_attribute.pkl"
    df_branch_file = "df_branch.pkl"
    df_bus_file = "df_bus.pkl"
    fast_screening_file = 'fast_screening_file.pkl'

    method = LODF_extended_fast(
        root_path = root_path,
        scenario = scenario, 
        approach = approach, 
        sim_model_file = sim_model_file, 
        CTG_file = CTG_file, 
        LODF_file = LODF_file,
        PTDF_file = PTDF_file,
        CTG_attribute_file=CTG_attribute_file,
        df_branch_file=df_branch_file,
        df_bus_file=df_bus_file,
        fast_screening_file = fast_screening_file)

    method.preprocessing(
        flg_read_LODF= True,)

    method.fast_screening_preprocessing()

    method.fast_screening()

    # method.analysis()
    pass